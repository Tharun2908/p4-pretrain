"""M6: Compare native BF16 and Transformer Engine BF16 on GPT-125M."""

import torch
import transformer_engine.pytorch as te

from model import GPT
from model_configs import MODELS


def convert_internal_linears(model):
    count = 0

    for block in model.transformer.h:
        locations = (
            (block.attn, "c_attn"),
            (block.attn, "c_proj"),
            (block.mlp, "c_fc"),
            (block.mlp, "c_proj"),
        )

        for parent, name in locations:
            original = getattr(parent, name)

            replacement = te.Linear(
                original.in_features,
                original.out_features,
                bias=original.bias is not None,
                params_dtype=original.weight.dtype,
                device=original.weight.device,
            )

            with torch.no_grad():
                replacement.weight.copy_(original.weight)
                if original.bias is not None:
                    replacement.bias.copy_(original.bias)

            assert torch.equal(replacement.weight, original.weight)
            if original.bias is not None:
                assert torch.equal(replacement.bias, original.bias)

            setattr(parent, name, replacement)
            count += 1

    return count


def relative_l2(actual, reference):
    return (
        (actual.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1e-12)
    ).item()


def main():
    torch.manual_seed(42)

    device = "cuda"
    cfg = MODELS["gpt-125m"]

    model = GPT(cfg).to(device)

    original_params = model.num_params(non_embedding=False)

    x = torch.randint(0, 50257, (1, 128), device=device)
    targets = torch.randint(0, 50257, (1, 128), device=device)

    print("=== M6 GPT-125M Transformer Engine BF16 ===")
    print("Model parameters:", original_params)
    print("Input shape:", tuple(x.shape))

    # Baseline: original PyTorch GPT
    with torch.autocast("cuda", dtype=torch.bfloat16):
        native_logits, native_loss = model(x, targets)

    native_loss.backward()

    native_logits = native_logits.detach().clone()
    native_loss_value = native_loss.item()

    native_grad = (
        model.transformer.h[0]
        .attn.c_attn.weight.grad.detach().clone()
    )

    print("Native BF16 loss:", native_loss_value)

    model.zero_grad(set_to_none=True)

    # Convert the four internal projections per block
    converted = convert_internal_linears(model)

    assert converted == 4 * cfg.n_layer

    assert (
        model.num_params(non_embedding=False)
        == original_params
    ), "Parameter count changed"

    assert (
        model.lm_head.weight
        is model.transformer.wte.weight
    ), "Embedding/output weight tying broken"

    print("TE linear layers:", converted)
    print("Parameter count preserved: True")
    print("Weight tying preserved: True")

    # Framework-matched BF16 control
    with torch.autocast("cuda", dtype=torch.bfloat16):
        te_logits, te_loss = model(x, targets)

    te_loss.backward()
    torch.cuda.synchronize()

    te_grad = (
        model.transformer.h[0]
        .attn.c_attn.weight.grad
    )

    assert torch.isfinite(te_logits).all()
    assert torch.isfinite(te_loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )

    print("TE BF16 loss:", te_loss.item())

    print(
        "Logits relative L2 error:",
        relative_l2(te_logits, native_logits),
    )

    print(
        "QKV weight-gradient relative L2 error:",
        relative_l2(te_grad, native_grad),
    )

    # Confirm a real optimizer update
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    weight = model.transformer.h[0].attn.c_attn.weight
    before = weight.detach().clone()

    optimizer.step()
    torch.cuda.synchronize()

    weight_change = (
        weight.detach() - before
    ).abs().max().item()

    print("Maximum QKV weight update:", weight_change)

    assert weight_change > 0

    print("\n[PASS] GPT-125M TE BF16 integration smoke test")


if __name__ == "__main__":
    main()
