"""M6: GPT-125M full-model FP8 training smoke test."""

import torch
import transformer_engine.pytorch as te

from transformer_engine.common.recipe import DelayedScaling, Format
from model import GPT
from model_configs import MODELS
from m6_te_gpt_smoke import convert_internal_linears


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
    count = convert_internal_linears(model)

    te_layers = [
        m for m in model.modules()
        if isinstance(m, te.Linear)
    ]

    assert count == 48
    assert len(te_layers) == 48

    assert (
        model.lm_head.weight
        is model.transformer.wte.weight
    )

    x = torch.randint(
        0, 50257, (1, 128), device=device
    )

    targets = torch.randint(
        0, 50257, (1, 128), device=device
    )

    print("=== M6 GPT-125M FP8 smoke test ===")
    print("TE linear layers:", len(te_layers))

    # Reference: TE BF16, identical weights and tokens.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref_logits, ref_loss = model(x, targets)

    ref_loss.backward()

    ref_logits = ref_logits.detach().clone()
    ref_loss_value = ref_loss.item()

    ref_grad = (
        model.transformer.h[0]
        .attn.c_attn.weight.grad.detach().clone()
    )

    model.zero_grad(set_to_none=True)

    recipe = DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=16,
    )

    # FP8 forward and backward.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with te.autocast(enabled=True, recipe=recipe):
            fp8_logits, fp8_loss = model(x, targets)

    fp8_loss.backward()
    torch.cuda.synchronize()

    fp8_grad = (
        model.transformer.h[0]
        .attn.c_attn.weight.grad
    )

    assert torch.isfinite(fp8_logits).all()
    assert torch.isfinite(fp8_loss)

    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name

    active_count = sum(layer.fp8 for layer in te_layers)

    print("FP8-active TE layers:", active_count)

    assert active_count == 48

    print("\n=== Numerical comparison ===")
    print(f"TE BF16 loss: {ref_loss_value:.6f}")
    print(f"TE FP8 loss:  {fp8_loss.item():.6f}")

    print(
        "Logits relative L2 error:",
        relative_l2(fp8_logits, ref_logits),
    )

    print(
        "QKV gradient relative L2 error:",
        relative_l2(fp8_grad, ref_grad),
    )

    # Verify optimizer update.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3
    )

    weight = model.transformer.h[0].attn.c_attn.weight
    before = weight.detach().clone()

    optimizer.step()
    torch.cuda.synchronize()

    change = (
        weight.detach() - before
    ).abs().max().item()

    print("Maximum QKV weight update:", change)

    assert change > 0
    assert model.lm_head.weight is model.transformer.wte.weight

    print("\n[PASS] GPT-125M FP8 training smoke test")


if __name__ == "__main__":
    main()
