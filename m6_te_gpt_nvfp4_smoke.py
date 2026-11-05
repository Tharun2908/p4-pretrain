"""M6: GPT-125M NVFP4 full-model training smoke test."""

import torch
import transformer_engine.pytorch as te

from transformer_engine.common.recipe import NVFP4BlockScaling
from model import GPT
from model_configs import MODELS
from m6_te_gpt_smoke import convert_internal_linears


def relative_l2(actual, reference):
    actual = actual.detach().float()
    reference = reference.detach().float()
    return (
        (actual - reference).norm()
        / reference.norm().clamp_min(1e-12)
    ).item()


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    available, reason = te.is_nvfp4_available(return_reason=True)
    assert available, reason

    cfg = MODELS["gpt-125m"]
    model = GPT(cfg).to("cuda")

    original_params = model.num_params(non_embedding=False)
    converted = convert_internal_linears(model)

    te_layers = [
        m for m in model.modules()
        if isinstance(m, te.Linear)
    ]

    assert converted == 48
    assert len(te_layers) == 48
    assert model.num_params(non_embedding=False) == original_params
    assert model.lm_head.weight is model.transformer.wte.weight

    x = torch.randint(
        0, 50257, (1, 128), device="cuda"
    )
    targets = torch.randint(
        0, 50257, (1, 128), device="cuda"
    )

    print("=== M6 GPT-125M NVFP4 smoke test ===")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model parameters:", original_params)
    print("TE linear layers:", len(te_layers))

    # NVFP4 RHT requires BF16 inputs.
    # Apply identical input casts to TE BF16 and NVFP4.
    def ensure_bf16_input(module, inputs):
        return (inputs[0].to(torch.bfloat16),)

    for te_layer in te_layers:
        te_layer.register_forward_pre_hook(ensure_bf16_input)

    print("TE projection inputs: explicitly BF16")

    # Reference: identical model weights and tokens, TE BF16
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

    # Full-model NVFP4 execution
    recipe = NVFP4BlockScaling()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        with te.autocast(enabled=True, recipe=recipe):
            nvfp4_logits, nvfp4_loss = model(x, targets)

    nvfp4_loss.backward()
    torch.cuda.synchronize()

    assert torch.isfinite(nvfp4_logits).all()
    assert torch.isfinite(nvfp4_loss)

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing grad: {name}"
        assert torch.isfinite(param.grad).all(), \
            f"Non-finite grad: {name}"

    active_layers = [
        m for m in te_layers
        if m.fp8 and isinstance(
            m.fp8_meta.get("recipe"),
            NVFP4BlockScaling,
        )
    ]

    print("NVFP4-active TE layers:", len(active_layers))

    assert len(active_layers) == 48

    nvfp4_grad = (
        model.transformer.h[0]
        .attn.c_attn.weight.grad
    )

    print("\n=== Numerical comparison ===")
    print(f"TE BF16 loss:  {ref_loss_value:.6f}")
    print(f"TE NVFP4 loss: {nvfp4_loss.item():.6f}")

    print(
        "Logits relative L2 error:",
        relative_l2(nvfp4_logits, ref_logits),
    )

    print(
        "QKV gradient relative L2 error:",
        relative_l2(nvfp4_grad, ref_grad),
    )

    # Optimizer update
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

    print("\n[PASS] GPT-125M NVFP4 training smoke test")


if __name__ == "__main__":
    main()
