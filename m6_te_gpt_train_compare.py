"""M6: controlled 12-step GPT-125M precision comparison."""

import gc
import torch
import transformer_engine.pytorch as te

from transformer_engine.common.recipe import (
    DelayedScaling,
    Format,
    NVFP4BlockScaling,
)
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

from model import GPT
from model_configs import MODELS
from m6_precision import convert_internal_linears


STEPS = 12
SEQUENCE_LENGTH = 128
SEED = 42


def relative_l2(actual, reference):
    actual = actual.float()
    reference = reference.float()
    return (
        (actual - reference).norm()
        / reference.norm().clamp_min(1e-12)
    ).item()


def run_mode(name, recipe, cfg, initial_state, batches):
    FP8GlobalStateManager.reset()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = GPT(cfg).to("cuda")
    model.load_state_dict(initial_state, strict=True)

    converted = convert_internal_linears(model)
    assert converted == 4 * cfg.n_layer

    te_layers = [
        module for module in model.modules()
        if isinstance(module, te.Linear)
    ]

    assert model.lm_head.weight is model.transformer.wte.weight

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    losses = []
    grad_norms = []
    first_logits = None
    first_grad = None

    torch.cuda.reset_peak_memory_stats()

    print(f"\n=== {name} ===", flush=True)

    for step, (x_cpu, y_cpu) in enumerate(batches):
        x = x_cpu.to("cuda")
        targets = y_cpu.to("cuda")

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            with te.autocast(
                enabled=recipe is not None,
                recipe=recipe,
            ):
                logits, loss = model(x, targets)

        assert torch.isfinite(loss), f"{name}: non-finite loss"

        loss.backward()

        for param_name, param in model.named_parameters():
            assert param.grad is not None, param_name
            assert torch.isfinite(param.grad).all(), param_name

        if step == 0:
            first_logits = logits.detach().float().cpu().clone()
            first_grad = (
                model.transformer.h[0]
                .attn.c_attn.weight.grad.detach().float().cpu().clone()
            )

            active = sum(layer.fp8 for layer in te_layers)

            expected = 0 if recipe is None else len(te_layers)

            assert active == expected, (
                f"{name}: active={active}, expected={expected}"
            )

            print(
                f"Quantized TE layers: {active}/{len(te_layers)}",
                flush=True,
            )

        # Global gradient norm BEFORE clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True,
        )

        optimizer.step()

        losses.append(loss.item())
        grad_norms.append(grad_norm.item())

        if step in (0, 1, 2, 5, 8, 11):
            print(
                f"Step {step:2d} | "
                f"loss={loss.item():.6f} | "
                f"grad_norm={grad_norm.item():.4f}",
                flush=True,
            )

    torch.cuda.synchronize()

    assert model.lm_head.weight is model.transformer.wte.weight

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    print(f"Peak allocated memory: {peak_gb:.3f} GiB")
    print(f"[PASS] {name}: {STEPS} optimizer steps")

    result = {
        "losses": losses,
        "grad_norms": grad_norms,
        "first_logits": first_logits,
        "first_grad": first_grad,
    }

    del model, optimizer
    gc.collect()
    FP8GlobalStateManager.reset()
    torch.cuda.empty_cache()

    return result


def main():
    torch.manual_seed(SEED)

    cfg = MODELS["gpt-125m"]

    print("=== M6 controlled training comparison ===")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", cfg.name)
    print("Steps:", STEPS)
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Micro-batch size: 1")
    print("Precision boundary: BF16 for all TE linears")

    # CPU initialization shared across all experiments.
    initial_model = GPT(cfg)
    initial_state = {
        key: value.detach().clone()
        for key, value in initial_model.state_dict().items()
    }
    del initial_model

    # Exactly the same token batches, in the same order.
    generator = torch.Generator(device="cpu").manual_seed(2027)

    batches = [
        (
            torch.randint(
                0, 50257, (1, SEQUENCE_LENGTH),
                generator=generator,
            ),
            torch.randint(
                0, 50257, (1, SEQUENCE_LENGTH),
                generator=generator,
            ),
        )
        for _ in range(STEPS)
    ]

    recipes = {
        "BF16": None,
        "FP8": DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=16,
        ),
        "NVFP4": NVFP4BlockScaling(),
    }

    results = {}

    for name, recipe in recipes.items():
        results[name] = run_mode(
            name, recipe, cfg, initial_state, batches
        )

    print("\n=== Loss trajectory comparison ===")
    print("Step     BF16         FP8          NVFP4")

    for step in range(STEPS):
        print(
            f"{step:3d}   "
            f"{results['BF16']['losses'][step]:.6f}   "
            f"{results['FP8']['losses'][step]:.6f}   "
            f"{results['NVFP4']['losses'][step]:.6f}"
        )

    print("\n=== First-step numerical differences ===")

    ref = results["BF16"]

    for name in ("FP8", "NVFP4"):
        result = results[name]

        output_error = relative_l2(
            result["first_logits"], ref["first_logits"]
        )

        gradient_error = relative_l2(
            result["first_grad"], ref["first_grad"]
        )

        print(
            f"{name}: "
            f"logits_error={output_error:.4%}, "
            f"QKV_grad_error={gradient_error:.4%}"
        )

    print("\n[PASS] Controlled 12-step comparison completed")


if __name__ == "__main__":
    main()
