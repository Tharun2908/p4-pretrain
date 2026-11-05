"""M5: BF16 + FlashAttention + TP gradient clipping smoke."""

import os
import torch
import torch.distributed as dist

from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.distributed.tensor import DTensor

from m5_tp_optimizer_smoke import build_models
from m5_tp_clip_smoke import tp_clip_grad_norm


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group("nccl", device_id=device)

    try:
        assert dist.get_world_size() == 2

        reference, parallel = build_models(device)

        generator = torch.Generator("cpu").manual_seed(2026)

        x = torch.randint(
            0, 50257, (2, 64), generator=generator
        ).to(device)

        targets = torch.randint(
            0, 50257, (2, 64), generator=generator
        ).to(device)

        with (
            torch.autocast("cuda", dtype=torch.bfloat16),
            sdpa_kernel(SDPBackend.FLASH_ATTENTION),
        ):
            logits_ref, loss_ref = reference(x, targets)
            logits_tp, loss_tp = parallel(x, targets)

        assert torch.isfinite(loss_ref).item()
        assert torch.isfinite(loss_tp).item()

        # Compare model outputs.
        logits_diff = (
            logits_ref.float() - logits_tp.float()
        ).abs().max()

        loss_diff = abs(loss_ref.item() - loss_tp.item())

        # Backpropagate through FlashAttention.
        loss_ref.backward()
        loss_tp.backward()

        for param in parallel.parameters():
            grad = param.grad
            if grad is None:
                continue
            local = (
                grad.to_local()
                if isinstance(grad, DTensor)
                else grad
            )
            assert torch.isfinite(local).all().item()

        # Compare global norms before applying clipping.
        ref_norm = torch.nn.utils.clip_grad_norm_(
            reference.parameters(), 0.1
        )

        tp_norm = tp_clip_grad_norm(
            parallel, 0.1, rank, device
        )

        norm_rel_diff = (
            abs(ref_norm.item() - tp_norm.item())
            / max(ref_norm.item(), 1e-12)
        )

        if rank == 0:
            print(
                f"[M5 BF16] max logits diff: "
                f"{logits_diff.item():.6e}"
            )
            print(
                f"[M5 BF16] loss diff: "
                f"{loss_diff:.6e}"
            )
            print(
                f"[M5 BF16] reference norm: "
                f"{ref_norm.item():.6f}"
            )
            print(
                f"[M5 BF16] TP norm: "
                f"{tp_norm.item():.6f}"
            )
            print(
                f"[M5 BF16] relative norm diff: "
                f"{norm_rel_diff:.6e}"
            )

        # BF16 and differing attention shapes can introduce
        # larger rounding differences than the FP32 tests.
        assert loss_diff < 0.03
        assert norm_rel_diff < 0.03
        assert ref_norm.item() > 0.1

        if rank == 0:
            print("[M5 BF16] PASS: BF16 + FlashAttention")
            print("[M5 BF16] PASS: TP global gradient clipping")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
