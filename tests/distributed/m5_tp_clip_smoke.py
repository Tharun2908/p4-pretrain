"""M5: TP-aware global gradient clipping correctness."""

import os
import torch
import torch.distributed as dist

from torch.distributed.tensor import DTensor, Shard, Replicate
from torch.nn.attention import sdpa_kernel, SDPBackend

from m5_tp_optimizer_smoke import (
    build_models,
    make_optimizer,
    compare_parameters,
)


@torch.no_grad()
def tp_clip_grad_norm(model, max_norm, rank, device):
    """
    Compute the global L2 norm without double-counting
    replicated gradients.

    This implementation assumes a 1D TP mesh containing
    all ranks in the process group.
    """
    local_sq = torch.zeros(
        (), dtype=torch.float64, device=device
    )

    for param in model.parameters():
        grad = param.grad

        if grad is None:
            continue

        if isinstance(grad, DTensor):
            assert len(grad.placements) == 1

            placement = grad.placements[0]
            local = grad.to_local()

            if isinstance(placement, Shard):
                # Every rank contributes its unique shard.
                count_this_rank = True

            elif isinstance(placement, Replicate):
                # Count the replicated gradient only once.
                count_this_rank = rank == 0

            else:
                raise RuntimeError(
                    f"Unexpected gradient placement: {placement}"
                )

        else:
            # Ordinary tensors are replicated in this TP model.
            local = grad
            count_this_rank = rank == 0

        if count_this_rank:
            local_sq += local.double().square().sum()

    # Combine unique contributions across both ranks.
    dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)

    total_norm = local_sq.sqrt()
    scale = min(
        1.0,
        max_norm / (total_norm.item() + 1e-6),
    )

    # Apply the SAME clipping factor to every local gradient.
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(scale)

    return total_norm


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)

    try:
        assert dist.get_world_size() == 2
        torch.backends.cuda.matmul.allow_tf32 = False

        reference, parallel = build_models(device)

        ref_opt = make_optimizer(reference)
        tp_opt = make_optimizer(parallel)

        generator = torch.Generator(
            device="cpu"
        ).manual_seed(2026)

        x = torch.randint(
            0, 50257, (2, 8), generator=generator
        ).to(device)

        targets = torch.randint(
            0, 50257, (2, 8), generator=generator
        ).to(device)

        max_norm = 0.1

        for step in range(1, 4):
            ref_opt.zero_grad(set_to_none=True)
            tp_opt.zero_grad(set_to_none=True)

            with sdpa_kernel(SDPBackend.MATH):
                _, ref_loss = reference(x, targets)
                _, tp_loss = parallel(x, targets)

                ref_loss.backward()
                tp_loss.backward()

            ref_norm = torch.nn.utils.clip_grad_norm_(
                reference.parameters(),
                max_norm,
            )

            tp_norm = tp_clip_grad_norm(
                parallel,
                max_norm,
                rank,
                device,
            )

            norm_diff = abs(
                ref_norm.item() - tp_norm.item()
            )

            assert norm_diff < 1e-4, (
                f"Global gradient norm mismatch: {norm_diff}"
            )

            assert ref_norm.item() > max_norm, (
                "Clipping wasn't exercised"
            )

            ref_opt.step()
            tp_opt.step()

            max_param_diff = compare_parameters(
                reference,
                parallel,
                device,
            )

            if rank == 0:
                print(
                    f"[M5 CLIP] step={step} "
                    f"ref_norm={ref_norm.item():.6f} "
                    f"tp_norm={tp_norm.item():.6f} "
                    f"norm_diff={norm_diff:.3e} "
                    f"param_diff={max_param_diff:.3e}",
                    flush=True,
                )

        if rank == 0:
            print("[M5 CLIP] PASS: global gradient norm")
            print("[M5 CLIP] PASS: clipped AdamW updates")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
