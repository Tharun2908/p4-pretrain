"""M5: end-to-end TP=2 training with standard AdamW."""

import os
import torch
import torch.distributed as dist

from torch.distributed.tensor import DTensor
from torch.nn.attention import sdpa_kernel, SDPBackend

from m5_tp_optimizer_smoke import build_models
from m5_tp_clip_smoke import tp_clip_grad_norm


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)

    try:
        assert dist.get_world_size() == 2

        _, model = build_models(device)

        decay = [p for p in model.parameters() if p.ndim >= 2]
        no_decay = [p for p in model.parameters() if p.ndim < 2]

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 0.1},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=6e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=False,
            foreach=False,
        )

        generator = torch.Generator("cpu").manual_seed(2026)

        x = torch.randint(
            0, 50257, (2, 64), generator=generator
        ).to(device)

        targets = torch.randint(
            0, 50257, (2, 64), generator=generator
        ).to(device)

        for step in range(1, 4):
            before = {
                name: local(p.detach()).clone()
                for name, p in model.named_parameters()
            }

            optimizer.zero_grad(set_to_none=True)

            with (
                torch.autocast("cuda", dtype=torch.bfloat16),
                sdpa_kernel(SDPBackend.FLASH_ATTENTION),
            ):
                _, loss = model(x, targets)

            assert torch.isfinite(loss).item()
            loss.backward()

            norm = tp_clip_grad_norm(
                model, 0.1, rank, device
            )
            assert torch.isfinite(norm).item()

            optimizer.step()

            max_update = torch.zeros((), device=device)

            for name, param in model.named_parameters():
                current = local(param.detach())
                assert torch.isfinite(current).all().item()

                diff = (current - before[name]).abs().max()
                max_update = torch.maximum(max_update, diff)

            dist.all_reduce(max_update, op=dist.ReduceOp.MAX)
            assert max_update.item() > 0

            if rank == 0:
                print(
                    f"[M5 STANDARD] step={step} "
                    f"loss={loss.item():.6f} "
                    f"grad_norm={norm.item():.6f} "
                    f"max_update={max_update.item():.3e}",
                    flush=True,
                )

        if rank == 0:
            print("[M5 STANDARD] PASS: BF16 FlashAttention")
            print("[M5 STANDARD] PASS: standard AdamW optimizer")
            print("[M5 STANDARD] PASS: TP gradient clipping")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
