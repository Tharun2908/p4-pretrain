"""M4: ZeRO-1 optimizer-state sharding smoke test."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.attention import sdpa_kernel, SDPBackend

from model import GPT
from model_configs import DEBUG_MODELS


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)

    try:
        torch.manual_seed(1337)

        model = GPT(DEBUG_MODELS["gpt-tiny"]).to(device)
        assert model.transformer.wte.weight is model.lm_head.weight

        ddp = DDP(model, device_ids=[local_rank])

        # Preserve the original AdamW weight-decay groups.
        decay = [p for p in model.parameters() if p.ndim >= 2]
        no_decay = [p for p in model.parameters() if p.ndim < 2]

        optimizer = ZeroRedundancyOptimizer(
            [
                {"params": decay, "weight_decay": 0.1},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            optimizer_class=torch.optim.AdamW,
            lr=6e-4,
            fused=True,
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(2026 + rank)

        x = torch.randint(
            0, 50257, (2, 64), generator=generator
        ).to(device)

        y = torch.randint(
            0, 50257, (2, 64), generator=generator
        ).to(device)

        for step in range(1, 4):
            optimizer.zero_grad(set_to_none=True)

            with (
                torch.autocast("cuda", dtype=torch.bfloat16),
                sdpa_kernel(SDPBackend.FLASH_ATTENTION),
            ):
                _, loss = ddp(x, y)

            assert torch.isfinite(loss).item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            mean_loss = loss.detach().clone()
            dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG)

            if rank == 0:
                print(
                    f"[ZeRO-1] step={step} "
                    f"loss={mean_loss.item():.6f}",
                    flush=True,
                )

        # Verify optimizer moments are distributed among ranks.
        local_moments = sum(
            state["exp_avg"].numel()
            for state in optimizer.optim.state.values()
            if "exp_avg" in state
        )

        counts = [None] * dist.get_world_size()
        dist.all_gather_object(counts, local_moments)

        total_params = sum(p.numel() for p in model.parameters())

        assert sum(counts) == total_params
        assert all(0 < count < total_params for count in counts)

        # Verify all DDP model replicas remain synchronized.
        flat = torch.cat([
            p.detach().float().flatten()
            for p in model.parameters()
        ])

        reference = flat.clone()
        dist.broadcast(reference, src=0)

        max_diff = (flat - reference).abs().max()
        dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)

        assert max_diff.item() == 0.0

        if rank == 0:
            print(
                f"[ZeRO-1] optimizer moment elements per rank: "
                f"{counts}"
            )
            print(
                f"[ZeRO-1] total parameter elements: "
                f"{total_params}"
            )
            print(
                f"[ZeRO-1] max replica difference: "
                f"{max_diff.item():.3e}"
            )
            print("[ZeRO-1] PASS: optimizer state sharded")
            print("[ZeRO-1] PASS: model replicas synchronized")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
