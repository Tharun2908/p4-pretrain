"""Compare DDP and nested FSDP updates on identical rank-local data."""

import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.nn.attention import sdpa_kernel, SDPBackend

from model import GPT
from model_configs import DEBUG_MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["grad", "full"], required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    torch.backends.cuda.matmul.allow_tf32 = False

    try:
        cfg = DEBUG_MODELS["gpt-tiny"]
        strategy = {
            "grad": ShardingStrategy.SHARD_GRAD_OP,
            "full": ShardingStrategy.FULL_SHARD,
        }[args.strategy]

        # Reset seed before EACH construction: identical initial weights.
        torch.manual_seed(1337)
        ddp_base = GPT(cfg).to(device)

        torch.manual_seed(1337)
        fsdp_base = GPT(cfg).to(device)

        ddp = DDP(ddp_base, device_ids=[local_rank])

        for i, block in enumerate(fsdp_base.transformer.h):
            fsdp_base.transformer.h[i] = FSDP(
                block,
                sharding_strategy=strategy,
                use_orig_params=True,
                device_id=device,
            )

        fsdp = FSDP(
            fsdp_base,
            sharding_strategy=strategy,
            use_orig_params=True,
            device_id=device,
        )

        # SGD isolates parameter-update correctness from AdamW state.
        ddp_opt = torch.optim.SGD(ddp.parameters(), lr=0.01)
        fsdp_opt = torch.optim.SGD(fsdp.parameters(), lr=0.01)

        generator = torch.Generator(device="cpu").manual_seed(2026 + rank)
        x = torch.randint(
            0, 50257, (2, cfg.block_size), generator=generator
        ).to(device)
        y = torch.randint(
            0, 50257, (2, cfg.block_size), generator=generator
        ).to(device)

        for step in range(1, 4):
            ddp_opt.zero_grad(set_to_none=True)
            fsdp_opt.zero_grad(set_to_none=True)

            with sdpa_kernel(SDPBackend.MATH):
                _, ddp_loss = ddp(x, y)
                _, fsdp_loss = fsdp(x, y)

            ddp_loss.backward()
            fsdp_loss.backward()

            ddp_norm = torch.nn.utils.clip_grad_norm_(
                ddp.parameters(), 1.0
            )
            fsdp_norm = fsdp.clip_grad_norm_(1.0)

            ddp_opt.step()
            fsdp_opt.step()

            if rank == 0:
                print(
                    f"[M4 parity] step={step} "
                    f"DDP loss={ddp_loss.item():.6f} "
                    f"FSDP loss={fsdp_loss.item():.6f} "
                    f"DDP norm={ddp_norm.item():.6f} "
                    f"FSDP norm={fsdp_norm.item():.6f}",
                    flush=True,
                )

        # Reconstruct a full, unsharded FSDP state for comparison.
        # This older state-dict API is sufficient for this tiny diagnostic.
        with FSDP.state_dict_type(
            fsdp,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            fsdp_state = fsdp.state_dict()

        ddp_state = {
            k: v.detach().cpu()
            for k, v in ddp.module.state_dict().items()
        }

        assert ddp_state.keys() == fsdp_state.keys()

        max_diff = max(
            (ddp_state[k] - fsdp_state[k]).abs().max().item()
            for k in ddp_state
        )

        if rank == 0:
            print(f"[M4 parity] max parameter difference: {max_diff:.3e}")

        assert max_diff < 5e-4, (
            f"DDP/FSDP parameter-update parity failed: {max_diff}"
        )

        if rank == 0:
            print(f"[M4 parity] PASS: DDP vs {strategy.name}")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
