"""M4: minimal nested-FSDP correctness smoke test."""

import argparse
import os
import torch
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.nn.attention import sdpa_kernel, SDPBackend

from model import GPT
from model_configs import DEBUG_MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', choices=['grad', 'full'], required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group("nccl", device_id=device)

    try:
        torch.manual_seed(1337)

        cfg = DEBUG_MODELS["gpt-tiny"]
        model = GPT(cfg).to(device)

        # Verify tied embeddings BEFORE wrapping.
        assert (
            model.transformer.wte.weight
            is model.lm_head.weight
        )

        strategy = {
            'grad': ShardingStrategy.SHARD_GRAD_OP,
            'full': ShardingStrategy.FULL_SHARD,
        }[args.strategy]

        # Wrap each transformer block independently.
        for i, block in enumerate(model.transformer.h):
            model.transformer.h[i] = FSDP(
                block,
                sharding_strategy=strategy,
                use_orig_params=True,
                device_id=device,
            )

        # The root wrapper owns the shared embedding/head weight.
        model = FSDP(
            model,
            sharding_strategy=strategy,
            use_orig_params=True,
            device_id=device,
        )

        # Simple optimizer for initial correctness testing.
        # We'll restore the benchmark's optimizer settings later.
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=6e-4, weight_decay=0.0
        )

        generator = torch.Generator(device=device)
        generator.manual_seed(2026 + rank)

        x = torch.randint(
            0, 50257, (2, cfg.block_size),
            generator=generator, device=device,
        )
        y = torch.randint(
            0, 50257, (2, cfg.block_size),
            generator=generator, device=device,
        )

        for step in range(1, 4):
            optimizer.zero_grad(set_to_none=True)

            with (
                torch.autocast("cuda", dtype=torch.bfloat16),
                sdpa_kernel(SDPBackend.FLASH_ATTENTION),
            ):
                _, loss = model(x, y)

            assert torch.isfinite(loss).item()

            loss.backward()

            # FSDP-aware global gradient clipping.
            grad_norm = model.clip_grad_norm_(1.0)

            optimizer.step()

            mean_loss = loss.detach().clone()
            dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG)

            if rank == 0:
                print(
                    f"[M4] step={step} "
                    f"loss={mean_loss.item():.6f} "
                    f"grad_norm={grad_norm.item():.4f}",
                    flush=True,
                )

        # Materialize full parameters and verify the tie survived.
        with FSDP.summon_full_params(model, recurse=True):
            embedding = model.module.transformer.wte.weight
            output = model.module.lm_head.weight

            assert embedding is output
            assert embedding.data_ptr() == output.data_ptr()

        if rank == 0:
            print("[M4] PASS: forward, backward, optimizer")
            print("[M4] PASS: embedding/output weight tying")
            print(f"[M4] PASS: nested {strategy.name} FSDP")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
