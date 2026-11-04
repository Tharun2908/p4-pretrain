"""M3 CPU preflight: DDP accumulation versus single-process reference.

Run with:
    torchrun --standalone --nproc_per_node=2 tests/ddp_gradient_check.py \
        --data /workspace/p4/m2-data/fineweb_tokens_v1/train.bin
"""

import argparse

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT
from model_configs import DEBUG_MODELS
from train import build_optimizer, setup_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)

    rank, world, _ = setup_dist("cpu")
    assert world in (2, 4), "Expected two or four ranks"

    try:
        cfg = DEBUG_MODELS["gpt-tiny"]
        T = cfg.block_size
        micro_bs = 1
        grad_accum = 2
        global_batch = world * micro_bs * grad_accum

        # Build identical models on each rank.
        torch.manual_seed(2026)
        ddp = DDP(GPT(cfg))

        torch.manual_seed(2026)
        reference = GPT(cfg)

        for a, b in zip(ddp.module.parameters(), reference.parameters()):
            assert torch.equal(a, b), "Initial parameters differ"

        # All ranks construct the same four real-text training examples.
        data = np.memmap(args.data, dtype="<u2", mode="r")
        starts = [i * (T + 1) for i in range(global_batch)]

        assert starts[-1] + T + 1 <= len(data)

        xs = torch.stack([
            torch.from_numpy(
                np.asarray(data[s:s+T], dtype=np.int64)
            )
            for s in starts
        ])

        ys = torch.stack([
            torch.from_numpy(
                np.asarray(data[s+1:s+T+1], dtype=np.int64)
            )
            for s in starts
        ])

        lr = 6e-4
        opt = build_optimizer(ddp.module, lr, 0.1, "cpu")
        ref_opt = build_optimizer(reference, lr, 0.1, "cpu")

        # Reference: one process sees the entire global batch.
        _, ref_loss = reference(xs, ys)
        ref_loss.backward()

        # DDP: each rank processes two microbatches.
        for micro in range(grad_accum):
            ddp.require_backward_grad_sync = (
                micro == grad_accum - 1
            )

            sample = rank * grad_accum + micro
            _, loss = ddp(
                xs[sample:sample+1],
                ys[sample:sample+1],
            )

            (loss / grad_accum).backward()

        # Compare every parameter gradient before clipping.
        max_grad_diff = 0.0
        parameters_checked = 0

        for name, ddp_param in ddp.module.named_parameters():
            ref_param = dict(reference.named_parameters())[name]

            assert ddp_param.grad is not None
            assert ref_param.grad is not None

            diff = (
                ddp_param.grad - ref_param.grad
            ).abs().max().item()

            max_grad_diff = max(max_grad_diff, diff)
            parameters_checked += 1

            torch.testing.assert_close(
                ddp_param.grad,
                ref_param.grad,
                rtol=1e-4,
                atol=2e-5,
                msg=f"Gradient mismatch: {name}",
            )

        # Match train.py's gradient clipping and optimizer update.
        torch.nn.utils.clip_grad_norm_(
            ddp.module.parameters(), 1.0
        )
        torch.nn.utils.clip_grad_norm_(
            reference.parameters(), 1.0
        )

        opt.step()
        ref_opt.step()

        max_weight_diff = 0.0

        for name, ddp_param in ddp.module.named_parameters():
            ref_param = dict(reference.named_parameters())[name]

            diff = (
                ddp_param - ref_param
            ).abs().max().item()

            max_weight_diff = max(max_weight_diff, diff)

            torch.testing.assert_close(
                ddp_param,
                ref_param,
                rtol=1e-4,
                atol=5e-5,
                msg=f"Optimizer update mismatch: {name}",
            )

        print(
            f"[rank {rank}] PASS | "
            f"parameters checked={parameters_checked} | "
            f"max gradient diff={max_grad_diff:.3e} | "
            f"max weight diff={max_weight_diff:.3e}",
            flush=True,
        )

        dist.barrier()

        if rank == 0:
            print(
                "PASS: DDP accumulated gradients and AdamW update "
                "match the single-process global-batch reference"
            )

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
