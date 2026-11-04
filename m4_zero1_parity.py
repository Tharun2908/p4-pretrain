"""M4: DDP AdamW versus DDP + ZeRO-1 AdamW parity."""

import os
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.attention import sdpa_kernel, SDPBackend

from model import GPT
from model_configs import DEBUG_MODELS
from train import build_optimizer, lr_at


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)

    torch.backends.cuda.matmul.allow_tf32 = False

    try:
        cfg = DEBUG_MODELS["gpt-tiny"]

        # Identical initial parameters.
        torch.manual_seed(1337)
        reference = GPT(cfg).to(device)

        torch.manual_seed(1337)
        candidate = GPT(cfg).to(device)

        ddp_ref = DDP(reference, device_ids=[local_rank])
        ddp_zero = DDP(candidate, device_ids=[local_rank])

        ref_opt = build_optimizer(
            reference, 6e-4, 0.1, str(device)
        )

        decay = [
            p for p in candidate.parameters()
            if p.ndim >= 2
        ]
        no_decay = [
            p for p in candidate.parameters()
            if p.ndim < 2
        ]

        zero_opt = ZeroRedundancyOptimizer(
            [
                {"params": decay, "weight_decay": 0.1},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            optimizer_class=torch.optim.AdamW,
            lr=6e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=True,
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(2026 + rank)

        x = torch.randint(
            0, 50257, (2, cfg.block_size), generator=generator
        ).to(device)
        y = torch.randint(
            0, 50257, (2, cfg.block_size), generator=generator
        ).to(device)

        for step in range(5):
            # Warmup and cosine decay from the actual trainer.
            lr = lr_at(step, 6e-4, warmup=2, total=4)

            for opt in (ref_opt, zero_opt):
                for group in opt.param_groups:
                    group["lr"] = lr
                opt.zero_grad(set_to_none=True)

            with sdpa_kernel(SDPBackend.MATH):
                _, ref_loss = ddp_ref(x, y)
                _, zero_loss = ddp_zero(x, y)

            ref_loss.backward()
            zero_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                reference.parameters(), 1.0
            )
            torch.nn.utils.clip_grad_norm_(
                candidate.parameters(), 1.0
            )

            ref_opt.step()
            zero_opt.step()

            # Check that ZeRO propagated LR changes to its
            # underlying rank-local optimizer.
            local_lrs = [
                group["lr"] for group in zero_opt.optim.param_groups
            ]
            assert all(abs(v - lr) < 1e-10 for v in local_lrs), (
                f"ZeRO local learning rates mismatch: {local_lrs}"
            )

            # Compare every parameter after the optimizer update.
            max_diff = torch.stack([
                (a.detach() - b.detach()).abs().max()
                for a, b in zip(
                    reference.parameters(),
                    candidate.parameters(),
                )
            ]).max()

            dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)

            loss_diff = abs(ref_loss.item() - zero_loss.item())

            if rank == 0:
                print(
                    f"[ZeRO-1 parity] step={step} "
                    f"lr={lr:.8f} "
                    f"loss_diff={loss_diff:.3e} "
                    f"max_param_diff={max_diff.item():.3e}",
                    flush=True,
                )

            assert max_diff.item() < 1e-5, (
                f"Parameter parity failed at step {step}"
            )

        if rank == 0:
            print("[ZeRO-1 parity] PASS: DDP vs ZeRO-1 AdamW")
            print("[ZeRO-1 parity] PASS: LR schedule propagated")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
