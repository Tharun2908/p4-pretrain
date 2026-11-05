"""M5: fused versus standard AdamW on TP=2 DTensor parameters."""

import os
import torch
import torch.distributed as dist

from torch.distributed.tensor import DTensor
from torch.nn.attention import sdpa_kernel, SDPBackend

from m5_tp_optimizer_smoke import build_models
from m5_tp_clip_smoke import tp_clip_grad_norm


def build_adamw(model, fused):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=6e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=fused,
        foreach=False,
    )


def local_tensor(p):
    if isinstance(p, DTensor):
        return p.to_local()
    return p


@torch.no_grad()
def compare_params(a, b, device):
    params_a = dict(a.named_parameters())
    params_b = dict(b.named_parameters())

    assert params_a.keys() == params_b.keys()

    max_diff = torch.zeros((), device=device)

    for name in params_a:
        x = local_tensor(params_a[name]).detach()
        y = local_tensor(params_b[name]).detach()

        assert x.shape == y.shape, name
        assert torch.isfinite(x).all().item(), name
        assert torch.isfinite(y).all().item(), name

        diff = (x - y).abs().max()
        max_diff = torch.maximum(max_diff, diff)

    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
    return max_diff.item()


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)

    try:
        assert dist.get_world_size() == 2

        # build_models resets the seed and constructs identical TP weights.
        _, standard = build_models(device)
        _, fused = build_models(device)

        standard_opt = build_adamw(standard, fused=False)
        fused_opt = build_adamw(fused, fused=True)

        generator = torch.Generator("cpu").manual_seed(2026)

        x = torch.randint(
            0, 50257, (2, 64), generator=generator
        ).to(device)

        targets = torch.randint(
            0, 50257, (2, 64), generator=generator
        ).to(device)

        for step in range(1, 4):
            standard_opt.zero_grad(set_to_none=True)
            fused_opt.zero_grad(set_to_none=True)

            with (
                torch.autocast("cuda", dtype=torch.bfloat16),
                sdpa_kernel(SDPBackend.FLASH_ATTENTION),
            ):
                _, standard_loss = standard(x, targets)
                _, fused_loss = fused(x, targets)

            standard_loss.backward()
            fused_loss.backward()

            standard_norm = tp_clip_grad_norm(
                standard, 0.1, rank, device
            )
            fused_norm = tp_clip_grad_norm(
                fused, 0.1, rank, device
            )

            # This is the compatibility operation we're testing.
            standard_opt.step()
            fused_opt.step()

            param_diff = compare_params(
                standard, fused, device
            )

            loss_diff = abs(
                standard_loss.item() - fused_loss.item()
            )
            norm_diff = abs(
                standard_norm.item() - fused_norm.item()
            )

            if rank == 0:
                print(
                    f"[M5 FUSED] step={step} "
                    f"loss_diff={loss_diff:.3e} "
                    f"norm_diff={norm_diff:.3e} "
                    f"param_diff={param_diff:.3e}",
                    flush=True,
                )

            assert loss_diff < 1e-5
            assert norm_diff < 1e-5
            assert param_diff < 1e-5, (
                f"Optimizer implementations diverged: {param_diff}"
            )

        if rank == 0:
            print("[M5 FUSED] PASS: fused AdamW with DTensor")
            print("[M5 FUSED] PASS: optimizer-update parity")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
