"""M5: TP=2 x DP=2 clipping and AdamW update parity."""

import os
import torch
import torch.distributed as dist

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.nn.attention import sdpa_kernel, SDPBackend

from model import GPT
from model_configs import DEBUG_MODELS
from tp_parallel import (
    parallelize_gpt_tp,
    reorder_qkv,
    sync_dp_grads_,
    clip_grad_norm_tp_,
)


def local(t):
    return t.to_local() if isinstance(t, DTensor) else t


def expected_shard(name, full, tp_rank):
    if name.endswith(("attn.c_attn.weight", "attn.c_attn.bias")):
        return reorder_qkv(full, 2).chunk(2, dim=0)[tp_rank]

    if name.endswith(("mlp.c_fc.weight", "mlp.c_fc.bias")):
        return full.chunk(2, dim=0)[tp_rank]

    if name.endswith(("attn.c_proj.weight", "mlp.c_proj.weight")):
        return full.chunk(2, dim=1)[tp_rank]

    return full


def optimizer_for(model):
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
        fused=False,
        foreach=False,
    )


def compare(reference, parallel, tp_rank, device, gradients):
    ref_params = dict(reference.named_parameters())
    tp_params = dict(parallel.named_parameters())

    assert ref_params.keys() == tp_params.keys()

    max_diff = torch.zeros((), device=device)

    for name, tp_param in tp_params.items():
        ref_param = ref_params[name]

        if gradients:
            expected = ref_param.grad
            actual = local(tp_param.grad)
        else:
            expected = ref_param.detach()
            actual = local(tp_param.detach())

        expected = expected_shard(name, expected, tp_rank)

        torch.testing.assert_close(
            actual, expected,
            rtol=5e-4, atol=5e-5,
            msg=lambda msg: f"{name}: {msg}",
        )

        max_diff = torch.maximum(
            max_diff,
            (actual - expected).abs().max(),
        )

    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
    return max_diff.item()


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)

    try:
        assert dist.get_world_size() == 4
        torch.backends.cuda.matmul.allow_tf32 = False

        mesh = init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("dp", "tp")
        )
        dp_mesh = mesh["dp"]
        tp_mesh = mesh["tp"]

        dp_rank = dp_mesh.get_local_rank()
        tp_rank = tp_mesh.get_local_rank()

        cfg = DEBUG_MODELS["gpt-tiny"]

        torch.manual_seed(1337)
        reference = GPT(cfg).to(device)

        torch.manual_seed(1337)
        parallel = GPT(cfg).to(device)
        parallelize_gpt_tp(parallel, tp_mesh)

        ref_opt = optimizer_for(reference)
        tp_opt = optimizer_for(parallel)

        g = torch.Generator("cpu").manual_seed(2026)

        for step in range(1, 4):
            # Every rank constructs the same global batch.
            x = torch.randint(
                0, 50257, (4, 8), generator=g
            ).to(device)
            y = torch.randint(
                0, 50257, (4, 8), generator=g
            ).to(device)

            # Each DP replica uses its own half.
            x_local = x.chunk(2, dim=0)[dp_rank].contiguous()
            y_local = y.chunk(2, dim=0)[dp_rank].contiguous()

            ref_opt.zero_grad(set_to_none=True)
            tp_opt.zero_grad(set_to_none=True)

            with sdpa_kernel(SDPBackend.MATH):
                _, ref_loss = reference(x, y)
                _, tp_loss = parallel(x_local, y_local)

                ref_loss.backward()
                tp_loss.backward()

            # Synchronize corresponding TP shards across DP.
            sync_dp_grads_(parallel, dp_mesh)

            grad_diff = compare(
                reference, parallel, tp_rank, device,
                gradients=True,
            )

            # Compare global pre-clipping gradient norms.
            ref_norm = torch.nn.utils.clip_grad_norm_(
                reference.parameters(), 0.1
            )

            tp_norm = clip_grad_norm_tp_(
                parallel, 0.1, tp_mesh=tp_mesh
            )

            norm_diff = abs(ref_norm.item() - tp_norm.item())

            assert norm_diff < 1e-4, norm_diff
            assert ref_norm.item() > 0.1

            ref_opt.step()
            tp_opt.step()

            param_diff = compare(
                reference, parallel, tp_rank, device,
                gradients=False,
            )

            if rank == 0:
                print(
                    f"[M5 HYBRID OPT] step={step} "
                    f"ref_norm={ref_norm.item():.6f} "
                    f"tp_norm={tp_norm.item():.6f} "
                    f"norm_diff={norm_diff:.3e} "
                    f"grad_diff={grad_diff:.3e} "
                    f"param_diff={param_diff:.3e}",
                    flush=True,
                )

        if rank == 0:
            print("[M5 HYBRID OPT] PASS: DP gradient averaging")
            print("[M5 HYBRID OPT] PASS: TP global clipping")
            print("[M5 HYBRID OPT] PASS: AdamW update parity")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
