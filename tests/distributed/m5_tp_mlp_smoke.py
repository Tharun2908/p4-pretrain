"""M5: verify 2-way tensor-parallel MLP forward/backward."""

import os
import torch
import torch.distributed as dist

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from model import MLP
from model_configs import MODELS


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)

    try:
        assert dist.get_world_size() == 2
        torch.backends.cuda.matmul.allow_tf32 = False

        cfg = MODELS["gpt-1.3b"]

        # Both models begin with identical parameters.
        torch.manual_seed(1337)
        reference = MLP(cfg).to(device)

        torch.manual_seed(1337)
        parallel = MLP(cfg).to(device)

        mesh = init_device_mesh("cuda", (2,))

        parallel = parallelize_module(
            parallel,
            mesh,
            {
                "c_fc": ColwiseParallel(),
                "c_proj": RowwiseParallel(),
            },
        )

        # Verify the actual local weight-shard dimensions.
        fc_shape = tuple(parallel.c_fc.weight.to_local().shape)
        proj_shape = tuple(parallel.c_proj.weight.to_local().shape)

        assert fc_shape == (4096, 2048)
        assert proj_shape == (2048, 4096)

        if rank == 0:
            print("[M5] MLP TP=2")
            print("[M5] local c_fc.weight:", fc_shape)
            print("[M5] local c_proj.weight:", proj_shape)

        # Identical input tensor on both ranks.
        g = torch.Generator(device="cpu").manual_seed(2026)
        x = torch.randn(
            2, 8, cfg.n_embd, generator=g
        ).to(device)

        x_ref = x.clone().requires_grad_(True)
        x_tp = x.clone().requires_grad_(True)

        # FP32 first, to isolate TP numerical correctness.
        y_ref = reference(x_ref)
        y_tp = parallel(x_tp)

        assert y_ref.shape == y_tp.shape == (2, 8, 2048)

        forward_diff = (y_ref - y_tp).abs().max()

        torch.testing.assert_close(
            y_tp, y_ref, rtol=1e-4, atol=1e-5
        )

        # Backward through both implementations.
        loss_ref = y_ref.square().mean()
        loss_tp = y_tp.square().mean()

        loss_ref.backward()
        loss_tp.backward()

        backward_diff = (
            x_ref.grad - x_tp.grad
        ).abs().max()

        torch.testing.assert_close(
            x_tp.grad, x_ref.grad,
            rtol=1e-4, atol=1e-5,
        )

        dist.all_reduce(forward_diff, op=dist.ReduceOp.MAX)
        dist.all_reduce(backward_diff, op=dist.ReduceOp.MAX)

        if rank == 0:
            print(
                "[M5] max forward difference:",
                f"{forward_diff.item():.3e}",
            )
            print(
                "[M5] max input-gradient difference:",
                f"{backward_diff.item():.3e}",
            )
            print("[M5] PASS: TP MLP forward parity")
            print("[M5] PASS: TP MLP backward input-gradient parity")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
