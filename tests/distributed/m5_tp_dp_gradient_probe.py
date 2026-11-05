"""M5: verify TP=2 x DP=2 gradient parity against full GPT."""

import os
import torch
import torch.distributed as dist

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.nn.attention import sdpa_kernel, SDPBackend

from model import GPT
from model_configs import DEBUG_MODELS
from tp_parallel import parallelize_gpt_tp, reorder_qkv


def local_grad(param):
    grad = param.grad
    assert grad is not None
    return grad.to_local() if isinstance(grad, DTensor) else grad


def sync_dp_grads(model, dp_group):
    """Average corresponding local gradient shards across DP replicas."""
    dp_size = dist.get_world_size(group=dp_group)

    for param in model.parameters():
        if param.grad is None:
            continue

        grad = local_grad(param)

        dist.all_reduce(
            grad,
            op=dist.ReduceOp.SUM,
            group=dp_group,
        )
        grad.div_(dp_size)


def expected_local_grad(name, grad, tp_rank):
    """Convert reference gradients to the TP parameter layout."""
    if name.endswith((
        "attn.c_attn.weight",
        "attn.c_attn.bias",
    )):
        return reorder_qkv(grad, 2).chunk(2, dim=0)[tp_rank]

    if name.endswith((
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
    )):
        return grad.chunk(2, dim=0)[tp_rank]

    if name.endswith((
        "attn.c_proj.weight",
        "mlp.c_proj.weight",
    )):
        return grad.chunk(2, dim=1)[tp_rank]

    # Embeddings, LayerNorms, and output biases are replicated.
    return grad


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

        parallel = parallelize_gpt_tp(parallel, tp_mesh)

        # Four samples: two for each DP replica.
        # Within a TP group, both ranks receive identical inputs.
        g = torch.Generator("cpu").manual_seed(2026)

        x_global = torch.randint(
            0, 50257, (4, 8), generator=g
        ).to(device)

        y_global = torch.randint(
            0, 50257, (4, 8), generator=g
        ).to(device)

        x_local = x_global.chunk(2, dim=0)[dp_rank].contiguous()
        y_local = y_global.chunk(2, dim=0)[dp_rank].contiguous()

        with sdpa_kernel(SDPBackend.MATH):
            _, ref_loss = reference(x_global, y_global)
            _, tp_loss = parallel(x_local, y_local)

            ref_loss.backward()
            tp_loss.backward()

        # Averaging the two DP-local losses should give
        # the same loss as processing the full global batch.
        mean_tp_loss = tp_loss.detach().clone()
        dist.all_reduce(
            mean_tp_loss,
            op=dist.ReduceOp.AVG,
            group=dp_mesh.get_group(),
        )

        torch.testing.assert_close(
            mean_tp_loss, ref_loss.detach(),
            rtol=1e-4, atol=1e-5,
        )

        # This is the key hybrid-parallel operation.
        sync_dp_grads(parallel, dp_mesh.get_group())

        ref_params = dict(reference.named_parameters())
        tp_params = dict(parallel.named_parameters())

        assert ref_params.keys() == tp_params.keys()

        max_grad_diff = torch.zeros((), device=device)
        n_checked = 0

        for name, tp_param in tp_params.items():
            expected = expected_local_grad(
                name, ref_params[name].grad, tp_rank
            )

            actual = local_grad(tp_param)

            assert expected.shape == actual.shape, name

            torch.testing.assert_close(
                actual,
                expected,
                rtol=3e-4,
                atol=3e-5,
                msg=lambda msg: f"{name}: {msg}",
            )

            diff = (actual - expected).abs().max()
            max_grad_diff = torch.maximum(max_grad_diff, diff)
            n_checked += 1

        dist.all_reduce(
            max_grad_diff, op=dist.ReduceOp.MAX
        )

        if rank == 0:
            print("[M5 HYBRID] TP=2, DP=2")
            print("[M5 HYBRID] Global batch=4, sequence=8")
            print(
                f"[M5 HYBRID] reference loss: "
                f"{ref_loss.item():.8f}"
            )
            print(
                f"[M5 HYBRID] averaged TP+DP loss: "
                f"{mean_tp_loss.item():.8f}"
            )
            print(
                f"[M5 HYBRID] parameter gradients checked: "
                f"{n_checked}"
            )
            print(
                f"[M5 HYBRID] max gradient diff: "
                f"{max_grad_diff.item():.3e}"
            )
            print("[M5 HYBRID] PASS: DP gradient synchronization")
            print("[M5 HYBRID] PASS: full-model gradient parity")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
