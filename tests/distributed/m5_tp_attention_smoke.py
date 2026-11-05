"""M5: TP=2 fused-QKV attention correctness test."""

import os
import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.nn.attention import sdpa_kernel, SDPBackend

from model import CausalSelfAttention
from model_configs import MODELS


class TPFusedAttention(CausalSelfAttention):
    """Same fused QKV computation, but with local attention heads."""

    def __init__(self, cfg, tp_size):
        super().__init__(cfg)
        assert cfg.n_head % tp_size == 0

        self.n_head = cfg.n_head // tp_size
        self.local_dim = cfg.n_embd // tp_size

    def forward(self, x):
        B, T, _ = x.shape

        # Local fused output: Q_local, K_local, V_local.
        q, k, v = self.c_attn(x).split(
            self.local_dim, dim=-1
        )

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        y = y.transpose(1, 2).contiguous()
        y = y.view(B, T, self.local_dim)

        # RowwiseParallel combines partial outputs.
        return self.c_proj(y)


def reorder_qkv(tensor, tp_size):
    """[Q_all, K_all, V_all] -> [Q0,K0,V0,Q1,K1,V1]."""
    assert tensor.shape[0] % (3 * tp_size) == 0

    local = tensor.shape[0] // (3 * tp_size)

    return (
        tensor.reshape(3, tp_size, local, *tensor.shape[1:])
        .transpose(0, 1)
        .contiguous()
        .reshape(tensor.shape)
    )


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
        tp_size = 2

        torch.manual_seed(1337)
        reference = CausalSelfAttention(cfg).to(device)

        torch.manual_seed(1337)
        parallel = TPFusedAttention(cfg, tp_size).to(device)

        # Reorder full QKV weights BEFORE sharding.
        with torch.no_grad():
            parallel.c_attn.weight.copy_(
                reorder_qkv(reference.c_attn.weight, tp_size)
            )
            parallel.c_attn.bias.copy_(
                reorder_qkv(reference.c_attn.bias, tp_size)
            )
            parallel.c_proj.load_state_dict(
                reference.c_proj.state_dict()
            )

        mesh = init_device_mesh("cuda", (tp_size,))

        parallel = parallelize_module(
            parallel,
            mesh,
            {
                "c_attn": ColwiseParallel(),
                "c_proj": RowwiseParallel(),
            },
        )

        qkv_shape = tuple(parallel.c_attn.weight.to_local().shape)
        proj_shape = tuple(parallel.c_proj.weight.to_local().shape)

        assert qkv_shape == (3072, 2048)
        assert proj_shape == (2048, 1024)

        if rank == 0:
            print("[M5 ATTENTION] TP=2, local heads=8")
            print("[M5 ATTENTION] local QKV weight:", qkv_shape)
            print("[M5 ATTENTION] local output weight:", proj_shape)

        generator = torch.Generator(device="cpu").manual_seed(2026)
        x = torch.randn(
            2, 8, cfg.n_embd, generator=generator
        ).to(device)

        x_ref = x.clone().requires_grad_(True)
        x_tp = x.clone().requires_grad_(True)

        with sdpa_kernel(SDPBackend.MATH):
            y_ref = reference(x_ref)
            y_tp = parallel(x_tp)

            forward_diff = (y_ref - y_tp).abs().max()

            torch.testing.assert_close(
                y_tp, y_ref, rtol=1e-4, atol=1e-5
            )

            y_ref.square().mean().backward()
            y_tp.square().mean().backward()

        backward_diff = (x_ref.grad - x_tp.grad).abs().max()

        torch.testing.assert_close(
            x_tp.grad, x_ref.grad,
            rtol=1e-4, atol=1e-5,
        )

        # Check local QKV weight gradients against corresponding
        # slices of the original model's fused weight gradients.
        expected_qkv_grad = reorder_qkv(
            reference.c_attn.weight.grad, tp_size
        ).chunk(tp_size, dim=0)[rank]

        actual_qkv_grad = parallel.c_attn.weight.grad.to_local()

        torch.testing.assert_close(
            actual_qkv_grad, expected_qkv_grad,
            rtol=1e-4, atol=1e-5,
        )

        qkv_grad_diff = (
            actual_qkv_grad - expected_qkv_grad
        ).abs().max()

        for value in (forward_diff, backward_diff, qkv_grad_diff):
            dist.all_reduce(value, op=dist.ReduceOp.MAX)

        if rank == 0:
            print(
                f"[M5 ATTENTION] max forward diff: "
                f"{forward_diff.item():.3e}"
            )
            print(
                f"[M5 ATTENTION] max input-gradient diff: "
                f"{backward_diff.item():.3e}"
            )
            print(
                f"[M5 ATTENTION] max QKV-weight-gradient diff: "
                f"{qkv_grad_diff.item():.3e}"
            )
            print("[M5 ATTENTION] PASS: fused-QKV TP forward/backward")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
