"""Tensor parallelism for P4's GPT architecture.

M5 initial scope: one-dimensional, two-rank tensor parallelism.
Embeddings, LayerNorms, residual streams and LM head stay replicated.
Attention and MLP projections are sharded.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch.distributed.tensor import DTensor, Shard, Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from model import CausalSelfAttention


def reorder_qkv(tensor, tp_size):
    """Convert [Q_all,K_all,V_all] to rank-major QKV ordering."""
    assert tensor.shape[0] % (3 * tp_size) == 0
    local = tensor.shape[0] // (3 * tp_size)

    return (
        tensor.reshape(3, tp_size, local, *tensor.shape[1:])
        .transpose(0, 1)
        .contiguous()
        .reshape(tensor.shape)
    )


class TPFusedAttention(CausalSelfAttention):
    """Fused QKV attention with local heads on each TP rank."""

    def __init__(self, cfg, tp_size):
        super().__init__(cfg)
        assert cfg.n_head % tp_size == 0

        self.n_head = cfg.n_head // tp_size
        self.local_dim = cfg.n_embd // tp_size

    def forward(self, x):
        B, T, _ = x.shape

        q, k, v = self.c_attn(x).split(
            self.local_dim, dim=-1
        )

        def split_heads(t):
            return t.view(
                B, T, self.n_head, self.head_dim
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(B, T, self.local_dim)
        )

        return self.c_proj(y)


def parallelize_gpt_tp(model, mesh):
    """Convert a normally initialized GPT to tensor-parallel GPT."""
    tp_size = mesh.size()

    if mesh.ndim != 1 or tp_size != 2:
        raise ValueError("M5 currently supports a 1D TP=2 mesh")

    cfg = model.cfg

    for i, block in enumerate(model.transformer.h):
        old_attn = block.attn

        tp_attn = TPFusedAttention(
            cfg, tp_size
        ).to(
            device=old_attn.c_attn.weight.device,
            dtype=old_attn.c_attn.weight.dtype,
        )

        with torch.no_grad():
            tp_attn.c_attn.weight.copy_(
                reorder_qkv(old_attn.c_attn.weight, tp_size)
            )

            if old_attn.c_attn.bias is not None:
                tp_attn.c_attn.bias.copy_(
                    reorder_qkv(old_attn.c_attn.bias, tp_size)
                )

            tp_attn.c_proj.load_state_dict(
                old_attn.c_proj.state_dict()
            )

        block.attn = tp_attn

        model.transformer.h[i] = parallelize_module(
            block,
            mesh,
            {
                "attn.c_attn": ColwiseParallel(),
                "attn.c_proj": RowwiseParallel(),
                "mlp.c_fc": ColwiseParallel(),
                "mlp.c_proj": RowwiseParallel(),
            },
        )

    # Keep token embedding and LM head tied.
    assert (
        model.transformer.wte.weight
        is model.lm_head.weight
    )

    return model



@torch.no_grad()
def sync_dp_grads_(model, dp_mesh):
    """Average local parameter gradients over corresponding DP ranks."""
    assert dp_mesh.ndim == 1
    group = dp_mesh.get_group()
    dp_size = dp_mesh.size()

    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue

        local = grad.to_local() if isinstance(grad, DTensor) else grad

        dist.all_reduce(local, op=dist.ReduceOp.SUM, group=group)
        local.div_(dp_size)


@torch.no_grad()
def clip_grad_norm_tp_(model, max_norm, tp_mesh=None):
    """Global L2 clipping across unique TP shards and replicas.

    Call AFTER DP gradient averaging in hybrid TP+DP training.
    tp_mesh=None preserves the original standalone TP=2 behavior.
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group required")

    if tp_mesh is None:
        if dist.get_world_size() != 2:
            raise RuntimeError("Supply tp_mesh for hybrid TP+DP")
        tp_group = dist.group.WORLD
        tp_rank = dist.get_rank()
    else:
        if tp_mesh.ndim != 1 or tp_mesh.size() != 2:
            raise RuntimeError("Expected a 1D TP=2 mesh")
        tp_group = tp_mesh.get_group()
        tp_rank = tp_mesh.get_local_rank()

    device = next(model.parameters()).device
    local_sq = torch.zeros((), dtype=torch.float64, device=device)

    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue

        if isinstance(grad, DTensor):
            if len(grad.placements) != 1:
                raise RuntimeError("Expected 1D DTensor placement")

            placement = grad.placements[0]
            local = grad.to_local()

            if isinstance(placement, Shard):
                include = True
            elif isinstance(placement, Replicate):
                include = tp_rank == 0
            else:
                raise RuntimeError(
                    f"Unsupported gradient placement: {placement}"
                )
        else:
            local = grad
            include = tp_rank == 0

        if include:
            local_sq += local.double().square().sum()

    # Sum only within each TP group, not across all DP replicas.
    dist.all_reduce(local_sq, op=dist.ReduceOp.SUM, group=tp_group)

    total_norm = local_sq.sqrt()
    if not torch.isfinite(total_norm).item():
        raise RuntimeError("Non-finite global TP gradient norm")

    scale = min(1.0, max_norm / (total_norm.item() + 1e-6))

    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(scale)

    return total_norm
