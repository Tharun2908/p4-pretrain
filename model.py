"""
p4/model.py — the GPT every P4 milestone trains.

GPT-2 lineage, deliberately: learned positional embeddings, pre-LayerNorm blocks, GELU MLP,
tied input/output embeddings. With vocab_size=50257 and bias=True this reproduces GPT-2's
published parameter counts EXACTLY (124,439,808 for small; 354,823,168 for medium) — a test
checks it. That external anchor is the point: when someone asks "how do you know your model
is right?", the answer is "it matches a number I didn't compute myself."

No dropout. Pretraining at our token budgets is far from overfitting, and dropout would add
an RNG-dependent op to every step that muddies throughput measurements for zero benefit.

Shape legend:  B batch · T sequence · C = n_embd · nh = n_head · hd = head_dim · V vocab
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_configs import GPTConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention. One fused Linear produces q, k and v in a single
    (C -> 3C) matmul instead of three (C -> C) ones: same FLOPs, one bigger kernel, fewer
    launches. (M5 note: fusing complicates tensor parallelism — a column-parallel split of
    the fused 3C dim must split q, k and v *each* by heads, not the concatenation naively.)
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.head_dim
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.RESIDUAL_PROJ = True          # gets depth-scaled init, see GPT._init_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)                              # 3 × (B, T, C)
        # split C into heads and move heads ahead of T: each head attends independently
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)          # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # softmax(q·kᵀ / sqrt(hd) + causal mask) · v, as ONE fused kernel. Which kernel
        # (flash / cuDNN / mem-efficient / math) is chosen by the CALLER via
        # torch.nn.attention.sdpa_kernel — the model stays backend-agnostic so train.py can
        # make the backend an explicit, recorded experimental variable.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)          # (B, nh, T, hd)

        # re-assemble heads side by side. transpose leaves a non-contiguous view; .view()
        # needs contiguous memory, hence .contiguous() — a real copy, visible in profiles.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """C -> 4C -> C. Holds 2/3 of each block's weights (8C² of 12C²), so most FLOPs live here."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU(approximate="tanh")   # GPT-2's original approximation
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.RESIDUAL_PROJ = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    """
    Pre-LN residual block:   x = x + attn(LN(x));   x = x + mlp(LN(x))

    Pre-LN (normalize the *input* to each sublayer) rather than the original Transformer's
    post-LN (normalize *after* adding): the residual stream x is never normalized, so
    gradients have an unobstructed identity path from the loss to the embeddings. That's
    what lets deep stacks train without careful LR warmup tricks.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),     # token id   -> vector
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),     # position   -> vector (learned)
            h=nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer)),
            ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),     # pre-LN needs one final norm
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # WEIGHT TYING: the output projection IS the input embedding matrix — one tensor,
        # two uses. "Which vector represents token i" and "which direction scores token i"
        # are the same question. Saves V·C params (38.6M at 125M — 31% of the model) and
        # wte gets gradient from both ends of the network.
        # (M4 note: FSDP must keep a tied param in ONE shard unit, or the tie breaks.)
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """
        GPT-2 init: N(0, 0.02) everywhere, biases zero, LayerNorm at identity (torch default).

        Depth scaling: every block ADDS two outputs (attn.c_proj, mlp.c_proj) into the
        residual stream. Sum 2L independent contributions and the stream's variance grows
        ~2L×. Shrinking those two projections' std by 1/sqrt(2L) keeps the stream's scale
        roughly depth-independent at init.

        Tied weight: apply() visits wte (Embedding) then lm_head (Linear) — the same tensor
        initialized twice, both times N(0, 0.02), so visiting order doesn't matter.
        """
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, "RESIDUAL_PROJ", False):
                std *= (2 * self.cfg.n_layer) ** -0.5
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx: (B, T) token ids.  targets: (B, T) next-token ids, -1 = ignore that position.
        Returns (logits (B, T, V), loss or None).
        """
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.cfg.block_size}")

        pos = torch.arange(T, device=idx.device)                   # (T,)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)  # (B,T,C) + (T,C) broadcast
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                   # (B, T, V)

        loss = None
        if targets is not None:
            # No explicit .float(): under bf16 autocast, cross_entropy already runs in fp32
            # internally. Casting first would materialize a full fp32 copy of the logits —
            # 2.5 GB at 1.3B with micro-batch 12 — for nothing.
            # PADDED VOCAB: ids 50257..50303 never appear as targets, so their logits only
            # ever receive "push down" gradient and the model learns to assign them ~0
            # probability. Harmless in training; mask them out at sampling time.
            # reshape, not view: targets built by slicing (idx[:, 1:]) are non-contiguous and
            # view() can't flatten them. reshape is free for contiguous logits and makes a
            # negligible int64 copy for sliced targets. (Bug caught in external review.)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=-1)
        return logits, loss

    def num_params(self, non_embedding: bool = True) -> int:
        """parameters() de-duplicates the tied tensor, so wte/lm_head is counted ONCE.
        non_embedding subtracts wpe only (wte is a real matmul via lm_head) — same
        convention as model_configs.n_params_non_embedding and bench.py's MFU."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n
