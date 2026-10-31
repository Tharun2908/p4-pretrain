"""
p4/model_configs.py — the three model sizes P4 uses, and the arithmetic behind them.

    python model_configs.py          # prints the table that DEC-006 is built on

Roles (DEC-006):
  gpt-125m  correctness/debugging model + the one clean from-scratch loss curve
  gpt-350m  optional middle point for the DDP-vs-ZeRO crossover trend
  gpt-1.3b  systems benchmarking model: throughput, ZeRO stages, TP, precision ablation

All three FIT under plain DDP on a 192 GB B200, which is what keeps the FSDP comparison
controlled at every size: sharding is always optional, so its cost is always measurable.
"""

from dataclasses import dataclass

from bench import gpt_flops_per_token


@dataclass(frozen=True)
class GPTConfig:
    name: str
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int = 1024
    # 50257 padded up to a multiple of 64. Ragged vocab dims hit slow matmul tiles in the
    # lm_head, which is one of the largest matmuls in a small model. Pinned here so no run
    # silently uses the unpadded size.
    vocab_size: int = 50304
    bias: bool = True

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head

    @property
    def n_params(self) -> int:
        """Total params, lm_head tied to wte (counted once). Estimate until model.py exists."""
        d, b = self.n_embd, int(self.bias)
        # weights 12d² (qkv 3d² + proj d² + mlp 8d²) · linear biases 9d · 2 LNs of (1+b)d each
        per_layer = 12 * d * d + b * 9 * d + 2 * (1 + b) * d
        return (self.vocab_size * d + self.block_size * d   # wte + wpe
                + self.n_layer * per_layer + (1 + b) * d)    # blocks + final LN

    @property
    def n_params_non_embedding(self) -> int:
        """Minus wpe only. wte stays in: tied to lm_head, it IS a matmul. Same as bench.py."""
        return self.n_params - self.block_size * self.n_embd

    @property
    def flops_per_token(self) -> float:
        return gpt_flops_per_token(self.n_params_non_embedding, self.n_layer,
                                   self.n_head, self.head_dim, self.block_size)

    def ddp_static_mem_gb(self, bytes_per_param: int = 16) -> float:
        """fp32 params(4) + grads(4) + Adam m,v(8). Activations EXCLUDED — they depend on
        micro-batch and checkpointing, and are what actually decides OOM."""
        return self.n_params * bytes_per_param / 1e9


MODELS: dict[str, GPTConfig] = {
    "gpt-125m": GPTConfig("gpt-125m", n_layer=12, n_head=12, n_embd=768),
    "gpt-350m": GPTConfig("gpt-350m", n_layer=24, n_head=16, n_embd=1024),
    # head_dim 128: the shape attention kernels are best tuned for.
    "gpt-1.3b": GPTConfig("gpt-1.3b", n_layer=24, n_head=16, n_embd=2048),
}


# Not experiment models: for CPU smoke tests of train.py on the laptop. Kept out of MODELS
# so they never appear in result tables or the param-count test sweep.
DEBUG_MODELS: dict[str, GPTConfig] = {
    "gpt-tiny": GPTConfig("gpt-tiny", n_layer=2, n_head=4, n_embd=64, block_size=64),
}


if __name__ == "__main__":
    hdr = f"{'model':<10} {'params':>8} {'hd':>4} {'GFLOP/tok':>10} {'attn %':>7} {'DDP static GB':>14} {'Chinchilla tok':>15}"
    print(hdr)
    print("-" * len(hdr))
    for c in MODELS.values():
        attn = 12 * c.n_layer * c.n_head * c.head_dim * c.block_size
        print(f"{c.name:<10} {c.n_params/1e6:>7.0f}M {c.head_dim:>4} "
              f"{c.flops_per_token/1e9:>10.2f} {attn/c.flops_per_token*100:>6.1f}% "
              f"{c.ddp_static_mem_gb():>14.1f} {20*c.n_params/1e9:>13.1f}B")
