"""
M6 precision utilities.

Replace only the four internal GPT block projections with
Transformer Engine Linear layers.

Keep embeddings, final language-model head, LayerNorm and
PyTorch SDPA unchanged.

Explicitly cast TE Linear inputs to BF16 for compatibility
with the default NVFP4 Random Hadamard Transform.
"""

import torch
import torch.nn as nn
import transformer_engine.pytorch as te


class BF16InputLinear(te.Linear):
    """TE Linear with an explicit BF16 input boundary."""

    def forward(self, inp, *args, **kwargs):
        inp = inp.to(dtype=torch.bfloat16)
        return super().forward(inp, *args, **kwargs)


def convert_internal_linears(model):
    """Convert GPT block projections without breaking weight tying."""

    count = 0

    for block in model.transformer.h:
        locations = (
            (block.attn, "c_attn"),
            (block.attn, "c_proj"),
            (block.mlp, "c_fc"),
            (block.mlp, "c_proj"),
        )

        for parent, name in locations:
            original = getattr(parent, name)

            if not isinstance(original, nn.Linear):
                raise TypeError(
                    f"Expected nn.Linear at {name}; "
                    f"found {type(original).__name__}"
                )

            replacement = BF16InputLinear(
                original.in_features,
                original.out_features,
                bias=original.bias is not None,
                params_dtype=original.weight.dtype,
                device=original.weight.device,
            )

            with torch.no_grad():
                replacement.weight.copy_(original.weight)

                if original.bias is not None:
                    replacement.bias.copy_(original.bias)

            assert torch.equal(
                replacement.weight, original.weight
            )

            if original.bias is not None:
                assert torch.equal(
                    replacement.bias, original.bias
                )

            setattr(parent, name, replacement)
            count += 1

    assert model.lm_head.weight is model.transformer.wte.weight

    return count
