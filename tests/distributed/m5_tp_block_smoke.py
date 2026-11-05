"""M5: complete transformer block TP=2 correctness test."""

import os

import torch
import torch.distributed as dist

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.nn.attention import sdpa_kernel, SDPBackend

from model import Block
from model_configs import MODELS
from m5_tp_attention_smoke import TPFusedAttention, reorder_qkv


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

        # Construct identical transformer blocks.
        torch.manual_seed(1337)
        reference = Block(cfg).to(device)

        torch.manual_seed(1337)
        parallel = Block(cfg).to(device)

        # Replace attention with our local-head implementation.
        tp_attn = TPFusedAttention(cfg, tp_size).to(device)

        with torch.no_grad():
            tp_attn.c_attn.weight.copy_(
                reorder_qkv(
                    reference.attn.c_attn.weight, tp_size
                )
            )
            tp_attn.c_attn.bias.copy_(
                reorder_qkv(
                    reference.attn.c_attn.bias, tp_size
                )
            )
            tp_attn.c_proj.load_state_dict(
                reference.attn.c_proj.state_dict()
            )

        parallel.attn = tp_attn

        mesh = init_device_mesh("cuda", (tp_size,))

        # Tensor-parallelize BOTH attention and MLP.
        parallel = parallelize_module(
            parallel,
            mesh,
            {
                "attn.c_attn": ColwiseParallel(),
                "attn.c_proj": RowwiseParallel(),
                "mlp.c_fc": ColwiseParallel(),
                "mlp.c_proj": RowwiseParallel(),
            },
        )

        if rank == 0:
            print("[M5 BLOCK] TP=2")
            print("[M5 BLOCK] Attention + MLP sharded")
            print("[M5 BLOCK] LayerNorm + residual replicated")

        generator = torch.Generator(
            device="cpu"
        ).manual_seed(2026)

        x = torch.randn(
            2, 8, cfg.n_embd, generator=generator
        ).to(device)

        x_ref = x.clone().requires_grad_(True)
        x_tp = x.clone().requires_grad_(True)

        # FP32 + math attention for correctness isolation.
        with sdpa_kernel(SDPBackend.MATH):
            y_ref = reference(x_ref)
            y_tp = parallel(x_tp)

            assert y_ref.shape == y_tp.shape == (2, 8, 2048)

            torch.testing.assert_close(
                y_tp, y_ref, rtol=1e-4, atol=1e-5
            )

            y_ref.square().mean().backward()
            y_tp.square().mean().backward()

        # Compare complete block output.
        output_diff = (
            y_ref.detach() - y_tp.detach()
        ).abs().max()

        # Compare input gradients through both residual paths.
        input_grad_diff = (
            x_ref.grad - x_tp.grad
        ).abs().max()

        torch.testing.assert_close(
            x_tp.grad, x_ref.grad,
            rtol=1e-4, atol=1e-5,
        )

        # Check that replicated LayerNorm gradients match.
        ln_diffs = []

        for name in ("ln_1", "ln_2"):
            ref_ln = getattr(reference, name)
            tp_ln = getattr(parallel, name)

            for param in ("weight", "bias"):
                ref_grad = getattr(ref_ln, param).grad
                tp_grad = getattr(tp_ln, param).grad

                torch.testing.assert_close(
                    tp_grad, ref_grad,
                    rtol=1e-4, atol=1e-5,
                )

                ln_diffs.append(
                    (tp_grad - ref_grad).abs().max()
                )

        ln_grad_diff = torch.stack(ln_diffs).max()

        # Compare locally sharded MLP weight gradients.
        fc_expected = (
            reference.mlp.c_fc.weight.grad
            .chunk(tp_size, dim=0)[rank]
        )
        fc_actual = (
            parallel.mlp.c_fc.weight.grad.to_local()
        )

        torch.testing.assert_close(
            fc_actual, fc_expected,
            rtol=1e-4, atol=1e-5,
        )

        fc_grad_diff = (
            fc_actual - fc_expected
        ).abs().max()

        metrics = {
            "Output": output_diff,
            "Input gradient": input_grad_diff,
            "LayerNorm gradient": ln_grad_diff,
            "MLP c_fc gradient": fc_grad_diff,
        }

        for value in metrics.values():
            dist.all_reduce(
                value, op=dist.ReduceOp.MAX
            )

        if rank == 0:
            for name, value in metrics.items():
                print(
                    f"[M5 BLOCK] {name} max diff: "
                    f"{value.item():.3e}"
                )

            print("[M5 BLOCK] PASS: full block forward")
            print("[M5 BLOCK] PASS: full block backward")
            print("[M5 BLOCK] PASS: LayerNorm gradients")
            print("[M5 BLOCK] PASS: sharded MLP gradients")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
