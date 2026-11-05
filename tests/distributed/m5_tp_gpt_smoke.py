"""M5: Full GPT tensor-parallel forward/backward parity."""

import argparse
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

from model import GPT
from model_configs import DEBUG_MODELS, MODELS
from m5_tp_attention_smoke import TPFusedAttention, reorder_qkv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["gpt-tiny", "gpt-1.3b"],
        default="gpt-tiny",
    )
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)

    try:
        assert dist.get_world_size() == 2
        torch.backends.cuda.matmul.allow_tf32 = False

        cfg = {**MODELS, **DEBUG_MODELS}[args.model]

        # Identical full-model initialization.
        torch.manual_seed(1337)
        reference = GPT(cfg).to(device)

        torch.manual_seed(1337)
        parallel = GPT(cfg).to(device)

        mesh = init_device_mesh("cuda", (2,))

        # Parallelize every transformer block.
        for i, block in enumerate(parallel.transformer.h):
            old_attn = block.attn

            tp_attn = TPFusedAttention(cfg, 2).to(device)

            with torch.no_grad():
                tp_attn.c_attn.weight.copy_(
                    reorder_qkv(old_attn.c_attn.weight, 2)
                )
                tp_attn.c_attn.bias.copy_(
                    reorder_qkv(old_attn.c_attn.bias, 2)
                )
                tp_attn.c_proj.load_state_dict(
                    old_attn.c_proj.state_dict()
                )

            block.attn = tp_attn

            parallel.transformer.h[i] = parallelize_module(
                block,
                mesh,
                {
                    "attn.c_attn": ColwiseParallel(),
                    "attn.c_proj": RowwiseParallel(),
                    "mlp.c_fc": ColwiseParallel(),
                    "mlp.c_proj": RowwiseParallel(),
                },
            )

        # Embeddings and output head remain tied and replicated.
        assert (
            parallel.transformer.wte.weight
            is parallel.lm_head.weight
        )

        if rank == 0:
            print(f"[M5 GPT] model={cfg.name}")
            print(f"[M5 GPT] layers={cfg.n_layer}")
            print("[M5 GPT] TP size=2")
            print("[M5 GPT] embedding/head tied=True")

        generator = torch.Generator(
            device="cpu"
        ).manual_seed(2026)

        x = torch.randint(
            0, 50257, (2, 8), generator=generator
        ).to(device)

        targets = torch.randint(
            0, 50257, (2, 8), generator=generator
        ).to(device)

        # Reference and TP full-model execution.
        with sdpa_kernel(SDPBackend.MATH):
            logits_ref, loss_ref = reference(x, targets)
            logits_tp, loss_tp = parallel(x, targets)

            torch.testing.assert_close(
                logits_tp, logits_ref,
                rtol=3e-4, atol=3e-5,
            )

            loss_ref.backward()
            loss_tp.backward()

        logits_diff = (
            logits_ref.detach() - logits_tp.detach()
        ).abs().max()

        loss_diff = (
            loss_ref.detach() - loss_tp.detach()
        ).abs()

        # Compare all trainable parameter gradients, accounting
        # for column/row sharding and fused-QKV reordering.
        ref_params = dict(reference.named_parameters())
        tp_params = dict(parallel.named_parameters())

        assert ref_params.keys() == tp_params.keys()

        max_grad_diff = torch.zeros((), device=device)

        for name, tp_param in tp_params.items():
            expected = ref_params[name].grad
            actual = tp_param.grad

            assert expected is not None, name
            assert actual is not None, name

            if name.endswith((
                "attn.c_attn.weight",
                "attn.c_attn.bias",
            )):
                expected = reorder_qkv(expected, 2)
                expected = expected.chunk(2, dim=0)[rank]

            elif name.endswith((
                "mlp.c_fc.weight",
                "mlp.c_fc.bias",
            )):
                expected = expected.chunk(2, dim=0)[rank]

            elif name.endswith((
                "attn.c_proj.weight",
                "mlp.c_proj.weight",
            )):
                expected = expected.chunk(2, dim=1)[rank]

            # DTensor gradients expose local tensor shards.
            if hasattr(actual, "to_local"):
                actual = actual.to_local()

            torch.testing.assert_close(
                actual, expected,
                rtol=3e-4, atol=3e-5,
                msg=lambda msg: f"{name}: {msg}",
            )

            diff = (actual - expected).abs().max()
            max_grad_diff = torch.maximum(max_grad_diff, diff)

        for value in (logits_diff, loss_diff, max_grad_diff):
            dist.all_reduce(value, op=dist.ReduceOp.MAX)

        if rank == 0:
            print(
                "[M5 GPT] max logits diff:",
                f"{logits_diff.item():.3e}",
            )
            print(
                "[M5 GPT] loss diff:",
                f"{loss_diff.item():.3e}",
            )
            print(
                "[M5 GPT] max parameter-gradient diff:",
                f"{max_grad_diff.item():.3e}",
            )
            print("[M5 GPT] PASS: full GPT forward parity")
            print("[M5 GPT] PASS: all parameter gradients")
            print("[M5 GPT] PASS: tied embeddings preserved")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
