"""M5: TP=2 AdamW parameter-update parity against full GPT."""

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
from model_configs import DEBUG_MODELS
from m5_tp_attention_smoke import TPFusedAttention, reorder_qkv


def build_models(device):
    cfg = DEBUG_MODELS["gpt-tiny"]

    torch.manual_seed(1337)
    reference = GPT(cfg).to(device)

    torch.manual_seed(1337)
    parallel = GPT(cfg).to(device)

    mesh = init_device_mesh("cuda", (2,))

    for i, block in enumerate(parallel.transformer.h):
        old = block.attn
        tp_attn = TPFusedAttention(cfg, 2).to(device)

        with torch.no_grad():
            tp_attn.c_attn.weight.copy_(
                reorder_qkv(old.c_attn.weight, 2)
            )
            tp_attn.c_attn.bias.copy_(
                reorder_qkv(old.c_attn.bias, 2)
            )
            tp_attn.c_proj.load_state_dict(
                old.c_proj.state_dict()
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

    assert parallel.transformer.wte.weight is parallel.lm_head.weight
    return reference, parallel


def make_optimizer(model):
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
        foreach=False,
        fused=False,
    )


def compare_parameters(reference, parallel, device):
    ref_params = dict(reference.named_parameters())
    tp_params = dict(parallel.named_parameters())
    assert ref_params.keys() == tp_params.keys()

    max_diff = torch.zeros((), device=device)

    with torch.no_grad():
        for name, tp_param in tp_params.items():
            expected = ref_params[name].detach()

            # Full tensor reconstruction gathers TP shards.
            actual = (
                tp_param.full_tensor()
                if hasattr(tp_param, "full_tensor")
                else tp_param
            ).detach()

            if name.endswith((
                "attn.c_attn.weight",
                "attn.c_attn.bias",
            )):
                expected = reorder_qkv(expected, 2)

            torch.testing.assert_close(
                actual,
                expected,
                rtol=5e-4,
                atol=5e-5,
                msg=lambda msg: f"{name}: {msg}",
            )

            diff = (actual - expected).abs().max()
            max_diff = torch.maximum(max_diff, diff)

    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
    return max_diff.item()


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)

    try:
        assert dist.get_world_size() == 2
        torch.backends.cuda.matmul.allow_tf32 = False

        reference, parallel = build_models(device)

        ref_opt = make_optimizer(reference)
        tp_opt = make_optimizer(parallel)

        generator = torch.Generator(
            device="cpu"
        ).manual_seed(2026)

        x = torch.randint(
            0, 50257, (2, 8), generator=generator
        ).to(device)

        targets = torch.randint(
            0, 50257, (2, 8), generator=generator
        ).to(device)

        for step in range(1, 4):
            ref_opt.zero_grad(set_to_none=True)
            tp_opt.zero_grad(set_to_none=True)

            with sdpa_kernel(SDPBackend.MATH):
                _, ref_loss = reference(x, targets)
                _, tp_loss = parallel(x, targets)

                ref_loss.backward()
                tp_loss.backward()

            # No clipping yet: isolate optimizer correctness.
            ref_opt.step()
            tp_opt.step()

            max_diff = compare_parameters(
                reference, parallel, device
            )

            if rank == 0:
                print(
                    f"[M5 OPT] step={step} "
                    f"ref_loss={ref_loss.item():.6f} "
                    f"tp_loss={tp_loss.item():.6f} "
                    f"max_param_diff={max_diff:.3e}",
                    flush=True,
                )

        if rank == 0:
            print("[M5 OPT] PASS: TP AdamW parameter-update parity")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
