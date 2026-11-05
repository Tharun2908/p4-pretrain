"""M5: compare interrupted/resumed vs uninterrupted DCP checkpoints."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from model import GPT
from model_configs import DEBUG_MODELS
from tp_parallel import parallelize_gpt_tp
from tp_checkpoint import TrainingState


RESUMED = Path("logs/m5_train_crash_probe")
CONTINUOUS = Path("logs/m5_train_continuous_probe")


def local(t):
    return t.to_local() if isinstance(t, DTensor) else t


def build_model_and_optimizer(device, tp_mesh):
    torch.manual_seed(1337)

    model = GPT(DEBUG_MODELS["gpt-tiny"]).to(device)
    parallelize_gpt_tp(model, tp_mesh)

    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=6e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=False,
        foreach=False,
    )
    return model, optimizer


def snapshot(model, optimizer):
    result = {}

    for name, param in model.named_parameters():
        assert param in optimizer.state, name

        values = {
            "weight": local(param.detach()).cpu().clone()
        }

        for key, value in optimizer.state[param].items():
            if torch.is_tensor(value):
                values[key] = local(value.detach()).cpu().clone()
            elif isinstance(value, (int, float)):
                values[key] = torch.tensor(value)
            else:
                raise TypeError(f"Unexpected optimizer state: {name}.{key}")

        result[name] = values

    return result


def load_checkpoint(root, step, device, tp_mesh, rank):
    folder = root / f"step_{step:08d}"

    metadata = json.loads(
        (folder / "metadata.json").read_text()
    )

    runtime = torch.load(
        folder / f"runtime_rank{rank}.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert metadata["completed_step"] == step
    assert runtime["completed_step"] == step

    model, optimizer = build_model_and_optimizer(
        device, tp_mesh
    )

    state = {
        "training": TrainingState(model, optimizer)
    }

    # Every rank participates in both DCP loads.
    dcp.load(state, checkpoint_id=str(folder))

    assert len(optimizer.state) > 0

    result = snapshot(model, optimizer)

    return result, runtime, metadata


def compare(a, b, runtime_a, runtime_b, meta_a, meta_b,
            step, rank, device):
    assert meta_a["signature"] == meta_b["signature"]
    assert a.keys() == b.keys()

    max_diff = 0.0
    nonexact = 0
    tensors_checked = 0

    for name in a:
        assert a[name].keys() == b[name].keys(), name

        for key in a[name]:
            x, y = a[name][key], b[name][key]

            assert x.shape == y.shape, (name, key)
            assert x.dtype == y.dtype, (name, key)

            tensors_checked += 1

            if not torch.equal(x, y):
                nonexact += 1

            if x.numel():
                diff = (x.double() - y.double()).abs().max().item()
                max_diff = max(max_diff, diff)

    cursor_match = (
        runtime_a["data_cursor"] == runtime_b["data_cursor"]
    )
    cpu_rng_match = torch.equal(
        runtime_a["cpu_rng"], runtime_b["cpu_rng"]
    )
    cuda_rng_match = torch.equal(
        runtime_a["cuda_rng"], runtime_b["cuda_rng"]
    )

    if not (cursor_match and cpu_rng_match and cuda_rng_match):
        nonexact += 1

    # Reduce verification results across all four GPU ranks.
    maximum = torch.tensor(
        max_diff, dtype=torch.float64, device=device
    )
    mismatches = torch.tensor(
        nonexact, dtype=torch.int64, device=device
    )

    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    dist.all_reduce(mismatches, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(
            f"[M5 COMPARE] step={step} "
            f"parameter_state_tensors_per_rank={tensors_checked}"
        )
        print(
            f"[M5 COMPARE] step={step} "
            f"max_model_optimizer_diff={maximum.item():.3e}"
        )
        print(
            f"[M5 COMPARE] step={step} "
            f"nonexact_checks_all_ranks={mismatches.item()}"
        )
        print(
            f"[M5 COMPARE] step={step} "
            f"data_cursor={runtime_a['data_cursor']}"
        )

    assert maximum.item() == 0.0, "Model/optimizer state mismatch"
    assert mismatches.item() == 0, "Non-identical checkpoint contents"


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)

    try:
        assert dist.get_world_size() == 4

        mesh = init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("dp", "tp")
        )
        tp_mesh = mesh["tp"]

        for step in (10, 20):
            a, runtime_a, meta_a = load_checkpoint(
                RESUMED, step, device, tp_mesh, rank
            )
            b, runtime_b, meta_b = load_checkpoint(
                CONTINUOUS, step, device, tp_mesh, rank
            )

            compare(
                a, b, runtime_a, runtime_b,
                meta_a, meta_b, step, rank, device
            )

            if rank == 0:
                print(f"[M5 COMPARE] PASS: step {step} exact match")

        if rank == 0:
            print("[M5 COMPARE] PASS: model and AdamW state")
            print("[M5 COMPARE] PASS: data cursor and RNG states")
            print("[M5 COMPARE] PASS: interrupted vs continuous")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
