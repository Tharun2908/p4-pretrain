"""M5: checkpoint recovery across two separate torchrun launches."""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh

from model import GPT
from model_configs import DEBUG_MODELS
from tp_parallel import parallelize_gpt_tp
from m5_dcp_roundtrip import (
    TrainingState,
    train_step,
    snapshot,
    max_snapshot_diff,
)

CKPT = "logs/m5_dcp_restart_ckpt"


def build_optimizer(model):
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
        fused=False,
        foreach=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["save", "resume"], required=True)
    args = parser.parse_args()

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
        dp_mesh, tp_mesh = mesh["dp"], mesh["tp"]
        dp_rank = dp_mesh.get_local_rank()

        torch.manual_seed(1337)
        model = GPT(DEBUG_MODELS["gpt-tiny"]).to(device)
        parallelize_gpt_tp(model, tp_mesh)

        optimizer = build_optimizer(model)
        state = {"training": TrainingState(model, optimizer)}

        expected_path = Path(
            f"logs/m5_dcp_restart_expected_rank{rank}.pt"
        )

        if args.phase == "save":
            # Step 1: checkpoint this completed optimizer update.
            train_step(
                model, optimizer, dp_rank,
                dp_mesh, tp_mesh, device, step=1
            )
            after_step1 = snapshot(model, optimizer)

            dcp.save(state, checkpoint_id=CKPT)

            # Step 2: uninterrupted reference trajectory.
            train_step(
                model, optimizer, dp_rank,
                dp_mesh, tp_mesh, device, step=2
            )
            after_step2 = snapshot(model, optimizer)

            torch.save(
                {
                    "after_step1": after_step1,
                    "after_step2": after_step2,
                },
                expected_path,
            )

            dist.barrier()
            if rank == 0:
                print("[M5 RESTART] Saved checkpoint after step 1")
                print("[M5 RESTART] Recorded uninterrupted step 2")
                print("[M5 RESTART] SAVE PHASE PASS", flush=True)

        else:
            # This phase starts in a NEW torchrun process.
            assert len(optimizer.state) == 0

            expected = torch.load(
                expected_path,
                map_location="cpu",
                weights_only=True,
            )

            dcp.load(state, checkpoint_id=CKPT)

            # Explicitly verify optimizer state was recovered.
            assert len(optimizer.state) > 0, (
                "Optimizer state was not populated by DCP"
            )

            restore_diff = max_snapshot_diff(
                expected["after_step1"],
                model, optimizer, device,
            )

            assert restore_diff < 1e-6

            # Replay step 2 from the restored checkpoint.
            train_step(
                model, optimizer, dp_rank,
                dp_mesh, tp_mesh, device, step=2
            )

            replay_diff = max_snapshot_diff(
                expected["after_step2"],
                model, optimizer, device,
            )

            if rank == 0:
                print(
                    f"[M5 RESTART] restored-state diff: "
                    f"{restore_diff:.3e}"
                )
                print(
                    f"[M5 RESTART] replayed-step diff: "
                    f"{replay_diff:.3e}"
                )

            assert replay_diff < 1e-6

            if rank == 0:
                print("[M5 RESTART] PASS: fresh-process DCP load")
                print("[M5 RESTART] PASS: AdamW state recovered")
                print("[M5 RESTART] PASS: deterministic step replay")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
