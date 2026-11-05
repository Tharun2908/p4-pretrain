"""Verify missing latest.json prevents automatic checkpoint quarantine."""

import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from tp_checkpoint import save_tp_checkpoint


ROOT = Path("logs/m5_train_missing_marker_probe")


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)

    try:
        if rank == 0:
            assert not ROOT.exists(), "Test directory already exists"

            target = ROOT / "step_00000001"
            target.mkdir(parents=True)

            (target / "preserve_me.txt").write_text(
                "Existing checkpoint contents must not be moved.\n"
            )

        dist.barrier()

        try:
            save_tp_checkpoint(
                root=ROOT,
                completed_step=1,
                model=None,
                optimizer=None,
                batches=SimpleNamespace(i=0),
                args=None,
                cfg=None,
                world=4,
                device=device,
            )
        except RuntimeError as exc:
            assert (
                "Cannot quarantine existing checkpoint" in str(exc)
            ), str(exc)

            print(
                f"[M5 MISSING MARKER] rank={rank} received expected error",
                flush=True,
            )
        else:
            raise AssertionError("Expected missing-marker rejection")

        passed = torch.tensor(1, device=device)
        dist.all_reduce(passed, op=dist.ReduceOp.SUM)
        assert passed.item() == 4

        if rank == 0:
            target = ROOT / "step_00000001"

            assert (target / "preserve_me.txt").is_file()
            assert not (ROOT / "latest.json").exists()
            assert not list(ROOT.glob(".step_00000001.orphan-*"))

            print("[M5 MISSING MARKER] PASS: all ranks rejected save")
            print("[M5 MISSING MARKER] PASS: existing files preserved")
            print("[M5 MISSING MARKER] PASS: no checkpoint published")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
