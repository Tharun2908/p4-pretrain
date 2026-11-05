"""Verify that checkpoint creation errors reach every distributed rank."""

import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from tp_checkpoint import save_tp_checkpoint


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group("nccl", device_id=device)

    try:
        root = Path("logs/m5_train_resume_probe")

        # This checkpoint already exists from our recovery tests.
        assert (root / "step_00000010").is_dir()

        try:
            # Model/optimizer are deliberately unused: the
            # existing-directory guard must fail first.
            save_tp_checkpoint(
                root=root,
                completed_step=10,
                model=None,
                optimizer=None,
                batches=SimpleNamespace(i=0),
                args=None,
                cfg=None,
                world=dist.get_world_size(),
                device=device,
            )
        except RuntimeError as exc:
            assert "Checkpoint already exists" in str(exc)
            print(
                f"[M5 ERROR] rank={rank} received expected error",
                flush=True,
            )
        else:
            raise AssertionError(
                "Existing checkpoint was not rejected"
            )

        passed = torch.tensor(1, device=device)
        dist.all_reduce(passed, op=dist.ReduceOp.SUM)

        assert passed.item() == dist.get_world_size()

        if rank == 0:
            print("[M5 ERROR] PASS: all ranks received error")
            print("[M5 ERROR] PASS: no distributed deadlock")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
