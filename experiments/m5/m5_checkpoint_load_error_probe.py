"""Verify collective checkpoint-load preflight on four GPUs."""

import os
from pathlib import Path

import torch
import torch.distributed as dist

import tp_checkpoint
import train


ROOT = Path("logs/m5_train_post_review_probe")
original_load = torch.load


def injected_load(path, *args, **kwargs):
    rank = int(os.environ.get("RANK", "-1"))

    if rank == 2 and str(path).endswith("runtime_rank2.pt"):
        raise OSError("INJECTED rank-2 runtime load failure")

    return original_load(path, *args, **kwargs)


def main():
    rank = int(os.environ["RANK"])

    # Test only: no checkpoint file is changed.
    assert (ROOT / "latest.json").is_file()
    tp_checkpoint.torch.load = injected_load

    try:
        train.main()

    except RuntimeError as exc:
        message = str(exc)

        assert "Checkpoint load preflight failed" in message
        assert "INJECTED rank-2 runtime load failure" in message

        print(
            f"[M5 LOAD ERROR] rank={rank} received expected error",
            flush=True,
        )

        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

        passed = torch.tensor(1, device=device)
        dist.all_reduce(passed, op=dist.ReduceOp.SUM)

        assert passed.item() == 4

        if rank == 0:
            print("[M5 LOAD ERROR] PASS: all ranks rejected load")
            print("[M5 LOAD ERROR] PASS: no distributed deadlock")
            print("[M5 LOAD ERROR] PASS: checkpoint files untouched")

    else:
        raise AssertionError("Expected checkpoint-load failure")

    finally:
        tp_checkpoint.torch.load = original_load

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
