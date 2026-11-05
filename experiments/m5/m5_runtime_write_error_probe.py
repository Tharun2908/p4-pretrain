"""M5: verify collective handling of rank-local checkpoint write errors."""

import os
from pathlib import Path

import torch
import torch.distributed as dist

import tp_checkpoint
import train


ROOT = Path("logs/m5_train_runtime_write_failure_probe")

original_replace = tp_checkpoint.os.replace


def injected_replace(src, dst):
    rank = int(os.environ.get("RANK", "-1"))

    if (
        rank == 2
        and str(src).endswith("runtime_rank2.pt.tmp")
    ):
        raise OSError("INJECTED rank-2 runtime write failure")

    return original_replace(src, dst)


# Inject an error only during rank 2's runtime-state publication.
tp_checkpoint.os.replace = injected_replace


def main():
    rank = int(os.environ["RANK"])

    try:
        train.main()

    except RuntimeError as exc:
        message = str(exc)

        assert "Rank-local checkpoint write failed" in message
        assert "INJECTED rank-2 runtime write failure" in message

        print(
            f"[M5 RUNTIME ERROR] rank={rank} received expected error",
            flush=True,
        )

        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

        passed = torch.tensor(1, device=device)
        dist.all_reduce(passed, op=dist.ReduceOp.SUM)

        assert passed.item() == 4

        dist.barrier()

        if rank == 0:
            assert not (ROOT / "latest.json").exists(), (
                "Incomplete checkpoint was incorrectly published"
            )

            print("[M5 RUNTIME ERROR] PASS: all ranks received error")
            print("[M5 RUNTIME ERROR] PASS: no distributed deadlock")
            print("[M5 RUNTIME ERROR] PASS: incomplete checkpoint not published")

    else:
        raise AssertionError(
            "Expected checkpoint write failure did not occur"
        )

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
