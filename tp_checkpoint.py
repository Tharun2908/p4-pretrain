"""M5: restartable DCP checkpoints for TP and TP+DP training."""

import json
import os
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
)


class TrainingState(Stateful):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_sd, optim_sd = get_state_dict(
            self.model, self.optimizer
        )
        return {
            "model": model_sd,
            "optimizer": optim_sd,
        }

    def load_state_dict(self, state):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )


def run_signature(args, cfg, world):
    """Configuration fields that must match on restart."""
    keys = (
        "strategy",
        "backend",
        "data",
        "micro_bs",
        "grad_accum",
        "steps",
        "lr",
        "lr_warmup",
        "weight_decay",
        "grad_clip",
        "seed",
        "compile",
        "deterministic",
    )

    result = {key: getattr(args, key) for key in keys}

    result.update(
        model=cfg.name,
        block_size=cfg.block_size,
        world_size=world,
    )
    return result


def validate_data(batches):
    if not hasattr(batches, "i"):
        raise RuntimeError(
            "M5 restart integration currently supports "
            "SyntheticBatches only"
        )


def save_tp_checkpoint(
    root, completed_step, model, optimizer,
    batches, args, cfg, world, device,
):
    """Collective save, with an atomic latest-checkpoint marker."""
    validate_data(batches)

    rank = dist.get_rank()
    root = Path(root)
    target = root / f"step_{completed_step:08d}"

    # All ranks must agree whether checkpoint creation succeeded.
    # Otherwise rank 0 could raise while other ranks wait at a barrier.
    create_error = [None]

    if rank == 0:
        try:
            if target.exists():
                if not target.is_dir() or target.is_symlink():
                    raise FileExistsError(
                        f"Checkpoint path is not a directory: {target}"
                    )

                latest_path = root / "latest.json"

                if not latest_path.is_file():
                    raise RuntimeError(
                        "Cannot quarantine existing checkpoint without "
                        f"a valid latest.json: {target}"
                    )

                if latest_path.exists():
                    latest = json.loads(latest_path.read_text())
                    latest_name = latest["directory"]

                    if (
                        not isinstance(latest_name, str)
                        or not latest_name.startswith("step_")
                        or not latest_name[5:].isdigit()
                    ):
                        raise ValueError(
                            f"Invalid latest checkpoint pointer: {latest_name}"
                        )

                    published_step = int(latest_name[5:])

                    if completed_step <= published_step:
                        raise FileExistsError(
                            f"Checkpoint already exists: {target}"
                        )

                # Only a newer, unpublished checkpoint attempt
                # can be quarantined. Never delete its contents.
                orphan = root / (
                    f".{target.name}.orphan-{uuid.uuid4().hex}"
                )
                target.rename(orphan)

                print(
                    f"[M5 CKPT] quarantined incomplete checkpoint: "
                    f"{target} -> {orphan}",
                    flush=True,
                )

            target.mkdir(parents=True)

        except Exception as exc:
            create_error[0] = f"{type(exc).__name__}: {exc}"

    dist.broadcast_object_list(create_error, src=0)

    if create_error[0] is not None:
        raise RuntimeError(create_error[0])

    dist.barrier()

    training = {
        "training": TrainingState(model, optimizer)
    }

    # Every rank participates in the distributed save.
    dcp.save(training, checkpoint_id=str(target))

    # Rank-specific runtime state.
    runtime = {
        "completed_step": completed_step,
        "data_cursor": batches.i,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device),
    }

    final = target / f"runtime_rank{rank}.pt"
    temporary = target / f"runtime_rank{rank}.pt.tmp"

    # Each rank writes its own runtime state atomically.
    # Collect failures before entering the next synchronization.
    runtime_error = None

    try:
        torch.save(runtime, temporary)
        os.replace(temporary, final)
    except Exception as exc:
        runtime_error = (
            f"rank {rank}: {type(exc).__name__}: {exc}"
        )

    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, runtime_error)

    failures = [error for error in errors if error is not None]

    if failures:
        raise RuntimeError(
            "Rank-local checkpoint write failed:\n"
            + "\n".join(failures)
        )

    dist.barrier()

    # Publish checkpoint metadata only after all distributed
    # model and rank-local runtime writes have completed.
    # Propagate rank-0 filesystem errors to every rank.
    publish_error = [None]

    if rank == 0:
        try:
            metadata = {
                "version": 1,
                "completed_step": completed_step,
                "signature": run_signature(args, cfg, world),
            }

            (target / "metadata.json").write_text(
                json.dumps(metadata, indent=2)
            )

            latest_tmp = root / "latest.json.tmp"
            latest = root / "latest.json"

            latest_tmp.write_text(
                json.dumps({"directory": target.name})
            )
            os.replace(latest_tmp, latest)

            print(
                f"[M5 CKPT] saved completed_step={completed_step} "
                f"path={target}",
                flush=True,
            )
        except OSError as exc:
            publish_error[0] = f"{type(exc).__name__}: {exc}"

    dist.broadcast_object_list(publish_error, src=0)

    if publish_error[0] is not None:
        raise RuntimeError(
            f"Checkpoint publication failed: {publish_error[0]}"
        )

    dist.barrier()


def load_tp_checkpoint(
    root, model, optimizer,
    batches, args, cfg, world, device,
):
    """Collective DCP load plus rank-specific runtime restoration."""
    validate_data(batches)

    rank = dist.get_rank()
    root = Path(root)

    # Validate every rank's checkpoint files before collective DCP load.
    preflight_result = None

    try:
        latest = json.loads(
            (root / "latest.json").read_text()
        )
        target = root / latest["directory"]

        metadata = json.loads(
            (target / "metadata.json").read_text()
        )

        if metadata["version"] != 1:
            raise RuntimeError("Unsupported checkpoint version")

        expected_signature = run_signature(args, cfg, world)

        if metadata["signature"] != expected_signature:
            raise RuntimeError(
                "Checkpoint configuration does not match "
                "the current training configuration"
            )

        runtime = torch.load(
            target / f"runtime_rank{rank}.pt",
            map_location="cpu",
            weights_only=True,
        )

        completed_step = metadata["completed_step"]

        if runtime["completed_step"] != completed_step:
            raise RuntimeError(
                "Checkpoint step mismatch between metadata and runtime"
            )

        preflight_result = (True, str(target), completed_step)

    except Exception as exc:
        preflight_result = (
            False,
            f"rank {rank}: {type(exc).__name__}: {exc}",
            None,
        )

    results = [None] * dist.get_world_size()
    dist.all_gather_object(results, preflight_result)

    failures = [r[1] for r in results if not r[0]]
    if failures:
        raise RuntimeError(
            "Checkpoint load preflight failed:\n"
            + "\n".join(failures)
        )

    # All ranks must load the same checkpoint step.
    choices = {(r[1], r[2]) for r in results}
    if len(choices) != 1:
        raise RuntimeError(
            "Ranks selected different checkpoints"
        )

    training = {
        "training": TrainingState(model, optimizer)
    }

    dcp.load(training, checkpoint_id=str(target))

    # Restore runtime state AFTER loading model/optimizer.
    batches.i = runtime["data_cursor"]
    torch.set_rng_state(runtime["cpu_rng"])
    torch.cuda.set_rng_state(
        runtime["cuda_rng"], device=device
    )

    dist.barrier()

    if rank == 0:
        print(
            f"[M5 CKPT] restored completed_step={completed_step} "
            f"data_cursor={batches.i}",
            flush=True,
        )

    return completed_step
