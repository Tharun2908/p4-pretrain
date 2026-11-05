"""M5: TP=2 x DP=2 distributed checkpoint round-trip."""

import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
)
from torch.nn.attention import SDPBackend, sdpa_kernel

from model import GPT
from model_configs import DEBUG_MODELS
from tp_parallel import (
    parallelize_gpt_tp,
    sync_dp_grads_,
    clip_grad_norm_tp_,
)

CHECKPOINT = "logs/m5_dcp_roundtrip_ckpt"


class TrainingState(Stateful):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_sd, optim_sd = get_state_dict(
            self.model, self.optimizer
        )
        return {"model": model_sd, "optimizer": optim_sd}

    def load_state_dict(self, state):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )


def local(t):
    return t.to_local() if isinstance(t, DTensor) else t


def snapshot(model, optimizer):
    """Copy all local weights and AdamW state for comparison."""
    result = {}

    for name, param in model.named_parameters():
        result[name] = {
            "weight": local(param.detach()).cpu().clone()
        }

        for key, value in optimizer.state[param].items():
            if torch.is_tensor(value):
                result[name][key] = (
                    local(value.detach()).cpu().clone()
                )

    return result


@torch.no_grad()
def max_snapshot_diff(expected, model, optimizer, device):
    actual = snapshot(model, optimizer)
    assert expected.keys() == actual.keys()

    maximum = 0.0

    for name in expected:
        assert expected[name].keys() == actual[name].keys(), name

        for key in expected[name]:
            a = expected[name][key]
            b = actual[name][key]

            assert a.shape == b.shape, (name, key)

            diff = (a.double() - b.double()).abs().max().item()
            maximum = max(maximum, diff)

    result = torch.tensor(maximum, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return result.item()


def train_step(model, optimizer, dp_rank, dp_mesh, tp_mesh,
               device, step):
    generator = torch.Generator("cpu").manual_seed(2026 + step)

    x = torch.randint(
        0, 50257, (4, 8), generator=generator
    ).to(device)
    y = torch.randint(
        0, 50257, (4, 8), generator=generator
    ).to(device)

    x = x.chunk(2, dim=0)[dp_rank].contiguous()
    y = y.chunk(2, dim=0)[dp_rank].contiguous()

    optimizer.zero_grad(set_to_none=True)

    with sdpa_kernel(SDPBackend.MATH):
        _, loss = model(x, y)
        loss.backward()

    sync_dp_grads_(model, dp_mesh)
    clip_grad_norm_tp_(model, 0.1, tp_mesh=tp_mesh)

    optimizer.step()
    return loss.item()


def main():
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
        dp_mesh = mesh["dp"]
        tp_mesh = mesh["tp"]
        dp_rank = dp_mesh.get_local_rank()

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
            foreach=False,
            fused=False,
        )

        state = {"training": TrainingState(model, optimizer)}

        # Step 1 creates weights and AdamW momentum/variance.
        loss1 = train_step(
            model, optimizer, dp_rank, dp_mesh, tp_mesh,
            device, step=1
        )

        saved_snapshot = snapshot(model, optimizer)

        # All ranks participate in distributed checkpoint saving.
        dcp.save(state, checkpoint_id=CHECKPOINT)

        if rank == 0:
            print(
                f"[M5 DCP] Saved after step 1; loss={loss1:.6f}",
                flush=True,
            )

        # Continue uninterrupted to step 2.
        train_step(
            model, optimizer, dp_rank, dp_mesh, tp_mesh,
            device, step=2
        )
        uninterrupted = snapshot(model, optimizer)

        # Restore the state saved AFTER step 1.
        dcp.load(state, checkpoint_id=CHECKPOINT)

        restore_diff = max_snapshot_diff(
            saved_snapshot, model, optimizer, device
        )

        # Replay step 2 with identical inputs.
        train_step(
            model, optimizer, dp_rank, dp_mesh, tp_mesh,
            device, step=2
        )

        resume_diff = max_snapshot_diff(
            uninterrupted, model, optimizer, device
        )

        if rank == 0:
            print(
                f"[M5 DCP] restored-state max diff: "
                f"{restore_diff:.3e}",
                flush=True,
            )
            print(
                f"[M5 DCP] resumed-step max diff: "
                f"{resume_diff:.3e}",
                flush=True,
            )

        assert restore_diff < 1e-6
        assert resume_diff < 1e-6

        if rank == 0:
            print("[M5 DCP] PASS: model checkpoint")
            print("[M5 DCP] PASS: AdamW optimizer state")
            print("[M5 DCP] PASS: resumed parameter updates")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
