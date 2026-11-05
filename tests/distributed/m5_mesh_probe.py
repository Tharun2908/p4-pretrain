"""M5: Verify TP=2 x DP=2 device mesh and process groups."""

import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group("nccl", device_id=device)

    try:
        assert dist.get_world_size() == 4

        mesh = init_device_mesh(
            "cuda",
            (2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        tp_mesh = mesh["tp"]
        dp_mesh = mesh["dp"]

        tp_rank = tp_mesh.get_local_rank()
        dp_rank = dp_mesh.get_local_rank()

        tp_group = dist.get_process_group_ranks(
            tp_mesh.get_group()
        )
        dp_group = dist.get_process_group_ranks(
            dp_mesh.get_group()
        )

        expected_tp = [0, 1] if rank < 2 else [2, 3]
        expected_dp = [rank % 2, rank % 2 + 2]

        assert tp_group == expected_tp
        assert dp_group == expected_dp

        # Validate that group collectives communicate
        # with exactly the expected partners.
        tp_sum = torch.tensor(
            float(rank + 1), device=device
        )
        dp_sum = tp_sum.clone()

        dist.all_reduce(
            tp_sum, group=tp_mesh.get_group()
        )
        dist.all_reduce(
            dp_sum, group=dp_mesh.get_group()
        )

        assert tp_sum.item() == sum(r + 1 for r in expected_tp)
        assert dp_sum.item() == sum(r + 1 for r in expected_dp)

        # Print ranks in order.
        for r in range(4):
            dist.barrier()
            if rank == r:
                print(
                    f"[M5 MESH] rank={rank} "
                    f"dp_rank={dp_rank} tp_rank={tp_rank} "
                    f"TP={tp_group} DP={dp_group} "
                    f"tp_sum={tp_sum.item():.0f} "
                    f"dp_sum={dp_sum.item():.0f}",
                    flush=True,
                )

        dist.barrier()

        if rank == 0:
            print("[M5 MESH] PASS: TP=2 x DP=2 topology")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
