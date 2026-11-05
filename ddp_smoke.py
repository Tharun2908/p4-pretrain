"""
p4/ddp_smoke.py — M0. Run this before anything else.

    cd /workspace/p4/p4-pretrain
    NCCL_DEBUG=INFO torchrun --nproc_per_node=4 ddp_smoke.py 2>&1 | tee /tmp/nccl_debug.txt
    torchrun --nproc_per_node=4 ddp_smoke.py --prediction "peak busbw 200-240 GB/s"

It does four things:

  1. Checks for co-tenant processes on the GPUs. On a shared node this is the leading
     explanation for a disappointing bandwidth number, so it's checked *first* rather
     than guessed at afterwards.
  2. Proves 4 ranks initialize and all-reduce returns the correct value.
  3. Measures all-reduce bandwidth across message sizes, timed on the SLOWEST rank.
  4. Writes runs/M0-001.json so the number enters the log like every other measurement.

On algbw vs busbw
-----------------
algbw = message_bytes / time. This is what your gradient sync actually experiences.
busbw = algbw * 2*(n-1)/n. This is the *ring all-reduce convention* used by NVIDIA's
nccl-tests, and exists so numbers are comparable across world sizes and against
published tables. NCCL may internally choose a tree algorithm instead of a ring at
some message sizes, in which case busbw no longer describes bytes physically crossing
the fabric -- it remains a comparable figure, not a physical one. Report both, and
don't confuse the convention for the physics.

Expected on 4×A100-SXM4 behind NVSwitch (NV12 every pair): peak busbw somewhere in the
200-240 GB/s region, plateauing past ~64 MB. That is a PREDICTION derived from typical
measured nccl-tests results, not a spec-sheet guarantee and not a pass/fail gate. Being
wrong about it is informative; that's why it gets written down first.
"""

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist

from bench import _env_snapshot

FP32_BYTES = 4


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}TB"


def check_contention() -> list[str]:
    """
    Look for other processes on the visible GPUs. On a shared cluster slice this is the
    first thing to rule out -- a busy NVSwitch fabric depresses collective bandwidth in
    exactly the way a transport fallback would, and the two are easy to confuse.

    MUST be called before init_process_group() and before any CUDA context is created.
    Inside a container, nvidia-smi reports *host* PIDs while os.getpid() returns the
    container-namespace PID, so self-filtering by PID does not work -- your own ranks
    show up as ~12 MiB "other processes" and the check cries wolf every run. Running
    before any context exists sidesteps the namespace problem entirely: anything visible
    at that point genuinely belongs to someone else.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=15).decode().strip()
    except Exception:
        return []
    return [ln for ln in out.splitlines() if ln.strip()]


def bench_allreduce(nbytes: int, warmup: int = 5, iters: int = 20) -> tuple[float, float]:
    """Return (algbw_GBps, busbw_GBps) timed on the slowest rank."""
    world = dist.get_world_size()
    device = torch.cuda.current_device()

    # Zeros, not ones: repeated in-place all-reduce multiplies values by world_size each
    # pass. It wouldn't overflow fp32 at this scale, but there's no reason to let the
    # buffer drift. NCCL doesn't compress zeros, so communication volume is identical.
    tensor = torch.zeros(nbytes // FP32_BYTES, dtype=torch.float32, device=device)

    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()

    start = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters

    # A collective finishes when its slowest participant finishes. Rank 0's local timing
    # is not the collective's timing, so take the max across ranks. This doubles as a
    # sync point between message sizes.
    t = torch.tensor(elapsed, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    elapsed = t.item()

    algbw = nbytes / elapsed / 1e9
    busbw = algbw * 2 * (world - 1) / world
    return algbw, busbw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", default="peak all-reduce busbw 200-240 GB/s, "
                                            "plateauing past ~64MB",
                    help="What you expect, written BEFORE you read the output.")
    ap.add_argument("--run-id", default="M0-001")
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--max-mb", type=int, default=2048,
                    help="Largest message size to sweep. Default 2GB: at 4 ranks the curve "
                         "was still climbing at 512MB, so the knee is further out.")
    args = ap.parse_args()

    # --- 0. contention check, BEFORE any CUDA context exists ------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))     # torchrun sets this pre-init
    others = check_contention() if local_rank == 0 else []
    if local_rank == 0 and others:
        print("\n  ⚠ other compute processes already on these GPUs before this job started:")
        for ln in others:
            print(f"      {ln}")
        print("  Bandwidth below expectation may be contention, not a transport problem.\n")

    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(local_rank)

    # --- 1. correctness -------------------------------------------------
    x = torch.tensor([float(rank)], device="cuda")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = world * (world - 1) / 2
    ok = abs(x.item() - expected) < 1e-6

    props = torch.cuda.get_device_properties(local_rank)
    print(f"[rank {rank}/{world}] host={socket.gethostname()} local_rank={local_rank} "
          f"gpu={props.name} {props.total_memory/1e9:.0f}GB "
          f"cc={props.major}.{props.minor} allreduce_ok={ok}", flush=True)
    dist.barrier()

    if not ok:
        if rank == 0:
            print("\n*** ALL-REDUCE RETURNED THE WRONG VALUE — stop and debug NCCL. ***")
        dist.barrier()
        dist.destroy_process_group()
        return

    # --- 2. bandwidth sweep ---------------------------------------------
    results = []
    nbytes = 1 << 20                                          # 1 MB
    while nbytes <= args.max_mb * (1 << 20):
        if nbytes > props.total_memory * 0.2:
            break
        algbw, busbw = bench_allreduce(nbytes)
        results.append((nbytes, algbw, busbw))
        nbytes <<= 1

    # --- 3. report + record ---------------------------------------------
    if rank == 0:
        print(f"\n  all-reduce bandwidth · world_size={world} · fp32 · slowest-rank timing")
        print(f"  {'size':>8} | {'algbw GB/s':>11} | {'busbw GB/s':>11}")
        print(f"  {'-'*8}-+-{'-'*11}-+-{'-'*11}")
        for nbytes, algbw, busbw in results:
            print(f"  {human(nbytes):>8} | {algbw:>11.1f} | {busbw:>11.1f}")

        peak = max(r[2] for r in results)
        peak_alg = max(r[1] for r in results)
        print(f"\n  peak busbw: {peak:.1f} GB/s   (algbw {peak_alg:.1f} GB/s)")

        # The peak is a headline number; DDP never sees it. Gradient buckets default to
        # 25MB, so the efficiency at bucket-scale messages is what actually governs
        # gradient-sync cost -- and it is the M7 tuning lever.
        near_bucket = min(results, key=lambda r: abs(r[0] - 25 * (1 << 20)))
        frac = near_bucket[2] / peak
        print(f"  at ~{human(near_bucket[0])} (DDP default bucket scale): "
              f"{near_bucket[2]:.1f} GB/s busbw = {frac*100:.0f}% of peak")
        if frac < 0.9:
            print(f"  → M7 hypothesis: raising bucket_cap_mb should recover part of that "
                  f"{(1-frac)*100:.0f}% gap, at the cost of less backward/comms overlap.")

        if peak > 150:
            verdict = ("Strong intra-node collective bandwidth, consistent with NVLink/"
                       "NVSwitch. Proceed to M1.")
        elif peak > 40:
            verdict = ("Below the NVLink-class range. Investigate before collecting scaling "
                       "numbers: co-tenant load, NCCL transport (rerun with NCCL_DEBUG=INFO), "
                       "GPU clocks/power state, container --shm-size, CPU scheduling noise.")
        else:
            verdict = ("Very slow for this topology. Do NOT collect scaling numbers yet. "
                       "Likely transport fallback or a node health problem — confirm against "
                       "nccl-tests all_reduce_perf to separate 'fabric is slow' from "
                       "'my PyTorch process-group config is slow'.")
        print(f"  verdict: {verdict}")

        print("\n  topology:")
        subprocess.run("nvidia-smi topo -m 2>/dev/null | head -n 8", shell=True)

        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        rec = {
            "run_id": args.run_id,
            "milestone": "M0",
            "question": "Do 4 ranks initialize, is all-reduce correct, and what bandwidth "
                        "does the PyTorch/NCCL stack actually deliver?",
            "prediction": args.prediction,
            "strategy": "allreduce-bench",
            "precision": "fp32",
            "config": {"world_size": world, "dtype": "fp32",
                       "timing": "max across ranks", "warmup": 5, "iters": 20},
            # Same snapshot every RunRecord gets: node (downward API), git SHA, versions.
            # Infra runs aren't blocked on a dirty tree, but dirtiness is still recorded.
            "env": {
                **_env_snapshot(),
                "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", ""),
                "NCCL_DEBUG_SUBSYS": os.environ.get("NCCL_DEBUG_SUBSYS", ""),
                "NCCL_ALGO": os.environ.get("NCCL_ALGO", "<unset:auto>"),
                "NCCL_NVLS_ENABLE": os.environ.get(
                    "NCCL_NVLS_ENABLE", "<unset:default>"
                ),
            },
            "metrics": {
                "peak_busbw_GBps": round(peak, 1),
                "peak_algbw_GBps": round(peak_alg, 1),
                "sweep": [{"bytes": n, "algbw_GBps": round(a, 1), "busbw_GBps": round(b, 1)}
                          for n, a, b in results],
                "contention_processes": others,
            },
            "outcome": verdict,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path = Path(args.outdir) / f"{args.run_id}.json"
        path.write_text(json.dumps(rec, indent=2))
        print(f"\n  [bench] wrote {path}")
        print(f"  [bench] PREDICTED: {args.prediction}")
        print(f"  [bench] ACTUAL:    peak busbw {peak:.1f} GB/s")
        print("  → paste into p4_lab_log.md §2 with the gap vs prediction.\n")

    # Tear the communicator down together; rank 0 has been printing and shelling out
    # while ranks 1-3 sat idle.
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
