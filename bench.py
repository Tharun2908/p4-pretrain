"""
p4/bench.py — instrumentation for every measured run in Project 4.

Purpose: make logging automatic. Manual logging always decays; by the time you're
debugging FSDP at 1am you will not hand-write a JSON record. So every training script
imports this and gets a structured record for free.

Usage in a training loop:

    from bench import RunRecord, ThroughputMeter, gpt_flops_per_token, peak_for

    peak, label = peak_for("bf16")          # auto-detects A100 / H200 / B200
    rec = RunRecord(
        run_id="M3-ddp4-001",
        question="How far below ideal 4x does DDP land at 125M?",
        prediction="~3.7x of ideal; all-reduce over NVLink should cost <10%",
        strategy="DDP", precision="bf16",
        config=dict(n_layer=12, n_head=12, n_embd=768, block_size=1024,
                    micro_bs=12, grad_accum=4, world_size=4),
    )
    meter = ThroughputMeter(
        flops_per_token=gpt_flops_per_token(n_params, 12, 12, 64, 1024),
        peak_flops=peak, peak_label=label, world_size=4,
    )

    for step in range(max_steps):
        meter.step_start()
        ...train step...
        meter.step_end(tokens=micro_bs * grad_accum * block_size * world_size, loss=loss.item())
        if step % 50 == 0 and rank == 0:
            print(meter.report())

    rec.finish(meter, notes="...")     # writes runs/M3-ddp4-001.json  (rank 0 only)

For the M6 three-way ablation on B200, the ONLY thing that changes per run is the
precision string -- peak_for() swaps the denominator for you and records which one
it used:

    peak, label = peak_for("fp8")       # 4.5e15 on B200, vs 2.25e15 for bf16
    ...
    # run identically otherwise, so the comparison is controlled

Then:  python bench.py --summarize runs/   →  markdown table for docs/training.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # allows --summarize on a laptop with no torch
    torch = None


# --------------------------------------------------------------------------
# Peak dense FLOPS, per GPU. Numbers are vendor spec sheet, non-sparse.
# MFU is only comparable if everyone uses the same denominator, so pin it here
# and never fudge it upward to make a result look better.
# --------------------------------------------------------------------------
PEAK_FLOPS: dict[str, dict[str, float]] = {
    "A100-40GB": {"bf16": 312e12, "fp16": 312e12, "tf32": 156e12, "fp32": 19.5e12},
    "A100-80GB": {"bf16": 312e12, "fp16": 312e12, "tf32": 156e12, "fp32": 19.5e12},
    "H100-SXM":  {"bf16": 989e12, "fp16": 989e12, "fp8": 1979e12, "tf32": 495e12},
    "H200":      {"bf16": 989e12, "fp16": 989e12, "fp8": 1979e12, "tf32": 495e12},
    "B200":      {"bf16": 2.25e15, "fp16": 2.25e15, "fp8": 4.5e15,
                  "fp4": 9.0e15, "tf32": 1.2e15},
    "V100":      {"bf16": 0.0,    "fp16": 125e12, "fp32": 15.7e12},
}


def detect_gpu_key() -> str:
    """Map torch's device name onto a PEAK_FLOPS key. Fails loudly rather than guessing."""
    if torch is None or not torch.cuda.is_available():
        return "unknown"
    name = torch.cuda.get_device_name(0)
    for key in ("B200", "H200", "H100", "A100-80GB", "A100-40GB", "V100"):
        probe = {"H100": "H100", "A100-80GB": "A100", "A100-40GB": "A100"}.get(key, key)
        if probe in name:
            if key.startswith("A100"):
                gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                return "A100-80GB" if gb > 60 else "A100-40GB"
            if key == "H100":
                return "H100-SXM"
            return key
    return f"unknown({name})"


def peak_for(precision: str, gpu_key: str | None = None) -> tuple[float, str]:
    """
    Returns (peak_flops_per_gpu, human_label) for the current GPU at a given precision.

    Use this instead of indexing PEAK_FLOPS by hand. The label it returns gets stored in
    the run record, which is what keeps the M6 three-way ablation interpretable: bf16,
    fp8 and fp4 are each divided by a *different* denominator, and six weeks from now a
    bare "mfu: 0.11" tells you nothing about which one.
    """
    key = gpu_key or detect_gpu_key()
    if key not in PEAK_FLOPS:
        raise KeyError(
            f"No peak-FLOPS entry for GPU '{key}'. Add it to PEAK_FLOPS with the vendor "
            f"DENSE (non-sparse) number, or MFU is meaningless."
        )
    if precision not in PEAK_FLOPS[key]:
        raise KeyError(
            f"'{key}' has no '{precision}' entry — does this GPU actually support it? "
            f"Available: {sorted(PEAK_FLOPS[key])}"
        )
    peak = PEAK_FLOPS[key][precision]
    return peak, f"{key}/{precision} = {peak:.3g} FLOPS dense"


def gpt_flops_per_token(n_params_non_embed: int, n_layer: int, n_head: int,
                        head_dim: int, seq_len: int) -> float:
    """
    PaLM-paper convention, same as nanoGPT's estimate_mfu:

        flops_per_token = 6*N + 12*L*H*Q*T

    6*N   : fwd(2) + bwd(4) matmul FLOPs per parameter.
    12*L*H*Q*T : the attention score/value matmuls (q·kᵀ and att·v), which scale with
                 sequence length and are *not* captured by parameter count. Dropping
                 this positive term would UNDERSTATE MFU. (An earlier version of this
                 comment said "inflates" — backwards; caught in review, ISSUE-005.)

    Known bias of this convention, kept for comparability with PaLM/nanoGPT: the term
    assumes FULL T×T attention, but causal flash/cuDNN kernels skip the masked upper
    triangle and do roughly half that work. So MFU is credited with attention FLOPs that
    never execute — an OVERstatement of up to ~attention_share/2 (≈7% at 125M, ≈4% at 1.3B,
    from model_configs.py). Say this before an interviewer does.

    n_params_non_embed: exclude token+position embeddings (they're a lookup, not a matmul).
    """
    return 6 * n_params_non_embed + 12 * n_layer * n_head * head_dim * seq_len


def count_non_embedding_params(model) -> int:
    n = sum(p.numel() for p in model.parameters())
    for attr in ("wpe", "pos_emb", "position_embeddings"):
        emb = getattr(getattr(model, "transformer", model), attr, None)
        if emb is not None and hasattr(emb, "weight"):
            n -= emb.weight.numel()
    return n


# --------------------------------------------------------------------------
class ThroughputMeter:
    """
    Tracks tok/s, MFU, peak memory. Discards the first `warmup` steps — the first
    iterations include CUDA autotuning, allocator growth, and (with torch.compile)
    the compile itself. Including them is the single most common way people
    under-report their own throughput.
    """

    def __init__(self, flops_per_token: float, peak_flops: float,
                 world_size: int = 1, warmup: int = 10, peak_label: str = ""):
        self.flops_per_token = flops_per_token
        self.peak_flops_per_gpu = peak_flops
        self.peak_flops = peak_flops * world_size   # aggregate peak across the job
        self.peak_label = peak_label                # e.g. "B200/fp8 = 4.5e+15 FLOPS dense"
        self.world_size = world_size
        self.peak_mem_override_gb = None
        self.warmup = warmup
        self.n_steps = 0
        self.tokens = 0
        self.elapsed = 0.0
        self.losses: list[float] = []
        self.step_times: list[float] = []   # per step (step mode) or per-window mean (window mode)
        self.timing_mode = "per-step"
        self._window_steps = 0
        self._t0 = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def step_start(self) -> None:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def step_end(self, tokens: int, loss: float | None = None) -> None:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()          # without this you time the launch, not the work
        dt = time.perf_counter() - self._t0
        self.n_steps += 1
        if loss is not None:
            self.losses.append(float(loss))
        if self.n_steps > self.warmup:
            self.tokens += tokens
            self.elapsed += dt
            self.step_times.append(dt)

    # ---- derived metrics -------------------------------------------------
    @property
    def tokens_per_sec(self) -> float:
        return self.tokens / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def mfu(self) -> float | None:
        if self.peak_flops <= 0:                     # CPU / unknown hardware
            return None
        return (self.tokens_per_sec * self.flops_per_token) / self.peak_flops

    @property
    def peak_mem_gb(self) -> float:
        if self.peak_mem_override_gb is not None:
            return self.peak_mem_override_gb
        if torch is None or not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1e9

    @property
    def step_time_p50(self) -> float:
        s = sorted(self.step_times)
        return s[len(s) // 2] if s else 0.0

    @property
    def step_time_p95(self) -> float:
        s = sorted(self.step_times)
        return s[int(len(s) * 0.95)] if s else 0.0

    def record_window(self, tokens: int, n_steps: int, dt: float,
                      loss: float | None = None) -> None:
        """
        Windowed timing: caller syncs ONCE per window of n_steps and reports the total.

        Why: step_end() syncs every step, which stops the CPU from queuing step N+1's kernels
        while step N runs. On small, fast steps (125M on a B200) that lost overlap is a
        measurement artifact of a few percent — the meter would be measuring itself. The
        caller is responsible for discarding warmup windows (this method doesn't).
        p50/p95 become percentiles over per-window mean step times.
        """
        self.timing_mode = "windowed"
        self.n_steps += n_steps
        self._window_steps += n_steps
        self.tokens += tokens
        self.elapsed += dt
        self.step_times.append(dt / n_steps)
        if loss is not None:
            self.losses.append(float(loss))

    def report(self) -> str:
        return (f"step {self.n_steps:>6} | {self.tokens_per_sec:>9,.0f} tok/s | "
                f"MFU {(f'{self.mfu*100:5.2f}%') if self.mfu is not None else '  n/a '} | peak {self.peak_mem_gb:5.1f} GB | "
                f"p50 {self.step_time_p50*1e3:6.1f} ms | p95 {self.step_time_p95*1e3:6.1f} ms"
                + (f" | loss {self.losses[-1]:.4f}" if self.losses else ""))

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "tokens_per_sec_per_gpu": round(self.tokens_per_sec / self.world_size, 1),
            "mfu": round(self.mfu, 5) if self.mfu is not None else None,
            "mfu_denominator_per_gpu": self.peak_flops_per_gpu,
            "mfu_denominator_label": self.peak_label,
            "flops_per_token": self.flops_per_token,
            "peak_mem_gb": round(self.peak_mem_gb, 3),
            "step_time_p50_ms": round(self.step_time_p50 * 1e3, 3),
            "step_time_p95_ms": round(self.step_time_p95 * 1e3, 3),
            "measured_steps": (self._window_steps if self.timing_mode == "windowed"
                               else max(0, self.n_steps - self.warmup)),
            "warmup_steps_discarded": self.warmup,
            "timing_mode": self.timing_mode,
            "final_loss": self.losses[-1] if self.losses else None,
            "min_loss": min(self.losses) if self.losses else None,
        }


# --------------------------------------------------------------------------
POD_OWNED_DIRS = ("runs", "plots", "logs")


def _env_snapshot() -> dict[str, Any]:
    """Captured automatically so a run is reproducible six weeks later."""
    env: dict[str, Any] = {
        # NODE_NAME comes from the downward API in the pod spec (spec.nodeName). hostname is
        # only the POD name, which says nothing about which physical machine ran this.
        "node": os.environ.get("NODE_NAME"),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", None),
    }
    if torch is not None and torch.cuda.is_available():
        env.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count_visible": torch.cuda.device_count(),
            "cuda": torch.version.cuda,
            "nccl": ".".join(map(str, torch.cuda.nccl.version())) if hasattr(torch.cuda, "nccl") else None,
        })
    try:
        env["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        # Dirty = uncommitted CODE. Pod-owned output dirs are excluded, otherwise the JSON
        # written by run N would block run N+1 until committed.
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".",
             *[f":(exclude){d}" for d in POD_OWNED_DIRS]],
            stderr=subprocess.DEVNULL).decode().strip()
        env["git_dirty"] = bool(dirty)
        env["git_dirty_files"] = dirty.splitlines()[:20]
    except Exception:
        env["git_sha"] = None
    for k in ("NCCL_DEBUG", "NCCL_P2P_DISABLE", "CUDA_VISIBLE_DEVICES",
              "TORCH_NCCL_BLOCKING_WAIT", "OMP_NUM_THREADS"):
        if k in os.environ:
            env[k] = os.environ[k]
    return env


@dataclass
class RunRecord:
    """One measured run. `question` and `prediction` are mandatory on purpose."""
    run_id: str
    question: str
    prediction: str
    strategy: str                       # DDP | FSDP-ZeRO3 | TP2xDP2 | single | ...
    precision: str                      # bf16 | fp8 | fp32
    config: dict[str, Any] = field(default_factory=dict)
    milestone: str = ""
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    env: dict[str, Any] = field(default_factory=_env_snapshot)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    outcome: str = ""                   # "as predicted" | "off by X because Y" | "failed: Z"

    def __post_init__(self) -> None:
        if not self.question.strip() or not self.prediction.strip():
            raise ValueError(
                "RunRecord requires a `question` and a `prediction` written BEFORE the run. "
                "A number with no prior expectation is not an experiment, it's a log line."
            )
        # Provenance guard. git_sha is stamped into the record; if the tree is dirty, that
        # SHA names code that did NOT run, and the record misstates its own origin.
        allow = os.environ.get("P4_ALLOW_DIRTY") == "1"
        self.env["allowed_dirty"] = allow
        if allow:
            return
        if self.env.get("git_sha") is None:
            raise ValueError("Not inside a git repo — this run's code can't be identified. "
                             "Run from the repo, or set P4_ALLOW_DIRTY=1 for a throwaway run.")
        if self.env.get("git_dirty"):
            files = "\n    ".join(self.env.get("git_dirty_files", []))
            raise ValueError(
                "Uncommitted code changes — the recorded git_sha would not be the code that ran.\n"
                f"    {files}\n"
                "Commit first (single-line: git commit -am \"<run-id>: ...\"), or set "
                "P4_ALLOW_DIRTY=1 for a throwaway debug run (flagged in the record)."
            )

    def finish(self, meter: ThroughputMeter | None = None, notes: str = "",
               outcome: str = "", outdir: str = "runs",
               extra: dict[str, Any] | None = None) -> Path | None:
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            return None                                  # only rank 0 writes
        if meter is not None:
            self.metrics = meter.as_dict()
        if extra:
            self.metrics.update(extra)          # e.g. compile_s, observed attention kernels
        self.notes = notes or self.notes
        self.outcome = outcome or self.outcome
        self.metrics["wall_clock_s"] = round(
            time.time() - time.mktime(time.strptime(self.started_at, "%Y-%m-%dT%H:%M:%S")), 1)
        Path(outdir).mkdir(parents=True, exist_ok=True)
        path = Path(outdir) / f"{self.run_id}.json"
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2))
        print(f"\n[bench] wrote {path}")
        print(f"[bench] PREDICTED: {self.prediction}")
        if meter is not None:
            print(f"[bench] ACTUAL:    {meter.report()}")
        print("[bench] → now paste this into §2 of p4_lab_log.md with the gap vs prediction.\n")
        return path


# --------------------------------------------------------------------------
def summarize(runs_dir: str) -> str:
    """Regenerate the results table from the JSON records. Used for docs/training.md."""
    rows = []
    for p in sorted(Path(runs_dir).glob("*.json")):
        try:
            d = json.loads(p.read_text())
            if not isinstance(d, dict) or "run_id" not in d:
                continue                                  # not one of ours
        except Exception:
            continue
        m, c = d.get("metrics", {}), d.get("config", {})
        rows.append({
            "run": d["run_id"],
            "strategy": d.get("strategy", ""),
            "prec": d.get("precision", ""),
            "gpus": c.get("world_size", 1),
            "tok/s": f'{m.get("tokens_per_sec", 0):,.0f}',
            "tok/s/gpu": f'{m.get("tokens_per_sec_per_gpu", 0):,.0f}',
            "MFU %": f'{m["mfu"]*100:.2f}' if m.get("mfu") is not None else "—",
            "MFU denom": (m.get("mfu_denominator_label") or "—").split(" = ")[0],
            "peak GB": f'{m.get("peak_mem_gb", 0):.1f}',
            "p50 ms": f'{m.get("step_time_p50_ms", 0):.1f}',
            "loss": f'{m.get("final_loss"):.4f}' if m.get("final_loss") is not None else "—",
            "outcome": d.get("outcome", "")[:60],
        })
    if not rows:
        return "_no runs recorded yet_"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", metavar="RUNS_DIR", help="emit a markdown table of all runs")
    a = ap.parse_args()
    if a.summarize:
        print(summarize(a.summarize))
    else:
        ap.print_help()
