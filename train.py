"""
p4/train.py — the one script every P4 run goes through.

Laptop CPU smoke test (no GPU, no record):
    python train.py --model gpt-tiny --device cpu --steps 30 --dry-run

M1 measured run on 1×B200:
    python train.py --model gpt-125m --compile --backend flash --micro-bs 32 --steps 200 \\
        --run-id M1-001 --question "..." --prediction "..."

Multi-GPU (M3+; the DDP path below is written but NOT validated until M3):
    torchrun --nproc_per_node=4 train.py ...

Measured runs construct their RunRecord BEFORE any training, so a missing prediction or a
dirty git tree fails in the first second — not after twenty minutes of GPU time.

Measurement design (DEC-008):
  * Windowed timing: sync once per --log-every steps, never per step (see bench.record_window).
  * Steps before --bench-warmup are trained but never counted (compile, autotuning, allocator
    growth all happen there).
  * Step 0 is timed separately and recorded as compile_s — that cost is real, just not
    steady-state throughput.
  * After step 0, one extra fwd+bwd runs under the profiler to record WHICH attention kernels
    actually executed. Asking for a backend and getting one are different things.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.distributed as dist

from bench import RunRecord, ThroughputMeter, peak_for
from model import GPT
from model_configs import DEBUG_MODELS, MODELS

ALL_MODELS = {**MODELS, **DEBUG_MODELS}
BACKENDS = {"flash": "FLASH_ATTENTION", "cudnn": "CUDNN_ATTENTION",
            "efficient": "EFFICIENT_ATTENTION", "math": "MATH"}
REAL_VOCAB = 50257              # ids >= this are vocab padding: never sampled as data


# ---- distributed plumbing --------------------------------------------------------------

def setup_dist(device: str = "auto"):
    """Use Gloo for CPU DDP and NCCL for GPU DDP."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world > 1:
        use_cuda = device.startswith("cuda") or (
            device == "auto" and torch.cuda.is_available()
        )

        if use_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA DDP requested without CUDA")
            torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl" if use_cuda else "gloo",
            device_id=torch.device(f"cuda:{local_rank}") if use_cuda else None,
        )

    return rank, world, local_rank


# ---- data ------------------------------------------------------------------------------

class SyntheticBatches:
    """
    Random token ids, pre-generated ON the device and cycled. Token values don't change the
    compute of a dense transformer, so this measures model throughput with the input
    pipeline removed entirely — the clean baseline M1 needs before M2 exists. A small pool
    (not one batch) so nothing degenerate happens if a kernel ever specializes on data.
    x and y are generated contiguous, not sliced, so no copy happens inside the timed loop.
    """

    def __init__(self, micro_bs: int, block: int, device: str, pool: int = 8, seed: int = 0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.xs = [torch.randint(0, REAL_VOCAB, (micro_bs, block), generator=g).to(device)
                   for _ in range(pool)]
        self.ys = [torch.randint(0, REAL_VOCAB, (micro_bs, block), generator=g).to(device)
                   for _ in range(pool)]
        self.i = 0

    def next(self):
        j = self.i % len(self.xs)
        self.i += 1
        return self.xs[j], self.ys[j]


class MemmapBatches:
    """
    nanoGPT-format token file: flat uint16 array (M2 will produce these). Random windows,
    x = tokens[i:i+T], y = tokens[i+1:i+T+1]. The memmap is re-opened per batch — numpy
    memmaps otherwise accumulate resident pages and look like a slow memory leak over a
    long run. Pinned memory + non_blocking lets the H2D copy overlap with compute.
    """

    def __init__(self, path: str, micro_bs: int, block: int, device: str, seed: int = 0):
        self.path, self.B, self.T, self.device = path, micro_bs, block, device
        n = len(np.memmap(path, dtype=np.uint16, mode="r"))
        if n <= block + 1:
            raise ValueError(f"{path}: {n} tokens, need > block_size+1 = {block + 1}")
        self.n = n
        self.rng = np.random.default_rng(seed)

    def next(self):
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        ix = self.rng.integers(0, self.n - self.T - 1, size=self.B)
        x = torch.from_numpy(np.stack([data[i:i + self.T] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + self.T] for i in ix]).astype(np.int64))
        if self.device.startswith("cuda"):
            return (x.pin_memory().to(self.device, non_blocking=True),
                    y.pin_memory().to(self.device, non_blocking=True))
        return x, y


# ---- optimizer & schedule ----------------------------------------------------------------

def build_optimizer(model: GPT, lr: float, weight_decay: float, device: str,
                    decay_param_ids: set[int] | None = None):
    """
    Weight decay on matrices only (dim >= 2: all Linear weights + the tied embedding).
    Biases and LayerNorm gains stay undecayed — decaying a LayerNorm gain toward zero
    fights the normalization it exists to undo. parameters() de-dups the tied tensor.
    fused=True: one kernel for the whole AdamW update instead of a loop over params.
    """
    params = list(model.parameters())
    if decay_param_ids is None:
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
    else:
        # FSDP may expose original parameter objects as 1D shards.
        decay = [p for p in params if id(p) in decay_param_ids]
        no_decay = [p for p in params if id(p) not in decay_param_ids]
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8,
                             fused=device.startswith("cuda"))


def lr_at(step: int, max_lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    """Linear warmup, cosine decay to min_ratio·max_lr. Irrelevant to throughput, but every
    run uses the same schedule so a benchmark run and a training run differ in length only."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


# ---- attention-kernel verification -------------------------------------------------------

ATTN_KERNEL_HINTS = ("flash", "fmha", "cudnn", "attention", "efficient", "sdpa")


def observe_attention_kernels(model, batches, ctx_fn) -> list[str]:
    """
    One fwd+bwd under the profiler; returns the names of CUDA kernels that look like
    attention. This is the ground truth for "which backend ran" — sdpa_kernel() restricts the
    allowed set, but compile, shape constraints or an unsupported case can still change
    what executes. No optimizer step: this doesn't advance training.
    """
    from torch.profiler import ProfilerActivity, profile
    x, y = batches.next()
    # CPU activity too: the aten op name (e.g. aten::_scaled_dot_product_cudnn_attention) is
    # the definitive record of which backend was dispatched; kernel names are corroboration.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with ctx_fn():
            _, loss = model(x, y)
        loss.backward()
        torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    names = {e.key for e in prof.key_averages()
             if any(h in e.key.lower() for h in ATTN_KERNEL_HINTS)}
    return sorted(names)          # no truncation: verify_backend must see every name


# aten op names are the definitive dispatch record, e.g. aten::_scaled_dot_product_cudnn_attention
FUSED_OP_TAGS = {"flash": "_scaled_dot_product_flash_attention",
                 "cudnn": "_scaled_dot_product_cudnn_attention",
                 "efficient": "_scaled_dot_product_efficient_attention"}


def verify_backend(requested: str, observed: list[str]) -> None:
    """
    Refuse to continue if the requested backend can't be confirmed from the profile.
    Applies to dry runs too — catching a mismatch is exactly what the tiny dry run is for.
      flash/cudnn/efficient: that backend's aten op must appear
      math:  NO fused attention op may appear (math decomposes into plain matmul/softmax)
      auto:  nothing enforced; whatever ran is recorded
    """
    fused = {b for b, tag in FUSED_OP_TAGS.items() if any(tag in n for n in observed)}
    if requested == "auto":
        return
    if requested == "math":
        if fused:
            raise RuntimeError(f"requested MATH but fused attention ran: {sorted(fused)}")
        return
    if requested not in fused:
        raise RuntimeError(f"requested backend '{requested}' not confirmed by the profiler. "
                           f"Fused backends observed: {sorted(fused) or 'none'}. "
                           f"Attention-like names: {observed}")


# ---- main ----------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt-125m", choices=sorted(ALL_MODELS))
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--strategy", choices=["ddp", "fsdp-grad", "fsdp-full"],
                   default="ddp", help="Distributed training strategy")
    p.add_argument("--deterministic", action="store_true",
                   help="Enable deterministic PyTorch algorithms for reproducibility")
    p.add_argument("--backend", default="auto", choices=["auto", *BACKENDS],
                   help="SDPA backend; 'auto' = PyTorch's own dispatch (still recorded)")
    p.add_argument("--data", default="synthetic", help="'synthetic' or path to a uint16 .bin")
    p.add_argument("--micro-bs", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=100, help="steps AFTER step 0")
    p.add_argument("--bench-warmup", type=int, default=20, help="steps trained but not measured")
    p.add_argument("--log-every", type=int, default=10, help="also the timing window size")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--lr-warmup", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dry-run", action="store_true",
                   help="no RunRecord: no prediction needed, dirty tree allowed. Never for results.")
    p.add_argument("--run-id")
    p.add_argument("--question")
    p.add_argument("--prediction")
    p.add_argument("--outdir", default="runs")
    p.add_argument("--notes", default="")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.deterministic:
        # Must be configured before initializing CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    rank, world, local_rank = setup_dist(args.device)
    master = rank == 0
    say = print if master else (lambda *a, **k: None)

    device = args.device
    if device == "auto":
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    on_cuda = device.startswith("cuda")
    cfg = ALL_MODELS[args.model]
    if args.strategy != "ddp":
        if world < 2 or not on_cuda:
            raise SystemExit("FSDP requires at least two CUDA ranks for M4")
        if args.compile:
            raise SystemExit("FSDP + torch.compile not yet validated in M4")
        if args.grad_accum != 1:
            raise SystemExit("FSDP gradient accumulation not yet validated in M4")

    # steps are counted AFTER step 0, so windows end at log_every, 2·log_every, … steps
    if args.bench_warmup % args.log_every:
        raise SystemExit("--bench-warmup must be a multiple of --log-every")
    if args.steps < args.bench_warmup + args.log_every:
        raise SystemExit("--steps must cover --bench-warmup plus at least one measured window")

    tokens_per_step = args.micro_bs * args.grad_accum * cfg.block_size * world
    config = dict(model=cfg.name, n_params=cfg.n_params, world_size=world, device=device,
                  compile=args.compile, deterministic=args.deterministic,
                  strategy=args.strategy,
                  backend=args.backend, data=args.data,
                  micro_bs=args.micro_bs, grad_accum=args.grad_accum,
                  block_size=cfg.block_size, tokens_per_step=tokens_per_step,
                  steps=args.steps, bench_warmup=args.bench_warmup, log_every=args.log_every,
                  lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                  precision="bf16" if on_cuda else "fp32", seed=args.seed)

    # ---- record FIRST: guards fire before any GPU time is spent --------------------------
    rec = None
    if not args.dry_run:
        if not (args.run_id and args.question and args.prediction):
            raise SystemExit("measured runs need --run-id, --question and --prediction "
                             "(or pass --dry-run for a throwaway run)")
        rec = RunRecord(run_id=args.run_id, question=args.question, prediction=args.prediction,
                        strategy=("single" if world == 1 else {
                            "ddp": "DDP",
                            "fsdp-grad": "FSDP-SHARD_GRAD_OP",
                            "fsdp-full": "FSDP-FULL_SHARD",
                        }[args.strategy]),
                        precision=config["precision"], config=config, notes=args.notes)

    torch.manual_seed(args.seed)                 # identical init on every rank
    if on_cuda:
        torch.set_float32_matmul_precision("high")

    model = GPT(cfg).to(device)
    raw_model = model

    # Capture identity and matrix classification BEFORE FSDP sharding.
    original_param_ids = {id(p) for p in raw_model.parameters()}
    decay_param_ids = {
        id(p) for p in raw_model.parameters() if p.dim() >= 2
    }

    ddp_model = None
    fsdp_model = None

    if args.strategy == "ddp":
        if world > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            ddp_model = model = DDP(
                model,
                device_ids=[local_rank] if on_cuda else None,
            )
    else:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
        )

        sharding = {
            "fsdp-grad": ShardingStrategy.SHARD_GRAD_OP,
            "fsdp-full": ShardingStrategy.FULL_SHARD,
        }[args.strategy]

        # Keep tied embedding/output weight in the ROOT owner.
        for i, block in enumerate(raw_model.transformer.h):
            raw_model.transformer.h[i] = FSDP(
                block,
                sharding_strategy=sharding,
                use_orig_params=True,
                device_id=local_rank,
            )

        fsdp_model = model = FSDP(
            raw_model,
            sharding_strategy=sharding,
            use_orig_params=True,
            device_id=local_rank,
        )

        if {id(p) for p in model.parameters()} != original_param_ids:
            raise RuntimeError("FSDP changed original parameter identities")

    if args.compile:
        model = torch.compile(model)

    # Optimizer is created AFTER FSDP wrapping, but retains the
    # original weight-decay versus no-decay classification.
    opt = build_optimizer(
        raw_model, args.lr, args.weight_decay, device,
        decay_param_ids=decay_param_ids if fsdp_model is not None else None,
    )

    if args.data == "synthetic":
        batches = SyntheticBatches(args.micro_bs, cfg.block_size, device, seed=args.seed + rank)
    else:
        batches = MemmapBatches(args.data, args.micro_bs, cfg.block_size, device, seed=args.seed + rank)

    if on_cuda:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        attn = ((lambda: sdpa_kernel(getattr(SDPBackend, BACKENDS[args.backend])))
                if args.backend != "auto" else nullcontext)
    else:
        amp = attn = nullcontext                 # CPU smoke test: fp32, default attention

    @contextmanager
    def fwd_ctx():
        with amp(), attn():
            yield

    peak, peak_label = (peak_for("bf16") if on_cuda else (0.0, "cpu: MFU not applicable"))
    meter = ThroughputMeter(flops_per_token=cfg.flops_per_token, peak_flops=peak,
                            world_size=world, warmup=0, peak_label=peak_label)

    def sync():
        if on_cuda:
            torch.cuda.synchronize()

    def train_step(step: int) -> torch.Tensor:
        lr = lr_at(step, args.lr, args.lr_warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = torch.zeros((), device=device)
        for micro in range(args.grad_accum):
            if ddp_model is not None:            # all-reduce grads only on the last micro-step
                ddp_model.require_backward_grad_sync = (micro == args.grad_accum - 1)
            x, y = batches.next()
            with fwd_ctx():
                _, loss = model(x, y)
            loss = loss / args.grad_accum
            loss_acc += loss.detach()
            loss.backward()
        if args.grad_clip > 0:
            if fsdp_model is not None:
                fsdp_model.clip_grad_norm_(args.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), args.grad_clip
                )
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss_acc

    say(f"[train] {cfg.name} ({cfg.n_params/1e6:.0f}M) · {device} · world {world} · "
        f"compile={args.compile} · backend={args.backend} · {tokens_per_step:,} tok/step")

    # ---- step 0: compile + first-touch costs, timed on their own -------------------------
    sync(); t0 = time.perf_counter()
    loss = train_step(0)
    sync(); compile_s = time.perf_counter() - t0
    say(f"[train] step 0 (incl. compile/autotune): {compile_s:.1f}s · loss {loss.item():.4f}")

    kernels = observe_attention_kernels(model, batches, fwd_ctx) if on_cuda else []
    if kernels:
        say(f"[train] attention kernels observed: {kernels}")
    if on_cuda:
        verify_backend(args.backend, kernels)       # fails here, before any measured window

    # ---- main loop: windowed timing ------------------------------------------------------
    first_loss = loss.item()
    mem_per_rank_gb = []
    sync(); t_last = time.perf_counter()
    for step in range(1, args.steps + 1):
        loss = train_step(step)
        if step % args.log_every == 0:
            sync()
            dt = time.perf_counter() - t_last
            lv = loss
            if world > 1:
                # Distributed throughput is limited by the slowest rank.
                # Reduce timing before logging, outside the measured window.
                elapsed = torch.tensor(
                    dt, dtype=torch.float64, device=device
                )
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                dt = elapsed.item()

                # Report mean loss across all ranks.
                dist.all_reduce(lv, op=dist.ReduceOp.AVG)
            lv = lv.item()

            # Memory reporting is outside the measured window.
            if on_cuda:
                local_mem = torch.tensor(
                    [torch.cuda.max_memory_allocated() / 1e9],
                    dtype=torch.float64,
                    device=device,
                )
                if world > 1:
                    gathered = [
                        torch.zeros_like(local_mem)
                        for _ in range(world)
                    ]
                    dist.all_gather(gathered, local_mem)
                    mem_per_rank_gb = [
                        round(x.item(), 3) for x in gathered
                    ]
                else:
                    mem_per_rank_gb = [round(local_mem.item(), 3)]

                meter.peak_mem_override_gb = max(mem_per_rank_gb)

            if step > args.bench_warmup:
                meter.record_window(tokens_per_step * args.log_every, args.log_every, dt, lv)
                say(meter.report())
            else:
                say(f"step {step:>6} | warmup (not measured) | {dt/args.log_every*1e3:.1f} ms/step | loss {lv:.4f}")
            # GPU is idle after the sync: .item(), the DDP loss all-reduce and printing are
            # reporting overhead, not training. Start the next window AFTER them.
            if world > 1:
                # Align the next window; exclude reporting/barrier time.
                dist.barrier()
            t_last = time.perf_counter()

    extra = {"compile_s": round(compile_s, 2), "first_loss": first_loss,
             "attn_kernels_observed": kernels,
             "peak_mem_gb_per_rank": mem_per_rank_gb if on_cuda else None}
    out = {**meter.as_dict(), **extra}
    if rec is not None:
        rec.finish(meter, notes=args.notes, outdir=args.outdir, extra=extra)
    if world > 1:
        dist.destroy_process_group()
    return out


if __name__ == "__main__":
    main()
