# P4 Training Systems — Benchmark Results

> **Project status (2026-10-31):** M1 completed on 1×NVIDIA B200. M0 has a historical 4×A100 interconnect baseline; **4×B200 NCCL rerun and M2–M7 training systems work remain pending**. This document is a growing results report, not a claim of completed distributed training.

## M1 — Single-GPU BF16 baseline

### Experimental setup

- **Hardware:** 1×NVIDIA B200 on `cl-worker36` (device total memory ≈183 GiB), NVIDIA driver 580.196.07; PyTorch 2.8.0+cu128; BF16 autocast; CUDA 12.8 runtime.
- **Architectures:** GPT-125M (≈124.5M parameters; 12 layers, 12 heads, width 768, head dimension 64) and GPT-1.3B (≈1.314B parameters; 24 layers, 16 heads, width 2048, head dimension 128). Both use 1024-token sequence length and padded vocabulary size 50304.
- **Workload:** synthetic token batches generated on device, pool of eight; `micro_bs=16`, `grad_accum=1`, 16,384 tokens per optimizer step; 200 steps after first step, 50 unmeasured warmup steps, 150 measured steps. Same model optimizer and data setup across each controlled pair. Input pipeline cost is intentionally excluded.
- **Optimization:** fused AdamW; BF16 autocast; native SDPA. `--compile` toggles `torch.compile`; SDPA backend pinned `flash` or `cudnn`; **profiled operation names verified** before accepting a measured run.
- **Measurement:** CUDA synchronized once per 10-step timing window, excluding warmup and periodic logging; aggregate tokens/sec, model FLOPs utilization (MFU), and `torch.cuda.max_memory_allocated()`. p50/p95 are **percentiles of 10-step window-averaged times**. First-step times include compilation, initialization/autotuning and cache effects, not isolated compiler duration.
- **MFU convention:** model training FLOPs/token ≈ `6 × non-embedding params + 12 × layers × heads × head_dim × sequence_length`; divide throughput × estimated FLOPs/token by **2.25e15 dense BF16 FLOP/s**. Approximate values: 0.855G FLOPs/token (125M), 8.47G (1.3B). This is *model FLOPs* over hardware peak, not the fraction of all SM cycles that perform useful compute.
- **Known causal-attention overcount:** the conventional `12LHQT` attention term assumes full `T×T` attention. Fused **causal** Flash/cuDNN kernels can skip much of the masked upper triangle, so this MFU **overcredits** FLOPs that need not execute. A rough half-of-attention-term adjustment implies about **6.5–7% relative overstatement at GPT-125M and 3.5–4% at GPT-1.3B**, ignoring tile/diagonal implementation details. For example, the five reported MFUs would be approximately **16.4%, 24.3%, 33.1%, 38.4%, 41.3%** under that *heuristic*, **not** profiler-counted executed-FLOP estimates. Published-convention results below are intentionally **unchanged** for comparability; same-model, same-context throughput ratios are unaffected.

### Five-run controlled baseline table

| Run | Model | Execution | Observed SDPA | Throughput (tok/s) | MFU | Step p50† | Peak allocated | First step‡ |
|---|---|---|---|---:|---:|---:|---:|---:|
| M1-001 | GPT-125M | Eager | Flash | 461,503 | 17.54% | 35.4 ms | 17.2 GB | 1.0 s |
| M1-002 | GPT-125M | Compiled | Flash | 685,273 | 26.05% | 23.9 ms | 12.3 GB | 11.8 s |
| M1-003 | GPT-1.3B | Eager | Flash | 91,087 | 34.30% | 179.6 ms | 57.5 GB | 1.7 s |
| M1-004 | GPT-1.3B | Compiled | Flash | 105,819 | 39.85% | 154.9 ms | 52.8 GB | 23.2 s |
| M1-005 | GPT-1.3B | Compiled | cuDNN | 113,663 | 42.81% | 143.9 ms | 52.8 GB | 3.4 s (warm cache) |

† A quantile of **window average** step times, not per-step latency. ‡ First step includes first-touch/autotune/compile overhead; M1-005 has a warm compiler cache and is not directly comparable as cold compile duration.

### Controlled comparisons

| Change | Baseline | Variant | Token throughput gain | MFU difference | Peak allocated memory difference |
|---|---:|---:|---:|---:|---:|
| `torch.compile` at GPT-125M, Flash fixed | M1-001 | M1-002 | **+48.5% (1.485×)** | +8.51 percentage points | **−28.5%** |
| `torch.compile` at GPT-1.3B, Flash fixed | M1-003 | M1-004 | **+16.2% (1.162×)** | +5.55 percentage points | **−8.2%** |
| cuDNN instead of Flash at compiled GPT-1.3B | M1-004 | M1-005 | **+7.41% (1.074×)** | +2.96 percentage points | 0.0% at reported precision |

### M1-006…009 — Compiled GPT-125M micro-batch scaling

The original five-run protocol fixed micro-batch 16 to isolate model-size/compiler/backend comparisons. A separate **batch-size sweep** tests whether the relatively low 125M MFU can be substantially improved just by providing more token rows to the GPU. Configuration remained 1×B200, GPT-125M, BF16, compiled Flash Attention (**fwd+bwd profiler-verified**), sequence length 1024, grad_accum 1, 200 post-initial steps (50 warmup + 150 measured; initial step 0 timed separately), repeated synthetic on-device batches. The batch size was varied; **effective global optimizer batch therefore varied too**, so these are systems-performance comparisons, not equal-batch training-quality experiments.

| Run | Micro-batch | Tokens/step | Throughput (tok/s) | MFU* | Step p50† | Peak allocated | Gain over prior batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1-006 | 16 | 16,384 | **686,697** | **26.11%** | 23.9 ms | 12.3 GB | — |
| M1-007 | 32 | 32,768 | **723,974** | **27.52%** | 45.2 ms | 22.8 GB | +5.43% |
| M1-008 | 64 | 65,536 | **757,145** | **28.78%** | 86.6 ms | 43.7 GB | +4.58% |
| M1-009 | 128 | 131,072 | **774,236** | **29.43%** | 169.5 ms | 85.6 GB | +2.26% |

*Unadjusted conventional MFU (causal-attention overcount disclosed above). † p50 over ten-step **window mean** times, not individually instrumented steps. Throughput is cumulative over the 150 measured steps.*

**Prediction calibration (original forecasts unchanged):** before the sweep, predictions for batches 16/32/64/128 were **685K/850K/1.0M/1.1M tok/s** and **26%/32%/38%/42% MFU**, respectively. M1-006 landed **+0.25%** relative to its 685K forecast; M1-007, M1-008, and M1-009 throughput came in **14.83%, 24.29%, and 29.61% below** their original forecasts. After M1-007, the expectation for M1-008 was revised to 750K–800K and was met. After M1-008, M1-009 was revised to 775K–800K and finished 764 tok/s below that range; none of those conditional revisions replaced the preregistered forecasts. Exact command/prediction wording is preserved in [`p4_lab_log.md`](p4_lab_log.md) and the run JSONs.

**Measured tradeoff:** batch 16→128 increased throughput by **12.75%** while peak allocated memory increased from **12.3 to 85.6 GB (6.96×)**. The incremental token-throughput benefit shrank with each batch doubling, even as memory rose steeply. The batch-16 repeat M1-006 reproduced M1-002 throughput to **+0.21%**. Batch 16 is therefore a reasonable **memory-efficient starting point** for M3; batch 32 may be considered if a small per-GPU speed improvement is worth the extra memory. These runs alone do not establish the best setting for fixed global batch size, model quality, power efficiency, or other models.

### Post-hoc two-parameter batch-scaling fit (M1-006…009)

The four measured *mean* step times (computed as `tokens_per_step / cumulative_tokens_per_second`, rather than the reported window-mean p50) admit the following ordinary least-squares model:

`mean_step_time ≈ 3.463 ms + 1.266 µs × tokens_per_step`

| Micro-batch | Measured mean step | Fitted mean step | Relative fit error |
|---:|---:|---:|---:|
| 16 | 23.86 ms | 24.20 ms | +1.45% |
| 32 | 45.26 ms | 44.95 ms | -0.70% |
| 64 | 86.56 ms | 86.43 ms | -0.15% |
| 128 | 169.29 ms | 169.39 ms | +0.06% |

The maximum absolute relative residual is **1.45%**. In this *empirical model*, the non-amortizing intercept is about **14.5%** of the observed batch-16 step. Taking batch size to infinity *within the fitted relationship* gives an asymptotic throughput of **≈790K tok/s** (≈**30.0% reported MFU**); batch 16 already delivers **≈87%** of that extrapolated limit. Only ≈**3.9 MFU percentage points** could be recovered from compiled 125M batch-16 performance by eliminating this model's amortizable fixed cost. This is smaller than the **13.74-point** observed difference versus compiled 1.3B / Flash at batch 16 (26.11% versus 39.85%). The remaining ≈**9.8 points** are *not explained by this particular 125M batch-overhead model*; they are **not a measured per-operator or model-width attribution**.

**Scope and next test:** this is a **post-hoc four-point fit**, not a preregistered prediction or a universal 30% GPU limit. Kernel schedules, per-token cost and memory pressure may change outside batches 16–128; the tiny residual on the same four fitted observations is not an independent validation. At M7, profile the fitted **≈3.46 ms/step intercept** (fused AdamW, enabled gradient clipping, gradient clearing, launch/host gaps are *candidate* contributors) and **≈1.266 µs/token slope** (GEMMs, attention, normalization, output projection and memory traffic). A fitted intercept is an accounting term, *not* a direct timing of those operations. For M3, repeat batch selection after measuring 4×B200 NCCL overhead rather than assuming that the one-GPU fit carries over to DDP.

**Interpretation, not established mechanism:** already at batch 16, MLP GEMMs operate on M=16,384 token rows (for example `[16,384×768] @ [768×3072]`), so the simplistic prior that 125M must have single-digit MFU because of its narrow layers was false. Larger micro-batches increase the M dimension, but operator-level profiling is still needed to distinguish matmul efficiency, launch overhead, memory bandwidth, and optimizer contribution. Cumulative throughput declined modestly over some 64/128 runs; no GPU clock/power/temperature traces were collected to diagnose the drift.

### Predictions and calibration

The preregistered predictions were **not retroactively edited**. Later conditional estimates were written before their associated runs using already-observed earlier M1 results; these are distinguished from original prior estimates in [`p4_lab_log.md`](p4_lab_log.md).

| Run | Original expectation | Observation | What we learned |
|---|---|---|---|
| M1-001 | 5% MFU, ≈132K tok/s | 17.54%, 461.5K | small-model prior underestimated GPU utilization by ~3.5× at 16K tokens/step |
| M1-002 | 8% MFU, ≈211K tok/s | 26.05%, 685.3K | compiler benefit larger than original and revised priors |
| M1-003 | 15% MFU, ≈40K tok/s | 34.30%, 91.1K | larger model improved measured MFU despite lower tokens/sec |
| M1-004 | 20% MFU, ≈53K tok/s | 39.85%, 105.8K | revised 38–45% range, conditional on M1-003, captured result |
| M1-005 | cuDNN ≈5% faster than Flash | cuDNN 7.41% faster | direction correct; result specific to this configuration |

### Interpretation and limitations

1. **Model size:** with identical 16,384-token steps and BF16 precision, GPT-1.3B eager achieved 34.30% MFU versus GPT-125M eager's 17.54%. A larger share of large matmuls is a plausible explanation; operator-level profiling remains a future task. This does **not** imply 1.3B processes more tokens/sec.
2. **Compilation:** compiled throughput rose by 48.5% at 125M, but only 16.2% at 1.3B. Peak allocated memory fell in both cases. Fusion and fewer launches are potential mechanisms; actual attribution requires M7 traces, not just end-to-end timings.
3. **Attention backend:** at compiled GPT-1.3B (sequence length 1024, head_dim 128), cuDNN outperformed Flash by 7.41%. Neither result establishes which is best for other head dimensions, sequence lengths or library versions.
4. **Backend-identity failure caught:** an initial M1-005 attempt requested cuDNN but executed Flash kernels and was **rejected** by `verify_backend`, rather than silently counted. A separate `TORCHINDUCTOR_CACHE_DIR` was associated with correct cuDNN dispatch in a diagnostic and full run. Incorrect cache reuse is a leading hypothesis, **not a conclusively isolated local root cause**. The permanent `run_train.sh` launcher now isolates Inductor cache directories by requested SDPA backend before Python startup; subsequent compiled tiny Flash/cuDNN smoke runs both verified their actual fwd/bwd kernels. See `ISSUE-004` in the lab log.
5. **Statistical and scope limitations:** each configuration has a single official 150-measured-step run, except the compiled GPT-125M batch-16 configuration, repeated as M1-006. Minor drift was observed in longer 1.3B runs and in batch-64/128 cumulative throughput; no cross-run confidence intervals or hardware clock/thermal trace. The causal-attention estimator issue is recorded separately as reconstructed `ISSUE-005`. Loss values on repeated random synthetic batches do **not** validate generalization, data preparation or a realistic pretraining loss curve. No data-loader throughput, multi-GPU scaling, FSDP or tensor-parallel results are implied by M1.

### Reproducing from saved artifacts

Each measured run produced `runs/M1-00{1..9}.json` (nine JSON artifacts), containing the run configuration, environment information, predictions, observed attention operations, and metrics. Original and conditional predictions and the experiment rationale are preserved in [`p4_lab_log.md`](p4_lab_log.md).

```bash
source /workspace/p4/p4env.sh
python -m pytest -q -rs
python bench.py --summarize runs/
```

For future repetitions, use `bash run_train.sh ... --compile --backend flash` (or `cudnn`) to set a **separate Inductor cache directory before Python starts**; keep profiler backend verification enabled. `setup_env.sh` checks and, on supported root GPU pods, installs native compiler tools if absent, while preserving the locked Python environment on the PVC.

## Remaining milestones

- **M0-B200:** rerun NCCL bandwidth on the 4×B200 node before distributed runs (historical 4×A100 M0-001 is not directly transferable).
- **M2:** CPU data pipeline (tokenization, deduplication, quality filtering, packing, sharding), followed by a meaningful real-data loss curve at 125M.
- **M3–M5:** 4-GPU DDP; FSDP/ZeRO memory/communication tradeoffs; 2-way tensor parallelism and checkpoint/resume.
- **M6:** controlled BF16/FP8/FP4 precision investigation.
- **M7:** operator-level bottleneck diagnosis, repeatability, and a measured optimization ablation.
- **M8:** finalize README, figures, and system narrative when subsequent results exist.
