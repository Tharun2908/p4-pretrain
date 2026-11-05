# M6 — B200 Mixed-Precision Training Results

## Summary

**Date:** 2026-11-05

**Status:** Single-GPU and four-GPU benchmark matrices completed.
Real-data convergence and selected low-precision state validations remain pending.

The same GPT architecture was compared at two model sizes using native BF16,
Transformer Engine BF16, FP8, and NVFP4.

**Principal observation:** At GPT-125M, both FP8 and NVFP4 were slower
than TE BF16. At GPT-1.3B, both low-precision modes delivered higher
aggregate throughput than TE BF16 in the recorded runs.

This is an observation about the tested implementation and workload,
not a universal performance claim about numerical formats.

## Experimental configuration

- Hardware: 1 NVIDIA B200, SM 10.0
- PyTorch: 2.8.0+cu128
- CUDA runtime: 12.8
- Transformer Engine: 2.19.0
- Sequence length: 1,024 tokens
- Micro-batch size: 8
- Gradient accumulation: 1
- Tokens per optimizer step: 8,192
- Training steps: step 0 plus steps 1–100
- Warmup: steps 1–20
- Measured: steps 21–100, grouped into eight 10-step windows
- Seed: 1337
- Synthetic on-device token batches
- Eager execution; torch.compile disabled
- Attention: PyTorch FlashAttention observed through SDPA
- Optimizer: AdamW, with the existing training-runner settings
- Gradient clipping: 1.0

All modes used the same training runner and numerical input tensors.
For TE modes, four internal projections per transformer block were
replaced with TE Linear layers. Embeddings, tied output projection,
LayerNorm, and attention remained outside that conversion.

The FP8 recipe used HYBRID DelayedScaling with amax history length of 16.
NVFP4 used the default NVFP4BlockScaling recipe.

The source results are committed RunRecord JSON files under `runs/`.

## GPT-125M results

| Precision | Run ID | Tokens/s | vs TE BF16 | p50 ms | p95 ms | Peak GB |
|---|---|---:|---:|---:|---:|---:|
| Native BF16 | `M6-125M-NBF16-001` | 414,541.6 | 0.9760x | 19.761 | 19.773 | 9.487 |
| TE BF16 | `M6-125M-TEBF16-001` | 424,734.4 | 1.0000x | 19.276 | 19.393 | 9.516 |
| FP8 | `M6-125M-FP8-001` | 371,908.3 | 0.8756x | 21.955 | 22.408 | 9.062 |
| NVFP4 | `M6-125M-NVFP4-001` | 272,373.8 | 0.6413x | 29.623 | 33.854 | 8.854 |

Single recorded trial per precision mode.

FP8 and NVFP4 were slower than TE BF16 at this model size.
Their extra quantization and transform work are plausible contributors,
but end-to-end slowdown has not been fully attributed to individual kernels.

## GPT-1.3B results

| Precision | Run ID | Tokens/s | vs TE BF16 | p50 ms | p95 ms | Peak GB |
|---|---|---:|---:|---:|---:|---:|
| Native BF16 | `M6-1P3B-NBF16-001` | 81,962.3 | 0.9951x | 100.014 | 100.174 | 37.963 |
| TE BF16 | `M6-1P3B-TEBF16-001` | 82,365.6 | 1.0000x | 99.477 | 99.676 | 37.997 |
| FP8 #1 | `M6-1P3B-FP8-001` | 90,203.7 | 1.0952x | 84.429 | 113.319 | 34.776 |
| FP8 #2 | `M6-1P3B-FP8-002` | 91,070.3 | 1.1057x | 84.414 | 110.921 | 34.776 |
| NVFP4 #1 | `M6-1P3B-NVFP4-001` | 89,738.9 | 1.0895x | 91.606 | 91.666 | 33.695 |
| NVFP4 #2 | `M6-1P3B-NVFP4-002` | 85,931.0 | 1.0433x | 91.551 | 110.203 | 33.695 |

FP8 delivered a measured throughput advantage of approximately
9.5–10.6% versus the single TE BF16 reference run.

NVFP4 delivered a measured throughput advantage of approximately
4.3–9.0% versus that reference.

Both low-precision modes also reduced peak allocated GPU memory.
These are peak allocated-memory measurements, not total device memory,
reserved allocator memory, or checkpoint size.

## Four-GPU GPT-1.3B DDP results

### Protocol

- Four NVIDIA B200 GPUs on one node
- Full GPT-1.3B replica on each rank
- Micro-batch 8 per GPU; global tokens/step 32,768
- Sequence length 1,024
- Eager execution, no gradient accumulation
- Seed 1337; rank-specific synthetic training batches
- 20 warmup and 80 measured optimizer steps
- Identical precision-independent training configuration
- Exact model-parameter verification after training

### Measured results

| Precision | Trial | Tokens/s | vs TE BF16 | p50 ms | p95 ms | Peak GB/GPU |
|---|---:|---:|---:|---:|---:|---:|
| TE BF16 | 1 | 293,397.4 | 1.0000x | 111.777 | 111.961 | 43.252 |
| TE BF16 | 2 | 293,999.4 | 1.0000x | 111.457 | 111.735 | 43.252 |
| FP8 | 1 | 343,722.9 | 1.1715x | 95.254 | 95.964 | 40.031 |
| FP8 | 2 | 344,077.0 | 1.1703x | 95.237 | 95.315 | 40.031 |
| NVFP4 | 1 | 314,354.0 | 1.0714x | 101.555 | 113.934 | 38.951 |
| NVFP4 | 2 | 319,308.8 | 1.0861x | 101.300 | 112.048 | 38.951 |

All three precision modes passed four-GPU correctness tests.
For GPT-1.3B, all 292 parameter tensors were bitwise identical
across four ranks at the end of each benchmark, and all checked
model parameter values were finite.

These tests do not independently establish equality of optimizer
states or Transformer Engine's internal scaling histories.

### Prediction versus observation

- TE BF16 first run: predicted approximately 285,000 tokens/s;
  actual 293,397 tokens/s, approximately 2.95% above prediction.
- FP8 first run: predicted approximately 310,000 tokens/s;
  actual 343,723 tokens/s, approximately 10.88% above prediction.
- NVFP4 first run: predicted approximately 325,000 tokens/s;
  actual 314,354 tokens/s, approximately 3.28% below prediction.

Across paired four-GPU trials, FP8 outperformed TE BF16 by
approximately 17.0–17.2%, and NVFP4 by approximately 7.1–8.6%.

FP8 was fastest in both trials. NVFP4 achieved the lowest
peak allocated GPU memory per rank.

TE BF16 throughput differed by 0.21% between trials, FP8 by
0.10%, and NVFP4 by 1.58%. Intermittent slow windows persisted
in NVFP4. Two trials per mode are insufficient for statistical
confidence intervals or broad performance generalizations.

### Weak-scaling interpretation

The per-GPU micro-batch size remained eight while GPU count
increased from one to four, making this a weak-scaling comparison.

Using the second single-GPU FP8 trial as reference, four-GPU
FP8 scaling efficiency was approximately 94.4%. That single-GPU
reference experienced slow timing windows, so this ratio may
overstate efficiency relative to uninterrupted steady-state
single-GPU performance.

Differences in scaling efficiency cannot be attributed solely
to collective communication without further profiling.

The distributed benchmark artifacts are:

- `runs/M6-1P3B-DDP4-TEBF16-001.json`
- `runs/M6-1P3B-DDP4-TEBF16-002.json`
- `runs/M6-1P3B-DDP4-FP8-001.json`
- `runs/M6-1P3B-DDP4-FP8-002.json`
- `runs/M6-1P3B-DDP4-NVFP4-001.json`
- `runs/M6-1P3B-DDP4-NVFP4-002.json`

## Kernel-level verification

At GPT-125M, the CUDA profiler showed:

- FP8 quantization kernels, including 1D and 2D casting
- FP8 delayed-scaling update kernels
- NVFP4 block-scaled CUTLASS GEMMs on Blackwell
- NVFP4 Hadamard transforms, quantization, and scaling kernels
- BF16 FlashAttention forward and backward operations

A profiled FP8 pass had 817 CUDA kernel launches.
A profiled NVFP4 pass had 1,389 launches.

These profile counts are diagnostic and do not directly explain
the complete step-time difference. The exact operand precision
of FP8 GEMMs was not conclusively identified from their kernel names.

## Timing variability

Both FP8 and NVFP4 experienced occasional slow 10-step windows
during GPT-1.3B runs.

FP8 repeat:
- Typical window-mean step time approximately 84.4 ms
- Slowest reconstructed window approximately 110.9 ms

NVFP4 repeat:
- Median window-mean step time approximately 91.6 ms
- Reported p95 window-mean step time approximately 110.2 ms

The p50 and p95 statistics describe 10-step window-average times,
not individual optimizer-step latency distributions.

A separate FP8 diagnostic monitored GPU activity:
- SM clock approximately 1,965 MHz during training
- Mostly 98–99% sampled GPU utilization
- One temporary utilization dip was observed
- No sustained GPU clock reduction was observed

Another FP8 diagnostic monitored cgroup CPU quota counters.
The measured slow window occurred with no CPU quota-throttling
events in that interval.

The cause of the intermittent slow windows remains unresolved.
Possible effects of contention or scheduling have not been excluded.

## Numerical correctness and limitations

BF16, FP8, and NVFP4 completed short controlled correctness tests
with finite reported losses, gradients, and optimizer updates.

The controlled 12-step GPT-125M numerical experiment is documented
in `docs/m6_smoke_results.md`.

Training on repeated synthetic batches is suitable for throughput
measurement but does not demonstrate equivalent model convergence,
held-out validation loss, or downstream quality.

MFU is reported for BF16 using the repository's conventional dense
BF16 peak denominator. For FP8 and NVFP4, MFU was intentionally
left unavailable because the model executes a mixture of precision
formats and lacks per-operation executed-FLOP accounting.

The original M6 speedup predictions are retained in
`docs/p4_lab_log.md`. See that file for the prediction-versus-actual
assessment and milestone status.

## Remaining work

1. Confirm exact FP8 GEMM operand precision with stronger kernel evidence.
2. Conduct longer real-data training and validation comparisons.
3. Verify optimizer-state and FP8 scaling-history agreement across ranks.
4. Characterize intermittent timing variation if further performance
   attribution is necessary.
5. Incorporate results into the final P4 training writeup.
