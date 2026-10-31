# M1 Batch-Size Sweep — Predictions

Date: 2026-10-31
Hardware: 1x NVIDIA B200
Model: GPT-125M
Precision: BF16
Attention: Flash
Compilation: enabled
Sequence length: 1024
Gradient accumulation: 1
Steps: 200
Warmup: 50

## Predictions recorded before experiments

| Run | Micro-batch | Predicted throughput | Predicted MFU |
|---|---:|---:|---:|
| M1-006 | 16 | 685,000 tok/s | 26% |
| M1-007 | 32 | 850,000 tok/s | 32% |
| M1-008 | 64 | 1,000,000 tok/s | 38% |
| M1-009 | 128 | 1,100,000 tok/s | 42% |

Hypothesis:
Larger batches increase the matrix multiplication M dimension
and amortize fixed overhead, improving GPU utilization.
Throughput gains should diminish as the GPU becomes saturated.

The batch-16 prediction uses M1-002 as an empirical baseline.
The larger-batch predictions are extrapolations, not measurements.

Record peak memory, throughput, MFU, and any OOM.
