# M6 — B200 Mixed-Precision Pretraining Protocol

Status: Single-B200 and four-B200 precision benchmark matrices completed, with distributed parameter-equality verification. Real-data convergence, optimizer/scaling-state validation, and exact FP8 GEMM operand verification remain pending. See `docs/m6_results.md` and `docs/m6_smoke_results.md`.

## Objective

Compare BF16, FP8, and NVFP4 training on NVIDIA B200
without confusing hardware capability, kernel throughput,
and end-to-end optimizer-step throughput.

## Model sizes and hardware

- GPT-125M: correctness and overhead characterization.
- GPT-1.3B: primary large-model performance comparison.
- One B200: core controlled precision benchmarks.
- Four B200s: follow-up distributed confirmation.

## Precision configurations

1. Native PyTorch BF16: preserve the M1 implementation.
2. Transformer Engine BF16: framework-matched control.
3. Transformer Engine FP8: explicit, recorded scaling recipe.
4. Transformer Engine NVFP4: Blackwell-specific mixed precision.

FP8 and NVFP4 do not imply all operations, parameters,
gradients, and optimizer states use that storage format.

Record exactly which linear operations are quantized.
Preserve sensitive higher-precision operations as needed.
Keep model architecture and weight tying unchanged.

## Software constraints

Current stack:
- PyTorch 2.8.0+cu128
- CUDA runtime 12.8
- cuDNN 9.10.2
- B200 SM 10.0
- CUDA toolkit 12.8 and development headers used for the original Transformer Engine extension build
- Transformer Engine 2.19 validated in isolated `/workspace/envs/p4-m6`
- Transformer Engine PyTorch extension compiled locally

The original pinned environment `/workspace/envs/p4` remains unchanged.
The replacement pod did not retain the CUDA toolkit, but the compiled
Transformer Engine extension remained usable in the persistent venv.
Rebuilding the extension may require restoring toolkit headers and nvcc.

Do not modify the pinned baseline environment blindly.
Record package versions, toolkit, and image provenance.

## Correctness checklist

- [x] Forward and backward complete on a tiny model.
- [ ] Actual low-precision kernels/recipes are verified.
- [x] Tied embedding/output weights remain tied.
- [x] Loss and gradient values are finite.
- [x] Optimizer parameters update correctly.
- [x] Quantization/scaling and higher-precision exceptions documented.
- [x] Short-run loss trajectories compared on identical tokens.
- [ ] Any numerical divergence or overflow investigated.

## Controlled benchmarking checklist

- [x] Same architecture, initialization, and seed.
- [x] Same dataset and token ordering.
- [x] Same sequence length and effective global batch.
- [x] Same optimizer, LR schedule, and gradient clipping.
- [x] Same GPU count, warmup, and measured-step budget.
- [x] Compiler and attention backend differences recorded.
- [x] Steady-state timing excludes compilation and initialization.
- [x] Multiple measurements for representative configurations.
- [x] Aggregate and per-GPU tokens/sec, step latency, and peak allocated memory reported.
- [ ] Training and held-out validation loss compared where valid evaluation data exists.
- [ ] Optional GPU power and energy/token measured with a documented sampling method.
- [x] Loss, gradient norms, and NaN/Inf behavior reported.
- [x] MFU denominators and precision assumptions documented.
- [x] Precision speedups compared to an appropriate BF16 control.

## Prediction discipline

The original M6 predictions in docs/p4_lab_log.md
remain unchanged. Record any revised expectations
separately and before the corresponding measurements.

Original hypotheses include:
- FP8/BF16 speedup: 1.2–1.5x.
- NVFP4/FP8 additional speedup: approximately 1.1–1.2x.
- Larger model may benefit more from lower precision.

These are hypotheses, not guaranteed outcomes.

## Scope

Short synthetic-data benchmarks establish throughput
and early numerical behavior, not model convergence
or downstream quality. A clean real-data loss trajectory
remains a separate project requirement.
