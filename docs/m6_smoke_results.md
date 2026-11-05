# M6 — Transformer Engine Precision Smoke Tests

## Status

GPT-125M BF16, FP8, and NVFP4 single-GPU smoke tests completed.

These are correctness and short-run numerical experiments.
They are not measured throughput runs, convergence studies, or
evidence of equivalent downstream model quality.

## Environment

- GPU: NVIDIA B200 (SM 10.0)
- PyTorch: 2.8.0+cu128
- CUDA runtime: 12.8
- CUDA toolkit: 12.8
- cuDNN: 9.10.2
- Transformer Engine: 2.19.0
- Transformer Engine core: CUDA 12
- Transformer Engine PyTorch extension: locally compiled
- Isolated environment: `/workspace/envs/p4-m6`
- Baseline environment: `/workspace/envs/p4` (unchanged)

The source-built extension required CUDA development headers and
include-path configuration. Runtime loading required access to
the installed NVIDIA wheel libraries.

These environment paths describe the experimental pod and are
not portable installation instructions.

## Model and precision boundaries

- Model: GPT-125M, 124,475,904 parameters
- Transformer blocks: 12
- Transformer Engine linear layers: 48
- Embeddings and language-model head: original PyTorch modules
- Output projection remains tied to token embeddings
- Attention: PyTorch scaled dot-product attention
- LayerNorm: PyTorch
- Master parameters: FP32
- Layer input boundary: BF16 in the controlled comparison
- BF16: Transformer Engine BF16 control
- FP8: DelayedScaling, HYBRID, amax history 16
- NVFP4: default NVFP4BlockScaling recipe

Quantized layer activation was checked, but executed low-precision
GEMM kernels have not yet been independently verified by profiling.

## Standalone linear-layer observations

At M=128, K=128, N=256, using identical weights and inputs:

| Metric | FP8 vs TE BF16 | NVFP4 vs TE BF16 |
| --- | ---: | ---: |
| Output relative L2 error | 4.08% | 14.61% |
| Weight-gradient relative L2 error | 7.82% | 16.93% |
| Input-gradient relative L2 error | 7.75% | 17.08% |
| Output cosine similarity | 0.999167 | 0.989316 |

The default NVFP4 recipe failed for the earlier M=32 configuration
with a cuBLASLt unsupported-algorithm error; M=128 succeeded.
The exact supported shape boundary remains unverified.

## GPT-125M framework-matched BF16 control

Replacing the 48 internal PyTorch Linear modules with Transformer
Engine Linear modules preserved:

- Model parameter count
- Tied embedding/output weights
- Native BF16 loss (11.0078125)
- Logits (relative L2 error 0.0)
- First-block QKV weight gradients (relative L2 error 0.0)
- Nonzero optimizer updates

This initial equivalence test used the original input-dtype policy.
The controlled training experiment below explicitly used BF16
inputs to every TE Linear in all three modes.

## Controlled 12-step GPT-125M experiment

Configuration:

- One B200
- Seed 42 for model initialization
- Synthetic token generator seed 2027
- Identical initial state and token batches across modes
- Sequence length 128; micro-batch size 1
- AdamW LR 3e-4, betas (0.9, 0.95), eps 1e-8
- Weight decay 0.1; gradient clipping 1.0
- BF16 inputs to all Transformer Engine linear layers
- 12 optimizer steps, no learning-rate schedule

| Metric | TE BF16 | FP8 | NVFP4 |
| --- | ---: | ---: | ---: |
| Active quantized layers | 0/48 | 48/48 | 48/48 |
| Step 0 loss | 10.989746 | 10.978516 | 10.998047 |
| Step 11 loss | 10.969238 | 10.952637 | 10.964844 |
| Step 0 gradient norm | 24.4858 | 23.8468 | 24.5607 |
| Step 11 gradient norm | 2.6864 | 2.6861 | 2.7022 |
| Peak allocated memory, GiB | 2.388 | 2.383 | 2.384 |

First-step relative L2 errors against TE BF16:

| Metric | FP8 | NVFP4 |
| --- | ---: | ---: |
| Logits | 28.7065% | 36.8936% |
| First-block QKV gradients | 38.0031% | 49.3303% |

All modes completed 12 optimizer updates with finite losses and
gradients. The synthetic random targets do not establish learning
convergence. The observed memory differences are negligible.

## Remaining validation

- Verify actual GEMM kernels via profiling.
- Integrate precision selection into the main training runner.
- Check multi-step stability and scaling-state behavior.
- Run matched, warmed-up throughput benchmarks.
- Compare larger-model behavior on GPT-1.3B.
- Validate longer training on meaningful token sequences.
- Confirm distributed correctness on four B200 GPUs.
- Record run IDs, software provenance, and experimental metrics.
