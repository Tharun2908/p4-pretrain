# M5 — Tensor Parallelism and Hybrid TP × DP

## Objective

Implement and validate tensor parallelism for the P4 GPT pretraining
benchmark, then combine TP with data parallelism.

The milestone focuses on:
- Numerical correctness against a non-parallel reference
- Hybrid TP × DP gradient and optimizer correctness
- Distributed checkpointing and crash recovery
- Throughput and memory tradeoffs

## Hardware and software

Validated environment:

- NVIDIA B200-SXM GPUs
- Single-node 2-GPU and 4-GPU configurations
- PyTorch 2.8.0 with CUDA support
- NCCL 2.27.3
- BF16 autocast and FlashAttention

## Implemented strategies

| Strategy | World size | Device mesh |
|---|---:|---|
| TP | 2 | TP=2 |
| Hybrid TP × DP | 4 | DP=2, TP=2 |

For the hybrid mesh:

- TP groups: ranks [0, 1] and [2, 3]
- DP groups: ranks [0, 2] and [1, 3]
- Ranks in the same TP group process identical input batches
- Different DP replicas process different input batches

## Implementation

Production modules:

- `tp_parallel.py`: tensor parallelism, DP gradient averaging, clipping
- `tp_checkpoint.py`: distributed checkpointing and recovery
- `train.py`: integration and benchmark execution

### Tensor parallelism

- Attention QKV projection: column-wise sharding
- Attention output projection: row-wise sharding
- MLP expansion projection: column-wise sharding
- MLP output projection: row-wise sharding
- Embeddings and LayerNorm: replicated
- LM head: replicated, with tied embedding weights preserved

Fused QKV weights must be reordered from
`[Q_all, K_all, V_all]` into rank-major QKV order before sharding.

### Hybrid TP × DP

Hybrid training averages corresponding local parameter gradients
across DP replicas.

Global gradient clipping accounts for each sharded contribution
once and each replicated parameter once, without double-counting
TP replicas.

## Reproducing correctness tests

The passing distributed tests are in `tests/distributed/`.

Run from the repository root:

```bash
export PYTHONPATH="$PWD:$PWD/tests/distributed"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=WARN
```

### Full GPT parity: TP=2

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  tests/distributed/m5_tp_gpt_smoke.py \
  --model gpt-tiny
```

Validated GPT-tiny results:

| Check | Result |
|---|---:|
| Maximum logits difference | 2.384e-07 |
| Loss difference | 0 |
| Maximum parameter-gradient difference | 1.788e-07 |
| Tied embeddings | Preserved |

Full GPT-1.3B forward and gradient parity was also validated.

### BF16 training: TP=2

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  tests/distributed/m5_tp_standard_adamw_smoke.py
```

Three training steps passed using BF16 FlashAttention,
standard AdamW and TP-aware gradient clipping.

### Hybrid gradient parity: TP=2 × DP=2

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  tests/distributed/m5_tp_dp_gradient_probe.py
```

Validated results:

- Reference loss: 10.83638573
- Hybrid loss: 10.83638573
- Parameter gradients checked: 28
- Maximum gradient difference: 2.533e-07

### Hybrid optimizer parity

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  tests/distributed/m5_tp_dp_optimizer_probe.py
```

Three optimizer updates matched the reference model:

- Maximum gradient-norm difference: 1.350e-07
- Maximum parameter difference: 3.822e-06
- DP averaging: PASS
- TP global clipping: PASS
- AdamW update parity: PASS

Other distributed tests cover MLP, attention, transformer blocks,
device-mesh layout, clipping, BF16 behavior, and DCP restart.

These tests require explicit `torchrun` execution and are not
automatically included in ordinary pytest runs.

## Distributed checkpointing

The checkpoint implementation uses PyTorch Distributed Checkpoint
(DCP) to save model and optimizer states.

Additional per-rank runtime state includes:

- Completed global step
- Synthetic-data cursor
- CPU RNG state
- CUDA RNG state

A configuration signature is checked when resuming.

The `latest.json` pointer is updated atomically after successful
model, optimizer and rank-local runtime writes.

### Reproduce a controlled restart

Choose a checkpoint directory that does not already exist.

First, train through step 2:

```bash
export ROOT=logs/m5_train_documentation_probe

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train.py \
  --model gpt-tiny \
  --strategy tp-dp \
  --backend flash \
  --micro-bs 2 \
  --grad-accum 1 \
  --steps 4 \
  --bench-warmup 0 \
  --log-every 1 \
  --grad-clip 0.1 \
  --checkpoint-dir "$ROOT" \
  --save-every 2 \
  --stop-after 2 \
  --dry-run
```

Then resume:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train.py \
  --model gpt-tiny \
  --strategy tp-dp \
  --backend flash \
  --micro-bs 2 \
  --grad-accum 1 \
  --steps 4 \
  --bench-warmup 0 \
  --log-every 1 \
  --grad-clip 0.1 \
  --checkpoint-dir "$ROOT" \
  --save-every 2 \
  --resume \
  --dry-run
```

The resumed run should execute global steps 3 and 4.

### Recovery validation

Validated scenarios include:

- DCP roundtrip
- Fresh-process restart
- Exact comparison with uninterrupted training
- Restart following an intentionally terminated worker
- Collective error propagation during checkpoint creation
- Collective handling of rank-local write failures
- Collective preflight for checkpoint loading
- Preservation and quarantine of incomplete checkpoints
- Rejection of automatic quarantine without `latest.json`

The uninterrupted and recovered runs matched in model state,
optimizer state, data cursors, and CPU/CUDA RNG states.

Diagnostic scripts are preserved under `experiments/m5/`.

## Official M5 benchmark results

Four clean-tree benchmark runs were recorded on 2026-11-05
using the M5 implementation at commit `b1d1583`.

All runs used:

- GPT-1.3B, sequence length 1024
- NVIDIA B200-SXM GPUs
- BF16 autocast and FlashAttention
- 120 training steps, including 30 warmup steps
- 90 measured training steps
- Gradient accumulation 1
- Global gradient clipping at 0.1
- Synthetic training data
- Seed 1337

### Measured results

| Run | Strategy | GPUs | Tokens/step | Tokens/s | MFU | Peak GB/GPU |
|---|---|---:|---:|---:|---:|---:|
| M5-DDP2-001 | DDP=2 | 2 | 16,384 | 148,011 | 27.87% | 43.2 |
| M5-TP2-001 | TP=2 | 2 | 16,384 | 80,806 | 15.22% | 39.4 |
| M5-DDP4-001 | DDP=4 | 4 | 32,768 | 288,584 | 27.17% | 43.2 |
| M5-HYBRID-001 | TP=2 x DP=2 | 4 | 32,768 | 148,641 | 13.99% | 39.4 |

### Step-time measurements

| Strategy | p50 | p95 |
|---|---:|---:|
| DDP=2 | 110.7 ms | 110.9 ms |
| TP=2 | 202.3 ms | 204.3 ms |
| DDP=4 | 113.5 ms | 113.8 ms |
| TP=2 x DP=2 | 213.2 ms | 260.1 ms |

These percentiles describe average step times within
10-step measurement windows, not individually timed steps.

### Interpretation

At matched global tokens per step:

- DDP=2 delivered approximately 1.83x TP=2 throughput.
- DDP=4 delivered approximately 1.94x hybrid throughput.
- TP and hybrid allocated 3.8 GB less peak GPU memory
  than their respective DDP comparisons.
- DDP throughput increased approximately 1.95x when
  scaling from two GPUs to four GPUs at fixed
  per-rank microbatch size.

The hybrid run exhibited late-window throughput variability:
cumulative throughput reached 153,244 tokens/s at step 100,
then finished at 148,641 tokens/s at step 120.
The logs do not establish its cause.

### Predictions versus observations

- DDP=2: predicted to outperform TP=2; confirmed.
- TP=2: predicted approximately 82,000 tokens/s;
  observed 80,806 tokens/s.
- DDP=4: predicted approximately 290,000 tokens/s;
  observed 288,584 tokens/s.
- Hybrid: predicted approximately 145,000 tokens/s;
  observed 148,641 tokens/s.

### Comparison limitations

The comparisons evaluate the implemented configurations,
not an isolated measurement of TP communication costs.

- DDP uses fused AdamW.
- TP and hybrid use single-tensor AdamW.
- Per-rank or per-TP-group microbatch sizes differ.
- Hybrid manually synchronizes DP gradients per parameter.
- TP adds intra-layer communication.

All four unmodified benchmark records are available at:

- `runs/M5-DDP2-001.json`
- `runs/M5-TP2-001.json`
- `runs/M5-DDP4-001.json`
- `runs/M5-HYBRID-001.json`

## Current limitations

1. Tensor-parallel degree is fixed at TP=2.
2. Hybrid mode uses the validated 2 × 2 mesh.
3. Gradient accumulation is restricted to 1 for TP/hybrid.
4. TP with `torch.compile` is not validated.
5. TP currently uses single-tensor AdamW.
6. Tested fused and foreach AdamW paths failed with the current
   mixture of ordinary tensors and DTensors.
7. Hybrid DP gradient synchronization uses one collective
   per parameter instead of gradient buckets.
8. Restart integration supports synthetic data and `--dry-run`.
9. Restart requires matching topology and training configuration.
10. Concurrent writers to one checkpoint directory are unsupported.
11. Catastrophic failures during collectives are not covered by
    the ordinary collective filesystem-error handling.

## Regression suite

Ordinary repository tests:

```bash
pytest -q
```

The existing regression suite passed 94 tests during M5 review.

Distributed correctness tests must be launched separately with
the appropriate GPU count.

Experimental scripts, including failed optimizer approaches,
are retained under `experiments/m5/` as diagnostic evidence.

## Conclusion

M5 establishes validated TP=2 and hybrid TP × DP training,
reference-model numerical parity, restartable distributed
checkpoints, and an initial performance baseline.

The implementation prioritizes correctness and reproducibility.
Communication optimization and broader parallel configurations
remain future work.
