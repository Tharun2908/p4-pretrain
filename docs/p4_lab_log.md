# P4 Lab Log — Distributed Pretraining From Scratch

**Started:** 2026-08-01
**Repo:** `/workspace/p4/p4-pretrain/`
**Rule:** no unmotivated repeat measurements; documented reproducibility checks are experiments. If it isn't in this file, it didn't happen.

---

## 0. How this log works (read once, then obey it)

Four registers, four different purposes. Don't merge them.

| Register | What goes in | Why it exists |
|---|---|---|
| **§2 Run Log** | Every measured run, one row | Produces the scaling / memory / MFU tables in `docs/training.md` |
| **§3 Issue Log** | Every thing that broke and cost >20 min | Produces "tell me about a hard bug" answers |
| **§4 Decision Log** | Every non-obvious choice + the alternative rejected | Produces "why did you do it that way" answers |
| **§5 Negative Results** | Things that didn't work, *with structural explanation* | Your highest-value register. See below. |

### The one non-negotiable field: **Prediction before the run**

Before launching any measured run, write down what you expect the number to be *and why*.
Then record the actual. The gap between prediction and reality is the entire interview story.

This is not ceremony. It is the mechanism that turns "I ran FSDP and got 41K tok/s" (worthless,
anyone can read a log) into "I predicted ZeRO-3 would cost ~15% throughput because the all-gather
overlaps with compute; it cost 34%, and profiling showed the reason was the reduce-scatter on the
backward pass sitting on the critical path at this layer width" (a hire signal).

A run with no prediction recorded is a run you cannot tell a story about later. You will not
remember what you expected. Write it down before you hit enter.

### What counts as a negative result worth keeping

Only if you can state the **structural reason**. "FSDP was slower" is a number.
"FSDP was slower *because* on a model that fits in memory, ZeRO-3 pays all-gather cost per layer
for a memory saving with no consumer" is a negative result. Same discipline as the GRPO write-up:
document *why the method was structurally disadvantaged*, not just that it lost.

If you can't explain it structurally yet, park it in §3 as an open issue until you can.

### Weekly (Sunday, 10 min)

1. Any run in §2 without a prediction? → fix the habit, not the entry.
2. Any issue in §3 open >7 days? → it's a scope problem, decide: solve or cut.
3. Any §5 entry without a structural explanation? → downgrade to §3 until you have one.
4. Update the fill-in-the-blank pitch in §6 with any number that got resolved.

---

## 1. Milestone status

All hardware is **4×B200 on one node** from M1 onward (DEC-005). Sizes per DEC-006.

| M# | Label | GPUs | Status | Blocking |
|----|-------|------|--------|----------|
| M0 | Env + NCCL sanity | 1 → 4 | ✅ B200 4-rank NCCL verified; 651.1 GB/s peak ring-equivalent bus bandwidth; repeat confirmed | — |
| M1 | Single-GPU baseline + MFU, 125M **and** 1.3B, eager vs compiled | **1** | ✅ 33 B200 tests; M1-001…005 controlled baselines + M1-006…009 batch sweep complete; backend-cache regression passed | — |
| M2 | Data pipeline (ingest→filter→exact/near dedup→group split→tokenize→shard) | **0** (CPU) | ✅ M2-001…006 implemented and committed; ✅ M2-007 500-doc real-data pilot audited; ⬜ final CPU training smoke / M2-007 commit pending | Larger real corpus + clean 125M loss curve remain separate |
| M3 | DDP 4-way + scaling efficiency table | 4 | ✅ Seven measured DDP runs; synchronization, deterministic replay checks, and results documentation completed | — |
| M4 | FSDP / ZeRO-1,2,3 — overhead **trend across 125M→1.3B** | 4 | ✅ Core ZeRO-1/2/3 benchmarks completed at both sizes; follow-ups documented | — |
| M5 | 2-way TP **at 1.3B** + checkpoint/resume | 2 / 4 | ✅ TP and hybrid correctness, checkpoint recovery, benchmarks, and documentation committed | — |
| M6 | bf16→fp8→fp4 ablation at 125M **and** 1.3B | **1** core + 4 confirm | ⬜ not started | M1 |
| M7 | MFU before→after with diagnosed bottleneck | 4 | ⬜ not started | M3 |
| M8 | `docs/training.md` + README + writeup | 0 | 🟡 M0/M1/M3/M4/M5 results consolidated; final M6/M7 integration and writeup pending | M6, M7 |

**Never cut:** M3, M4, M7, one clean from-scratch loss curve (125M), one diagnosed instability.

---

## 1b. GPU budget & pod workflow

On this cluster **1 GPU schedules in seconds; 4 GPUs can take hours.** So the scarce resource is
4-GPU *sessions*, and the rules exist to spend them only on measurement.

| Work | Request | Why |
|---|---|---|
| `setup_env.sh` first build / verify | 1×B200 | must confirm `sm_100` + SDPA backends on the target arch |
| M2 tokenize / dedup / pack | 0 GPUs, many CPUs | CPU-bound; holding a GPU for it is waste |
| M1 single-GPU baselines | 1×B200 | by definition |
| Writing / debugging DDP, FSDP, TP code | 1 GPU, `world_size=1` | most bugs show at world 1; only comms bugs need ≥2 |
| Checkpoint/resume logic | 1 | logic is rank-agnostic; the real `kill -9` test is 4 |
| M6 precision ablation (core) | 1×B200 | precision speedup is a per-GPU compute property; 1 GPU isolates it from comms |
| M3/M4/M5/M7 measured runs, M6 confirm, M5 kill test | 4×B200 | scaling *is* the measurement |

1. **Never debug on a 4-GPU pod.** Arrive with a run queue already dry-run on 1 GPU
   (`world_size=1`, few steps), execute back-to-back, release.
2. **Every 4-GPU session starts with `ddp_smoke.py --run-id M0-B200-<nn>`** (~10 s): free
   bandwidth sample per session + co-tenant check before anything gets measured.
3. **Delete the 1-GPU pod before requesting 4.** It holds one of the node's B200s against you,
   and `ml-pvc` is RWO: if it ever sits on a *different* node, the 4-GPU pod hangs in
   `ContainerCreating` with a Multi-Attach error.
4. **Env is hardware-portable, measurements are not.** torch 2.8 / CUDA 12.8 runs on A100/H200/B200,
   so a 1-GPU debug session can land on any free GPU — but measured runs only on B200 (DEC-005).
5. **New pod → `bash setup_env.sh` → `source /workspace/p4/p4env.sh`.** Fast path ~10 s.

---

## 2. Run Log

Every measured run. `bench.py` writes the JSON; this table is the human-readable index.
Regenerate the table any time with `python bench.py --summarize runs/ > /tmp/table.md`.

### Entry template

```
### RUN <id>  —  <one-line what this run answers>
- **Date / HW:** 2026-08-__ · 4×A100-40GB SXM (cl-worker__) · torch __ · NCCL __
- **Config:** model=125M · block=1024 · micro_bs=__ · grad_accum=__ · strategy=DDP · precision=bf16
- **Question:** (the single thing this run is supposed to settle)
- **PREDICTION (written before launch):** __ tok/s, __ GB peak, __ % MFU — because __
- **Actual:** __ tok/s · __ GB peak · __ % MFU · loss @ __ steps = __
- **Gap vs prediction:** __ (if >20% off, that's a §5 or §3 entry — don't let it slide)
- **Decision / next:** __
- **Artifacts:** `runs/<id>.json`, `plots/<id>.png`
```

### RUN M0-001 — Does NCCL come up on 4 GPUs, and is the interconnect actually NVLink? ✅ *(A100 reference — pre-DEC-005)*

- **Date / HW:** 2026-08-02 · 4×A100-SXM4-40GB (`p4-training-a100`) · torch 2.4.1+cu124 ·
  NCCL 2.20.5+cuda12.4 · driver 570.211.01
- **Question:** Do 4 ranks initialize, does all-reduce return the right answer, and what
  bandwidth does the PyTorch/NCCL stack actually deliver?
- **PREDICTION (written before launch):** peak all-reduce busbw **200–240 GB/s**, plateauing
  past ~64 MB.
- **Actual:** **195.0 GB/s peak busbw** (algbw 130.0) at 512 MB, all ranks `allreduce_ok=True`.
  Repeat run: 195.3 GB/s → **run-to-run variance 0.15%**, so single-run measurements on this
  node are trustworthy.
  Curve: 1MB 16.3 · 8MB 92.8 · 32MB 150.2 · 64MB 159.6 · 128MB 178.0 · 256MB 187.4 · 512MB 195.0
- **Gap vs prediction:**
  - *Magnitude:* 195 vs a 200–240 range — **2.5% under the low end.** Prior was sound.
  - *Plateau: WRONG.* Curve was still climbing at 512 MB (+4% from 256→512), nowhere near flat.
    The 64 MB plateau guess was anchored on published 8-GPU DGX nccl-tests curves; at 4 ranks
    the ring is shorter, so 512 MB divides to only ~5 MB per channel per ring step across 24
    channels — not enough to amortize per-chunk overhead. Knee is further out. Sweep now
    extends to 2 GB by default.
- **What the NCCL log added that the table didn't:**
  1. `NVLS multicast support is not available` · `0 nvls channels` → **no NVLink SHARP** on this
     node. Ring/tree only, no in-network reduction, so the ring busbw convention genuinely
     describes the traffic here. *B200 prediction: Blackwell NVSwitch likely DOES have NVLS; if
     it engages, ring-convention busbw stops describing physical traffic and may exceed what a
     ring could do. Do not be confused by it at M6.*
  2. `via P2P/CUMEM/read` on all 24 channels → positive confirmation NVLink carries the
     collective. Not SHM, not sockets.
  3. `Failed to open libibverbs.so` → `Using network Socket`. **Harmless today** (`nNodes 1`,
     everything P2P) but the node has 8× mlx5 NICs the container can't reach. If the fenced
     multi-node phase ever unfences, NCCL falls back to TCP and throughput collapses. See ISSUE-002.
- **Decision / next:** M0 passes. Proceed to M1. Extend sweep to 2 GB on next opportunity to
  locate the actual knee (low priority — see the M7 hypothesis below, the large-message region
  is academic for DDP).
- **Artifacts:** `runs/M0-001.json`, `/tmp/nccl_debug.txt`
- **⚠ Record hygiene:** a second invocation overwrote `runs/M0-001.json` with the placeholder
  `--prediction "your number here"`. Measurement survived, prediction didn't. Restore the real
  prediction string by hand, and use `--run-id` for repeat runs.

### DERIVED — M7 hypothesis, handed over by M0 before any training loop exists

PyTorch DDP's default gradient bucket is **25 MB**. At 32 MB this node delivers 150 GB/s busbw,
**77% of the 195 GB/s peak**. A 125M model in bf16 has ~250 MB of gradients → ~10 buckets, each
transferring at three-quarters of achievable bandwidth.

- **Hypothesis:** raising `bucket_cap_mb` recovers part of that ~23% gap.
- **Counter-pressure (why it isn't free):** fewer, larger buckets means gradient sync starts
  later in the backward pass, so less comms/compute overlap. The optimum is interior, not at
  either extreme.
- **Test at M7:** sweep `bucket_cap_mb` ∈ {25, 50, 100, 250}, report tok/s and MFU for each.
- **Why this is a good story:** the lever was identified from interconnect measurements *before*
  the training loop existed, rather than found by trial and error afterwards.

> Why this run matters and isn't just a smoke test: the measured busbw is the denominator for
> every DDP/FSDP explanation you'll give later. "DDP scaled to 3.6× of ideal at 4 GPUs, and the
> gap is gradient all-reduce — which I know cost me X ms/step because I measured the bus at
> Y GB/s" is a complete answer. Without this number you're guessing.

### M0 B200 NCCL validation — 2026-11-04

- **Hardware:** 4× NVIDIA B200, one node (`cl-worker36`), NV18 topology.
- **Protocol:** FP32 all-reduce, 4 ranks, 1 MiB–2 GiB message sweep, 5 warmup and 20 timing iterations per message size.
- **Prediction M0-B200-001:** peak algorithm bandwidth above the earlier A100 reference of 130.2 GB/s.
- **Actual M0-B200-001:** peak algorithm bandwidth **434.0 GB/s**, ring-equivalent bus bandwidth **651.1 GB/s**, at 2 GiB; all-rank correctness passed.
- **Prediction M0-B200-002:** repeat peak algorithm bandwidth within 10%.
- **Actual M0-B200-002:** **433.6 GB/s** algorithm, **650.5 GB/s** bus; algorithm-bandwidth difference **−0.09%**.
- **Interpretation:** bandwidth continued increasing through the largest tested message. The ring-equivalent bus-bandwidth metric is a convention, not a direct measurement of physical NVLink traffic. NCCL reported NVLS capability, but this alone does not establish which collective algorithm handled each size.
- **Limit:** comparison with the earlier A100 result is cross-generation and uses different best message sizes; it is not a controlled hardware-only speedup.
- **Artifacts:** `runs/M0-B200-001.json`, `runs/M0-B200-002.json`.

### M1 integration checks — 2026-10-31 (SMOKE ONLY; NOT performance baselines)

- **Hardware / code:** 1× NVIDIA B200, `cl-worker36`; driver 580.196.07; PyTorch 2.8.0+cu128; pytest 9.2.0; initial GPU validation at commit `3b21eb7`.
- **Correctness:** 33 tests passed, 0 skipped (`python -m pytest -q -rs`).
- **Eager / auto SDPA:** `gpt-tiny`, micro_bs=4, block=64, 40 steps, 20 warmup steps; profiler verified Flash forward and backward; 78,108 tok/s, first step 3.7s, peak 0.2 GB; loss 10.8577→10.1794.
- **Compiled / cuDNN SDPA:** same tiny config, `--compile --backend cudnn`; profiler verified cuDNN forward and backward; 80,973 tok/s, first step 14.1s, peak 0.2 GB; loss 10.8566→10.1776.
- **Caveat:** differing attention backends mean tiny-run speed difference is **not** an isolated compile benefit. First-step times contain compilation, autotuning and first-touch overhead. The container needed a C compiler (`gcc`/`g++`) for Triton; apt-installed system compilers are not persisted on the PVC after pod recreation.

### M1-001…005 — registered predictions and measurement protocol

- **Prediction recorded:** 2026-10-31, **before the first measured 125M/1.3B run**. Original hypotheses remain unchanged below. Subsequent *revised* expectations were recorded in the respective run's `--prediction` string after earlier results were known; they are marked as conditional updates and are not substituted for the original priors.
- **Hardware / precision:** 1× B200, `cl-worker36`, BF16 autocast; dense BF16 peak denominator = 2.25e15 FLOP/s; GPT-125M ≈ 0.855G estimated training FLOPs/token, GPT-1.3B ≈ 8.47G.
- **Controlled protocol:** `--micro-bs 16 --grad-accum 1 --steps 200 --bench-warmup 50 --log-every 10`, sequence length 1024, synthetic on-device batches (16,384 tokens/optimizer step), fused AdamW, 150 measured steps; eager/compiled pairs explicitly use Flash. M1-005 changes *only* the attention backend to cuDNN relative to M1-004. Each run profiles observed fused attention operations.
- **MFU accounting caveat:** `bench.py` uses the PaLM/nanoGPT convention `6N + 12LHQT` FLOPs/token, including a full `T×T` attention term. Fused **causal** Flash/cuDNN attention can skip much of the masked upper triangle, so this convention *overcredits* attention FLOPs (roughly half of the attention term, subject to block/tile effects). The reported MFUs are **not** causal-adjusted and are kept for published-convention comparability; the heuristic relative overcount is ~6.5–7% at 125M and ~3.5–4% at 1.3B. These estimates do not establish executed FLOPs. See reconstructed ISSUE-005.
- **Original pre-benchmark predictions:**
  - **M1-001, 125M eager/Flash:** 5% MFU, ≈132K tok/s; anticipated poor utilization from narrower layers and overhead.
  - **M1-002, 125M compiled/Flash:** 8% MFU, ≈211K tok/s, ≈+60% throughput over eager; expected kernel fusion to help a small model.
  - **M1-003, 1.3B eager/Flash:** 15% MFU, ≈40K tok/s; expected improved MFU, lower tok/s.
  - **M1-004, 1.3B compiled/Flash:** 20% MFU, ≈53K tok/s, ≈+33% over eager; expected relatively less compiler gain than 125M.
  - **M1-005, 1.3B compiled/cuDNN:** cuDNN ≈5% faster than Flash, low confidence; shape-specific fused attention kernels might help. Original magnitude depended on original M1-004 estimate, not yet measured values.

| Run | Model | Mode / observed attention | Predicted MFU | Actual tok/s | Actual MFU | Step p50* | Peak allocated | First step** |
|---|---|---|---:|---:|---:|---:|---:|---:|
| M1-001 | 125M | Eager / Flash | 5% | **461,503** | **17.54%** | 35.4 ms | 17.2 GB | 1.0 s |
| M1-002 | 125M | Compiled / Flash | 8% | **685,273** | **26.05%** | 23.9 ms | 12.3 GB | 11.8 s |
| M1-003 | 1.3B | Eager / Flash | 15% | **91,087** | **34.30%** | 179.6 ms | 57.5 GB | 1.7 s |
| M1-004 | 1.3B | Compiled / Flash | 20% | **105,819** | **39.85%** | 154.9 ms | 52.8 GB | 23.2 s |
| M1-005 | 1.3B | Compiled / cuDNN | +5% vs Flash | **113,663** | **42.81%** | 143.9 ms | 52.8 GB | 3.4 s (warm cache) |

*Step-time p50 and p95 are calculated from **averages of 10-step windows**, not individually instrumented steps. **First-step timing includes compile/autotune/initialization, and cache state differs across runs; the M1-005 3.4 s first step is warm-cache and cannot be compared as cold compilation cost.*

### RUN M1-001 — GPT-125M eager BF16 / Flash, 1×B200

- **Question:** Single-GPU baseline throughput, MFU, memory at 125M.
- **PREDICTION (original):** 5% MFU, 132,000 tok/s, ≈124 ms/step.
- **Actual:** **461,503 tok/s**, **17.54% MFU**, **35.4 ms window-mean p50**, **17.2 GB peak allocated**; first step **1.0s**; loss 10.9790→5.6386 on repeated synthetic batches.
- **Gap vs prediction:** **3.50×** higher tok/s (≈+250%) and 3.51× expected MFU. The simple small-model utilization prior was too pessimistic; 16,384 token positions create substantial matmul parallel work. This is a **hypothesis about why**, not attribution established through per-op profiling.
- **Backend evidence:** profiler observed Flash forward and backward.
- **Artifact:** `runs/M1-001.json`.

### RUN M1-002 — GPT-125M compiled BF16 / Flash, 1×B200

- **Question:** How much does `torch.compile` change steady-state performance at 125M when the attention backend is held fixed?
- **PREDICTION (original):** 8% MFU, ≈211,000 tok/s. **Conditional revision (after M1-001, before M1-002):** 18–22% MFU, modest improvement over observed eager baseline.
- **Actual:** **685,273 tok/s**, **26.05% MFU**, **23.9 ms window-mean p50**, **12.3 GB peak allocated**, first step **11.8s**.
- **Gap:** **3.25×** original tok/s prediction; also above revised 18–22% MFU band. Versus M1-001: **+48.5% throughput (1.485×)**, **+8.51 percentage points MFU**, and **−28.5% peak allocated memory**.
- **Interpretation:** Compiler optimization materially improved this fixed workload. Kernel fusion, memory traffic and launch amortization are plausible causes, but exact causal attribution requires M7 profiling.
- **Backend evidence:** profiler observed Flash forward and backward.
- **Artifact:** `runs/M1-002.json`.

### RUN M1-003 — GPT-1.3B eager BF16 / Flash, 1×B200

- **Question:** Does increasing model width/depth improve MFU despite lowering token throughput?
- **PREDICTION (original):** 15% MFU, ≈40,000 tok/s (≈410 ms/step).
- **Actual:** **91,087 tok/s**, **34.30% MFU**, **179.6 ms window-mean p50**, **57.5 GB peak allocated**; first step **1.7s**; loss 11.2096→10.1926 on repeated synthetic batches.
- **Gap:** **2.28×** original tok/s prediction, ≈2.29× predicted MFU. Compared with 125M eager, MFU is **1.96× higher** while tok/s is **≈5.07× lower**.
- **Interpretation:** Larger-matrix utilization plausibly explains the MFU improvement; no per-operator profile has isolated the cause. Synthetic repeated data is not a quality/generalization metric.
- **Backend evidence:** profiler observed Flash forward and backward.
- **Artifact:** `runs/M1-003.json`.

### RUN M1-004 — GPT-1.3B compiled BF16 / Flash, 1×B200

- **Question:** Does the same compiler optimization yield a smaller *relative* speedup at 1.3B than at 125M?
- **PREDICTION (original):** 20% MFU, ≈53,000 tok/s. **Conditional revision (after M1-003, before M1-004):** 38–45% MFU (≈101–120K tok/s).
- **Actual:** **105,819 tok/s**, **39.85% MFU**, **154.9 ms window-mean p50**, **52.8 GB peak allocated**, first step **23.2s**.
- **Gap:** **2.00×** original tok/s prediction, but within conditional revised band. Versus M1-003: **+16.2% throughput (1.162×)**, **+5.55 points MFU**, **−8.2% peak allocated memory**.
- **Interpretation:** Larger model shows less relative compilation gain than 125M under the same benchmark protocol; profile in M7 to test whether compute/launch balance explains the difference.
- **Observation:** cumulative throughput drifted from ~107.6K to 105.8K tok/s during the 150 measured steps; cause unknown.
- **Backend evidence:** profiler observed Flash forward and backward.
- **Artifact:** `runs/M1-004.json`.

### RUN M1-005 — GPT-1.3B compiled BF16 / cuDNN, 1×B200

- **Question:** Is cuDNN faster than Flash for this specific compiled model shape, with all other settings held constant?
- **PREDICTION (original):** cuDNN ≈5% faster than Flash, low confidence. **Conditional magnitude (after M1-004):** ~111,000 tok/s, 41.8% MFU.
- **First attempt:** profiler saw **Flash** despite `--backend cudnn` and `verify_backend` aborted before measured windows; **not a valid cuDNN result**. See ISSUE-004.
- **Diagnostic:** with `TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/p4-inductor/m1-005-cudnn`, a separate 10-step dry run confirmed cuDNN forward/backward, 114,949 tok/s (diagnostic only; no warmup).
- **Actual (200 steps / 50 warmup / 150 measured):** **113,663 tok/s**, **42.81% MFU**, **143.9 ms window-mean p50**, **52.8 GB peak allocated**, first step **3.4s with cached compilation**.
- **Gap:** **+7.41% cuDNN versus Flash** at the same compiled GPT-1.3B configuration, versus predicted +5% (difference +2.41 percentage points). Conditional 111K throughput prediction was within ~2.4% of measured result.
- **Backend evidence:** profiler saw `aten::_scaled_dot_product_cudnn_attention` and backward; kernel names included `cudnn_generated_...sm100...`.
- **Artifact:** `runs/M1-005.json`.

### M1-006…009 — GPT-125M micro-batch sweep (predictions registered before runs)

- **Date / HW / stack:** 2026-10-31 · 1×B200 (`cl-worker36`), driver 580.196.07, torch 2.8.0+cu128, BF16; GPT-125M compiled with observed **Flash forward and backward** throughout.
- **Question:** at fixed model, context length and execution backend, how much throughput can a larger per-GPU micro-batch recover, and how much activation memory does it cost?
- **Protocol:** micro_bs ∈ {16, 32, 64, 128}; `grad_accum=1`, sequence length 1024, 200 post-initial optimizer steps (50 unmeasured warmup + 150 measured; separate step 0 includes compilation), `--log-every 10`, fused AdamW, reused synthetic on-device batches. `run_train.sh` selected the Flash-only persistent Inductor cache **before** Python startup; actual kernels were profiler-verified. The batch size *is the experimental variable*, so the effective optimizer batch also changes; the results address systems throughput, **not** equal-effective-batch convergence.
- **Original sweep predictions (kept unchanged after seeing results):** M1-006 = **685K tok/s, 26% MFU** (reproduction of M1-002); M1-007 = **850K, 32%**; M1-008 = **1.0M, 38%**; M1-009 = **1.1M, 42%**. Rationale: larger matrix-multiply `M = micro_bs × 1024` could improve utilization, but scaling should eventually flatten. The extrapolations were uncertain.
- **Conditional revisions, made only after earlier run results:** after M1-007, before M1-008, **750K–800K tok/s**; after M1-008, before M1-009, **775K–800K tok/s**. Revisions are not substituted for original predictions.

| Run | Micro-bs | Tokens/step | Original prediction tok/s / MFU | Actual tok/s | MFU* | Window-mean p50 | Peak allocated | Gain vs previous |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| M1-006 | 16 | 16,384 | 685,000 / 26% | **686,697** | **26.11%** | 23.9 ms | 12.3 GB | — |
| M1-007 | 32 | 32,768 | 850,000 / 32% | **723,974** | **27.52%** | 45.2 ms | 22.8 GB | +5.43% |
| M1-008 | 64 | 65,536 | 1,000,000 / 38% | **757,145** | **28.78%** | 86.6 ms | 43.7 GB | +4.58% |
| M1-009 | 128 | 131,072 | 1,100,000 / 42% | **774,236** | **29.43%** | 169.5 ms | 85.6 GB | +2.26% |

*MFU follows the unchanged full-attention PaLM/nanoGPT convention; see caveat above. Throughput is the final cumulative report across 150 measured steps, not an instantaneous endpoint sample. Step p50 is calculated from 10-step window-average times.*

### RUN M1-006 — Reproduce compiled GPT-125M / Flash baseline at batch 16

- **PREDICTION (before run):** ≈685,000 tok/s, ≈26% MFU, anchored to M1-002.
- **Actual:** **686,697 tok/s**, **26.11% MFU**, 23.9 ms p50, 12.3 GB peak allocated, 150 measured steps, Flash fwd/bwd verified.
- **Gap vs prediction:** **+0.25% tok/s** versus 685K; **+0.11 percentage points MFU** versus 26%. Relative to M1-002 (685,273 tok/s), the repeat was **+0.21%** with matching reported p50 and memory. This supports close repeatability across these two runs, not a statistical confidence interval.
- **Artifact:** `runs/M1-006.json`.

### RUN M1-007 — Increase GPT-125M micro-batch from 16 to 32

- **PREDICTION (before run):** ≈850,000 tok/s, 32% MFU; larger token-axis matrix dimensions would improve utilization.
- **Actual:** **723,974 tok/s**, **27.52% MFU**, 45.2 ms p50, 22.8 GB peak allocated, 150 measured steps, Flash fwd/bwd verified.
- **Gap vs original prediction:** throughput **−14.83%** relative to predicted 850K, MFU **−4.48 percentage points**. Compared with M1-006, **+5.43% throughput** for **+10.5 GB** peak allocated memory; the per-step token count doubled.
- **Interpretation:** batch 16 already presented substantial matmul `M = 16,384`; scaling improved throughput only modestly. Exact kernel bottleneck remains unprofiled.
- **Artifact:** `runs/M1-007.json`.

### RUN M1-008 — Increase GPT-125M micro-batch from 32 to 64

- **PREDICTION (original, before sweep):** ≈1,000,000 tok/s, 38% MFU. **Conditional revision before this run:** ≈750K–800K tok/s, after observing M1-007.
- **Actual:** **757,145 tok/s**, **28.78% MFU**, 86.6 ms p50, 43.7 GB peak allocated, 150 measured steps, Flash fwd/bwd verified.
- **Gap vs original prediction:** throughput **−24.29%**, MFU **−9.22 percentage points**. The observed result fell within the revised throughput band. Compared with M1-007: **+4.58% throughput** at **+20.9 GB** peak allocated memory.
- **Observation:** cumulative throughput gradually moved from ~768.4K early to 757.1K tok/s by the end of the measured run; GPU clocks, temperature and power were not logged, so the cause remains unknown.
- **Artifact:** `runs/M1-008.json`.

### RUN M1-009 — Increase GPT-125M micro-batch from 64 to 128

- **PREDICTION (original, before sweep):** ≈1,100,000 tok/s, 42% MFU. **Conditional revision before this run:** ≈775K–800K tok/s, after observing M1-008.
- **Actual:** **774,236 tok/s**, **29.43% MFU**, 169.5 ms p50, 85.6 GB peak allocated, 150 measured steps, Flash fwd/bwd verified.
- **Gap vs original prediction:** throughput **−29.61%**, MFU **−12.57 percentage points**. Final throughput was **764 tok/s (0.10%) below** the revised range's lower bound. Compared with M1-008: **+2.26% throughput** at **+41.9 GB** peak allocated memory.
- **Observation:** cumulative throughput declined from ~789.3K early to ~774.2K by the end; no thermal, clock or power data was gathered, so do not assign a cause.
- **Artifact:** `runs/M1-009.json`.

### Batch-sweep finding and implication for M3

- **Observed:** batch 16→128 (8× tokens per optimizer step) increased steady-state tok/s **+12.75%** (686,697→774,236) while **peak allocated memory rose 6.96×** (12.3→85.6 GB). Incremental throughput gains narrowed **+5.43% → +4.58% → +2.26%** as batch doubled. The fixed-batch M1-002/M1-004 model-size comparison remains controlled, but does **not** identify each model's maximum attainable MFU.
- **Structural explanation, stated as hypothesis:** at batch 16, a GPT-125M MLP multiply already has shapes **[16,384×768] × [768×3,072]** (`M=16,384`, `K=768`, `N=3072`), providing meaningful parallel work. Larger M increases work but need not improve kernel efficiency much once launch/matmul utilization is reasonably good. This experiment alone does not diagnose whether the remaining gap is compute, memory traffic, optimizer, or other overhead; defer causal attribution to M7.
- **M3 decision:** use micro_bs=16 as a memory-efficient starting point (rather than treating it as the proven optimal throughput setting); compare micro_bs=32 if useful for per-GPU throughput. For a *fixed global batch*, micro-batch and gradient accumulation must be adjusted jointly, so this sweep must not be presented as an optimization-quality comparison.
- **Reproducibility note:** backend-specific Inductor caches, compiler check in `setup_env.sh` and the `run_train.sh` launcher were verified on 1×B200. Compiled `gpt-tiny` Flash and cuDNN smoke tests each executed the requested fwd/bwd kernels; these 10-step diagnostics are not benchmark results. M1-006 independently reproduced M1-002 throughput within 0.21%.

### Post-hoc model of batch scaling — fitted after M1-009 (not a prior prediction)

- **Data and method:** for M1-006…009, convert each final **cumulative** throughput to the *mean* measured step time using `micro_bs × 1024 / tok_per_sec`; fit `step_ms = intercept + slope × tokens_per_step` by ordinary least squares. Do **not** fit the p50 window-mean statistic in place of mean duration.
- **Model:** `step_time ≈ 3.463 ms + 1.266 µs/token × tokens_per_step`. Observed mean step durations **23.86, 45.26, 86.56, 169.29 ms** at batch 16/32/64/128; fitted values **24.20, 44.95, 86.43, 169.39 ms**. Maximum relative residual **1.45%** across the **same four points used for fitting**.
- **Conditional extrapolation, not a physical bound:** model-implied asymptotic throughput ≈**790K tok/s**, ≈**30.0% MFU** under the unchanged convention. Batch 16 already achieves **≈87%** of that extrapolated limit. The fitted fixed term corresponds to **≈14.5%** of batch-16 step time. Eliminating it within this 125M fit accounts for ≈**3.9 MFU percentage points**, versus the observed **13.74-point** compiled 125M↔1.3B/Flash MFU difference at batch 16. The ≈**9.8-point remainder** is an unexplained *cross-model* difference, **not proof of a width-specific kernel bottleneck**; 1.3B itself may have batch-amortizable costs.
- **M7 profiling target (hypotheses only):** isolate the fitted ≈**3.46 ms/step** intercept into actual measured costs: fused AdamW step, enabled `clip_grad_norm_`, `zero_grad(set_to_none=True)`, and any host/launch gaps. Examine the ≈**1.266 µs/token** slope using GEMM utilization, attention, normalization, logits/output projection and memory traffic. These components have **not yet** been separately timed; a regression intercept should not be labeled optimizer time.
- **Calibration lesson:** the initial model-size MFU forecasts were **too pessimistic (roughly 2–3.5× below observed MFU)**; the original larger-batch sweep forecasts then overshot throughput **by 14.83%, 24.29%, and 29.61%** at batches 32/64/128. The first structural error emphasized width but ignored matmul `M = micro_bs × T`; the second assumed increasing `M` would continue reducing per-token cost. A better next prior is to estimate the **batch-amortizable share of step time** and test that hypothesis explicitly. Original prior values and timestamp history are left untouched.
- **Validation limits:** four points, two fitted parameters and no held-out batch-size experiment. Extrapolation beyond batch 128, inference about DDP, exact causal source of the intercept, or a hardware-imposed 30% ceiling is **not established**. Batch 96 is a possible *optional* held-out interpolation test, not required for M1 completion.

### M1 synthesis — supported findings, caveats, and next action

- **Model-size effect:** Eager MFU rose **17.54% → 34.30%** when switching from 125M to 1.3B with the same batch/sequence configuration, while tok/s fell **461.5K → 91.1K**. This is a fixed-configuration observation, not a universal law that larger models yield higher MFU.
- **Compiler effect:** at fixed size and Flash backend, `torch.compile` improved steady-state throughput **1.485× at 125M** versus **1.162× at 1.3B**; peak allocated memory fell **28.5%** and **8.2%**, respectively. Root causes require per-op profiling later in M7.
- **Backend effect:** cuDNN beat Flash by **7.41%** at compiled GPT-1.3B (head_dim 128, sequence length 1024). This result does not generalize to other shapes or PyTorch/cuDNN versions without testing.
- **Batch-size effect:** GPT-125M compiled / Flash batch 16→128 gained **+12.75% throughput** at **6.96×** peak allocated memory; gains diminished with each doubling. See M1-006…009; effective optimizer batch changed, so this is a systems throughput comparison only.
- **MFU disclosure:** all reported numbers retain the full-attention FLOPs convention. Causal fused attention credits masked work as if executed, so the absolute MFU is overestimated by approximately half the attention share (≈6.5–7% *relative* at 125M and ≈3.5–4% at 1.3B). No profiler-derived executed-FLOP correction is claimed.
- **Measurement limits:** one full measured run per configuration except the 125M compiled batch-16 baseline, which was repeated as M1-006; no across-run error bars; 150 measured optimizer steps per run, 10-step averaged windows, hardware vendor theoretical BF16 peak as MFU denominator; on-device **repeated synthetic** batches remove input pipeline costs and do not assess language-model quality. `torch.cuda.max_memory_allocated()` reports allocated memory, not reserved/system-wide usage. A full tokenized real-data loss curve remains a separate M2+ deliverable.
- **Reproducibility:** original predictions were preserved. Successful runs verified the *executed* attention backend, not only the requested flag. First-step timings include first touch/autotuning and potentially cached compilation; don't treat them as pure compile latency.
- **Close M1:** preserve **nine** JSON artifacts (M1-001…009), the original prediction record, updated documentation, compiler-cache launcher and native-compiler setup in Git; defer detailed bottleneck attribution and repeated confidence intervals to M7. **Next: M2 CPU data pipeline** (tokenize, dedup, quality-filter, pack, shard); release the 1-GPU pod once artifacts are verified/committed.

### M3 B200 DDP scaling — 2026-11-04

- **Hardware:** 1/2/4×B200 on `cl-worker36`, BF16, FlashAttention.
- **Protocol:** per-GPU microbatch 16, sequence length 1024, gradient accumulation 1, synthetic on-device batches, 200 optimizer steps after step 0, 50 warmup steps and 150 measured steps. Window timing uses the maximum rank duration.
- **Question:** how does GPT-125M versus GPT-1.3B weak-scale under eager versus compiled DDP?
- **Efficiency:** aggregate N-GPU tokens/s divided by N times the matched M1 single-GPU tokens/s.
- **Predictions:** recorded before each run and preserved verbatim in the linked JSON artifacts.

| Run | Model | GPUs | Mode | Predicted tok/s | Actual tok/s | Efficiency | MFU | Peak GB/GPU |
|---|---|---:|---|---:|---:|---:|---:|---:|
| M3-001 | gpt-125m | 4 | Eager | 1.50–1.70M | 1,734,601 | 93.96% | 16.49% | 17.7 |
| M3-002 | gpt-125m | 4 | Compiled | 2.40–2.60M | 2,438,818 | 88.97% | 23.18% | 12.8 |
| M3-003 | gpt-1.3b | 4 | Eager | 310–345K | 335,788 | 92.16% | 31.61% | 62.8 |
| M3-004 | gpt-1.3b | 4 | Compiled | 368–398K | 366,670 | 86.63% | 34.52% | 58.0 |
| M3-005 | gpt-125m | 2 | Compiled | 1.26–1.34M | 1,190,023 | 86.83% | 22.62% | 12.8 |
| M3-006 | gpt-1.3b | 2 | Compiled | 180–196K | 188,660 | 89.14% | 35.53% | 58.0 |
| M3-007 | gpt-125m | 2 | Compiled | 1.20–1.23M | 1,222,981 | 89.23% | 23.25% | 12.8 |

**Prediction versus observation**

- M3-001, M3-002, M3-003, M3-006 and M3-007 landed inside their predicted throughput intervals.
- M3-004 finished about 0.36% below its 368K tok/s lower prediction bound.
- M3-005 finished about 5.6% below its 1.26M tok/s lower prediction bound. Its measured windows contained an early timing slowdown; M3-007 was a preregistered repeat.
- M3-007 achieved 1.223M tok/s versus M3-005's 1.190M, approximately 2.77% higher, with substantially more stable timing windows.

**Findings and limits**

- Four-GPU weak-scaling efficiencies: **93.97%** (125M eager), **88.97%** (125M compiled), **92.16%** (1.3B eager), **86.63%** (1.3B compiled).
- Compilation improved four-GPU throughput by approximately **40.6%** at 125M and **9.2%** at 1.3B, while lowering peak allocated memory for both.
- The two-GPU 125M repeat achieved 89.23% efficiency, close to the four-GPU 88.97% result. The difference is too small to establish a scaling advantage.
- These are **weak-scaling throughput measurements**, not fixed-global-batch strong scaling. All runs use repeated synthetic data; falling loss does not establish generalization.
- The 125M loss curves differed considerably between nominally identical default-mode runs, despite matching seed/configuration/Git SHA. A separate two-rank parameter-sync probe returned zero maximum inter-rank difference after 20 steps.
- Two deterministic replays using `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` matched all **15/15 logged loss values** to four decimals, but did not establish bitwise identity or isolate the original source of variation. An opt-in `--deterministic` flag was subsequently added in commit `a8e7b97`.
- **Metadata audit (2026-11-05):** The original JSON records reported `warmup_steps_discarded=0` even though the training loop excluded the first **50** warmup steps. Commit `884955d` corrected the recorded metadata with explicit audit annotations. Throughput and other measurement values were preserved.
- No NCCL communication/computation overlap trace, clock analysis, or causal explanation of residual scaling loss is claimed; defer root-cause profiling to M7.

**Artifacts:** `runs/M3-001.json` through `runs/M3-007.json`.

### M4 B200 FSDP sharding — 2026-11-04

- **Hardware/protocol:** 4×B200, eager BF16, FlashAttention, synthetic
  on-device batches, microbatch 16/GPU, sequence length 1024,
  gradient accumulation 1, 200 training steps, 50 excluded warmup
  steps, 150 measured steps.
- **Implementation:** ZeRO-1 via DDP + `ZeroRedundancyOptimizer`;
  ZeRO-2/3 via nested PyTorch FSDP with
  `SHARD_GRAD_OP` (ZeRO-2-like) and `FULL_SHARD` (ZeRO-3-like),
  `use_orig_params=True`, tied embedding/head weights owned by the
  root FSDP unit, fused AdamW, and FSDP-aware gradient clipping.
- **Correctness:** ZeRO-1 passed FP32/AdamW parameter-update and
  learning-rate schedule parity against DDP. Both FSDP strategies
  passed three-step
  FP32/SGD parameter-update parity against DDP (maximum observed
  difference 0.000e+00), BF16/Flash/AdamW execution tests, optimizer
  parameter-group checks, and GPT-125M/1.3B smoke tests.

| Run | Model | Strategy | Throughput (tok/s) | Peak GB/GPU | MFU |
|---|---|---|---:|---:|---:|
| M3-001 | 125M | DDP eager baseline | 1,734,601 | 17.68 | 16.49% |
| M4-006 | 125M | ZeRO-1 | 1,583,664 | 17.0 | 15.05% |
| M4-001 | 125M | ZeRO-2 initial | 1,380,219 | 16.6 | 13.12% |
| M4-002 | 125M | ZeRO-3 | 1,553,383 | 16.3 | 14.76% |
| M4-003 | 125M | ZeRO-2 repeat | 1,641,148 | 16.6 | 15.60% |
| M3-003 | 1.3B | DDP eager baseline | 335,788 | 62.8 | 31.61% |
| M4-007 | 1.3B | ZeRO-1 | 330,372 | 54.9 | 31.11% |
| M4-004 | 1.3B | ZeRO-2 | 342,653 | 51.0 | 32.26% |
| M4-005 | 1.3B | ZeRO-3 | 340,166 | 46.3 | 32.03% |

**Prediction versus observation**

- M4-001: throughput and memory predictions met, but severe slow
  measured windows reduced cumulative throughput.
- M4-002: memory prediction met; throughput exceeded the upper
  predicted bound of 1.50M tok/s.
- M4-003: repeat throughput and memory predictions met; throughput
  improved by 18.9% over M4-001.
- M4-004: memory prediction met; throughput exceeded the upper
  predicted bound of 335K tok/s.
- M4-005: throughput and memory predictions met.
- M4-006: throughput prediction 1.55–1.75M tok/s and memory
  prediction 17.0–17.5 GB/GPU both met. Slow timing windows
  affected the last 20 measured steps.
- M4-007: throughput prediction 300–350K tok/s and memory
  prediction 54–60 GB/GPU both met.

**Findings**

- ZeRO-1 at 125M saved approximately 3.8% peak allocated memory
  against DDP; at 1.3B it saved approximately 12.6%.
- At 1.3B, ZeRO-1 measured 330,372 tok/s, approximately 1.6%
  below the eager DDP baseline, while saving 7.9 GB/GPU.
- ZeRO-1 at 125M had per-rank peak allocations of
  17.001, 16.913, 16.915 and 16.912 GB (0.089 GB spread).
- ZeRO-1 at 1.3B had per-rank peak allocations of 54.903, 54.900, 54.886, 54.886 GB (spread 0.017 GB).


- At 125M, peak memory fell from 17.68 GB/GPU under DDP to 16.6
  with ZeRO-2 and 16.3 with ZeRO-3: approximately 6.1% and 7.8%
  savings.
- At 1.3B, peak memory fell from 62.8 GB/GPU under DDP to 51.0
  with ZeRO-2 and 46.3 with ZeRO-3: approximately 18.8% and
  26.3% savings.
- At 1.3B, ZeRO-3 measured 340,166 tok/s versus DDP's 335,788;
  this 1.3% difference is too small to establish an intrinsic
  throughput advantage without repeated controlled measurements.
- ZeRO-2 at 125M varied from 1.380M to 1.641M tok/s between two
  nominally identical runs. The underlying cause of the slow timing
  windows was not isolated.
- Loss trajectories varied across runs and strategies despite
  matching seeds/settings. BF16/AdamW parameter-update parity and
  deterministic multi-run reproducibility remain unverified.
- Measurements report PyTorch peak *allocated* GPU memory, not
  reserved memory or total device usage. Synthetic repeated-token
  training does not establish generalization.
- Comparisons use **eager DDP** baselines, not compiled DDP.
  The original warmup metadata discrepancy was corrected in commit
  `884955d`; measured throughput values were unchanged.

**Artifacts:** `runs/M4-001.json` through `runs/M4-007.json`.

**M4 status:** Core benchmark matrix completed: DDP versus ZeRO-1,
ZeRO-2 and ZeRO-3 at 125M and 1.3B. Remaining follow-ups include
BF16/AdamW numerical reproducibility, additional performance
repetitions, and causal communication profiling. Small throughput
differences should not be interpreted as definitive strategy rankings.

---

### RUN M6-001/002/003 — bf16 vs fp8 vs fp4 on B200 (PREDICTION LOGGED EARLY, run is weeks out)

- **Date / HW:** TBD · 4×B200 · precision is the only variable; everything else held fixed
- **Question:** what does descending precision actually buy on a 125M model, and where does it stop?
- **PREDICTION (written 2026-08-01, before any hardware access):**
  - fp8 vs bf16: **1.2–1.5× throughput, not 2×.** Peak doubles, but only the matmuls move to fp8 —
    layernorm, softmax, the optimizer step, gradient all-reduce, and the fp8 scale/cast overhead
    all stay put. Amdahl caps the gain at the matmul share of step time, and a *small* model
    spends proportionally more time in that non-matmul tail than a large one would.
  - fp4 vs fp8: **smaller marginal gain again**, same reason, compounding. Possibly ~1.1–1.2×.
    Watch for loss divergence / range issues — fp4 dynamic range is tiny.
  - **MFU-vs-own-peak will DECREASE monotonically** across bf16 → fp8 → fp4, because the
    denominator doubles each step while the numerator doesn't. This is the expected shape,
    not a regression.
- **Actual:** _fill in_
- **Gap vs prediction:** _fill in_
- **Decision / next:** _fill in_

- **ADDENDUM 2026-10-15 (DEC-006, written before any B200 run):** M6 now runs at **125M and 1.3B**,
  core measurement on **1×B200** (isolates per-GPU compute from comms), one 4-GPU confirm per precision.
  **Scaling prediction:** if the gain is bounded by the non-matmul share of the step, the fp8/bf16
  speedup is **larger at 1.3B than at 125M** (1.3B has a bigger matmul share — attention term alone
  drops 13.2% → 7.1% of FLOPs/token, and layer width d²-scales the matmuls). If the speedup does
  *not* grow with size, the Amdahl explanation is wrong and something else is the bottleneck → §5.

> If the measured curve matches this prediction, the writeup line is: *"low-precision training
> gains on small models are bounded by the non-matmul fraction of the step, which is why fp8
> gave 1.Nx rather than the 2x the spec sheet implies."* If it *doesn't* match, that gap is the
> more interesting result and goes in §5 with a structural explanation.

---


### RUN M5-DDP2-001 — DDP versus TP on two B200 GPUs

- **Date / HW:** 2026-11-05; 2x NVIDIA B200-SXM.
- **Code:** `b1d1583`; PyTorch 2.8.0; BF16; FlashAttention.
- **Config:** GPT-1.3B; DDP=2; micro_bs=8; sequence=1024;
  global tokens/step=16,384; fused AdamW; grad_clip=0.1.
- **Protocol:** 120 steps; 30 warmup; 90 measured;
  10-step timing windows.
- **Question:** How does DDP=2 compare with TP=2 at matched
  global tokens per optimizer step?
- **Prediction:** DDP=2 would outperform TP=2 under the
  current optimizer and communication implementations.
- **Actual:** 148,011 tok/s; 27.87% MFU; 43.2 GB peak/GPU;
  p50=110.7 ms; p95=110.9 ms.
- **Gap vs prediction:** Direction confirmed; DDP was 1.83x
  faster than the measured TP=2 configuration.
  The original prediction specified no numeric throughput.
- **Decision / next:** Compare against TP=2 with the same
  global tokens/step; retain optimizer differences as a caveat.
- **Artifacts:** `runs/M5-DDP2-001.json`.

### RUN M5-TP2-001 — Tensor parallelism on two B200 GPUs

- **Date / HW:** 2026-11-05; 2x NVIDIA B200-SXM.
- **Code:** `b1d1583`; PyTorch 2.8.0; BF16; FlashAttention.
- **Config:** GPT-1.3B; TP=2; micro_bs=16 per TP group;
  sequence=1024; global tokens/step=16,384;
  single-tensor AdamW; grad_clip=0.1.
- **Protocol:** 120 steps; 30 warmup; 90 measured;
  10-step timing windows.
- **Question:** How does TP=2 compare with DDP=2 at matched
  global tokens per optimizer step?
- **Prediction:** Approximately 82,000 tok/s, below DDP=2.
- **Actual:** 80,806 tok/s; 15.22% MFU; 39.4 GB peak/GPU;
  p50=202.3 ms; p95=204.3 ms.
- **Gap vs prediction:** 1,194 tok/s below prediction
  (approximately -1.46%).
- **Decision / next:** TP saved 3.8 GB peak memory per GPU,
  but throughput was substantially lower. Do not attribute
  the full difference to communication alone.
- **Artifacts:** `runs/M5-TP2-001.json`.

### RUN M5-DDP4-001 — Four-GPU data-parallel baseline

- **Date / HW:** 2026-11-05; 4x NVIDIA B200-SXM.
- **Code:** `b1d1583`; PyTorch 2.8.0; BF16; FlashAttention.
- **Config:** GPT-1.3B; DDP=4; micro_bs=8 per rank;
  sequence=1024; global tokens/step=32,768;
  fused AdamW; grad_clip=0.1.
- **Protocol:** 120 steps; 30 warmup; 90 measured;
  10-step timing windows.
- **Question:** How does DDP=4 compare with hybrid TP=2 x
  DP=2 at matched global tokens per step?
- **Prediction:** Approximately 290,000 tok/s, with DDP
  outperforming hybrid.
- **Actual:** 288,584 tok/s; 27.17% MFU; 43.2 GB peak/GPU;
  p50=113.5 ms; p95=113.8 ms.
- **Gap vs prediction:** 1,416 tok/s below prediction
  (approximately -0.49%).
- **Decision / next:** DDP scaled 1.95x relative to the
  DDP=2 run at fixed per-rank microbatch size.
- **Artifacts:** `runs/M5-DDP4-001.json`.

### RUN M5-HYBRID-001 — Hybrid TP=2 x DP=2

- **Date / HW:** 2026-11-05; 4x NVIDIA B200-SXM.
- **Code:** `b1d1583`; PyTorch 2.8.0; BF16; FlashAttention.
- **Config:** GPT-1.3B; TP=2 x DP=2;
  micro_bs=16 per TP group; sequence=1024;
  global tokens/step=32,768;
  single-tensor AdamW; grad_clip=0.1.
- **Protocol:** 120 steps; 30 warmup; 90 measured;
  10-step timing windows.
- **Question:** How does hybrid TP x DP compare with DDP=4
  at matched global tokens per step?
- **Prediction:** Approximately 145,000 tok/s, below DDP=4.
- **Actual:** 148,641 tok/s; 13.99% MFU;
  39.4 GB peak/GPU; p50=213.2 ms; p95=260.1 ms.
- **Gap vs prediction:** 3,641 tok/s above prediction
  (approximately +2.51%).
- **Observation:** Cumulative throughput reached
  153,244 tok/s at step 100 and declined to
  148,641 tok/s by step 120.
- **Decision / next:** DDP=4 was 1.94x faster while
  hybrid used 3.8 GB less peak memory per GPU.
  Investigate throughput variability only in a separate,
  controlled follow-up experiment.
- **Artifacts:** `runs/M5-HYBRID-001.json`.

## 3. Issue Log

Anything that broke and cost >20 minutes. Fill the root cause in even after you've moved on —
that's the field that makes it a story.

### Entry template

```
### ISSUE <id> — <symptom in your own words>
- **Date / cost:** 2026-08-__ · __ min
- **Symptom (exact):** error string or observed behaviour, verbatim
- **Hypotheses tried, in order:** 1) __ (ruled out by __) 2) __ (ruled out by __) 3) __
- **Root cause:** __
- **Fix:** __
- **Generalizable lesson:** __ ← the interview-usable part; skip it and the entry is worthless
```

### ISSUE-001 — Contention check cried wolf on my own ranks

- **Date / cost:** 2026-08-02 · ~10 min (caught on first run, before it cost anything)
- **Symptom:** every run printed "⚠ other compute processes visible on these GPUs" listing
  3 processes at 12 MiB each — on an otherwise idle node.
- **Hypotheses tried:** 1) genuine co-tenants (ruled out: PIDs incremented by ~300 between
  consecutive runs, tracking my own launches; 12 MiB is exactly a bare CUDA context;
  3 processes, not a stable number) 2) stale contexts (ruled out: same pattern every run).
- **Root cause:** inside a container, `nvidia-smi` reports **host** PIDs while `os.getpid()`
  returns the **container-namespace** PID. The self-filter `not ln.startswith(str(os.getpid()))`
  could never match, so my own ranks registered as strangers. Only 3 of 4 appeared because
  rank 0 queried before the 4th rank had finished creating its context — a race.
- **Fix:** run the check *before* `init_process_group()` and before any CUDA context exists,
  using `LOCAL_RANK` from the environment to pick the reporting rank. Anything visible at that
  point genuinely belongs to someone else, so no PID filtering is needed at all.
- **Generalizable lesson:** PID namespaces don't cross the container boundary, and any tool that
  reports host-level state (`nvidia-smi`, `/proc`, `ps` on a mounted host proc) will disagree
  with in-container identity. When correlating container processes with host-reported resource
  usage, match on something namespace-invariant — or restructure so the correlation isn't needed.

### ISSUE-002 — No libibverbs in the container (deferred, not blocking)

- **Date / cost:** 2026-08-02 · 0 min (observed, not debugged)
- **Symptom:** `NCCL INFO Failed to open libibverbs.so[.1]` → `NET/Socket : Using [0]eth0` →
  `Using network Socket`, despite the node exposing 8× mlx5 NICs plus `mlx5_bond_0`.
- **Impact today:** none. Single node (`nNodes 1`), all collectives go P2P over NVLink; the
  network path is never exercised.
- **Impact if multi-node ever unfences:** severe. NCCL would carry inter-node traffic over TCP
  sockets instead of RDMA. Any multi-node scaling number collected in this state would be
  measuring the wrong machine.
- **Fix (deferred):** container needs `libibverbs` + RDMA device access; the pod spec needs the
  IB resources exposed. Verify with `NCCL_DEBUG=INFO` showing `NET/IB` before trusting any
  multi-node result.
- **Generalizable lesson:** NCCL fails *silently and gracefully* to a slower transport. It does
  not error. Always confirm the chosen transport in the debug output rather than assuming the
  hardware in `nvidia-smi topo -m` is the hardware being used.

### ISSUE-003 — Two broken tests in the first model.py suite, caught by external review before first run

- **Date / cost:** 2026-10-16 · 0 min of compute (caught in review, before any run)
- **Symptom (would have been):** (a) `test_can_overfit_a_single_batch` raising
  `view size is not compatible with input tensor's size and stride`; (b) `test_initial_loss_is_near_uniform`
  failing at loss ≈ 10.32 vs the 10.83 uniform baseline.
- **Root cause (a):** `targets.view(-1)` on `idx[:, 1:]` — shape (4, 31), strides (32, 1). Flattening
  via `view` needs `stride[0] == size[1]·stride[1]`; 32 ≠ 31. Fix: `reshape`.
- **Root cause (b):** the test scored the *inputs* as targets. With tied embeddings the true-token logit
  is ≈ |wte[t]|² / std(residual) ≈ 0.02·C/√2 — **a copy bias that grows linearly with width**
  (~0.9 at C=64, ~11 at C=768). The test was measuring that, not init quality. Fix: next-token targets;
  added a separate test that asserts the copy bias exists, documenting it instead of just dodging it.
- **Also changed:** GPU backend test now measures a bf16 *precision floor* (bf16 math vs fp32 math) and
  asserts each fused backend's error ≤ 2× floor, instead of a guessed absolute tolerance.
- **Generalizable lessons:** tests sliced from training-style data need the same non-contiguous inputs
  real training produces; and a test threshold derived by reasoning is a hypothesis — I'd flagged these
  two tests as "most likely to need calibration," which was the right suspicion for the wrong reason.
  Interview-usable: *tied embeddings give an untrained model a structural prior toward copying its input.*

### ISSUE-004 — Requested cuDNN, but compiled GPT-1.3B executed Flash from cache

- **Date / HW:** 2026-10-31 · 1×B200 (`cl-worker36`), PyTorch 2.8.0+cu128.
- **Symptom (exact):** M1-005 started with `--compile --backend cudnn`, but the profiler recorded `aten::_scaled_dot_product_flash_attention` and Flash backward; `verify_backend` raised `RuntimeError: requested backend 'cudnn' not confirmed by the profiler` *before* any measured window. That failed attempt is not a valid cuDNN benchmark.
- **Observations:** Previous run M1-004 compiled the same GPT-1.3B graph with Flash; failing M1-005 first step took ~3.1s versus 23.2s on earlier compiled Flash run. A diagnostic after isolating Inductor cache took ~21.7s on first step and showed cuDNN operations. Full isolated-cache run completed and verified cuDNN, yielding 113,663 tok/s.
- **Leading hypothesis (not fully proven):** Inductor graph/cache reuse failed to reflect the requested SDPA backend when the model shape remained the same. A short first step and the resolution with cache isolation are consistent with stale compiled graph reuse; they are not by themselves proof of the exact upstream cache-key defect.
- **Fix / mitigation:** use an isolated `TORCHINDUCTOR_CACHE_DIR` for each compiled backend (e.g. `/workspace/.cache/p4-inductor/cudnn` or `/flash`); `run_train.sh` now sets that environment variable **before Python starts**, and `setup_env.sh` configures the persistent cache root and checks native compilers on GPU pods. The Flash→cuDNN separation was smoke-validated on compiled `gpt-tiny`, with profiler-confirmed forward/backward in both cases; the exact failing 1.3B warm-cache sequence was not rerun after the launcher change. Keep `verify_backend` as a hard failure for measured runs; never rename a Flash run as cuDNN.
- **Generalizable lesson:** a configuration flag expresses intent; the profiler reveals what executed. In systems experiments, guard the validity of the independent variable **before** collecting headline metrics.
- **Follow-up (optional):** isolate the precise cache layer with a minimal same-directory backend-switch repro and compare against a newer PyTorch version, if useful; the local cache isolation and profiler guard are sufficient for M1. Not needed to close M1.


### ISSUE-005 — train.py / bench.py review corrected attention-FLOPs interpretation (reconstructed)

- **Provenance:** reconstructed on 2026-10-31 from the uploaded `bench.py` docstring and M1 review notes. The original review-round log entry was lost when ISSUE-004 was reused for the compiled-backend cache failure. **This is not a verbatim recovery of the missing entry**; precise review date, time cost and any other findings are not available in the uploaded sources.
- **Symptom:** two different questions about the `12 × L × H × Q × T` attention term were conflated: (a) should a FLOPs estimate include attention score/value matmuls beyond the parameter-count `6N` term? **Yes.** (b) does the conventional full-matrix estimate accurately count executed FLOPs for causal fused attention? **Not exactly.**
- **Review finding:** omitting the attention term entirely **understates** the conventional estimate, whereas counting the entire `T×T` matrix for causal attention can **overstate actual masked-kernel work**. The earlier comment in `bench.py` had reversed the first point (called it an inflation); the corrected source now distinguishes the two.
- **Correction:** retain the PaLM/nanoGPT formula for comparability, clearly document its causal-attention upper-triangle overcount and avoid portraying convention-based MFU as profiler-derived executed FLOPs. Approximate relative bias: ~6.5–7% for GPT-125M and ~3.5–4% for GPT-1.3B under a half-attention-term heuristic; kernel tile details make these *estimates*, not exact corrections.
- **Generalizable lesson:** a model FLOPs formula is a reporting convention, not an instruction-level hardware counter. A term can be essential to the conventional estimator while still overcrediting work skipped by a specialized causal implementation. Review mathematical assumptions and source-code comments separately.
- **Resolution:** documentation now discloses the limitation; the stale `ISSUE-004` code reference is updated to `ISSUE-005`, preserving ISSUE-004 for the separately verified backend-cache incident.


### ISSUE-006 — TorchInductor failed to find a C compiler on the new B200 pod

- **Date / HW:** 2026-11-04 · 4×B200 · PyTorch 2.8.0+cu128.
- **Symptom:** compiled DDP failed on all four ranks with
  `torch._inductor.exc.InductorError: RuntimeError: Failed to find C compiler`.
- **Diagnosis:** eager DDP had succeeded, but `torch.compile` requires a native compiler for generated code. The persistent Python environment survived pod recreation; system-installed compiler packages did not.
- **Fix:** install `build-essential` in the container. Existing `setup_env.sh` already checks for `gcc`, `g++` and `make`; run it on each new GPU pod instead of only sourcing the previously generated environment file.
- **Validation:** compiled GPT-125M DDP smoke test passed, followed by successful M3-002 and M3-004 measured runs.
- **Lesson:** a persistent Python environment does not imply persistent system dependencies. Verify native build tools whenever GPU containers are recreated.

### ISSUE-007 — Compiled DDP loss trajectories differed between identical-seed runs

- **Date / HW:** 2026-11-04 · 2×B200 · GPT-125M compiled/Flash/BF16.
- **Symptom:** M3-005 and M3-007 used identical recorded configurations, seed 1337 and Git SHA `73e0292`, yet final losses were **7.4523** and **9.3490**.
- **Synchronization check:** a separate 20-step four-rank compiled DDP probe measured maximum inter-rank parameter difference **0.000e+00**.
- **Deterministic experiment:** two 200-step two-rank replays enabled `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. All **15/15 logged loss values matched to four decimals**, including final loss **10.5975** in both runs.
- **Interpretation:** deterministic configuration yielded reproducible logged trajectories in the tested setup. This supports numerical nondeterminism as a possible explanation for the earlier variation, but the exact operation or cause has **not** been isolated. No bitwise parameter-equivalence test was performed.
- **Performance caveat:** replay throughputs differed (**1,004,079** vs **1,099,709 tok/s**), despite matching logged losses. Reproducibility of the training trajectory does not establish performance stability.
- **Status:** mitigation validated; root-cause identification remains open.
- **Lesson:** fixed random seeds, synchronized DDP parameters and deterministic numerical execution are distinct guarantees. Test them independently.


---

## 4. Decision Log

Non-obvious choices only. If the alternative was never plausible, don't log it.

### Entry template

```
### DEC <id> — <decision>
- **Date:** 2026-08-__
- **Chose:** __
- **Over:** __
- **Because:** __
- **What would change my mind:** __
```

### DEC-009 — Keep deterministic execution optional

- **Date:** 2026-11-04.
- **Chose:** add an opt-in `--deterministic` flag to `train.py`, recording the mode in experiment configuration. Enable PyTorch deterministic algorithms and set a default `CUBLAS_WORKSPACE_CONFIG=:4096:8` when requested.
- **Over:** enabling deterministic execution for every training benchmark, or leaving the reproducibility controls only in temporary diagnostic scripts.
- **Because:** performance benchmarking and numerical reproducibility answer different questions. M3's original throughput measurements should retain their execution settings, while future correctness runs need an explicit, reproducible mode.
- **Validation:** two deterministic replays matched all 15 logged losses; the committed flag passed a two-GPU compiled-DDP smoke test.
- **Limit:** deterministic mode is not a guarantee of bitwise cross-system reproducibility and may alter performance. Its cuBLAS setting uses `setdefault`, so any existing environment override must be checked.
- **Commit:** `a8e7b97`.
- **What would change my mind:** evidence that deterministic execution has negligible cost for the representative workload and offers preferable reliability as the default.

### DEC-008 — How train.py measures: windowed sync, separate compile time, observed kernels

- **Date:** 2026-10-16
- **Chose:** (1) sync once per `--log-every` window, never per step; (2) `--bench-warmup` steps trained
  but never counted; (3) step 0 timed alone and recorded as `compile_s`; (4) one profiled fwd+bwd after
  step 0 records which attention ops/kernels *actually* ran; (5) the RunRecord is built *before*
  training so missing-prediction / dirty-tree guards fire in second one, not after the run.
- **Over:** bench's per-step `step_end()` sync (still available, now unused by train.py)
- **Because:** a per-step sync stops the CPU from queuing step N+1 while step N runs. On short steps
  (125M on B200) that's a measurement artifact of plausibly a few percent — the meter measuring itself.
  `sdpa_kernel()` restricts which backends are *allowed*; it doesn't prove which one *executed*
  (compile, shape limits, unsupported cases can still change it) — so the record stores the aten op
  name, e.g. `aten::_scaled_dot_product_cudnn_attention`, as ground truth.
- **Cheap M7 side-experiment this enables:** per-step vs windowed sync at 125M quantifies the artifact.
- **What would change my mind:** if windows hide step-time variance worth seeing (stragglers at M3) —
  then add a per-step mode for diagnostics only, never for headline numbers.

### DEC-005 — Everything on 4×B200; A100 dropped as the dev rig (supersedes DEC-002)

- **Date:** 2026-10-15
- **Chose:** all milestones M1–M7 on B200 (all cluster B200s sit on one worker node)
- **Over:** DEC-002's split — A100 for mechanics, B200 for M6 only
- **Because:** availability, not preference. A 4×A100 request sat in queue for days; B200 is
  reachable. DEC-002's real objection was *mixing* hardware (MFU has peak FLOPS in the denominator,
  so it isn't comparable across generations) — running everything on one generation removes that
  objection entirely. Remaining problem — 125M can't fill a B200 — is solved by DEC-006, not by
  staying on A100. With applications opening January, queue delay costs more than anything A100 bought.
- **Kept from the A100 work:** M0-001 stays as the Ampere reference point; ISSUE-001/002 still apply;
  DEC-004's topology reasoning gets redone for the B200 node at M0-B200.
- **Costs accepted:** new env (torch 2.8 / CUDA 12.8, `sm_100`), M0 redone on B200, and two things to
  *check rather than assume* on the new node: NVLS channels in NCCL (if present, ring-convention
  busbw stops describing physical traffic) and which SDPA backend actually runs (silent fallback =
  the ISSUE-002 pattern).
- **What would change my mind:** if B200 access becomes as contended as A100 was. Code stays
  hardware-agnostic (`peak_for()` auto-detects), so falling back costs a re-baseline, not a rewrite.

### DEC-006 — Three model sizes with distinct roles; 1.3B is the systems model

- **Date:** 2026-10-15 · prompted by an external review (Fable) — adopted with modifications
- **Chose:** 125M = correctness + the one full loss curve · 1.3B = systems benchmarking (throughput,
  ZeRO stages, TP, precision) · 350M = optional middle point for the crossover trend
- **Over:** (a) original plan, 125M/350M only; (b) the review's suggested 1B–3B
- **Because:** at 125M a 4×B200 run measures *"how fast can four B200s execute a workload too small
  for them"* rather than anything about scaling. From `model_configs.py`:

  | model | params | head_dim | GFLOP/tok | attn share | DDP static (fp32 Adam) | Chinchilla tokens |
  |---|---|---|---|---|---|---|
  | gpt-125m | 124M | 64 | 0.86 | 13.2% | 2.0 GB | 2.5B |
  | gpt-350m | 355M | 64 | 2.42 | 12.5% | 5.7 GB | 7.1B |
  | gpt-1.3b | 1314M | 128 | 8.47 | 7.1% | 21.0 GB | 26.3B |

  **Every size fits under plain DDP on a 192 GB B200** (even 3B would, at ~48 GB static). So
  DEC-001's controlled condition — sharding is optional, therefore its cost is measurable — holds at
  *all* sizes, and M4 upgrades from one negative result to a **trend: ZeRO-3 overhead vs model size**.
  1.3B also rescues M5 (TP at hidden 768 is pure overhead; at 2048 it means something) and turns M6
  into a falsifiable scaling claim.
- **Where I disagreed with the review:** 3B adds little over 1.3B for the trend at ~2.3× the cost,
  and its "meaningful loss trajectory" doesn't hold up — Chinchilla for 1.3B alone is 26B tokens,
  ~17–23 h of 4×B200 at 30–40% MFU. Not on a shared cluster, and not needed: throughput reaches
  steady state in a few hundred steps, and token *values* don't change compute, so benchmark runs
  can loop a small tokenized set. **The one clean loss curve stays at 125M** (2.5B tokens, cheap),
  which also keeps M2 small.
- **What would change my mind:** if ZeRO-3 overhead at 1.3B is already within noise of DDP, the
  trend has no second point worth plotting — then add 350M rather than going bigger.

### DEC-007 — Environment lives entirely on the PVC, built from a committed lockfile

- **Date:** 2026-10-15
- **Chose:** `setup_env.sh` — uv + a uv-managed Python 3.11 + venv + package cache all under
  `/workspace`; installs only from `requirements.lock` (committed); verifies on every run
- **Over:** installing into the image per pod (the previous-projects pattern that repeatedly cost hours)
- **Because:** P4 runs for weeks and pods can't be held, so pod deletion must cost seconds, not an
  install. Interpreter on the PVC (not the image's) means the venv survives an image change too.
  Lockfile means the 4×B200 env is byte-identical to the 1-GPU env it was debugged in. `uv pip sync`
  (not `install`) removes anything not in the lock, so drift can't accumulate. A hash guard refuses
  to run if `requirements.in` was edited without `--relock`.
- **Verified 2026-10-15 (sandbox):** PyPI torch 2.8.0 resolves against the CUDA 12.8 runtime
  (`nvidia-cuda-runtime-cu12` 12.8.90, `nvidia-nccl-cu12` 2.27.3) → `sm_100` covered with no
  special index. Cold build, fast path, stale-lock guard and relock all tested.
- **Deferred:** Transformer Engine (needed for fp8/fp4 at M6). Highest-risk dependency — may need
  the CUDA toolkit to build its torch extension. Handle when M6 starts; `torchao` float8 is the
  pure-pip fallback for the fp8 point.
- **Only thing the env can't control:** the node's NVIDIA driver (must be ≥ 570 for CUDA 12.8 /
  Blackwell). `setup_env.sh` prints it on every run.

### DEC-004 — TP group placement on the A100 node: (0,1) and (2,3), for host reasons only

- **Date:** 2026-08-01
- **Observed topology (`p4-training-a100`, 4×A100-SXM4-40GB, driver 570.211.01, torch 2.4.1+cu124):**
  `NV12` between **every** GPU pair. An A100 SXM4 has 12 NVLink3 ports total and cannot hold
  12 dedicated links to each of three peers — so this is an **NVSwitch** fabric giving uniform
  any-to-any bandwidth, not point-to-point NVLink.
  NUMA: GPU0/1 → node 1 (CPUs 16-31,144-159); GPU2/3 → node 7 (CPUs 112-127,240-255).
  *(Correction: the P4 plan doc recorded NUMA nodes 3/5 — that was cl-worker37, a different node.)*
- **Chose:** TP=2 groups (0,1) and (2,3)
- **Over:** any other pairing, and over the plan doc's stated rationale
- **Because:** the plan doc said TP groups "must sit on an NVLinked pair." On this fabric that
  constraint **does not exist** — GPU0↔GPU3 is as fast as GPU0↔GPU1. The remaining reason to
  prefer (0,1)/(2,3) is *host-side*: each pair shares a NUMA node and its own NICs (PXB), so
  dataloader pinned buffers and H2D copies stay local instead of crossing UPI. Real, but a
  much weaker effect than the doc implied, and unlikely to be measurable at 125M.
- **What would change my mind:** if M7 profiling shows the run is input-bound, test `numactl`
  binding — that's when host-side locality starts to matter.
- **Interview value:** "I checked the topology and found the standard NUMA-aware TP placement
  advice didn't apply on an NVSwitch node" beats following the advice without looking.

### DEC-002 — B200 for M6; A100 stays the dev rig for M1–M5 · ⛔ SUPERSEDED by DEC-005 (2026-10-15)

- **Date:** 2026-08-01
- **Chose:** all mechanics (M1–M5) on 4×A100-40GB; M6 precision ablation on 4×B200
- **Over:** (a) doing everything on B200 since it's available, (b) H200 for M6 as originally planned
- **Because:**
  - *Against B200 for mechanics:* 125M at block 1024 cannot saturate 2.25 PFLOPS of dense bf16.
    The run would be launch-overhead/dataloader bound, MFU would read single-digit, and mixing
    hardware across M1–M5 would silently corrupt the scaling tables — MFU has peak hardware
    FLOPS in the denominator and is therefore *not comparable across GPU generations*.
  - *For B200 over H200 at M6:* Blackwell's 2nd-gen Transformer Engine has native FP4 where
    Hopper has FP8 only. That turns the planned 2-point ablation (bf16 vs fp8) into a 3-point
    one (bf16 → fp8 → fp4) on hardware very few candidates have touched. Strictly better version
    of the same asset.
- **What would change my mind:** if the Blackwell software stack fights back (see §3). Fallback
  is the original H200 fp8-vs-bf16 two-point ablation, which is still a good result.
- **Retrospective correction (2026-10-31, after M1):** the *single-digit MFU* claim was **falsified** on 1×B200: GPT-125M eager reached **17.54%**, compiled/Flash **26.05%**, and compiled batch-128 **29.43%** under the stated PaLM/nanoGPT convention. The original claim remains above as a historical prior; the newer evidence supersedes it. At batch 16 with 1024-token context, the token-axis matmul dimension is already **M=16,384**, e.g. MLP GEMM `[16,384×768] @ [768×3072]`. Reasoning solely from small model width ignored this substantial M dimension; the input batches were already on GPU, so the prediction of dataloader-bound measured steps was also inapplicable to M1. The observation does **not** establish the fraction of SM cycles doing useful work or a root-cause profile.
- **Confirmed 2026-08-01:** the A100 pod runs torch 2.4.1+cu124 — correct for sm_80, but it has
  no `sm_100`, so **this image cannot be reused for the B200 runs.** M6 needs a separate
  CUDA 12.8+ image. Build and test it *before* holding a scarce B200 slot.
- **Prerequisite, unverified:** P4 venv must be built on CUDA 12.8+ / torch with `sm_100` in
  `torch.cuda.get_arch_list()`. The thesis env (torch 2.3.1+cu121) will *not* run on Blackwell —
  it fails with "no kernel image is available for execution on the device", not a graceful
  fallback. Check before scheduling a B200 pod:
  `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"`

### DEC-003 — Report both tok/s and MFU-vs-own-peak in the M6 table

- **Date:** 2026-08-01
- **Chose:** report throughput *and* MFU measured against each format's own dense peak
- **Over:** MFU alone, or a single fixed denominator across all three precisions
- **Because:** they answer different questions. tok/s answers "did it get faster." MFU-vs-own-peak
  answers "how well am I using the machine in this mode." Reporting only the first hides the
  efficiency story; reporting only the second looks like a regression, because
  **MFU will likely *fall* as precision descends** — fp8 doubles the denominator but won't
  double the numerator (see PREDICTION in the M6 run stub).
- **What would change my mind:** nothing; this is a reporting-hygiene decision, not a result.

### DEC-001 — Model sized to *fit* on one GPU, deliberately

- **Date:** 2026-08-01
- **Chose:** 125M (and later 350M), which fit comfortably on 4×A100-40GB without sharding
- **Over:** a model large enough to *require* FSDP
- **Because:** if the model requires sharding, FSDP is a necessity and there's nothing to measure.
  Because it fits, ZeRO-1/2/3 become a controlled comparison against DDP on identical work —
  which is the thing that produces the memory-vs-comms tradeoff table. The constraint is the experiment.
- **What would change my mind:** if the comparison turns out to be uninteresting at this scale
  (deltas within noise), bump to 350M where activation memory starts to bite.
- **Update 2026-10-15:** extended, not replaced, by DEC-006 — on 192 GB B200s the "fits under DDP"
  condition holds up to 1.3B (and beyond), so the controlled comparison survives at every size.

---

## 5. Negative Results Register

Structural explanation required. See §0.

### Entry template

```
### NEG <id> — <what didn't work>
- **Setup:** __
- **Expected:** __
- **Observed:** __
- **Structural reason it was always going to lose:** __
- **When would it win instead:** __ ← this sentence is what separates you from someone who just reports a number
- **Confidence:** high / medium / low — and what would raise it
```

_(no entries yet — but expect M4 to produce one)_

---

## 6. Interview pitch — fill in as numbers land

_Future pitch template only; do not present these milestones as completed until measured._

> "Beyond my applied LLM work I pretrained GPT models from scratch on a 4×NVLinked node — built
> the data pipeline myself, then implemented DDP, FSDP across ZeRO stages, and 2-way tensor
> parallelism, measuring the memory-vs-communication tradeoff directly. I took training MFU from
> __% to __% by fixing __, and I have an fp8-vs-bf16 throughput result on Blackwell."

| Question | Answer (fill as measured) |
|---|---|
| Scaling efficiency at 4 GPUs? | DDP hit __× of ideal 4×; gap is all-reduce grad sync, held down by NVLink at __ GB/s busbw |
| What did ZeRO-3 cost you? | −__% throughput for −__% peak memory; on a model that fits, DDP wins — FSDP is a big-model tool |
| MFU story? | M1 single B200 GPT-125M eager 17.54% → compiled 26.05% (+48.5% tok/s); batch 16→128 reached 29.43% but cost 6.96× peak allocated memory for +12.75% tok/s; MFU is convention-based and causal attention overcount is disclosed; root-cause profiling deferred to M7 |
| Fault tolerance? | Checkpoint/resume survives a mid-run `kill -9`; tested by __ at step __, resumed to identical loss trajectory |
| fp8? | __× throughput over bf16 at __ loss delta; watched range/overflow in __ |
| Hardest bug? | ISSUE-004: requested cuDNN but compiler cache reused Flash; profiler guard caught backend mismatch |
| Data pipeline proof? | M2: 500 real FineWeb-Edu docs; no qualifying Jaccard≥0.8 pairs among 124,750 audited pairs; 429,605 train + 47,014 validation GPT-2 tokens; sharded files reconstruct byte-for-byte; 500-doc pilot, not a model-quality study |
| Something that didn't work? | NEG-__ |


---

## 7. M2 — CPU Data Pipeline: Implementation, Audits, and Reproducibility

**Dates:** 2026-11-02 (pilot checks and data artifacts). **Hardware:** CPU pod `ml-cpu`, namespace `thme5878`, Python 3.11.17 in `/workspace/envs/p4`, using the PVC under `/workspace`. No GPU was needed for the pipeline tests or the real-data pilot.

**Status as of this log revision:** M2-001…006 **implemented, tested, and committed**; M2-007 **real-data pilot produced and audited**, with its adapter tests passing independently. The final M2-007 full-suite rerun, M2-007 commit, and `gpt-tiny` real-data optimizer-step smoke test **have not been shown as completed** in the recorded outputs. A larger corpus and a meaningful GPT-125M loss curve are **not yet complete**. The last confirmed M2-006 commit is `b887a08`.

### 7.1 Objective and end-to-end contract

Build a CPU-first, reproducible pretraining data path that consumes text and emits data compatible with the **existing** `train.py` `MemmapBatches` reader:

```text
Hugging Face source (M2-007) → raw.jsonl + provenance + source manifest
  → ingest.py                  validate, NFC-normalize, SHA-256
  → filter_stream.py           rule-based rejection statistics
  → exact_dedup.py            disk-backed SQLite SHA-256 uniqueness
  → near_dedup.py             3-word shingles → MinHash → LSH candidates
                                → exact Jaccard verification → union-find groups
  → split.py                  seeded, group-aware train/validation manifest
  → tokenize.py               GPT-2 IDs + one EOT per document → flat uint16 LE .bin
  → shard.py                  ordered token shards + offsets + SHA-256 manifest
```

- **Data outside Git:** pilot and intermediate files remain in `/workspace/p4/m2-data/`, not in the public source repository. Code, tests, and human-readable findings belong in Git.
- **Failure safety:** ingestion, quality filtering, exact dedup, and near dedup use temporary-output publication; split/tokenize/shard stages publish a complete new output directory, and refuse existing destinations. These mechanisms protect against ordinary processing errors; no power-loss/crash durability guarantee is claimed.
- **Memory:** ingest tracks seen IDs in RAM; near dedup retains document signatures/shingle sets/groups in RAM; `split.py` retains group records. Exact dedup's fingerprint index is SQLite-backed. The 500-document pilot does **not** establish scalability to billions of documents.
- **Reproducibility boundary:** the same *input bytes, order (where applicable), seed, code, tokenizer, and configuration* should produce the same output. Near-dedup representatives and `group-000...` labels depend on input order; group-level split assignment uses stable member identities, but may change if group membership changes.

### 7.2 M2-001 — Streaming ingestion

- **Implemented:** `data_pipeline/ingest.py`, fixture `data_pipeline/fixtures/sample.jsonl`, tests `tests/test_ingest.py`.
- **Contract:** require nonempty string `id`/`source`, string `text`, reject duplicate `(source,id)` identities, normalize CRLF/CR to LF, NFC-normalize Unicode, strip outside whitespace, skip blank normalized text, compute SHA-256 from normalized UTF-8 text, and write validated JSONL in a deterministic order.
- **Safety:** same-directory temp file, flush/fsync, `os.replace`; a failed input does not replace a preexisting output.
- **Synthetic fixture:** 3 accepted documents, including an exact duplicate of the first text under a separate identity; ingestion deliberately does not deduplicate content.
- **Tests:** six ingestion tests passed when first implemented. **Scaling caveat:** seen document identities are an in-memory set.

### 7.3 M2-002 — Quality rules and streaming filter

- **Implemented:** `data_pipeline/quality_filter.py`, `data_pipeline/filter_stream.py`, `tests/test_quality_filter.py`, `tests/test_filter_stream.py`.
- **Default rules, applied in order:** minimum 30 words; maximum 100,000 words; Unicode alphabetic fraction of non-whitespace characters at least 0.50; duplicated nonblank-line fraction no more than 0.30. A rejected document is attributed to its **first** failing rule.
- **Output:** accepted original JSONL rows, counts for seen/accepted/rejected/blank lines, and four rejection reasons; atomic replacement on successful completion.
- **Initial tests:** 18 tests passed across ingestion and quality components. For the three-document smoke fixture, a deliberately lowered `--min-words 3` accepted all three.
- **Real-data pilot (M2-007):** 500 seen, 500 accepted, **0 rejected** across all four rules. This is consistent with an already curated FineWeb-Edu slice; it **does not** validate usefulness or fairness of these thresholds on raw Common Crawl.

### 7.4 M2-003 — Exact and near-duplicate detection

**Exact deduplication (`data_pipeline/exact_dedup.py`):**

- Recomputes and validates `sha256` from each record's text. `INSERT OR IGNORE` into a temporary SQLite primary-key index retains the **first** occurrence of identical normalized text regardless of `source`/`id`; subsequent records are removed.
- Synthetic fixture: **3 seen → 2 unique, 1 removed**. Existing output remains intact if input validation fails. Test suite at this point: **54 passed, 3 skipped**.
- Real FineWeb-Edu slice: **500 seen → 500 unique; 0 exact duplicates removed**.

**Near deduplication (`minhash_lsh.py`, `near_dedup.py`):**

| Parameter | Pilot setting |
|---|---:|
| Shingling | Case-folded word 3-grams; punctuation ignored for matching |
| Signature | 112 deterministic MinHash values; seed 42 |
| LSH layout | 14 bands × 8 rows |
| Verification threshold | Exact shingle-set Jaccard ≥ 0.8 |
| Group construction | Union-find connected components; all documents indexed |
| Representative | First input occurrence per detected component |

- LSH returns *candidate pairs*, not confirmed duplicates. Only exact Jaccard verification creates a similarity edge. Transitive union-find groups can connect A and C through B when A and C are not directly above threshold.
- **Controlled candidate-recall experiment:** 23 synthetic documents (3 groups of 6 shifted, overlapping documents + 5 unrelated). There are **253** possible pairs, of which **36** have true Jaccard ≥ 0.8. LSH returned **33** candidates; **31** were true matches and **2** were nonmatches. Thus pairwise candidate recall **31/36 = 86.11%**, candidate precision **31/33 = 93.94%**, and **5 true pairs were missed**. Relative to all-pairs exact Jaccard, **(253−33)/253 = 86.96% fewer candidate-verification comparisons**; this is *not* a wall-time speedup.
- **Connected-component recovery on the same synthetic fixture:** 8 detected components with sizes **[6, 6, 6, 1, 1, 1, 1, 1]**. All three intended six-document components were recovered, despite the five missed pairwise edges. This is fixture-specific success, **not** a guarantee of complete leakage prevention.
- **Real FineWeb-Edu slice:** 500 documents; **0 LSH candidate pairs, 0 verified edges, 500 singleton groups**, 0 documents removed. Single observed CPU near-dedup runtime **10.249 s elapsed** (10.057 s user CPU / 0.065 s system); about **48.8 documents/s** on this specific sample. No repeated timings or scaling fit were performed.
- **Independent exhaustive real-data audit:** 500 documents give **124,750** unordered pairs. A mathematically safe shingle-set size-ratio bound pruned **107,665** pairs; **17,085** pairwise intersections/unions were evaluated; **0** pairs reached Jaccard ≥ 0.8. Audit runtime **0.33 s** for the separate precomputed-shingle comparison loop. This supports the conclusion that the real pilot contains **zero qualifying 3-gram/Jaccard-0.8 pairs**; it cannot measure recall when no positives exist.
- **Limits:** MinHash/LSH retrieval is probabilistic, and missed edges can split true components. The in-memory MinHash/Shingle-set implementation is a small-corpus reference, not a demonstrated large-scale deduplicator.

### 7.5 M2-004 — Deterministic group-aware splitting

- **Implemented:** `data_pipeline/split.py`; tests `tests/test_split.py` (**7 passed**).
- **Principle:** assign each detected component *as one unit* to train or validation. The canonical key hashes sorted `(source,id)` member identities, not the order-sensitive `group-000...` label. Seeded SHA-256 ranking gives deterministic group assignments for fixed membership; the new output directory contains `train.jsonl`, `val.jsonl`, `manifest.json`.
- **Synthetic experiment:** 8 detected groups split **6 train / 2 validation** at requested `val_fraction=0.25`, seed 42. The group-weighted original member counts were **21 train / 2 validation**, or only **8.7%** of original documents in validation. Group-count balancing is **not** document- or token-count balancing.
- **Real FineWeb-Edu experiment:** all 500 groups were singletons, yielding **450 train / 50 validation** documents and groups, seed 42, fraction 0.10.
- **Leakage qualification:** detected groups do not cross splits by construction; unseen/undetected similarities, or a change in how the groups were generated, can still produce leakage. The split manifest records the particular assignment; it is not a proof of corpus-level independence.

### 7.6 M2-005 — GPT-2 tokenization and flat packing

- **Implemented:** `data_pipeline/tokenize.py`; tests `tests/test_tokenize.py` (**6 passed**).
- **Tokenizer:** `tiktoken.get_encoding('gpt2')`; actual vocabulary **50,257**, EOT token **50,256**. This is distinct from the model's **50,304 padded vocabulary**. Ordinary text is tokenized with `encode_ordinary` so literal strings resembling special tokens remain ordinary text.
- **Format:** append one EOT ID after each document; write one continuous, little-endian `uint16` token stream per split, not precomputed independent 1,024-token blocks. Manifests contain source/output SHA-256 hashes, document counts, content-token counts, boundary counts, and totals. `train.py` samples next-token windows from the flat stream.
- **Synthetic split:** train **3,009** tokens across 6 representative documents, validation **1,004** tokens across 2. The **1,004-token validation file is insufficient for the 1,024-token reader**. `MemmapBatches` was separately tested on CPU with `(B,T)=(2,1024)` for train and `(2,64)` for validation; both input-target shifts and vocab bounds passed. This is a compatibility test, not a validation-loss experiment.
- **Real FineWeb-Edu split:** both files are long enough for the `train.py` `block_size=1024` reader. See the final token-count table in §7.9. Real-data GPT training backward/optimizer-step verification is **pending**.

### 7.7 M2-006 — Lossless binary sharding

- **Implemented:** `data_pipeline/shard.py`; tests `tests/test_shard.py` (**7 passed**).
- **Protocol:** split the flat `uint16` stream into ordered, size-limited shard files. Record per-shard filenames, start/end-exclusive token offsets, token/EOT counts, and SHA-256 checksums; refuse malformed input and publish only after all checks succeed. The source file's byte stream is **never retokenized**.
- **Synthetic trial:** 2,048-token shard cap produced **2 training shards** (2,048 + 961 tokens) and **1 validation shard** (1,004 tokens). Reassembling shard bytes in order matched both original `.bin` files exactly.
- **Real trial:** 100,000-token shard cap produced **5 train shards** and **1 validation shard**. Both reconstructed token streams matched the original token counts and SHA-256 values exactly.
- **Reader boundary:** individual shard files are **not automatically valid `MemmapBatches` inputs**. Training currently consumes the original flat `.bin` files; a shard-aware reader with a documented cross-shard window policy is a separate task.
- **Most recent complete full regression suite shown after M2-006:** **86 passed, 3 skipped, 0 failures**. Three skips were reported; their reasons were not rechecked in this log.

### 7.8 M2-007 — FineWeb-Edu real-data provenance and extraction

**Source:** `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`, streamed `train` split. The extraction manifest records revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. Selection was **the first 500 nonempty streamed documents**—*not* a randomized or representative sample. The adapter lives in `data_pipeline/fetch_fineweb.py` with offline tests `tests/test_fetch_fineweb.py` (**5 passed**).

| Provenance item | Recorded value |
|---|---|
| Dataset / configuration | `HuggingFaceFW/fineweb-edu` / `sample-10BT` |
| Revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| License (per source manifest) | ODC-By 1.0; Common Crawl terms also apply |
| Export requested / rows seen | 500 / 500 |
| Documents exported / blank rows skipped | 500 / 0 |
| Raw export SHA-256 | `cbf13c8ebd881bb4e41c35e2d4301549800d932780d3883d022ed457be001462` |
| Provenance SHA-256 | `5772c2995cc9ef5845e25c26c499cc92722d05161f6e6394a60c60c3cafc3da7` |

- `raw.jsonl` contains the input text/IDs; `provenance.jsonl` retains source URL, Common Crawl dump, date, and file path where present. `source_manifest.json` records the selection rule, revision, license/terms information, counts, and file checksums. The existing `ingest.py` intentionally strips extra metadata, so provenance is kept separately.
- **Provenance boundary:** recording a dataset-level license and URL does not independently clear rights for every crawled page. Do not publish the raw FineWeb sample in the source Git repository without a separate review.
- Real-data ingestion wrote **500 documents**, skipping 0 empty. Quality filtering accepted **500/500**, with zero rejections under any of the four configured rules. Exact/near dedup retained **500/500**, with zero confirmed duplicates; the exhaustive audit appears in §7.4.
- The pinned dataset revision, selected-row ordering, input sha256, and code should be retained when attempting reproduction. No separate independent dataset-card/revision audit is claimed here; these are values reported by the successfully executed extraction manifest.

### 7.9 Real-data final output and checksums

| Quantity | Train | Validation | Total |
|---|---:|---:|---:|
| Representative documents (singleton groups) | 450 | 50 | **500** |
| GPT-2 content tokens | 429,155 | 46,964 | **476,119** |
| EOT boundary tokens | 450 | 50 | **500** |
| Total stored token IDs | **429,605** | **47,014** | **476,619** |
| Storage as uint16 | 859,210 bytes | 94,028 bytes | 953,238 bytes |
| 100,000-token-cap shards | 5 | 1 | **6** |
| `block_size=1024` reader length check | Passed | Passed | — |

**Verified source-to-binary hashes:**

| File or manifest | SHA-256 |
|---|---|
| `fineweb_split_v1/manifest.json` | `9d28c558ec47ef9624e3b4794f0a1a701a0e6e69971172c6fe6e636db5b15f4c` |
| `fineweb_tokens_v1/train.bin` | `9349ea092c19a94fafb924b7409ccef00a9f2deabfba476fd5dbc379b80b4c43` |
| `fineweb_tokens_v1/val.bin` | `62acffe82571907bfefa88bb0a03a083b49ce70bce128fff76ce631d224d84a3` |

The tokenization and sharding manifests independently reported matching **total tokens, EOT counts, and stream SHA-256** values for both splits. The synthetic trial also passed direct byte-for-byte shard reassembly. On the real pilot, counts/checksums matched; a separate direct byte-by-byte reassembly loop was not reported. Those distinctions are kept explicit.

**Current data paths on the PVC:**

```text
/workspace/p4/m2-data/
  fineweb_pilot_500/        raw.jsonl, provenance.jsonl, source_manifest.json
  fineweb_ingested.jsonl
  fineweb_filtered.jsonl
  fineweb_exact_dedup.jsonl
  fineweb_near_dedup.jsonl
  fineweb_split_v1/         train.jsonl, val.jsonl, manifest.json
  fineweb_tokens_v1/        train.bin, val.bin, tokenization_manifest.json
  fineweb_shards_v1/        train-*.bin, val-*.bin, shard_manifest.json
```

### 7.10 Measurement interpretation, limitations, and next actions

**Established:** the CPU pipeline processes an actual 500-document web-text sample without a reported failure, produces deterministic structures and checksummed binary outputs, and preserves detected duplicate groups across the train/validation split. Both real-data flat token streams satisfy the existing training reader's length condition. The Jaccard audit validates the zero-near-duplicate finding for **this exact sample and similarity definition**.

**Not established:** any improvement in model loss/quality; statistical representativeness; recall on a real dataset containing confirmed near-duplicates; large-corpus performance, disk usage, or memory bounds for the near-dedup index; true training performance from these token files; elimination of unseen cross-split leakage; or dataset-wide licensing clearance. One synthetic sample is a code-validation fixture, not an empirical benchmark of LSH recall across the web.

**Explicit follow-up checks (not yet verified in the supplied run outputs):**

1. Run `python -m pytest -q` with the M2-007 files added. The previous full suite reported **86 passed, 3 skipped**; the five new exporter tests passed separately. **91 passed / 3 skipped is an expectation, not an observed full-suite result.**
2. Smoke-test a **real-data** CPU forward, next-token loss, backward, and AdamW update with `train.py --model gpt-tiny --device cpu --data /workspace/p4/m2-data/fineweb_tokens_v1/train.bin --micro-bs 1 --steps 1 --bench-warmup 0 --log-every 1 --lr-warmup 1 --dry-run`. This is not a 125M pretraining result.
3. Commit `data_pipeline/fetch_fineweb.py`, `tests/test_fetch_fineweb.py`, and this lab-log update after verifying the above. Last *confirmed* push was M2-006; no M2-007 push outcome was supplied.
4. For a meaningful clean GPT-125M from-scratch loss curve, scale to an appropriately sized reproducible real-text dataset, monitor PVC/cache usage, record dedup/quality statistics and validation-token coverage, then reserve a GPU. Keep this objective separate from upcoming M3 4-GPU communication/scaling comparisons.

**Decision for M3:** M2's pilot validates the data-path format and audit workflow, but is **not** a prerequisite to keep enlarging the corpus before coding/debugging DDP. Do not confuse pipeline completeness with the much larger training-data budget required for model-quality experiments.


## 8. M5 — Tensor Parallelism and Hybrid TP × DP

### 8.1 Objective and experimental scope

**Objective:** Implement tensor parallelism for the P4 GPT architecture,
combine it with data parallelism, verify numerical correctness, and
evaluate the performance and recovery tradeoffs.

**Hardware:** Single-node NVIDIA B200-SXM GPUs.

**Software:** PyTorch 2.8.0, CUDA-enabled runtime, NCCL 2.27.3,
BF16 autocast, and FlashAttention.

Two distributed configurations were validated:

| Strategy | World size | Parallelism |
|---|---:|---|
| TP | 2 | TP=2 |
| Hybrid | 4 | TP=2 × DP=2 |

The hybrid device mesh is organized as DP × TP.

- TP groups: [0,1], [2,3]
- DP groups: [0,2], [1,3]

The milestone prioritizes correctness before communication optimization.

### 8.2 M5-001 — Tensor-parallel implementation

**Implementation:** `tp_parallel.py`, integrated into `train.py`.

The attention and MLP projections are distributed as follows:

| Projection | Parallelization |
|---|---|
| Attention fused QKV | Column-wise |
| Attention output | Row-wise |
| MLP expansion | Column-wise |
| MLP output | Row-wise |

Embeddings, LayerNorms, and the LM head remain replicated.
The input embedding and LM head continue sharing their weight.

The fused QKV projection requires rank-major reordering before
column-wise sharding. Without this reordering, a TP rank would not
receive the intended local Q, K, and V partitions.

The standalone TP implementation passed forward, loss, backward,
and parameter-gradient parity tests against the reference GPT.

**GPT-tiny reference comparison:**

| Measurement | Observed |
|---|---:|
| Maximum logits difference | 2.384e-07 |
| Loss difference | 0 |
| Maximum parameter-gradient difference | 1.788e-07 |
| Tied embeddings | Preserved |

GPT-1.3B full-model parity was also validated:

- Maximum logits difference: approximately 1.168e-05
- Loss difference: approximately 9.537e-07
- Maximum parameter-gradient difference: approximately 1.136e-06

These results establish numerical parity within the measured
floating-point tolerances, not bitwise identity.

### 8.3 M5-002 — Hybrid TP × DP correctness

**Implementation:** `sync_dp_grads_()` and
`clip_grad_norm_tp_()` in `tp_parallel.py`.

Each TP group receives identical training examples, while different
DP replicas receive different examples.

Corresponding local parameter gradients are averaged across DP ranks
after backward propagation.

Global clipping sums squared gradients from unique TP shards and
counts replicated gradients only once per TP group.

**Hybrid gradient reference test:**

| Measurement | Observed |
|---|---:|
| Global batch | 4 |
| Sequence length | 8 |
| Reference loss | 10.83638573 |
| Hybrid loss | 10.83638573 |
| Parameters checked | 28 |
| Maximum gradient difference | 2.533e-07 |

**Hybrid optimizer reference test:**

Three FP32 update steps passed with DP gradient averaging, global
gradient clipping, and standard AdamW.

- Maximum gradient-norm difference: 1.350e-07
- Maximum parameter difference: 3.822e-06

BF16 FlashAttention training also passed with standard AdamW,
gradient clipping, and decreasing loss across three steps.

**Interpretation:** The hybrid implementation matches the reference
training computation within numerical tolerance for the tested
architecture, data, and optimizer settings.

### 8.4 M5-003 — Distributed checkpointing and recovery

**Implementation:** `tp_checkpoint.py`.

PyTorch Distributed Checkpoint saves model and optimizer state.
Additional per-rank runtime files preserve:

- Completed global training step
- Synthetic-data cursor
- CPU RNG state
- CUDA RNG state

A saved configuration signature is verified before resumption.

The `latest.json` marker is published after successful checkpoint
writes, using atomic file replacement.

**Verified recovery experiments:**

1. Distributed checkpoint roundtrip.
2. Fresh-process restart with restored optimizer and runtime state.
3. Interrupted versus uninterrupted training comparison.
4. Worker SIGKILL followed by restart from the last published step.
5. Collective reporting of checkpoint directory creation errors.
6. Collective reporting of rank-local runtime write errors.
7. Collective preflight rejection of unreadable runtime state.
8. Quarantine of a newer incomplete checkpoint directory.
9. Rejection of automatic quarantine when `latest.json` is absent.

The uninterrupted and resumed training experiments matched exactly
for compared model and optimizer tensors, data cursors, and CPU/CUDA
RNG states.

The checkpoint comparison covered 112 state tensors per rank in the
tested hybrid GPT-tiny configuration.

**Recovery boundary:** These guarantees were tested for synthetic
training with a matching topology and configuration. They are not
claims of arbitrary-topology checkpoint portability or recovery from
every possible collective failure.

### 8.5 M5-004 — Official performance benchmarks

**Date:** 2026-11-05.

**Code:** `b1d1583`, clean working tree at benchmark launch.

**Hardware:** Single-node NVIDIA B200-SXM GPUs.

**Protocol:** GPT-1.3B, BF16 autocast, FlashAttention,
sequence length 1024, synthetic data, seed 1337,
gradient accumulation 1, gradient clipping 0.1,
120 steps with 30 warmup and 90 measured steps.

| Strategy | GPUs | Tokens/step | Tokens/s | MFU | Peak GB/GPU |
|---|---:|---:|---:|---:|---:|
| DDP=2 | 2 | 16,384 | 148,011 | 27.87% | 43.2 |
| TP=2 | 2 | 16,384 | 80,806 | 15.22% | 39.4 |
| DDP=4 | 4 | 32,768 | 288,584 | 27.17% | 43.2 |
| TP=2 x DP=2 | 4 | 32,768 | 148,641 | 13.99% | 39.4 |

**Measured comparison:**

- On two GPUs, DDP throughput was approximately 1.83x TP.
- On four GPUs, DDP throughput was approximately 1.94x
  the hybrid strategy.
- TP/hybrid used 3.8 GB less peak memory per GPU.
- DDP throughput increased approximately 1.95x from
  two to four GPUs at the same per-rank microbatch size.

**Step-time p50 / p95:**

- DDP=2: 110.7 / 110.9 ms
- TP=2: 202.3 / 204.3 ms
- DDP=4: 113.5 / 113.8 ms
- Hybrid: 213.2 / 260.1 ms

Step-time percentiles are calculated from averages of
10-step measurement windows, not individual GPU steps.

**Prediction assessment:**

- DDP=2: predicted to outperform TP=2; confirmed.
- TP=2: predicted 82,000; actual 80,806 tok/s (-1.46%).
- DDP=4: predicted 290,000; actual 288,584 tok/s (-0.49%).
- Hybrid: predicted 145,000; actual 148,641 tok/s (+2.51%).

**Variability:** Hybrid cumulative throughput declined from
153,244 tok/s at step 100 to 148,641 tok/s at step 120.
The available logs do not identify the cause.

**Interpretation boundary:** The comparison evaluates the
implemented systems, not isolated TP communication overhead.
DDP used fused AdamW while TP/hybrid used single-tensor
AdamW. Microbatch configurations and communication patterns
also differed.

**Official artifacts:**

- `runs/M5-DDP2-001.json`
- `runs/M5-TP2-001.json`
- `runs/M5-DDP4-001.json`
- `runs/M5-HYBRID-001.json`

### 8.6 Decisions, negative results, and limitations

**Decision 1 — Preserve earlier optimizer behavior.**

Restrict standard single-tensor AdamW to TP and hybrid modes.
Retain the existing optimizer selection for DDP, ZeRO-1, and FSDP.

**Decision 2 — Keep TP degree fixed at 2.**

The current architecture and correctness tests validate TP=2.
Larger TP degrees require separate validation.

**Decision 3 — Prioritize correctness over communication tuning.**

Retain explicit per-parameter DP gradient averaging for the M5
baseline. Bucketed reductions are a future optimization.

**Negative result — Fused and foreach AdamW.**

The tested fused and foreach AdamW paths failed with the current
mixture of DTensors and ordinary tensors. This is a limitation of
the tested implementation and software configuration, not evidence
that fused optimization is universally incompatible with TP.

**Current restrictions:**

- TP=2, or TP=2 × DP=2.
- TP gradient accumulation restricted to 1.
- TP with `torch.compile` not validated.
- Checkpoint integration restricted to synthetic-data dry runs.
- Checkpoint resumption requires matching configuration and topology.
- Concurrent checkpoint writers are unsupported.

### 8.7 Regression evidence and next actions

During the M5 review, the ordinary repository suite passed:

**94 passed, 0 failures.**

Distributed validation was executed separately using `torchrun`.
The relocated GPT-tiny TP parity, BF16 training, hybrid gradient
parity, and hybrid optimizer parity tests passed.

The checkpoint implementation was additionally tested against
simulated filesystem failures and controlled restart scenarios.

**Repository locations:**

- `tp_parallel.py`
- `tp_checkpoint.py`
- `train.py`
- `tests/distributed/`
- `experiments/m5/`
- `docs/m5_tensor_parallel.md`

**Next actions:**

1. **Completed:** M5 regression suite passed (94 tests during
   the original M5 review).
2. **Completed:** Generated checkpoint artifacts were excluded from Git.
3. **Completed:** M5 implementation committed as `b1d1583`.
4. **Completed:** Four official clean-tree benchmark records collected
   on 2026-11-05 and committed as `56d8ffa`.
5. **Completed:** Historical warmup metadata corrected in `884955d`;
   the expanded regression suite passed 95 tests.
6. **Deferred:** Further communication and optimizer optimizations
   remain separate from the verified M5 baseline.

**M5 conclusion:** TP and hybrid TP × DP have passed the reported
reference-model correctness checks. Restartable DCP checkpointing
has passed controlled recovery and failure-injection experiments.
The current throughput comparisons demonstrate a performance
tradeoff, but do not isolate its individual causes.
