# P4 Lab Log — Distributed Pretraining From Scratch

**Started:** 2026-08-01
**Repo:** `/workspace/p4/p4-pretrain/`
**Rule:** nothing gets measured twice. If it isn't in this file, it didn't happen.

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
| M0 | Env + NCCL sanity | 1 → 4 | ✅ A100 ref (195 GB/s) · ✅ 1×B200 env verified · ⬜ 4×B200 NCCL rerun | 4-GPU session |
| M1 | Single-GPU baseline + MFU, 125M **and** 1.3B, eager vs compiled | **1** | ✅ 33 B200 tests; tiny CUDA smokes; five measured B200 baselines complete | — |
| M2 | Data pipeline (tokenize→dedup→filter→pack→shard) | **0** (CPU) | ⬜ not started | — |
| M3 | DDP 4-way + scaling efficiency table | 4 | ⬜ not started | M1, M2 |
| M4 | FSDP / ZeRO-1,2,3 — overhead **trend across 125M→1.3B** | 4 | ⬜ not started | M3 |
| M5 | 2-way TP **at 1.3B** + checkpoint/resume | 4 (logic: 1) | ⬜ not started | M4 |
| M6 | bf16→fp8→fp4 ablation at 125M **and** 1.3B | **1** core + 4 confirm | ⬜ not started | M1 |
| M7 | MFU before→after with diagnosed bottleneck | 4 | ⬜ not started | M3 |
| M8 | `docs/training.md` + README + writeup | 0 | 🟡 M1 results section drafted; final writeup pending | M4, M7 |

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

### M1 synthesis — supported findings, caveats, and next action

- **Model-size effect:** Eager MFU rose **17.54% → 34.30%** when switching from 125M to 1.3B with the same batch/sequence configuration, while tok/s fell **461.5K → 91.1K**. This is a fixed-configuration observation, not a universal law that larger models yield higher MFU.
- **Compiler effect:** at fixed size and Flash backend, `torch.compile` improved steady-state throughput **1.485× at 125M** versus **1.162× at 1.3B**; peak allocated memory fell **28.5%** and **8.2%**, respectively. Root causes require per-op profiling later in M7.
- **Backend effect:** cuDNN beat Flash by **7.41%** at compiled GPT-1.3B (head_dim 128, sequence length 1024). This result does not generalize to other shapes or PyTorch/cuDNN versions without testing.
- **Measurement limits:** one full measured run per configuration; no across-run error bars; 150 measured optimizer steps per run, 10-step averaged windows, hardware vendor theoretical BF16 peak as MFU denominator; on-device **repeated synthetic** batches remove input pipeline costs and do not assess language-model quality. `torch.cuda.max_memory_allocated()` reports allocated memory, not reserved/system-wide usage. A full tokenized real-data loss curve remains a separate M2+ deliverable.
- **Reproducibility:** original predictions were preserved. Successful runs verified the *executed* attention backend, not only the requested flag. First-step timings include first touch/autotuning and potentially cached compilation; don't treat them as pure compile latency.
- **Close M1:** preserve five JSON artifacts and documentation in Git; defer detailed bottleneck attribution and repeated confidence intervals to M7. **Next: M2 CPU data pipeline** (tokenize, dedup, quality-filter, pack, shard); release the 1-GPU pod once artifacts are committed.

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
- **Fix / mitigation:** use an isolated `TORCHINDUCTOR_CACHE_DIR` for each compiled backend (e.g. `/workspace/.cache/p4-inductor/m1-005-cudnn`); keep `verify_backend` as a hard failure for measured runs; never rename a Flash run as cuDNN.
- **Generalizable lesson:** a configuration flag expresses intent; the profiler reveals what executed. In systems experiments, guard the validity of the independent variable **before** collecting headline metrics.
- **Follow-up (optional):** reproduce the mismatch with fresh and warm caches under identical code, inspect compile cache key behavior and backend versions; fix permanently in setup/runtime or upgrade after controlled validation. Not needed to close M1.


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
| MFU story? | M1 single B200 GPT-125M eager 17.54% → compiled 26.05%; steady-state +48.5% tok/s; exact bottleneck attribution deferred to M7 |
| Fault tolerance? | Checkpoint/resume survives a mid-run `kill -9`; tested by __ at step __, resumed to identical loss trajectory |
| fp8? | __× throughput over bf16 at __ loss delta; watched range/overflow in __ |
| Hardest bug? | ISSUE-004: requested cuDNN but compiler cache reused Flash; profiler guard caught backend mismatch |
| Something that didn't work? | NEG-__ |
