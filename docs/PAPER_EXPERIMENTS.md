# Paper experiments: the flow, step by step

Written 2026-09-28 on branch `tm-opt`, and smoke-tested on Llama-3.1-8B. It can be re-run later by anyone,
without the sessions that wrote it. The user approved the flow on 2026-09-28. The full runs follow
`results/paper/PROTOCOL.md`, with the user's decisions: CUDA-graph latency is primary, eager supplementary; 5
rounds; lm-eval's default MMLU scoring; a Qwen BF16 MMLU out-of-memory rule.

- **Scripts:** `experiments/paper/`, one per step, plus `run_all.sh`.
- **Environment:** every path is an environment variable with a documented default (`paper_common.py`).
- **Scheduling:** no Slurm; everything runs locally on one RTX PRO 6000.

## What is measured

**Models:** Llama-3.1-8B, Mistral-7B-v0.3, Phi-4 and Qwen3.8-27B, at their pinned revisions (`sm120/eval/common.py`).

**Ours = TM-OPT+TC, the paper's final method** (the user's decision, recorded in `results/tm_opt/REPORT_QR.md`):
- **Element types:** E2M1 tiles use FourOverSix scaling; E0M3 tiles use alpha = 1.
- **Activations:** FourOverSix with per-token scales (`four_over_six_rows`).
- **Evaluation convention:** NativeLinear convention (c): the quantized Linears run on the SM120 kernels.
- **Calibration:** identical settings for every model (`--tm-opt --tile-grad-tc --param ste --lr 0.02
  --init-logit -1 --epochs 20 --eval-every 2 --no-dev --no-eval`, deterministic). The only exception is Qwen's
  micro-batch: 2 × accumulation 4.
- **Tile sizes:** 8x64 and 16x64 in the main tables, 256x64 in the appendix.

| experiment | step | what |
|---|---|---|
| 1 | 03 | WikiText-2 and C4 perplexity (`run_ppl_deploy.py`); paired ΔNLL ± 2 SE against FourOverSix and NVFP4 |
| 2 | 04 | Downstream accuracy through lm-eval 0.4.11 (`run_lmeval_deploy.py`, NativeLinear only): MMLU 5-shot; ARC-Challenge, ARC-Easy, HellaSwag and PIQA 0-shot. Paired accuracy differences ± 2 SE |
| 3.1 | 05 | End-to-end prefill latency, eager and CUDA-graph captured |
| 3.2 | 06 | GEMM (and activation-quantizer) kernel latency per text-Linear shape, and summed per forward |

### Policies

Accuracy policies (steps 03, 04), `paper_common.ACCURACY_POLICIES`:

| label | artifact | kernel |
|---|---|---|
| `bf16` | none (the model as loaded) | none |
| `nvfp4` | `<model>_nvfp4` | `auto_stock`: stock weights-on-A, CTA width from the tile table |
| `fo6` | `<model>_fo6` (FourOverSix) | `auto_stock` |
| `ours-8x64` | `<model>_tc_8x64` | `n8k64_wB` (weights on B; its single build) |
| `ours-16x64` | `<model>_tc_16x64` | `auto`: the mixed weights-on-A set (n16k64_wA), width from the tile table |
| `ours-256x64` | `<model>_tc_256x64` (exported as 16x64 granules) | `auto` |

Latency policies (step 05: `05_prefill_latency.py` and `paper_common.LATENCY_POLICIES`) add three kinds of row:
- **`bf16`:** a reference row.
- **`nvfp4-wB`, `fo6-wB`:** the same NVFP4 / FourOverSix artifacts on `stock_wB`, the same-placement references for
  8x64.
- **`ours-<unit>-nvfp4act`:** the same TM-OPT+TC artifact with only the activation quantizer switched to NVFP4
  (`nvfp4_rows`). **This row is latency only:**
  - the maps were calibrated with FourOverSix activations, so it is never evaluated for accuracy;
  - it is an explicit install option (`NM.install(..., activation_quantizer='nvfp4_rows')`), not a new artifact;
  - the install record says so (`activation_quantizer_override: true`);
  - every table labels it "latency only".

## Prerequisites

**Hardware:**
- one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (sm_120, 96 GB);
- **nothing else may run on the GPU.** Step 00 checks this, and steps 05 and 06 refuse to start otherwise.

**Software:**
- **The Python environment:** default `/home/dev/.conda/envs/n16k64`, with torch 2.9.0+cu128, transformers 5.16.1
  and lm-eval 0.4.11 (`sm120/requirements.lock.txt`; step 00 checks the versions).
- **CUDA 13.1** for the SM120 kernels, at `PAPER_CUDA_HOME`.
- **The kernels built:** n16k64_wA and its _n16/_n32/_n64 widths, stock_wA and its widths, n8k64_wB and stock_wB.
  See `docs/BUILD_AND_USE.md` §3.
- **The RTX PRO 6000 tile table** `sm120/configs/nvidia_rtx_pro_6000_blackwell_workstation_edition.json`
  (committed in b37e489). Both width-selecting sets must pick it up (step 00 checks this).

**Data:**
- the model snapshots in the HF cache at their pinned revisions;
- the calibration records (`prepare_*_data.py` outputs) under the data roots;
- the lm-eval datasets. They download on first use at pinned revisions (`lm_eval_datasets.py`, including
  `cais/mmlu` at c30699e8). Allow network access the first time (`HF_HUB_OFFLINE=0`, the default here): MMLU's 57
  subjects took several minutes to fetch.

### Environment variables (`experiments/paper/paper_common.py`)

| variable | default | what |
|---|---|---|
| `PAPER_REPO` | the repository containing the scripts | |
| `PAPER_PYTHON` | `/home/dev/.conda/envs/n16k64/bin/python` | interpreter for every step |
| `PAPER_CUDA_HOME` | `/home/dev/.conda/envs/mixfp4-cuda131` | CUDA 13.1 (the SM120 kernel libraries) |
| `HF_HOME` | `/home/dev/.cache/huggingface` | Hugging Face cache |
| `HF_HUB_OFFLINE` | `0` | set `1` once every model and dataset is cached |
| `PAPER_DATA_LLAMA8B` | `/home/dev/n16k64_campaign/cost_comparison/data` | Llama-3.1-8B's calibration data root |
| `PAPER_DATA_ROOT` | `/home/dev/n16k64_campaign/multimodel/data` | Mistral, Phi-4 and Qwen's data root |
| `PAPER_MAPS_REF` | `/home/dev/n16k64_campaign/tm_opt/runs` | committed TM-OPT+TC maps, reused when their sha256 matches |
| `PAPER_ARTIFACTS_REF` | `/home/dev/n16k64_campaign/deploy_eval/artifacts` | Parts 2–3 artifacts, compared by hash (not used) |
| `PAPER_OUT` | `/home/dev/n16k64_campaign/paper` | outputs of the real runs |
| `PAPER_SMOKE_OUT` | `/home/dev/n16k64_campaign/paper_smoke` | outputs of `--smoke` runs (never results) |

Every step sets `PYTHONPATH=<repo>/sm120:<repo>`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and the allocator. Steps 01–04
use `expandable_segments:True`, as every recorded calibration and evaluation did (Qwen's calibration peaks at
90 of 96 GiB). Steps 05–06 use the default caching allocator, as Part R did.

## Conventions common to every step

- **Filters:**
  - `--models llama8b,mistral7b,phi4,qwen27b` and `--units 8x64,16x64,256x64` restrict a step;
  - `--policies` takes the step's own labels (step 02: `nvfp4,fo6,tc`);
  - `--force` redoes finished work;
  - `--out DIR` overrides the output root.
- **Resumable:** every step skips what it has finished. A finished item has a complete report.
  - An unfinished artifact or training directory is moved aside as `*.failed_<time>`.
  - A finished one that is redone (`--force`, `--train`) is moved aside as `*.previous_<time>`.
  - An unfinished lm-eval report resumes at the task level.
- **Logs:**
  - every command's output is written to `<out>/logs/<step>_<model>_<policy>.log`;
  - `<out>/commands.log` gets every exact command line, with its start, end and exit code;
  - it also gets each map reuse and each latency round's shuffled policy order.
- **One process per (model, policy)** for accuracy, and per (model, policy, round) for latency. A crash therefore
  loses one item, and memory and allocator state never carry over between policies.
- **`--smoke`:**
  - models: Llama-3.1-8B only;
  - policies: every one except 256x64 (whose kernel path is 16x64's);
  - sizes: tiny (4 windows per corpus; `--limit 5`; one latency round with two lengths; one GEMM shape at two
    token counts);
  - outputs go to `PAPER_SMOKE_OUT`. **Smoke numbers are plumbing checks, not results.**

## The steps

Dependency order: **00 → 01 → 02 → {03, 04, 05, 06} → 07**. 03–06 only read 02's artifacts. Run them one after
another: there is one GPU, and 05–06 need it idle.

Times below are per model: Llama / Mistral / Phi-4 / Qwen. Sources:
- **measured:** the smoke run on this machine (2026-09-28, Llama);
- **recorded:** earlier runs of the same code on this GPU, as cited.

### 00 — environment and build check (`00_check.py`)

- **Purpose:** fail early.
- **Command:** `$PAPER_PYTHON experiments/paper/00_check.py`
- **Checks:**
  - the GPU is idle;
  - the pinned versions;
  - all ten kernel configurations load (sha256s recorded);
  - the tile table exists, matches this GPU's slug, and both kernel sets load it (sha256 recorded);
  - every model snapshot and calibration record is present;
  - the committed-map manifest is present.
- **Output:** `<out>/00_check.json`. Exits 1 on any failure.
- **Time:** 6 s (measured). **GPU:** none beyond loading the kernel libraries.

### 01 — calibration: TM-OPT+TC maps (`01_calibrate.py`)

- **Purpose:** one map per (model, unit), `<out>/maps/<model>_<unit>/map.pt`, with `calibration.json`.
- **Command:** `$PAPER_PYTHON experiments/paper/01_calibrate.py [--train]`
- **Reuse or train:**
  - If `PAPER_MAPS_REF/<run>/map.pt` has the committed sha256 (`experiments/paper/maps.sha256.json`: the 12
    committed `tc_*` / `q_tc_*` records), it is copied: "reused".
  - Otherwise, or with `--train`, it is trained: `run_train_map.py` with the method's flags, `--no-dev --no-eval`,
    deterministic. The record then says whether the new map equals the committed one bitwise. In Task 1,
    deterministic no-dev training reproduced the committed map exactly on this GPU for every pair it retrained:
    Llama, Mistral and Phi-4 at all three units, and Qwen at 8x64 (`results/nodev_cost`).
- **Checks:** sha256 against the manifest; training exit code and complete report.
- **Time:**
  - reusing: seconds;
  - training, per unit (recorded, `results/nodev_cost/cost.md`, deterministic, no dev): **8.6 / 7.5 / 14.8 / 76.0
    min**. Qwen's no-dev figure was measured for 8x64 only. Its with-dev runs took 79.7–79.9 min at all three units
    (`results/tm_opt/REPORT_QR.md`), so 16x64 and 256x64 are taken as the same 76 min;
  - all 12 maps: **≈ 5.3 h**.
- **GPU memory (training peak, allocated):** 40.5 / 35.9 / 59.1 / 90.1 GiB.
- **Smoke (measured):**
  - reuse of Llama 8x64 and 16x64 (sha256 match);
  - a 1-epoch training of Llama 16x64 into the smoke directory: 88 s, deterministic, lean, TC tile gradient;
  - the 1-epoch map is not used.

### 02 — deployment artifacts (`02_export.py`)

- **Purpose:** per model, `<out>/artifacts/<model>_{nvfp4,fo6,tc_8x64,tc_16x64,tc_256x64}` (maps also get
  `.mixfp4map`).
- **Command:** `$PAPER_PYTHON experiments/paper/02_export.py`
- **Export and ownership:**
  - each artifact comes from `export_map_artifact.py --ownership`;
  - maps: the ownership check runs on their own kernel (n16k64_wA for 16x64 and 256x64, n8k64_wB for 8x64);
  - baselines: it runs on stock_wA, then again on stock_wB (`ownership_wb.py`), the 8x64 same-placement latency
    reference.
- **Placement independence:** one artifact serves both placements. The scale layout does not depend on the operand
  (`sf_buffer_size(rows, k)`), and NativeLinear places the scales for the kernel it is given.
- **Checks:**
  - the exporter's own: weights equal the calibration record, candidates equal sm120's bitwise, packed weights equal
    the fake-quant weight;
  - ownership: every element's hardware-decoded value and executed format.
  - `<name>.export.json` records whether the weights hash equals the Parts 2–3 artifact (`PAPER_ARTIFACTS_REF`).
- **Time:**
  - Llama: 68–76 s per artifact (measured);
  - the others, from the Parts 2–3 export and ownership times: ≈ 6 / 6 / 11 / 20 min per model (5 artifacts);
  - total **≈ 45 min**.
- **GPU:** the BF16 model on the GPU (15–51 GiB of weights) plus one module's quantization working set.
- **Smoke (measured):** 4 artifacts (nvfp4, fo6, tc_8x64, tc_16x64), 4.9 min. Every artifact equals its Parts 2–3
  counterpart bitwise, and every ownership check is exact on stock_wA and stock_wB.

### 03 — Experiment 1: perplexity (`03_ppl.py`)

- **Purpose:** WikiText-2 and C4 perplexity of the six accuracy policies, convention (c). One `run_ppl_deploy.py`
  process per (model, policy) writes `<out>/ppl/<model>/<policy>/report.json`, with the NLL of every window.
- **Command:** `$PAPER_PYTHON experiments/paper/03_ppl.py`
- **Windows:** the released protocol's: WikiText-2 test in 2048-token windows, and 256 C4 validation crops, batch 1.
  They are validated against the published record where one exists, and against sm120's reference token hashes.
- **Checks:**
  - native coverage in every forward: every quantized Linear ran natively, and the number of calls is exact;
  - non-finite NLL stops the run.
- **Time:**
  - per process, the fixed part: load 27–93 s, data 12–19 s, install 7–30 s;
  - evaluation per policy (recorded, Parts 2–3 on this GPU): native ≈ 27 / 25 / 46 / 470–520 s; BF16 52 / 53 /
    97 / 458 s;
  - per model: **≈ 9 / 9 / 15 / 64 min**; total **≈ 1.6 h**.
- **GPU:** the BF16 policy holds the BF16 model (weights 15.0 / 13.5 / 27.3 / 51.0 GiB, Part R) plus one window's
  activations and logits; the native policies need about a third of that (Part R: 5.6 GiB of weights for Llama,
  18.5 GiB for Qwen).
- **Smoke (measured):** 5 policies, 4 windows per corpus. About 70 s per process (load and data dominate); 5.8 min in
  all. **Every policy's per-window NLL is bitwise identical to the Parts 2–3 record** on those windows. The runs now
  go through the tile table: the widths differ, the bits do not.

### 04 — Experiment 2: downstream accuracy (`04_downstream.py`)

- **Purpose:** lm-eval accuracy of the six policies, NativeLinear only (no fake quant). One `run_lmeval_deploy.py`
  process per (model, policy) writes `<out>/lmeval/<model>/<policy>/report.json`.
- **Command:** `$PAPER_PYTHON experiments/paper/04_downstream.py`
- **Settings:**
  - tasks `mmlu,arc_challenge,arc_easy,hellaswag,piqa`, with `--num-fewshot mmlu=5`; the rest are lm-eval defaults
    (0-shot);
  - lm-eval 0.4.11 with its default BOS handling; batch 16 (Qwen 8);
  - the datasets pinned (`lm_eval_datasets.py`).
- **MMLU's score:** lm-eval's own group aggregate, `weight_by_size`: the micro-average over all 14,042 questions.
  The report records this, and the per-subject and per-category metrics.
- **Recorded per example:** its correctness under the primary metric (acc_norm; MMLU acc), and the per-choice
  log-likelihoods (`loglikelihoods`).
- **Recorded per task:** `sample_digest`, a hash of lm-eval's doc / prompt / target hashes of every example, and a
  12-character prompt-hash prefix per example. Step 07 requires equal digests across the policies, i.e. the same
  samples.
- **Checks:**
  - native coverage: every quantized Linear ran in every forward; the unscoped Linears (Qwen's vision tower) never
    ran;
  - step 07 recomputes every accuracy from the examples, and it must equal lm-eval's value.
- **Time:** estimated per (model, policy).
  - **Fixed part:** model load 27 s (Llama) to 94 s (Qwen), plus ≈ 1–3 min for MMLU's task construction and
    scoring (63 s in the smoke).
  - **MMLU:** the smoke's model time, scaled by the full-to-smoke token ratio. The 14,042 5-shot prompts are 9.6M
    tokens with Llama's tokenizer, against the smoke's 170k (× 56.7). With the other tokenizers they are 11.0M
    (Mistral), 9.6M (Phi-4) and 10.0M (Qwen).
  - **The four 0-shot tasks:** Task 3's full-run estimates (`results/lmeval_ready/READINESS.md`), upper estimates.

  | model | MMLU 5-shot, per native policy | 4 × 0-shot, per native policy | per native policy | BF16 | six policies |
  |---|---:|---:|---:|---:|---:|
  | Llama-3.1-8B | ≈ 10 min | ≈ 3 min | ≈ 13 min | ≈ 20 min | ≈ 1.4 h |
  | Mistral-7B-v0.3 | ≈ 11 min | ≈ 3 min | ≈ 14 min | ≈ 22 min | ≈ 1.6 h |
  | Phi-4 | ≈ 16 min | ≈ 3.5 min | ≈ 20 min | ≈ 35 min | ≈ 2.3 h |
  | Qwen3.8-27B | ≈ 35–50 min | ≈ 45–48 min | ≈ 1.4–1.7 h | ≈ 1.8–2.2 h | **≈ 9–10 h** |

  - **Qwen's MMLU 5-shot, explicitly:** 14,042 questions are 1,756 forwards of 8 prompts (mean 711 tokens).
    - The Qwen probe below timed such forwards at 718 ms (8 × 512) and 1,605 ms (8 × 1024), eager ≈ graph
      (GPU-bound): ≈ 0.18–0.20 ms per token, ≈ 31 min for 10.0M tokens.
    - lm-eval adds padding and per-request work: on Llama its MMLU forwards cost ≈ 1.6× the pure prefill per token.
      Hence ≈ 35–50 min per native policy.
    - BF16 is ≈ 1.6× slower on Qwen's batched prefills (Part R, 4 × 2048): ≈ 55–80 min.
    - **Over the six policies, MMLU on Qwen alone is ≈ 3.5–5 h.**
  - **Qwen's HellaSwag is close behind:** 40,168 requests in 5,021 batches of 8 short sequences. Each forward is
    host-bound at ≈ 0.4–0.47 s: Qwen's linear attention runs HF's pure-PyTorch fallback, because flash-linear-attention
    is not installed. That is ≈ 30–39 min per policy.
- **GPU:**
  - **0-shot tasks:** BF16 17 / 15 / 29 / 54 GiB; native 8 / 6 / 11 / 21 GiB (recorded, Task 3).
  - **MMLU raises the peak.** lm-eval keeps the BF16 logits and their log-softmax, [batch × longest 5-shot prompt ×
    vocabulary], at once. In the smoke, Llama peaked at 37.6 GiB (BF16) and 29.9 GiB (native).
  - **Qwen BF16 is the largest:** ≈ 80 GiB estimated. That is 54 GiB of model plus 2 × 11.6 GiB for
    8 × 3,138 × 248k, within the 96 GB but the closest to it.
  - **If it runs out of memory:** rerun that (model, policy) with `--batch-size 4`, and record it as a deviation.
- **Smoke (measured):**
  - **What ran:** 5 policies with `--limit 5`: MMLU gets 5 questions per subject (285); the other tasks get 5
    examples each.
  - **Time:** 17 min. BF16 took 7.2 min, including the first download of MMLU's 57 subsets. Each native policy took
    2.4 min: load 27 s, MMLU 72 s (8.6 s of it in the model), 6–8 s per 0-shot task.
  - **Checks, all passed:**
    - every quantized Linear ran natively in every forward (224 / 224), on the tile table's widths;
    - every accuracy recomputed from the examples equals lm-eval's (step 07);
    - resume: one task removed from a finished report, and the rerun kept the other four. The re-run task's examples
      and log-likelihoods were bit-identical;
    - same samples: the five policies have equal `sample_digest` on all five tasks.
  - **Peak GPU:** 37.6 GiB for BF16 and 29.9 GiB for native, both on MMLU. lm-eval takes a vocabulary-sized log-softmax
    over batch × the longest 5-shot prompts.
  - **Observation:** in 5–10 % of the smoke's MMLU questions, depending on the policy, the top two choices have equal
    log-likelihoods. lm-eval computes the per-choice log-likelihoods in the model's dtype, BF16 (its `softmax_dtype`
    is unset by default), so choices closer than BF16's resolution tie. lm-eval's `np.argmax` then takes the earlier
    choice.
    - This is lm-eval's standard behaviour, and every policy is scored the same way.
    - Step 07 reports the tie rate per policy.
    - Changing it would mean a non-default setting, which the user would have to decide.

### 05 — Experiment 3.1: prefill latency (`05_prefill_latency.py`, worker `bench_prefill.py`)

- **Purpose:** end-to-end prefill latency.
  - **Policies:** BF16, NVFP4, FourOverSix; Ours with FourOverSix activations (deployed) and with NVFP4
    activations (latency only), at 8x64, 16x64 and 256x64; and, for 8x64, NVFP4 and FourOverSix on stock_wB.
  - **Shapes:** batch 1 × {128, 256, 512, 1024, 2048, 4096, 8192} and 4 × 2048 (`--shapes` changes them).
- **Command:** `$PAPER_PYTHON experiments/paper/05_prefill_latency.py [--rounds 5]`
- **Protocol (Part R's):**
  - one process per (model, policy, round), `<out>/latency/<model>/<policy>/round<r>.json`;
  - 5 rounds, the policy order shuffled per round (seed 20260928 + round, logged).
- **Eager timing:** `sm120/bench/model.py prefill`, the function Part R timed: 2 warm-ups, then 7 forwards timed
  with CUDA events; the median.
- **CUDA graph:** the same forward, captured and replayed: 2 warm-up replays, then 7 timed. The graph's logits are
  compared bitwise with an eager forward.
- **Host-bound flag:**
  - set when the graph is more than 5% faster than eager;
  - without a graph, when the host's enqueue time reaches 90% of the eager time.
- **Reported:** the median over rounds; ratios paired within rounds.
  - **The CUDA-graph numbers are the primary ones** (main tables; the user's decision).
  - **Eager is supplementary** (appendix), with the host-bound flag.
- **Registered check:** every shape is captured, and the graph's logits equal eager's bitwise. A failure stops the
  step, and a recorded failure is not skipped on a re-run.
- **Checks:**
  - an idle GPU;
  - strict install, no fallback;
  - every NativeLinear ran;
  - the tile table in use (the kernel-set description is recorded, with calls per width).
- **Time:**
  - **Per process:** load 5–8 s (from the page cache; a cold load of Qwen took 93 s), install 6–31 s.
  - **Per shape:** 2 + 7 timed eager forwards and 3 enqueue forwards. The graph adds a reference, 2 warm-ups and the
    capture (eager speed), then 10 replays.
  - **Per model** (55 processes: 11 policies × 5 rounds): Llama ≈ 32 min, Mistral ≈ 32 min, Phi-4 ≈ 55 min, and
    Qwen ≈ 4.2 h, whose eager forwards take 0.47 s (1×128) to 3.9 s (1×8192). **Total ≈ 6 h.**
  - **Knobs:** `--rounds 3` (−40 %), `--shapes` (1×8192 is Qwen's costliest), `--no-graph` (about −45 %). They are not
    the protocol; use them only if the user changes it.
- **GPU:** measured 28.5 GiB for Qwen with FourOverSix (probe); Qwen BF16 adds ≈ 33 GiB of weights (≈ 60–65 GiB,
  from Part R's 55.9 GiB prefill peak plus the graph's pool).
- **Smoke (measured):**
  - **What ran:** 9 policies, 1 round, 1×128 and 1×2048, 3 repetitions: 2.5 min (about 15 s per process).
  - **A capture problem, found and fixed in the harness.** In the first smoke run, the captured forwards were slower
    than eager at 1×2048 and their logits differed (BF16 included). transformers treats CUDA-graph capture as tracing
    and then builds an explicit causal mask, which moves SDPA off the flash kernel. `bench_prefill.py` now keeps the
    eager mask decision during capture only (its docstring explains how).
  - **After the fix:**
    - the graph logits are bitwise equal to eager in 18 of 18 (policy, shape) pairs;
    - every 1×128 point is host-bound: graph 7.4–16 ms against eager 30–39 ms;
    - no 1×2048 point is: graph and eager agree within 1 %.
  - **Eager is noisy where it is host-bound.** At 1×128 the NVFP4 artifacts' eager forwards took 30–31 ms and the
    others 35–37 ms, while the graphs agree (7.4–7.6 ms on stock wA). Part R saw the same kind of spread (±17 % at
    1×512). Use the graph numbers at host-bound points.
  - **Single-round ratios at 1×2048, graph** (Part R's 5-round values in brackets): Ours 16x64 vs FourOverSix +1.7 %
    [+1.7 %]; Ours 8x64 vs FourOverSix on wB +4.7 % [+5.3 %].

### 06 — Experiment 3.2: GEMM latency (`06_gemm_latency.py`, worker `bench_gemm.py`)

- **Purpose:** kernel time per text-Linear shape at T = 128 … 8192 tokens (4×2048 is T = 8192), and per forward.
- **Command:** `$PAPER_PYTHON experiments/paper/06_gemm_latency.py`
- **How:** each call is one NativeLinear forward (the activation quantizer, then the GEMM); CUPTI gives each
  kernel's time (median of 20).
- **Configurations:**
  - stock_wA through the tile table: FourOverSix weights with the FourOverSix quantizer, and NVFP4 weights with the
    NVFP4 quantizer. NVFP4 and FourOverSix share this GEMM; only the quantizer kernel differs;
  - stock_wB;
  - the mixed kernel through the tile table, with the real TC 16x64 and 256x64 tags;
  - n8k64_wB with the real 8x64 tags.
- **The real tags:** per projection, the module with the most E0M3 tiles, as the tile-table tuning did.
- **Per-forward sums:** the per-projection times × module counts. The quantizer sums use the reuse measured in step
  05 (q/k/v and gate/up share one quantization).
- **Output:** `<out>/gemm/<model>.json`.
- **Time:** ≈ 2–5 min per model: artifact loading and hash checks take 20–80 s, then 168–504 kernel measurements
  at ≈ 0.4 s each. **≈ 12 min in all.**
- **GPU:** small (one module per projection).
- **Smoke (measured):** q_proj at T = 128 and 2048, five configurations: 24 s.
  - At T = 2048: stock wA 46.9 µs, mixed 16x64 49.1 µs (+4.7 %); n8k64_wB 51.6 µs against stock wB 46.5 µs
    (+11.0 %). Part R's GEMM overheads were +4.1 % and +11.1 %.
  - Quantizers: FourOverSix 23.8 µs, NVFP4 21.8 µs.
  - Step 05 recorded the reuse exactly as expected: k/v reuse q's quantization, up reuses gate's (4 quantizer
    launches per Llama layer for 7 GEMMs).

### 07 — tables (`07_tables.py`)

- **Purpose:** `<out>/tables/main.md` (8x64 and 16x64), `appendix.md` (256x64 and the per-shape GEMM detail) and
  `tables.json`.
- **Command:** `$PAPER_PYTHON experiments/paper/07_tables.py`
- **Tables:**
  - **Perplexity** and paired ΔNLL (nats per token = Δ log PPL) ± 2 SE over the same windows (token hashes
    compared), against FourOverSix and NVFP4.
  - **Downstream accuracies** and paired differences in percentage points ± 2 SE over the same examples; the mean
    over the five tasks has SE √(Σ SE²)/5.
  - **Prefill:** CUDA-graph ms (median of rounds) in the main tables. Eager ms in the appendix, with † for
    host-bound points. Ours / reference − 1 uses the same activation quantizer, paired within rounds.
  - **Checks:** equal `sample_digest` across the policies of each (model, task). A missing digest is also an error
    outside `--smoke`.
  - **GEMM per-forward sums** and ratios (16x64 and 256x64 against stock_wA; 8x64 against stock_wB and stock_wA),
    with the NVFP4-vs-FourOverSix quantizer sums.
- **Missing results show as '—'.**
- **Time:** seconds, CPU only.
- **Smoke (measured):** 0.07 s; every section renders from the smoke outputs (`PAPER_SMOKE_OUT/tables/`), labeled
  as smoke output.

## Running everything, and re-running parts

```bash
cd /home/dev/NVFP4-RaZeR
experiments/paper/run_all.sh                       # 00 ... 07, stops at the first failure; re-run to continue
experiments/paper/run_all.sh --smoke               # the smoke flow (Llama only), into PAPER_SMOKE_OUT
```

Single model or policy (each step skips what is finished; `--force` redoes it):

```bash
PY=/home/dev/.conda/envs/n16k64/bin/python
$PY experiments/paper/03_ppl.py --models phi4 --policies ours-16x64
$PY experiments/paper/04_downstream.py --models qwen27b --policies fo6 --tasks mmlu
$PY experiments/paper/05_prefill_latency.py --models mistral7b --units 8x64 --rounds 5
$PY experiments/paper/06_gemm_latency.py --models llama8b --force
$PY experiments/paper/01_calibrate.py --models llama8b --units 16x64 --train     # retrain instead of reusing
$PY experiments/paper/07_tables.py
```

- **A crashed lm-eval process:** re-running step 04 resumes that (model, policy) at the next unfinished task.
- **A crashed latency round:** re-running step 05 repeats only the missing (model, policy, round).

## Collecting results into the repository

```bash
$PY experiments/paper/collect_results.py          # PAPER_OUT -> results/paper (records and tables; see its docstring)
```

- **Collected:** the environment check, `commands.log`, and the calibration and export records. Also the per-window
  perplexity records, and the lm-eval records (gzipped compact JSON with per-example correctness, per-choice
  log-likelihoods and sample hashes). Also the latency and GEMM records, and the tables.
- **Not collected:** the artifacts, the maps and the per-command logs.

## Total time and the critical path

Everything runs on one GPU, one step after another, so the critical path is the sum.

| step | Llama-3.1-8B | Mistral-7B-v0.3 | Phi-4 | Qwen3.8-27B | all |
|---|---:|---:|---:|---:|---:|
| 00 check | | | | | < 1 min |
| 01 maps, reused (retrained) | s (26 min) | s (23 min) | s (44 min) | s (3.8 h) | < 1 min (5.3 h) |
| 02 artifacts | 6 min | 6 min | 11 min | 20 min | ≈ 45 min |
| 03 perplexity | 9 min | 9 min | 15 min | 64 min | ≈ 1.6 h |
| 04 downstream | 1.4 h | 1.6 h | 2.3 h | 9–10 h | ≈ 15 h |
| 05 prefill latency | 32 min | 32 min | 55 min | 4.2 h | ≈ 6 h |
| 06 GEMM latency | 2.5 min | 2.5 min | 2 min | 5 min | ≈ 12 min |
| 07 tables | | | | | seconds |
| **total, maps reused** | **≈ 2.2 h** | **≈ 2.4 h** | **≈ 3.7 h** | **≈ 15 h** | **≈ 24 h** |

- **With the 12 maps retrained** (`01 --train`): ≈ 29 h.
- **Uncertainty:** ≈ ±25 %, mostly in Qwen's lm-eval estimate. It is extrapolated from Llama's smoke, the Qwen
  probe and Task 3's 20-document runs.
- **The critical path is Qwen:** ≈ 15 of the 24 h, and within it the downstream step (≈ 9–10 h).
  - Of that, MMLU 5-shot on Qwen over the six policies is ≈ 3.5–5 h, and HellaSwag ≈ 3–4 h.
  - Both are slow for the same reason: Qwen's linear attention runs HF's PyTorch fallback, which is host-bound for
    short sequences.
- **A convenient order:** the three smaller models first, since each model's steps are independent of the others'.
  Their complete results arrive in ≈ 8.3 h:
  ```bash
  experiments/paper/run_all.sh --models llama8b,mistral7b,phi4
  experiments/paper/run_all.sh --models qwen27b
  ```

### The Qwen probe (feasibility, not a result)

**What it answered:** whether Qwen's forward can be CUDA-graph captured, and how long MMLU-shaped batches take.

- **Run:** `bench_prefill.py` on Qwen3.8-27B with the Parts 2–3 FourOverSix artifact (`auto_stock`), shapes 1×128,
  8×512, 8×1024, 4×2048 and 1×8192, 3 repetitions, into `PAPER_SMOKE_OUT/probe/`. It took 3 min.
- **Capture works:** Qwen's hybrid linear-attention forward captures at every shape, and the logits are bitwise equal
  to eager (with the mask handling above).
- **Host-bound points:** 1×128 (eager 469 ms, graph 57.5 ms) and 1×8192 (eager 3,861 ms, graph 1,813 ms). At 8192
  tokens, the PyTorch chunked delta rule loops over 128 chunks per layer.
- **GPU-bound points:** 8×512, 8×1024 and 4×2048, where graph and eager agree within 2 %.
- **Other records:** the quantizer reuse (the linear-attention in_proj_z/b/a reuse in_proj_qkv's quantization);
  peak 28.5 GiB.

## Design choices

- **Reuse before retraining.**
  - The committed maps are copied when their sha256 matches the manifest; `--train` retrains.
  - Retraining is deterministic, and Task 1 showed that it reproduces the committed maps bitwise on this GPU. Reuse
    therefore changes the time, not the result, and saves ≈ 5.3 h.
- **One artifact per (model, kind), placement-independent.**
  - There is no separate "wB artifact": the same file runs on stock_wA and stock_wB.
  - Both placements are covered by an ownership check (the baselines on both kernels; the maps on their own).
- **"Ours with NVFP4 activations" is an install option, not an artifact.**
  - `NM.install(..., activation_quantizer=...)` switches the quantizer and records the override.
  - Accuracy steps never pass it, and the latency tables label it.
- **The downstream step runs NativeLinear only, as requested.** Task 3 compared fake (c) with native on smoke
  subsets; fake is 3–5× slower (`results/lmeval_ready/READINESS.md`).
- **MMLU is lm-eval's own group aggregate:** micro-averaged, `weight_by_size`. The subjects and categories are kept.
  The dataset is pinned by revision.
- **The latency protocol is Part R's.** Fresh processes, shuffled rounds and within-round pairing, now with the tile
  table.
  - **Added: CUDA-graph replay, and a host-bound flag,** so points dominated by launch overhead are visible. Part R's
    1×512 spread (up to ±17 %) came from such points.
  - **The captured forward must be the eager one.** transformers switches its mask construction under capture, so
    the harness keeps the eager decision there, and checks the logits bitwise.
- **GEMM per-forward sums weight each projection by its module count**, using the most-E0M3 module's tags, as the
  tile table did. For the mixed kernels this is the conservative choice.
- **Nothing is selected or tuned on WikiText, C4 or the downstream tasks.** The maps come from the calibration data
  alone; the evaluation settings are lm-eval's defaults (plus MMLU 5-shot).

## Changes to shared code made for this flow

- **`run_lmeval_deploy.py`:**
  - `--num-fewshot TASK=N`;
  - group tasks (MMLU: lm-eval's aggregate, per-subject metrics, examples keyed `<subject>:<doc_id>`);
  - per-choice log-likelihoods of every multiple-choice example;
  - `--resume` at the task level;
  - the recorded few-shot count of a group's subtasks.
  - A variable-name clash is fixed: when a native policy ran first in a process, its forward-hook handle was
    overwritten, and the hook could not be removed at the end.
- **`lm_eval_datasets.py`:** pins `cais/mmlu`.
- **`sm120/mixfp4_sm120/model.py`:** `install(..., activation_quantizer=None)`. It records
  `activation_quantizer` and `activation_quantizer_override` in the install report. The default is unchanged: the
  artifact's own quantizer.
