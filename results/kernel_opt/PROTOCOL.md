# Kernel optimization (branch `kernel-opt`) — protocol

Written 2026-09-30 on branch `kernel-opt` (cut from `tm-opt` at 74058af), before any registered GPU run of this
work; the hashes and time are in `registration.json`. Deviations and later optimizations are appended at the end.
Requested by the user (relayed by nvfp4-razer-c9, 2026-09-30).

## Rules from the request

- **Goal:** make FlipQuant (ours)'s mixed-format GEMM faster: n16k64_wA for the 16x64 and 256x64 maps, and n8k64_wB
  for the 8x64 maps.
- **No output bit may change.** Every new build must give outputs bitwise identical to the current kernels'. A change
  that cannot be bitwise (e.g. one that alters the accumulation order) is stopped and reported before it is adopted,
  because the paper's accuracy numbers would then need re-evaluation.
- **tm-opt's committed paper results are not changed.** This branch reports before vs after. The user decides whether
  the paper's latency numbers are re-measured.
- **No ncu, no hardware counters, no root, no clock locking.** Explanations come from the SASS census, no-dispatch
  ceilings and micro-benchmarks.
- **No data selection or tuning** on WikiText, C4 or zero-shot. Tile tuning uses synthetic operands only.

## Builds

- **Before:** `sm120/build`, the builds behind every paper result. Their library sha256s are in
  `results/paper/registration_gemm_isolated.json`; for example, n8k64_wB is `8d504d10…`.
- **After:** `/home/dev/n16k64_campaign/kernel_opt/build` (`SM120_BUILD_DIR`), all configurations built by
  `sm120/build.py` from the kernel-opt sources.

## Optimization 1: narrow token tiles for the weights-on-B kernel (8x64 maps)

**What changes.**
- New configurations `n8k64_wB_m64`, `n8k64_wB_m32` and `n8k64_wB_m16` (`sm120/mixfp4_sm120/configs.py`):
  - CTA tile M × 64 × 128, with M = 64, 32 or 16 tokens;
  - the ping-pong kernel schedule, with 4 MMA warps in a 1x4 arrangement;
  - the same granule (8 weight columns × 64 K), the same dispatch (16 joint arms per warp: 2 n-atoms × 2 k-blocks),
    and activations pinned E2M1.
- **Why the output cannot change:**
  - each output element is computed by the same MMA instruction sequence over the same k-tiles in the same order as
    in n8k64_wB;
  - the CTA tile only decides which outputs a CTA computes, not how each output is accumulated;
  - the narrow-N weights-on-A builds (n16k64_wA_n16/32/64) rely on the same argument and are verified bitwise
    (`sm120/tests/test_select.py`).
- **Kernel hooks.** These are compile-time hooks, inactive in every existing configuration
  (`sm120/kernel/LOCAL_CHANGES.md`):
  - `MIXFP4_TILE_M`;
  - `MIXFP4_PINGPONG`;
  - the narrow-M scale-factor path. The collective loads the whole 128-row SFA block and reads the CTA's sub-tile,
    mirroring the existing narrow-N SFB path.
- **Selection:** the width-selecting set `KernelSet('mixed_wB')`, i.e. kernel `'auto_wB'`, over
  {16: m16, 32: m32, 64: m64, 128: n8k64_wB}.
  - Its table rows (`sm120/configs/<gpu>.json`, key `mixed_wB`) come from
    `sm120/bench/tune_tiles.py --families mixed_wB --cold --update`. That is the fastest width per (shape, token
    bucket) by isolated launches on cold weights: the deviation-2 repetition, median of 20.
  - The tuner runs on synthetic operands, with each projection's tags taken from the densest module of the paper run's
    TC 8x64 map (`--maps8`).
  - The existing families' rows (`mixed`, `stock`) and the table's top-level meta are kept byte for byte.

### Gates (registered)

Every gate must pass before any measurement is reported as a result. A failure stops the work and is reported, with no
improvising.

- **G1, inert.** Every configuration is rebuilt from the kernel-opt sources, and its patched and unpatched SASS
  (`sass_sha256`, `unpatched_sass_sha256`) must equal the before build's.
  - The configurations in `sm120/build` are compared with those builds.
  - `n16k64_wA_8x1`, `n16k64_wA_sk` and `stock_wA_sk` are not in `sm120/build`. They are compared with builds made
    by the same `build.py` from a `git archive` of tm-opt 74058af.
- **G2, census.** The patcher's per-site OMMA census of each new build must equal its expected census:
  - {0: 4M, 2: 4M}, i.e. 256, 128 and 64 per site;
  - no predicated OMMA (`build.py` enforces both).
- **G3, self-tests.**
  - `build.py --selftest` for each new build: the upstream driver on randb tags must PASS patched and FAIL unpatched,
    and the library census must equal the driver's.
  - `pytest sm120/tests/test_gemm.py sm120/tests/test_select.py` with `SM120_BUILD_DIR` set to the after builds; every
    test must pass. The new builds are in `test_gemm.py`'s MIXED list, and `test_select.py` runs every family,
    including `mixed_wB`, with 8x64 maps and a batch-invariance test.
- **G4, kernel bitwise.** `experiments/kernel_opt/check_bitwise.py` (docstring), against n8k64_wB from `sm120/build`:
  - candidates: the three new builds, n8k64_wB rebuilt, and the `mixed_wB` set with the tuned table;
  - every distinct projection shape of all four models (Llama-3.1-8B, Mistral-7B-v0.3, Phi-4, Qwen3.8-27B);
  - real weights: the densest and the lower-median modules of each TC 8x64 artifact;
  - synthetic weights: all-E2M1, all-E0M3 and random 30 % tags, the last also with a bias;
  - 15 token counts from 1 to 8192;
  - through NativeLinear with FourOverSix activations, on four paths: the fused quantizer + GEMM, the reuse of a
    cached quantized input, the unfused path, and a CUDA-graph replay (real modules, T = 1, 16, 64).
  - Rule: every comparison is bitwise equal.
- **G5, model bitwise.** `experiments/kernel_opt/check_model_logits.py`: per model (all four), the TC 8x64 artifact
  installed on n8k64_wB (before) and on `auto_wB` (after).
  - Eager logits at 1x1, 1x16, 1x100, 1x2048 and 4x512 must be bitwise equal.
  - Every scoped Linear must be native and must have run.

### Measurements (registered)

- **M1, GEMM.** `experiments/kernel_opt/bench_ab_isolated.py`, the deviation-2 method
  (`results/paper/PROTOCOL_GEMM_ISOLATED.md`, deviation 1 included):
  - isolated launches, cold weights by rotation plus a 512 MiB read-flush, and CUPTI kernel times;
  - 3 rounds with rotated order, 30 repetitions and 3 warm-ups per configuration per round;
  - telemetry, the registered checks, and typical and worst tags.
  - **Configurations:**
    - n8k64_wB_{typical, worst} (before);
    - auto_wB_{typical, worst} (after);
    - stock_wA (FourOverSix on `auto_stock`) and stock_wB (FourOverSix on stock_wB), both from `sm120/build`.
  - **Tokens:** 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 and 8192.
  - **Scope:** all four models, every text-Linear shape.
  - **Added check:** before and after give bitwise equal outputs on the timed operands.
  - **Reported:**
    - per model and T, the per-forward GEMM sum (Σ over projections of modules × time);
    - after / before − 1, typical and worst;
    - both relative to stock_wA and stock_wB;
    - the width chosen per shape.
- **M2, prefill.** `experiments/kernel_opt/ab_e2e.py --what prefill` runs `experiments/paper/bench_prefill.py`
  unchanged.
  - **Policies:**
    - ours-8x64 (before: n8k64_wB);
    - ours-8x64-opt (after: `auto_wB` with `SM120_BUILD_DIR` = the after builds);
    - fo6 (`auto_stock`) and fo6-wB (stock_wB), both from `sm120/build`.
  - **Shapes:** all four models at the paper's prompt shapes (1x128 … 1x8192 and 4x2048), 7 repetitions.
  - **Rounds:** 5, with the policy list rotated by r − 1 in round r. The reported value is the median over rounds of
    each process's median, as in step 05.
  - **Checks:** the graph's logits equal eager's bitwise at every shape. Every library a process loaded is a build of
    the expected directory (by sha256).
- **M3, decode.** `ab_e2e.py --what decode` runs `experiments/paper_extra/bench_decode.py` unchanged, with the same
  policies.
  - **Models and settings:** Experiment D's three models (Llama, Mistral, Phi-4) and its settings (batch 1, 4, 16 ×
    prompt 512, 2048; 64 tokens).
  - **Rounds:** 5, rotated as in M2.
  - **Checks:** the graph's first 33 greedy tokens equal an eager StaticCache decode's; the build check as in M2.
- **Reporting.**
  - Before vs after as ratios, with the spread over rounds; no significance claims beyond it.
  - The fo6-wB comparison is no longer like-for-like after optimization 1: stock_wB has no narrow-M build, since
    CUTLASS's stock collective lacks the narrow-M SFA path. Its numbers are shown as recorded, labelled so.
  - The same-placement narrow reference is pending the user's decision.
- **Adoption.** The optimization is adopted on kernel-opt if G1–G5 pass. Its speed is reported whatever it is.

### Before registration (disclosed, not results)

Before this protocol, two exploratory runs were made on the GPU, as the feasibility check the request asked for. Neither
is a result:
- an isolated cold-L2 CUPTI A/B of the three narrow builds against n8k64_wB, stock_wA and stock_wB, on q_proj and gate
  shapes (scratch `quick_ab.py`);
- a 750-comparison bitwise check on 7 shapes with none, all and random tags and real Llama/Qwen TC maps (scratch
  `bitwise_wB.py`). All 750 were equal.
