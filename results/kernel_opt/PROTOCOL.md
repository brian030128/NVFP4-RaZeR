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

---

## Amendment 1 (optimization 1b): a cooperative 128 × 64 weights-on-B tile for mid T

Written 2026-09-30 after optimization 1's results (`REPORT.md`, 540e1da), before any registered GPU run of 1b. The hashes
and time are in `registration_1b.json`. Everything of optimization 1 not named here applies unchanged.

**Why.** After optimization 1, M1 still shows +18…+26 % GEMM overhead against stock_wA at T = 256, and +10…+15 % at
T = 512. The cause is the CTA count.
- At T = 256, Llama's down_proj runs 64 CTAs of 128 × 128 on 188 SMs: 52.3 µs, against stock_wA's cooperative
  128 × 64 tile at 32.9 µs.
- Optimization 1's ping-pong 64 × 64 build is slower there (59.0 µs): it runs 4 MMA warps per tile instead of 8.

**What changes.**
- **New build `n8k64_wB_n64`** (`sm120/mixfp4_sm120/configs.py`), using existing macros only (no kernel source changes):
  - the cooperative schedule (8 MMA warps) with a 128 (tokens) × 64 (weights) × 128 CTA tile;
  - 2x4 warps: each warp owns 64 tokens × 16 weight columns (4 m-atoms, 2 n-atoms). That is the per-warp shape, blob
    and 16-arm dispatch of `n8k64_wB_m64`, so each output has the same MMA sequence;
  - `MIXFP4_PERM_N=64`; expected census {0: 256, 2: 256}.
- **`KernelSet('mixed_wB')`** gains this build under the string key `'128x64'`. Only the table selects it; the fallback
  rule uses the int token widths only.
  - `select.py` adds `key_order` and `parse_key`.
  - The two consumers that sort width keys (`experiments/paper_extra/bench_decode.py`, `check_model_logits.py`) now
    sort with `key_order`. For int keys the order is unchanged.
- **The `mixed_wB` table rows are re-tuned** over the five builds with optimization 1's method:
  `tune_tiles.py --families mixed_wB --cold --update`, synthetic operands, the TC 8x64 maps' densest-module tags.
  - The `mixed` and `stock` rows and the table's top-level meta stay byte for byte.
  - Optimization 1's table is kept as `opt1/table_opt1.json`.

**Gates.** They are optimization 1's, re-run on the changed set, and a failure stops the work:
- **G1:** all configurations other than the four narrow ones and `n8k64_wB_n64` keep their before-build SASS. Checked by
  `check_sass.py` against `sm120/build` and the tm-opt-source builds; the four narrow ones are treated as new.
- **G2:** `n8k64_wB_n64`'s census, with no predicated OMMA.
- **G3:** `build.py --selftest` for `n8k64_wB_n64`, then `pytest sm120/tests/test_gemm.py sm120/tests/test_select.py`
  with the after builds (`n8k64_wB_n64` is in MIXED; test_select runs the family with the new key).
- **G4:** `check_bitwise.py --candidates n8k64_wB_n64 --set mixed_wB`, with the re-tuned table.
- **G5:** `check_model_logits.py` with the re-tuned table.

**Measurements.** Optimization 1's methods, with outputs under `/home/dev/n16k64_campaign/kernel_opt/opt1b/`:
- **M1:** `bench_ab_isolated.py --opt1-table results/kernel_opt/opt1/table_opt1.json`. It adds optimization 1's set
  (`auto_wB1`) to the configurations: stock_wA, stock_wB, n8k64_wB (before), auto_wB1 (optimization 1) and auto_wB (1b),
  typical and worst tags.
- **M2:** `ab_e2e.py --what prefill --out <opt1b>/e2e`. The policies are optimization 1's; ours-8x64-opt is now the 1b
  set.
- **M3, decode:** run only if the re-tuned table changes a choice at the decode buckets (T ≤ 16) of a shape of Llama,
  Mistral or Phi-4. Otherwise decode executes the same builds as optimization 1's M3, and those records stand. The
  table diff decides, and it is recorded.
- **Report:** `ab_report.py --src <opt1b> --tag ab1b`. It adds the 1b set's change against optimization 1's set to
  the tables.

**Before registration (disclosed, not results).** Exploratory, as the feasibility check:
- `n8k64_wB_n64` (2x4) was built, and so was a 1x8 variant: one n-atom per warp, a 4-arm dispatch, and
  `MIXFP4_LDSM_B=2`, since the default x4 LDSM fails CuTe's copy assert.
- A quick isolated cold CUPTI timing on 6 shapes × 6 token counts (scratch `quick_ab2.py`):
  - 0 of 144 outputs differed from sm120/build's n8k64_wB;
  - the 2x4 build beat the 1x8 one in 32 of 36 cells;
  - the 1x8 build would have been the best choice in 1 cell, by 0.6 %.
- The 1x8 variant was dropped and is not in `configs.py`.

**Note to amendment 1 (disclosed, 2026-09-30 16:23 UTC): M3 deferred by user request.**
- The user paused 1b, relayed by nvfp4-razer-c9, before its decode measurement.
- The orchestrator (`run_measurements.sh`, `ab_e2e.py`) was stopped during Qwen prefill round 4. The prefill process
  already running was left to finish.
- The remaining Qwen prefill processes ran as `ab_e2e.py --what prefill`, with the same settings and checks.
- No decode process was started, so no partial decode record exists.
- M3 will be run later as a decode-only run (`ab_e2e.py --what decode`, the same policies, settings, rotation and
  output directory), when the user says start. 1b is not reported as done until then.

---

## Amendment 2 (#2): frequency-aware dispatch — the all-E2M1 pattern first

Written 2026-09-30, before any registered GPU run of #2. The hashes and time are in `registration_2.json`. The user
decided to run it (relayed by nvfp4-razer-c9); optimization 1b's decode (M3) stays paused. Everything of optimization 1
not named here applies unchanged.

**Why.**
- The mixed kernels pay a fixed dispatch cost with every k_tile all-E2M1: +2.0–2.6 % over the no-dispatch build (C2,
  C3, and `e0m3/test_bc` +2.5 %).
- Each warp picks its k_tile arm through a balanced binary tree of log2(arms) compares (`dispatch_pattern`).
- In the paper's maps 82–95 % of all (warp, k_tile) patterns are pattern 0, i.e. all-E2M1 (`results/kernel_opt/dispatch`).
- All-E0M3 patterns are 0.00 % (16x64, 8x64) and at most 1.3 % (256x64).

**What changes.**
- **Hook** `MIXFP4_DISPATCH_FREQ=1` in the collective's `dispatch_pattern`:
  - if the pattern is 0, run arm 0 (one compare, the fall-through path);
  - else enter the unchanged tree, so every other pattern keeps its path plus one jump.
  - Unset, it is the upstream dispatch.
- **No all-E0M3 test.** The coordinator's first sketch tested all-E0M3 second; the histogram above makes that a compare
  added to the 5–18 % non-zero patterns, to help 0–1.3 %, so it is not done.
- **Every arm computes exactly as before.**
- **Builds:** `sm120/build.py --all --define MIXFP4_DISPATCH_FREQ=1` into `/home/dev/n16k64_campaign/kernel_opt/build_freq`,
  with the same configuration names. The define is recorded in each manifest as `extra_defines`; `--define` is a new
  optional build.py flag.
  - Dispatching builds covered: n16k64_wA, _8x1, _n64, _n32, _n16, _sk; n8k64_wB, _m64, _m32, _m16, _n64.
  - Measured: the deployed families, KernelSet('mixed') (16x64 and 256x64 maps) and KernelSet('mixed_wB') (8x64).
  - The Stream-K (_sk) and 8x1 builds are covered by the gates but not measured.
  - The tile table is unchanged, so the same widths are chosen before and after.
  - Optimization 1b's builds (`build`) and `sm120/build` are untouched.

**Gates** (a failure stops #2 and is reported):
- **G1′:**
  - n16k64_wA and n8k64_wB rebuilt from the hooked sources without the define have `sm120/build`'s patched and
    unpatched SASS (`build_hookcheck`).
  - In `build_freq`, every configuration without a dispatch (stock_*, n16k64_wA_nodisp) has its default SASS.
- **G2′:** every dispatching build in `build_freq` keeps its expected OMMA census, with no predicated OMMA (build.py
  enforces both).
- **G3′:**
  - `build.py --selftest --define MIXFP4_DISPATCH_FREQ=1` for n16k64_wA, n16k64_wA_n16, n8k64_wB and n8k64_wB_m16
    (PASS patched / FAIL unpatched);
  - `pytest sm120/tests/test_gemm.py sm120/tests/test_select.py` with `SM120_BUILD_DIR` = `build_freq`, all passing.
- **G4′:** `check_bitwise.py` against `sm120/build`, 0 differences, for both families:
  - `--family wA`: n16k64_wA, _n64, _n32, _n16, _8x1 and the `mixed` set from `build_freq`, on real 16x64 and 256x64
    modules plus synthetic 16x64 tags;
  - `--family wB`: n8k64_wB, _m64, _m32, _m16, _n64 and the `mixed_wB` set.
- **G5′:** `check_model_logits.py`, logits bitwise equal on all 4 models:
  - 16x64 and 256x64 artifacts: before = set:mixed from `sm120/build`, after = set:mixed from `build_freq`;
  - 8x64 artifact: before = n8k64_wB from `sm120/build`, after = set:mixed_wB from `build_freq`.

**Measurements:**
- **M1′: `experiments/kernel_opt/bench_ab2_isolated.py`,** the deviation-2 method (`bench_ab_isolated.run`, unchanged
  per repetition), all four models, every projection.
  - Tokens 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192.
  - Configurations:
    - stock_wA, stock_wB;
    - the 16x64 and 256x64 maps on the `mixed` set, from `sm120/build` (before) and `build_freq` (after);
    - the 8x64 map on the `mixed_wB` set, from `build` (optimization 1b, before) and `build_freq` (after);
    - typical and worst tags.
  - Check: after equals before bitwise on the timed operands.
  - Reported, as per-forward GEMM sums:
    - after vs before;
    - 16x64 and 256x64 vs stock_wA, before → after, at every T (requested by the coordinator);
    - 8x64 vs stock_wA and stock_wB.
- **C2′: `experiments/kernel_opt/c2_freq.py`,** 4096³:
  - stock_wA, nodisp; n16k64_wA default / freq with all-E2M1, real map and all-E0M3 tags;
  - the weights-on-B pair: stock_wB, n8k64_wB default / freq with 8x64 tags.
  - Modes b2b (C2's), isolated and sustained (NVML), each mode run for every configuration before the next mode.
  - Check: bitwise equal.
- **M2′ (conditional): end-to-end CUDA-graph prefill.**
  - It runs only if M1′ shows a per-forward GEMM gain of ≥ 1 % (well above M1's per-round spread of ≤ 0.7 pp) for a
    unit at T ≥ 128.
  - It would be registered as a further amendment, with its policies, before running.

**Before registration (disclosed, not a result).**
- The hook's n16k64_wA build (SASS 82eadf6d…, the same census as default) was timed once at 4096³, isolated, 3 rounds
  (scratch `quick_freq.py`). Outputs were bitwise equal.
- freq vs default: all-E2M1 +0.5 % (the rounds overlap), real map −1.2 %, all-E0M3 +1.6 % (the added jump).
- This suggests the fixed dispatch cost is not the tree's compares. M1′ and C2′ decide.

**Deviation 1 to amendment 2 (2026-09-30 18:12 UTC): the weights-on-A bitwise gate failed on a script assertion.**
- G4′ `--family wA` stopped 4 s after starting, before any comparison. `real_modules` asserted that the 256x64
  artifacts declare a 256x64 granule, but they store their tags as 16x64 granules (policy "TM-OPT+TC 256x64 as 16x64
  granules").
- The assertion now checks the family's granule (16x64 for weights on A, 8x64 for weights on B). Nothing else in the
  script changed.
- The chain resumes at G4′ (`experiments/kernel_opt/run_opt2_resume.sh`). The gates already passed (G3′ self-tests,
  G1′, G1′/G2′ SASS, G3′ pytest) stand.
- **The same misreading affected the dispatch-skip ceiling analysis** (f2297dc): it expanded the 256x64 masks by 16 a
  second time. Recomputed in `results/kernel_opt/dispatch` (correction noted there), the 256x64 figures quoted in this
  amendment's "Why" change only slightly:
  - all-E0M3 patterns are at most 1.2 % (not 1.3 %);
  - non-zero patterns are 5–19 % (not 5–18 %).
  - The design is unaffected.
