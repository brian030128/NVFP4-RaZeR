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

## Amendment 2b (#2's M2′): end-to-end CUDA-graph prefill, default vs MIXFP4_DISPATCH_FREQ

Written 2026-09-30 after M1′ and C2′ finished (18:43 UTC) and before any M2′ run. The hashes and time are in
`registration_2b.json`. The coordinator relayed "finish #2 as registered … e2e only if it beats noise".

**Why M2′ runs.** Amendment 2 made M2′ conditional on a per-forward GEMM gain of ≥ 1 % for a unit at T ≥ 128 in M1′.
The typical tags meet it (`results/kernel_opt/opt2/ab2_tables.md`):
- 8x64: 20 (model, T) cells at T = 256 … 8192, on all four models; −1.00 to −1.55 %.
- 256x64: Llama-3.1-8B T = 256, Phi-4 T = 128, Qwen3.8-27B T = 256 and 512; −1.07 to −1.40 %.
- 16x64: Qwen3.8-27B T = 256; −1.09 %.
- In every one of these 25 cells the per-round range excludes 0; the least negative single round is −0.77 %.

**What runs.** `experiments/kernel_opt/ab2_e2e.py` runs the paper's per-process script `bench_prefill.py`
unchanged. It uses M2's checks (`ab_e2e.prefill_check`) and a build check per policy.
- **Policies:** each unit before and after, and one reference.

  | policy | map | kernel set | build directory | role |
  |---|---|---|---|---|
  | ours-16x64 | TC 16x64 | `auto` | `sm120/build` | before |
  | ours-16x64-freq | TC 16x64 | `auto` | `build_freq` | after |
  | ours-256x64 | TC 256x64 | `auto` | `sm120/build` | before |
  | ours-256x64-freq | TC 256x64 | `auto` | `build_freq` | after |
  | ours-8x64-opt | TC 8x64 | `auto_wB` | kernel-opt `build` (optimization 1b) | before |
  | ours-8x64-freq | TC 8x64 | `auto_wB` | `build_freq` | after |
  | fo6 | FourOverSix | `auto_stock` | `sm120/build` | the deployment reference |

  - `auto` is the `mixed` set.
  - The tile table is `sm120/configs`' (unchanged), so before and after choose the same widths.
- **Shapes:** all four models at the paper's prompt shapes (1x128 … 1x8192 and 4x2048), 7 repetitions.
- **Rounds:** 5, with the policy list rotated by r − 1 in round r. This is 140 processes, about 4 h.
- **Checks** (a failure stops the run):
  - the graph's logits equal eager's bitwise at every shape;
  - every library a process loaded is a build of its policy's directory, by sha256;
  - the `build_freq` builds carry `MIXFP4_DISPATCH_FREQ=1`.
- **Reported** by `ab2_e2e_report.py`, per model and shape (the M2 convention):
  - the median over rounds of each process's median;
  - after vs before per unit, with the range over rounds pairing round r of both policies;
  - both vs fo6.
  - No significance claims beyond the round range. Earlier prefill runs varied by about 1 % per process (D1b), which
    is the size of the effect expected here: M1′'s 0.4–1.5 % GEMM gains, diluted by the non-GEMM share.
- **Adoption is unchanged:** #2 is adopted on kernel-opt because G1′–G5′ passed. M2′ reports its speed.

**Deviation 1 to amendment 2b (2026-09-30 19:12 UTC): M2′ stopped by user request.**
- Relayed by the coordinator: the expected end-to-end effect (~0.3–1 %) is at the per-process noise level. #2's
  end-to-end effect will be measured together with A′ and A, as one cumulative end-to-end measurement registered when
  those land.
- The runner was stopped between processes at 19:12:12. The process it had just started (Mistral-7B-v0.3,
  ours-8x64-opt, round 1) ran to completion on its own at 19:12:46 and passes the registered checks.
- Completed: Llama-3.1-8B, all 7 policies × 5 rounds, and 5 policies of Mistral-7B-v0.3's round 1. That is 40
  records, all complete and passing the registered checks.
- They are kept in `/home/dev/n16k64_campaign/kernel_opt/opt2/e2e` (`PARTIAL.md` there), marked partial, and are not
  used for conclusions.

## Amendment 3 (A′): 256x64 maps on a 4-arm kernel with 32-row granules

Written 2026-09-30, after #2's M2′ was stopped (deviation 1 to amendment 2b) and before any registered GPU run of
A′. The hashes and time are in
`registration_3.json`.
- The coordinator (nvfp4-razer-c9) relayed the user's standing goal: bring the FlipQuant 16x64 GEMM family as close to
  stock_wA latency as possible, at every T. A′ is its second step, after #2.
- Optimization 1b's decode (M3) stays paused. The decisive-margin rule and the E0M3 amendment (i)–(iii) are not
  adopted.
- Everything of optimization 1 not named here applies unchanged.

**Why.**
- 256x64 maps run today on n16k64_wA, whose weight granule is one 16-row m-atom.
  - That is 2 flag bits per k_block, 4 per k_tile, and a 16-arm tree.
- A 4x2 warp of the 128-row CTA panel owns two m-atoms: weight rows 16w … 16w + 15 and 64 + 16w … 64 + 16w + 15.
  - A 256x64 tile covers two whole panels, so a 256x64 map always gives both m-atoms the same format.
- With the upstream parameter `MIXFP4_A_ATOMS_PER_GRANULE=2`, the two m-atoms become one granule that takes the first
  atom's flag. That is 1 bit per k_block, 2 per k_tile, and 4 arms.
  - Each arm executes the same MMA instructions, with the same operands and accumulation order, as the n16k64_wA arm it
    replaces. So outputs are bitwise equal on every map that is uniform over 128-row panels.
  - Per k_tile it reads half the flags, the tree is 2 compares deep instead of 4, and there are 4 arm bodies instead
    of 16.
  - #2 found the kernels issue-bound, with the dispatch's fixed cost not in the tree's compares. So fewer flag reads
    are the plausible lever.

**What changes.**
- **No kernel source change.** `sm120/mixfp4_sm120/configs.py` gains the configurations below: the four widths of the
  `mixed` family with the A granule set to 2 atoms (blob generator `A_ATOMS=2`).
  | configuration | width | expected census (E2M1 / E0M3 OMMAs) |
  |---|---|---|
  | n16k64_wA_g32 | 128 | 128 / 128 |
  | n16k64_wA_g32_n64 | 64 | 64 / 64 |
  | n16k64_wA_g32_n32 | 32 | 32 / 32 |
  | n16k64_wA_g32_n16 | 16 | 16 / 16 |
- **`KernelConfig.map_tile_rows = 128`** for them. The granule's rows are not contiguous, so a map must be uniform over
  whole 128-row panels.
  - `NativeLinear` accepts a map declared at a finer tile (the 256x64 artifacts store their tags as 16x64 granules)
    only after `artifact.uniform_flags` verifies that its tags are uniform over the panels, a partial last panel over
    its real rows. Otherwise it raises.
  - The artifact compatibility list uses the same rule.
  - `artifact.retile` re-declares a PackedWeight at a coarser tile after the same verification.
- **Routing:**
  - `KernelSet('mixed256')` is a new family of the four builds.
  - `model.install(kernel='auto_256')` selects it.
  - It reads the `mixed` family's tile-table rows (`select.TABLE_FAMILY`), so every call runs at the same width as
    today's path.
  - `'auto'` is unchanged: the builds live in `build_A1`, and `sm120/build` is untouched.
- **Builds:** `sm120/build.py --all` into `/home/dev/n16k64_campaign/kernel_opt/build_A1`, from the A′ sources.
  - CPU only, with no GPU visible, before registration.
  - Listed with their SASS hashes and source hashes in `registration_3.json`.
- **Tests:** `sm120/tests/test_g32.py` (new) and `sm120/tests/test_select.py` (the `mixed256` family's maps).
- **Gate scripts:**
  - `check_bitwise.py --family g32`;
  - `check_granule_map.py` (new);
  - `check_sass.py --before-roots` (the kernel-opt build directories of earlier optimizations).
- **Measurement scripts:** `bench_abA1_isolated.py`, `c2_g32.py`, `abA1_report.py`, `run_A1.sh`.

**Gates** (`experiments/kernel_opt/run_A1.sh`, in this order; a failure stops A′ and is reported):
- **G3″:** `build.py --selftest` for the four g32 builds into `build_A1`: PASS patched / FAIL unpatched.
  - The upstream driver derives its granule map from the thread-to-row layout, so it tags the non-contiguous granules
    correctly.
- **G1″ / G2″ (`check_sass.py`):**
  - G1″: every configuration other than the four g32 builds, as built in `build_A1`, has the patched and unpatched SASS
    of its before build. The before build is searched in `sm120/build`, then kernel-opt's `build` (optimization 1b),
    then `build_e0m3` (the XOR diagnostics), then `build_tmopt`.
  - G2″: the four g32 builds have their expected census, with no predicated OMMA.
- **G3″ pytest:** `test_gemm.py`, `test_select.py` and `test_g32.py`, all passing.
  - `SM120_BUILD_DIR` = `build_A1`; n16k64_wA from `sm120/build` is the reference.
- **G-span (`check_granule_map.py`):** the granule map, measured with the decode probe.
  - n16k64_wA executes every 16-row block's own tag.
  - Every g32 width executes blocks {8p + w, 8p + w + 4} for w < 4, taking the first block's tag.
  - So every granule lies inside one 128-row panel and one 256-row tile, and every 256x64 tile is a union of granules.
  - On a 48-row weight (Qwen3.8's partial panel), every real block executes its own tag.
- **G4″ (`check_bitwise.py --family g32`):** 0 differences against n16k64_wA from `sm120/build`.
  - Candidates: the four g32 builds and the `mixed256` set.
  - Weights: the real modules of the four TC 256x64 artifacts (densest and lower-median per shape), plus synthetic
    256x64 maps (all E2M1, all E0M3, random 30 %, random 30 % + bias).
    - Each weight is re-declared at 256x64 by `retile`. That checks each artifact really is a 256x64 map.
  - 15 token counts from 1 to 8192; the fused, reuse and unfused paths, and CUDA graphs.
- **G5″ (`check_model_logits.py --unit 256x64`):** logits bitwise equal on all 4 models at 1x1, 1x16, 1x100, 1x2048
  and 4x512.
  - Before = set:mixed (`sm120/build`); after = set:mixed256 (`build_A1`).

**Measurements:**
- **M1″ (`bench_abA1_isolated.py`):** the deviation-2 method, all four models, every projection.
  - Tokens 1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (M1′'s list).
  - Configurations:
    - stock_wA;
    - the 256x64 maps on `mixed` from `sm120/build` (before);
    - the same maps on `mixed256` from `build_A1` (after);
    - typical and worst tags.
  - Checks: after equals before bitwise on the timed operands, and the same width is chosen for every call.
  - Reported as per-forward GEMM sums: after vs before with the per-round range, and 256x64 vs stock_wA before → after
    at every T.
- **C2″ (`c2_g32.py`):** 4096³.
  - Configurations: stock_wA, nodisp, and n16k64_wA (`sm120/build`) and n16k64_wA_g32, each with all-E2M1 tags, the
    real 256x64 map (Llama-3.1-8B layer-0 o_proj) and all-E0M3 tags.
  - Modes: b2b, isolated and sustained.
  - Check: bitwise equal.
- No end-to-end run is part of this amendment. If M1″ shows a per-forward GEMM gain of ≥ 1 % at T ≥ 128, one is proposed.
- **Adoption** is proposed to the coordinator, not automatic: route 256x64 artifacts to `mixed256` (`auto_256`) if
  every gate passes and M1″ shows no per-forward regression beyond its per-round range at any T.

**Before registration (disclosed, not results).**
- The granule map was probed once, inline, on worktree builds of n16k64_wA_g32 and n16k64_wA_g32_n16. It was exactly
  the map G-span requires.
- The worktree builds were timed once (scratch `quick_g32.py`, 19:21–19:23 UTC). The method was isolated single
  calls with CUPTI, 3 rotated rounds × 15, with WARM weights (no rotation or flush, unlike M1″'s cold weights). Every
  output was bitwise equal to today's path.
  - 4096³, 128-wide builds, g32 vs n16k64_wA: all-E2M1 −0.7 % (116.6 vs 117.4 µs), the real 256x64 map −1.9 %,
    all-E0M3 −6.4 %. g32 runs all-E0M3 no slower than all-E2M1.
  - Llama-3.1-8B shapes with the real 256x64 map, at the `mixed` table's widths:

    | shape | T = 1 | T = 16 | T = 64 | T = 256 | T = 1024 |
    |---|---|---|---|---|---|
    | q_proj 4096x4096 | −12.8 % (16) | −12.4 % (16) | −6.9 % (32) | −6.8 % (64) | +0.6 % (128) |
    | down_proj 4096x14336 | −13.7 % (16) | −14.1 % (16) | −7.6 % (32) | −7.1 % (64) | −0.8 % (64) |
    | gate_proj 14336x4096 | −1.5 % (16) | −2.3 % (16) | −7.5 % (64) | −0.5 % (128) | −1.9 % (128) |

    The width is in parentheses.
  - With cold weights, small T is closer to memory-bound, so the registered M1″ may show less. These numbers are not
    results.

## Amendment 4: re-tune the tile tables with the deviation-2 method

Written 2026-09-30, after A′'s adoption (2b60b01) and before any GPU run of the re-tune. The hashes and time are in
`registration_4.json`.

The user approved it, relayed by the coordinator: "Tile-table re-tune with the deviation-2 method (cold, isolated), done
identically for the mixed rows (also used by mixed256) and the stock rows, so the comparison stays fair. Gates: the
width-equivalence tests. M1: full T list, 4 models, 16x64 and 256x64 vs stock_wA, new tables vs current tables.
Report the cells whose width changed and their effect."

**Why.**
- The `mixed` and `stock` rows of `sm120/configs/<gpu>.json` date from the paper run (2026-09-28). They were measured
  as CUPTI over back-to-back calls on L2-warm weights: 20 calls, one pass.
- M1′ and M1″ (cold, isolated) show that the width choice matters at mid T.
  - At T = 128 the table runs Llama/Mistral q, o and down (4096x4096, 4096x14336) at width 32 where stock runs 64.
    These cells are +6.6 to +11 % vs stock, and they keep 256x64 at +3.6 % there after A′.
  - At T = 512 it runs k and v at 32 where stock runs 64: +5 to +7.5 %.
- The warm tuning's same-width gaps differ from M1's cold ones by up to 5× (e.g. +20 % vs +4 % in the same cell).
- A family's widths are bitwise interchangeable (tests/test_select.py), so any table is output-neutral.

**What changes.**
- **`sm120/bench/tune_tiles.py`** gains backward-compatible `--rounds R` / `--iters I` (with `--cold`): R rotated rounds
  × I isolated launches per width, taking the median of the per-round medians. The defaults (1, 20) are the earlier
  `--cold`.
- **One tuning run** (`--families mixed,stock --cold --rounds 3 --iters 30`) covers both families in the same process,
  with the same method, the same shapes (every Linear shape of the four models) and the same buckets (1 … 8192).
  - Builds: `SM120_BUILD_DIR` = `build_freq`. These are #2's builds, the 16x64 path; its stock builds have
    `sm120/build`'s SASS (G1′).
  - Mixed tags: the tuner's rule (per projection, the module with the most E0M3 tiles), from the paper artifacts' 16x64
    maps. Those maps are byte-identical to the ones the current table used.
  - Output: `results/kernel_opt/retune/tables/<gpu>.json`, plus the raw file with the per-round values. The tracked
    table is unchanged until adoption.
  - `mixed256` reads the `mixed` rows (`select.TABLE_FAMILY`), so it gets the new rows too.
- **`check_model_logits.py`** gains `--before-table` / `--after-table` and `--unit fo6`, and records both sets' widths.
- **New scripts:** `bench_ab_retune.py`, `abR_report.py`, `run_retune.sh`.

**Gates** (a failure stops the re-tune):
- **GW1:** `pytest sm120/tests/test_select.py` with `SM120_BUILD_DIR` = `build_freq`, and again with `build_A1`, all
  passing. It checks that every family's widths are bitwise equal, and batch invariance.
- **GW2:** `check_model_logits.py` on all 4 models × 5 shapes. The logits must be bitwise equal between the current and
  the new table for:
  - 16x64 on set:mixed (`build_freq`);
  - 256x64 on set:mixed256 (`build_A1`);
  - FourOverSix on set:stock (`sm120/build`).
- **In M1‴:** every new-table call's output equals its current-table call's bitwise on the timed operands.

**Measurement M1‴ (`bench_ab_retune.py`):** the deviation-2 method, all four models, every projection, tokens 1 … 8192
(the 12 counts).
- Configurations, each under the current and the new table:
  - stock_wA;
  - 16x64, typical and worst (set:mixed from `build_freq`);
  - 256x64, typical and worst (set:mixed256 from `build_A1`).
- Reported by `abR_report.py`:
  - per unit, new vs current, as per-forward sums with the round ranges;
  - 16x64 and 256x64 vs stock_wA, with both on the current table → both on the new table;
  - every (projection, T) cell whose width changed, with its time change.
- **Adoption** is proposed, not automatic: replace the tracked table's `mixed` and `stock` rows with the new ones if no
  unit's per-forward sum regresses beyond its round range at any (model, T). The result is reported either way.

**Before registration:** there was no GPU run of the re-tune.

**Amendment 4's result (2026-09-30 22:20 UTC):** every gate passed, but M1‴ showed regressions beyond the round range.
- They were at T = 32–64 on every unit, and for stock at T = 128: 9 to 16 (model, T) cells per unit.
- There were also gains at T = 256 and T = 1024.
- So the new rows are not adopted, and the tracked table is unchanged (`results/kernel_opt/retune/REPORT.md`).
- A disclosed diagnostic (`results/kernel_opt/retune/diag/`) found the cause. `tune_tiles.py --cold` flushes L2 and
  then runs the GEMM on activations quantized before the flush, so they are cold. M1 and inference quantize after
  the flush, so they are warm.
  - The tuner's order adds 1.0–2.6 µs at small T and re-ranks the widths.
  - M1's order reproduces M1's times and the current table's choices at those cells.

## Amendment 4b: the re-tune with the activations warm

Written 2026-09-30, after amendment 4's result and before any GPU run of 4b, except the smoke test disclosed below.
Hashes: `registration_4b.json`. It is the same re-tune the user approved, measured under the deviation-2 method's
actual condition.

**What changes from amendment 4:**
- `tune_tiles.py --act-warm`: with `--cold`, the kernel's activation quantizer (FourOverSix rows) runs after the flush
  and before each timed GEMM, which reads those fresh activations. Only the weights are cold.
- The tuning run adds `--act-warm`. Everything else of amendment 4 is unchanged: both families in one run, 3 rotated
  rounds × 30, `build_freq`, the paper artifacts' 16x64 tags.
- The chain (`run_retune_b.sh`) writes to `results/kernel_opt/retune/b` and `/home/dev/n16k64_campaign/kernel_opt/retune_b`.

**Gates, M1‴, reporting and the adoption criterion:** exactly amendment 4's, with the 4b table as "new". The current
table is still the tracked one.

**Before registration (disclosed):** the new option was smoke-tested once on Llama-3.1-8B's shapes, stock family, 1
round × 8, into a scratch directory.
- It reproduced M1's times and choices at the diagnostic's cells. For example, stock 4096x4096 at T = 64: width 32,
  8.70 µs.
- Nothing from it is used.

## Amendment 5 (#4): epilogue tile and tile-scheduler order, for the mixed 16x64 family and stock alike

Written 2026-10-01, after B's exploration (ec2fbb1) and before any registered GPU run of #4. The hashes and time are in
`registration_5.json`. B (per-warp run tables) was explored and not registered
(`results/kernel_opt/B/EXPLORATION.md`), so it has no amendment and #4 takes the number 5.

The user approved it, relayed by the coordinator: "#4, epilogue tile / raster-swizzle for the 16x64 family, with
stock_wA tuned the same way. Same reporting." Earlier gaps and decisions are unchanged:
- the 4b table is not adopted (its adoption awaits the user);
- #1b M3, the margin rule and E0M3 (i)–(iii) stay on hold.

**Why** (disclosed exploration, `results/kernel_opt/4/exploration/`; warm activations, cold weights, 3 × 30; every
variant bitwise equal to its family's default):
- **Epilogue tile.** Auto resolves to 64 × 32 for these tiles: the 64 × 32 builds have the auto builds' SASS, and they
  timed within ±0.25 %, a noise control. (`quick_E_part2.out` labels auto "128x32"; the SASS shows it is 64 × 32.)
  - The 64 × 64 tile keeps 4 mainloop stages; shared storage grows from 88,064 to 100,352 bytes.
  - 64 × 64 was faster for both families at width 128 (Llama shapes, T = 256 / 1024 / 4096): mixed −0.05 to −1.73 %,
    stock −0.09 to −0.95 %.
  - 128 × 64 costs mainloop stages (stock 4 → 2) and was +1 to +33 % slower.
  - At width 64 (CTA tile 128 × 64) the 64 × 64 tile costs a mainloop stage (6 → 5, both families). Timed after
    `build_4` was built (`quick_E3.py`, 8 cells where the table runs width 64):
    - −0.1 to −2.1 % mixed and −0.04 to −1.0 % stock on 7 cells;
    - Qwen down (5120x17408) at T = 128: +4.9 % mixed, +3.6 % stock.
    - So the sets use the tile at width 128 only.
- **Tile-scheduler raster order and swizzle.** The persistent scheduler takes both at run time. Most (shape, T) cells
  move both families by similar small amounts (−0.1 to −1.2 %).
  - At T = 128, where mixed runs narrow (width 32) and stock at 64, rastering along M gains mixed −2.8 % (4096x4096)
    and −3.6 % (4096x14336) against stock's −0.8 to −1.0 %.
  - Rastering along M costs +3.5 % at T = 512 for 14336x4096. So the choice is per (shape, T).

**What changes.**
- **Epilogue tile hook.** In `src/mixed_nvfp4_gemm.cu` (vendored; LOCAL_CHANGES updated), `-DMIXFP4_EPI_TILE_M/N`
  sets the epilogue tile of any warp arrangement; `sm120/csrc` does the same for the stock builds. Unset, the choice
  is upstream (identical SASS).
- **New builds:** n16k64_wA_e64 and n16k64_wA_n64_e64, and stock_wA_e64 and stock_wA_n64_e64 (the 64 × 64 tile at
  widths 128 and 64).
  - The width-64 builds are built and gated (G2, pytest, G4) but used by no set (see Why).
- **Sets:** `mixed_e` = {16: n16k64_wA_n16, 32: n16k64_wA_n32, 64: n16k64_wA_n64, 128: n16k64_wA_e64}, and `stock_e`
  the same with the stock builds.
  - They read the `mixed` / `stock` width rows, so every call runs at today's width.
  - Only the width-128 build and the scheduler setting differ from today's sets.
- **Scheduler, host side only:**
  - `sm120/csrc`: `sm120_gemm_ex` / `sm120_linear_ex` take (raster, swizzle) per call, with no global state.
  - `Kernel.gemm` / `gemm_ptr` take `schedule=`; `KernelSet.schedule(m, k, t)` reads the table's optional `schedule`
    rows; NativeLinear and the isolated harness pass it on every path.
  - Without schedule rows every call uses (0, 1), the scheduler's defaults, exactly as before.
- **`tune_tiles.py --schedule`:** tunes (raster, swizzle) per (shape, bucket) at each family's table width, with the
  amendment 4b method (cold weights, activations quantized after the flush, 3 rotated rounds × 30), for `mixed_e` and
  `stock_e` in one run.
  - A setting replaces (0, 1) only if its median is at least 0.5 % below the default's and every one of its rounds is
    below every round of the default's (a decisive margin).
  - The output is the tracked width table plus `schedule` rows: `results/kernel_opt/4/table/`.
- **Scripts:** `check_bitwise.py --table`; `check_sass.py` checks new unpatched builds (E2M1 only, no predication);
  `bench_ab4_isolated.py`; `ab4_report.py`; `run_4.sh`.
- **Builds:** `build.py --all` into `/home/dev/n16k64_campaign/kernel_opt/build_4` from the #4 sources (CPU only,
  before registration).

**Gates** (`run_4.sh`; a failure stops #4):
- **G3:** `build.py --selftest` for n16k64_wA_e64 and n16k64_wA_n64_e64 (the stock builds have no self-test).
- **G1 / G2 (`check_sass.py`):**
  - G1: every other configuration keeps its before build's SASS (`sm120/build`, kernel-opt `build`, `build_e0m3`,
    `build_A1`).
  - G2: the two mixed builds have their census; the two stock builds are E2M1-only; nothing is predicated.
- **Schedule tuning (SCHED_TUNE):** produces the table; it is not a gate.
- **G3 pytest:** on `build_4`:
  - `test_gemm.py`, with the e64 builds;
  - `test_select.py`: `mixed_e` and `stock_e` widths bitwise equal;
  - `test_select.py`'s new `test_schedules_bitwise_equal`: every setting (raster 0/1/2 × swizzle 1/2/4/8), at every
    width of `mixed_e` and `stock_e`, bitwise equal to (0, 1) through NativeLinear. So the gate does not depend on
    which settings the tuning picks.
- **G4:** `check_bitwise.py --family wA`, n16k64_wA_e64, n16k64_wA_n64_e64 and set:mixed_e with the new table (so
  every schedule row runs), 0 differences against n16k64_wA from `sm120/build`.
- **G5:** logits bitwise equal on all 4 models × 5 shapes:
  - 16x64: set:mixed (`sm120/build`) vs set:mixed_e with the new table;
  - FourOverSix: set:stock vs set:stock_e with the new table.

**Measurement M1 (`bench_ab4_isolated.py`):** the deviation-2 method, 4 models × tokens 1 … 8192.
- Configurations:
  - stock_wA (before) vs stock_e with the new table (after);
  - 16x64 typical and worst on `mixed` (`sm120/build`, before) vs `mixed_e` with the new table (after);
  - #2's freq path (`build_freq`) as a reference;
  - the 256x64 path (TM-OPT+TC 256x64 on `mixed256` from `build_A1`, A′, current table), typical and worst, as a
    reference. #4 does not change it; it gives the 256x64 gap to stock before and after.
- Checks: after = before bitwise on the timed operands; the same widths.
- Reported by `ab4_report.py`, per model and T:
  - after vs before for stock and 16x64;
  - the 16x64 gap vs stock_wA, both before → both after;
  - the 256x64 gap vs stock_wA before and vs stock after;
  - the scheduler settings the calls used.
- #4 is measured on the default-dispatch builds, so its effect is separate from #2's. #2's freq builds are timed
  alongside for reference.

**Adoption** is proposed to the coordinator, not automatic. Adopt the width-128 64 × 64 builds and the schedule rows
for both families if no unit's per-forward sum is slower beyond its round range at a (model, T) where anything
changed.

**Before registration (disclosed, not results).**
- The exploration builds (`build_E`, worktree `wtE`) and scripts are in `results/kernel_opt/4/exploration/`.
- The raster / swizzle exploration used a first version of the knob (a library-global setter). It was replaced by
  per-call parameters before registration; the device code is identical.
- The width-64 timing (`quick_E3.py` / `.out`, see Why) ran on the `build_4` libraries after `build.py --all`.
- Smoke tests, run once into a scratch directory; nothing from them is used:
  - `tune_tiles.tune_schedule` on two Llama shapes at buckets 16 and 256, 1 round × 3. It wrote a table.
  - `test_schedules_bitwise_equal`: 6 passed.
  - `bench_ab4_isolated.py` on Llama k_proj and o_proj at T = 16 and 256, 1 round × 3, with that table and two rows
    forced to non-default settings. The settings reached the calls, and all 60 bitwise checks were equal.

## Amendment 6 (t0): the cause of "nodisp is slower than stock_wA", and its fix — the blob's site-0 prmt tags dropped

Written 2026-10-01, before any registered GPU run of t0. The hashes and time are in `registration_6.json`.

The user asked, through the coordinator: "find the cause of 'n16k64_wA_nodisp is slower than stock_wA' (+0.8 to +1.8 %
in C3/C2; +1.2 % at 4096^3 in your #2 split) and fix it, bitwise-safe". The pending decisions stay pending: #4 adoption,
the 256x64 extension, 4b, B′ and the cumulative e2e.

**Cause** (disclosed exploration, `results/kernel_opt/t0/exploration/`):
- **SASS, CPU only.** stock_wA and n16k64_wA_nodisp from `build_4` (same sources, same epilogue tile) were compared.
  - The MMA warps' k_tile loop is 91 vs 95 instructions. It is identical opcode for opcode except +4 PRMT in nodisp.
  - Over the whole kernel the difference is +8 PRMT and −5 other instructions. Loads, barriers, syncs, the producer
    and the epilogue are the same.
- **The 4 PRMTs** are `PRMT Rd, Rs, 0x3210, Rs`: identity copies of the A scale-factor register.
  - The MMA blob emits a `prmt.b32` before every MMA's scale operand. It is the site tag that tells the SASS patcher
    which OMMAs to switch to E0M3; its selector names the site.
  - ptxas keeps one per (m-atom, k_block), so 4 per k_tile and MMA warp.
  - The real kernel's arms carry the same 4 per k_tile, the all-E2M1 arm included.
- **The MMA order also differs.** The blob is m-major; CUTLASS issues n outer with the m-atoms alternating, so stock
  has 16 OMMAs with operand `.reuse` per k_tile against nodisp's 2.
- **Timing** (`quick_N.py`): the M1 condition plus back-to-back, 3 rounds, 7 cells from 4096x4096 at T = 16 to
  4096³; every output bitwise equal to stock_wA. Against stock_wA:
  - nodisp: +0.9 to +2.0 %;
  - tags dropped: −0.6 to +0.3 %;
  - CUTLASS's order with tags: +0.5 to +1.8 %;
  - both: −0.4 to +0.4 %.
- So the tags are the whole gap, and the order is not part of it.

**What changes.**
- **`sm120/kernel/scripts/gen_mixed_mma_blob.py`** (vendored, now locally modified):
  - `TAG0=0` drops the tags of site-0 MMAs, which then read their scale word directly. The E0M3 sites keep theirs.
  - `ORDER=n` issues the MMAs in CUTLASS's order. It is diagnostic only, used by the exploration.
  - With neither set, the output is byte-identical (checked for the 7 blob shapes in use).
- **`sm120/kernel/scripts/patch_mixed_nvfp4_gemm.py`** (vendored, now locally modified): `--untagged-site0` reads
  each OMMA's site from the reaching definitions of its scale register over the kernel's control-flow graph.
  - All reaching definitions are tags of one site s: site s. None is an E0M3 tag: site 0. Anything else is an error.
  - Without the flag, the patcher is unchanged.
  - Checked on all 74 existing tagged builds in 6 build directories: the analysis gives every OMMA the same site as
    the strict parser.
- **`LOCAL_CHANGES.md` / `.patch`** cover both scripts; the patch reproduces all four modified files from upstream.
- **`sm120/build.py`** passes `--untagged-site0` exactly for configurations generated with `TAG0=0`, and requires
  them to declare their census.
- **New builds (`configs.py`):**
  - n16k64_wA{,_n64,_n32,_n16}_t0 and n16k64_wA_g32{,_n64,_n32,_n16}_t0: each base plus `TAG0=0`, with the same
    census.
  - n16k64_wA_nodisp_t0: the diagnostic ceiling.
- **Sets:** `mixed_t0` and `mixed256_t0`, reading the `mixed` width rows.
- **Builds**, CPU only, from the t0 sources, before registration:
  - `build_T`: all 37 configurations. `--all` built the first 4; the remaining 33 were built by 8 parallel
    `build.py --config` processes. `--all` is the same per-configuration build in a loop.
  - `build_Tfreq`: the four 16x64 t0 builds with `MIXFP4_DISPATCH_FREQ=1` (#2).
- **Tests:**
  - `test_gemm.py` with the 16x64 t0 builds and `test_g32.py` with the g32 t0 builds;
  - `test_select.py`: its random 256x64 maps for every `mixed256*` family, and batch invariance for the t0 sets.
- **Scripts:** `check_sass.py --only-new`, `bench_abT_isolated.py`, `c2_t0.py`, `abT_report.py`, `run_T.sh`.

**Gates** (`run_T.sh`; a failure stops t0):
- **G3:** `build.py --selftest` for the 8 t0 builds: the upstream self-test, PASS patched and FAIL unpatched under
  random tagging, with the driver patched by `--untagged-site0`.
- **G1 / G2 (`check_sass.py`):**
  - G1 on `build_T`: the 28 existing configurations keep their patched and unpatched SASS. The before roots are
    `sm120/build`, kernel-opt `build`, `build_e0m3`, `build_A1` and `build_4`.
  - G2: the 9 new builds have their census and nothing is predicated; `build_Tfreq`'s four likewise.
- **G3 pytest:** `test_gemm.py`, `test_select.py` and `test_g32.py` on `build_T`; `test_select.py -k mixed_t0` on
  `build_Tfreq`.
- **G4 (`check_bitwise.py`):** 0 differences against n16k64_wA from `sm120/build` for:
  - the 16x64 t0 builds and set:mixed_t0, on `build_T` and on `build_Tfreq`;
  - the g32 t0 builds and set:mixed256_t0, on `build_T`.
- **G5:** whole-model logits bitwise equal on 4 models × 5 shapes:
  - 16x64: set:mixed vs set:mixed_t0;
  - 16x64 freq: set:mixed (`build_freq`) vs set:mixed_t0 (`build_Tfreq`);
  - 256x64: set:mixed256 (`build_A1`) vs set:mixed256_t0 (`build_T`).

**Measurement M1** (`bench_abT_isolated.py`): the deviation-2 method, 4 models × tokens 1 … 8192, the current tile
table.
- Configurations:
  - stock_wA, the reference;
  - nodisp → nodisp_t0 (FourOverSix artifact);
  - 16x64 typical and worst, default dispatch (`sm120/build` → `build_T`);
  - 16x64 typical and worst, #2's dispatch (`build_freq` → `build_Tfreq`);
  - 256x64 typical and worst (`build_A1` → `build_T`).
- Checks: after = before bitwise on the timed operands, and nodisp_t0 = stock_wA; the same widths.
- Reported by `abT_report.py`: after vs before per unit, and each unit's gap vs stock_wA before → after.

**C2‴** (`c2_t0.py`): 4096³, back-to-back, isolated and sustained, 3 rounds. Configurations: stock_wA, nodisp,
nodisp_t0, and {default, t0, freq, freq t0} × {all-E2M1, real map, all-E0M3}.

**Adoption** is proposed to the coordinator, not automatic: adopt the t0 builds for 16x64 and 256x64, with both
dispatch variants, if no unit is slower beyond its round range at any (model, T).

**Before registration (disclosed, not results).**
- The exploration builds (`build_N`, worktree `wtN`), `quick_N.py` / `.out`, `sass_loops.py` and the loop dumps are
  in `results/kernel_opt/t0/exploration/`.
- The two `ORDER` builds of the exploration (n16k64_wA_nodisp_on and _t0on: nodisp with `ORDER=n`, and with `TAG0=0`
  as well) were dropped from `configs.py` before registration.
- `validate_cfg.py` / `.out` is the check of the reaching-definitions analysis on the 74 existing tagged builds.
- On the finished `build_T`, `check_sass.py` was run once, CPU only, as a pre-check before registering. It passed
  G1 for the 28 existing configurations and G2 for the 9 new ones. The chain runs it again as the gate.
- One timing probe (`quick_T.py` / `.out`): the real dispatch kernels (n16k64_wA from `sm120/build` and `build_freq`
  vs their t0 builds), Llama-3.1-8B's o_proj map and an all-E2M1 map, 4096x4096 at T = 256 / 1024 / 4096, M1's
  condition, 3 × 30. Every output was bitwise equal.
  - t0 vs tagged: −0.3 to −1.2 % (default dispatch) and −0.0 to −0.8 % (#2's dispatch) on the real map;
  - −0.8 to −1.1 % and −1.2 to −1.6 % on the all-E2M1 map.
  - Nothing from it is used.

## Amendment 6b (t0): the patcher resolves jump tables; the t0 chain runs again from the start

Written 2026-10-01, after amendment 6's chain stopped at G3, and before any further registered GPU run of t0. The hashes
and time are in `registration_6b.json`.

The stop and why it happened are recorded in `results/kernel_opt/t0/stop_g3/STOP.md` (df73edf); the record stays.

The coordinator approved the fix proposed there, with conditions:
- an unresolvable BRX stays a hard error;
- the 74 existing builds' site assignment is shown unchanged;
- the t0 directories are rebuilt from scratch, with manifests;
- re-registration comes before the chain re-runs from the start;
- the self-test gate is kept.

**What changes from amendment 6.** One thing: `patch_mixed_nvfp4_gemm.py --untagged-site0` resolves indirect branches.
- ptxas can compile a switch to `LDC Rx, c[0x2][..]; BRX Rx imm`. CUTLASS's warp-role dispatch in the upstream
  self-test driver is one; our library kernels contain none.
- The patcher reads the function's `.nv.constant2` section from the cubin (`cuobjdump -xelf`). Each BRX then branches
  to (word + its next PC + imm) for every word of the section, a superset of its real targets.
- A BRX without a table, or a computed target that is not an instruction, is an error.
- In the self-test kernels the table is the role switch: 4 words, e.g. `[0x16b0, 0x1050, 0x25e0, 0x1050]`.
  - 0x1050 is the instruction after the BRX.
  - 0x16b0 and 0x25e0 each follow an unconditional branch, and no direct branch reaches them.
- `LOCAL_CHANGES.md` / `.patch` are updated, and the patch still reproduces all four modified files from upstream.
- Everything else is amendment 6's: sources, builds, sets, scripts, gates, M1, C2‴ and the adoption criterion.

**Validation** (CPU only; `results/kernel_opt/t0/6b/validate_6b.py`, `.out`):
- On the 74 existing tagged mixed libraries (6 build directories), the analysis gives every OMMA the strict parser's
  site. None of them has a BRX.
- On the 21 existing tagged self-test executables, all of which have the BRX, it gives every OMMA the strict parser's
  site. This checks the new path against ground truth.
- The 8 t0 self-test executables left by the stopped G3 are classified with equal per-site counts. Each equals its
  library's census, e.g. n16k64_wA_t0 512 / 512.

**Builds.** The patcher is a recorded build input of every configuration, so both directories were rebuilt from the
6b sources, CPU only, before registration:
- `build_T`: all 37 configurations, 10 parallel `build.py --config` processes;
- `build_Tfreq`: the four 16x64 t0 builds with `MIXFP4_DISPATCH_FREQ=1`.

Every manifest's recorded sources equal the files registered in `registration_6b.json`. All 41 builds have the patched
and unpatched SASS that `registration_6.json` recorded, so the change alters no library.

**The chain.** `run_T.sh` runs from the start, unchanged. The stopped run's output directory was moved to
`/home/dev/n16k64_campaign/kernel_opt/T_stop_g3` and is kept.

## Amendment 7: adoption of t0 (16x64), #4 (both families) and the 4b widths (both families)

Written 2026-10-01, before any registered GPU run of this amendment. The hashes and time are in `registration_7.json`.

**The user's decisions**, relayed by the coordinator: "go on all pending decisions". This amendment carries out 1, 3
and 5 of them.
1. **Adopt t0 for the 16x64 family, for both dispatch variants (default and freq).** This is a disclosed deviation
   from amendment 6's adoption criterion. Every 16x64 median improves, and the flagged cells are ≤ +0.35 %, about the
   chance level for 3 rounds (`results/kernel_opt/t0/REPORT.md`). t0 is not adopted for 256x64.
2. (not this amendment)
3. **Adopt #4 for both families:** the 64 × 64 epilogue tile at width 128, plus the per-call scheduler rows.
4. (amendment 8: #4 on the 256x64 path)
5. **Adopt the 4b tile table for both families,** a disclosed deviation from amendment 4's criterion. 18 of 4b's 20
   flagged cells are identical computations, and stock must be tuned as well as the mixed kernels for an honest
   comparison. The schedule rows are re-tuned at the 4b widths.

Still on hold: #1b M3, the decisive-margin rule, E0M3 (i)–(iii). The paper numbers (tm-opt) stay untouched until the
user decides to re-measure.

**What changes.**
- **New builds (`configs.py`):**
  - n16k64_wA_e64_t0: #4's width-128 build without the site-0 tags. It joins the t0 variants, census {0: 512, 1: 512}.
  - n16k64_wA_g32_e64: A′'s width-128 build with the 64 × 64 epilogue tile, census {0: 128, 1: 128}, built now for
    amendment 8.
- **New sets (`select.py`):**
  - `mixed_ko` = {16: n16k64_wA_n16_t0, 32: …_n32_t0, 64: …_n64_t0, 128: n16k64_wA_e64_t0};
  - `stock_ko` = `stock_e` = {16, 32, 64: the stock builds; 128: stock_wA_e64}.
  - Both read the 'mixed' / 'stock' width rows of the adopted table `sm120/configs/<gpu>.ko.json` (`TABLE_FILE`).
  - The paper sets and the paper table `sm120/configs/<gpu>.json` are unchanged.
- **The adopted table** is 4b's widths (`results/kernel_opt/retune/b/tables/`) plus `schedule` rows for `mixed_ko`
  and `stock_ko`. The rows are tuned by `tune_tiles.py --schedule` at the 4b widths with amendment 5's method: cold
  weights, activations quantized after the flush, 3 rotated rounds × 30, and the decisive-margin rule.
  - Every row is re-tuned, not only those whose width 4b changed. The mixed builds changed too (t0), and the cost is
    the same.
  - One set of rows serves both dispatch variants; the scheduler order is host-side.
  - The chain writes the table to `results/kernel_opt/7/table/`. Adoption copies it to
    `sm120/configs/<gpu>.ko.json` after the gates pass.
- **Routing (`model.py`):**
  - `install(kernel='auto')` runs 16x64 maps on `mixed_ko`, and `'auto_stock'` (FourOverSix, NVFP4) runs on
    `stock_ko`.
  - Either falls back to the paper set when its builds are not in the build directory, with a routing note.
  - 256x64 maps stay on `mixed256` (A′) until amendment 8. `'auto_mixed'` follows `'auto'`'s 16x64 set.
  - `'paper_mixed'` / `'paper_stock'` / `'paper_256'` select the paper sets.
- **#2's dispatch** stays a build-time variant: the `mixed_ko` builds built with `MIXFP4_DISPATCH_FREQ=1`. Whether
  it, or B′, becomes the default is decided on B′'s results.
- **Tests:**
  - `test_gemm.py` with n16k64_wA_e64_t0; `test_g32.py` with n16k64_wA_g32_e64;
  - `test_select.py`: batch invariance of `mixed_ko`, and all 12 scheduler settings bitwise equal for `mixed_ko` and
    `stock_ko`;
  - `test_g32.py`'s routing test now expects the adopted sets.
- **Script:** `run_7.sh`.
- **Builds**, CPU only, from the amendment 7 sources, before registration:
  - `build_7`: all 39 configurations, 10 parallel `build.py --config` processes;
  - `build_7freq`: the four `mixed_ko` builds with `MIXFP4_DISPATCH_FREQ=1`.

**Gates** (`run_7.sh`; a failure stops the adoption):
- **G3:** `build.py --selftest` for n16k64_wA_e64_t0 and n16k64_wA_g32_e64.
- **G1 / G2 (`check_sass.py`):**
  - G1 on `build_7`: the 37 existing configurations keep their patched and unpatched SASS. The before roots are
    `sm120/build`, kernel-opt `build`, `build_e0m3`, `build_A1`, `build_4` and `build_T`.
  - G2: the 2 new builds have their census, and nothing is predicated; `build_7freq`'s four likewise.
- **Schedule tuning (SCHED_TUNE):** produces the adopted table; it is not a gate.
- **G3 pytest:** `test_gemm.py`, `test_select.py` and `test_g32.py` on `build_7`; `test_select.py -k mixed_ko` on
  `build_7freq`.
- **G4 (`check_bitwise.py`):** 0 differences against n16k64_wA (`sm120/build`) for:
  - n16k64_wA_e64_t0 and set:mixed_ko with the adopted table, on `build_7` and `build_7freq`;
  - n16k64_wA_g32_e64 (family g32), on `build_7`.
- **G5:** whole-model logits bitwise equal on 4 models × 5 shapes, each paper set vs its adopted set with the adopted
  table:
  - 16x64: set:mixed vs set:mixed_ko;
  - 16x64 with #2's dispatch: set:mixed from `build_freq` vs set:mixed_ko from `build_7freq`;
  - FourOverSix: set:stock vs set:stock_ko.

**No M1 here.** The adopted path is the combination of t0, #4 and 4b, and is measured as such in the cumulative
registered run (decision 7): combined vs the paper builds, plus the end-to-end harness. Each part has its own
measurement in amendments 4b, 5 and 6b.

**After the gates:**
1. Copy the tuned table to `sm120/configs/<gpu>.ko.json`.
2. Add a note on the adoption to `results/kernel_opt/4/REPORT.md`, `retune/REPORT.md`, `t0/REPORT.md` and
   `docs/BUILD_AND_USE.md`.
3. Commit and push.

### Amendment 7: the run, the adoption and its deviations

`run_7.sh` ran on 2026-10-01 from 04:47 to 05:24 UTC (registration 0ccb0a9). Every gate passed:
- **G3:** n16k64_wA_e64_t0 and n16k64_wA_g32_e64 PASS.
- **G1:** the 37 existing configurations keep their SASS.
- **G2:** the 2 new builds and `build_7freq`'s four have their census, with no predication.
- **pytest:** 1,151 passed on `build_7`, 9 on `build_7freq`.
- **G4:** 31,680 comparisons (13,440 / 13,440 / 4,800), 0 differences.
- **G5:** logits bitwise equal for 16x64, for 16x64 with #2's dispatch, and for FourOverSix.

The schedule tuning set 62 of 224 `mixed_ko` rows and 92 of 224 `stock_ko` rows to a non-default setting
(`results/kernel_opt/7/table/`).

**The adoption.** The tuned table was copied to `sm120/configs/<gpu>.ko.json`. With that, `auto` and `auto_stock`
route as registered, and BUILD_AND_USE and the #4, retune and t0 reports carry a note.

**Deviations, disclosed:**
1. **No 256x64 work.** After registration the user decided against any 256x64 work for now.
   - The amendment 8 announced above (#4 on the 256x64 path) is cancelled. It was never registered, and the number
     8 goes to the next registered amendment.
   - n16k64_wA_g32_e64 stays in `configs.py`, because removing it would change the `configs.py` hash that every
     `build_7` manifest records. It was built and gated as registered, but no set uses it.
2. **256x64 routing restored.** The registered `model.py` sent a 256x64 map whose g32 builds are absent to the
   adopted 16x64 set. The user asked for today's 256x64 routing, unchanged.
   - So, after the chain, that fallback is the paper `mixed` set again: `mixed256` when built, else `mixed`.
   - `test_g32.py`'s routing test was updated to match and re-run: it passed both on `build_7` and on `sm120/build`.

## Amendment 8: the E0M3-fraction sweep (C3k)

Written 2026-10-01, before any GPU run of the sweep. The protocol is `results/kernel_opt/c3k/PROTOCOL.md` and the
hashes are in `results/kernel_opt/c3k/registration.json`.

- **The request.** The user asked for GEMM latency against the E0M3 tile share on the adopted path (both dispatch
  variants), its no-dispatch ceiling and the paper kernel, each against its own stock. It runs before B′'s GPU work.
- **Descriptive only.** Nothing is tuned or adopted from it.
- **New:**
  - the no-dispatch ceiling of the adopted path: `nodisp_ko` = n16k64_wA_nodisp_{n16,n32,n64}_t0 and _e64_t0, in
    `configs.py` / `select.py`, built CPU-only in `build_C3k`;
  - `select.py`'s `SCHEDULE_FAMILY`, so that `nodisp_ko` reads `mixed_ko`'s scheduler rows;
  - `c3k_fraction.py`, `c3k_analyze.py`, `run_c3k.sh`.
- **Disclosed:** the sweep and analysis scripts were smoke-tested once into a scratch directory, on one shape, T = 16
  and 512, and 1 round × 2. Nothing from it is used.


## Amendment 9: the cumulative registered run — the combined 16x64 build against today's paper builds

Written 2026-10-01, before any registered GPU run of this amendment. The hashes and time are in `registration_9.json`.

**The user's decision 7**, relayed by the coordinator: "after B′, one registered run with the combined build (freq or
B′ per B′'s results + t0 + e64 + scheduling + 4b …). Gates, then M1 (combined vs today's paper builds) and the e2e
prefill/decode harness, 4 models." The later schedule change limits it to the 16x64 family, plus stock/FourOverSix and
the paper-build references. There is no 256x64 or 8x64 work.

**The dispatch is #2's.** B′ was explored and does not supersede it (`results/kernel_opt/Bprime/EXPLORATION.md`,
1884314). C3k showed #2's dispatch is the faster variant at real map densities (f ≈ 2–4 %).

**What is measured.** This amendment adds no build and changes no library source; it adds only the scripts below.
- **The combined 16x64 build:** `mixed_ko` from `build_7freq`, i.e. amendment 7's four builds with
  `MIXFP4_DISPATCH_FREQ=1`, on the adopted table `sm120/configs/<gpu>.ko.json`. That is #2's dispatch, t0, #4's
  epilogue tile at width 128, the 4b widths and the scheduler rows.
  - Disclosed: the scheduler rows were tuned with the default dispatch (`build_7`). One set of rows serves both
    dispatch variants (amendment 7).
- **The adopted stock set:** `stock_ko` from `build_7`, on the same table.
- **The references, today's paper builds:** `mixed` and `stock` from `sm120/build`, on the paper table
  `sm120/configs/<gpu>.json`.
- **In M1 only:** the adopted path with the default dispatch (`mixed_ko` from `build_7`). It gives #2's dispatch vs the
  default on the real maps.

**Gates** (`run_cum.sh`; a failure stops the run):
- **G0 (`check_provenance.py`), the registered files and builds are the ones measured:**
  - every file listed in the registration has its sha256;
  - every build used has its registered library sha256, SASS sha256 and defines, and its library file matches its
    manifest. These are `build_7freq`'s four builds, `build_7`'s four `mixed_ko` and four `stock_ko` builds, and the
    eight paper builds of `mixed` and `stock` in `sm120/build`;
  - the `build_7` and `build_7freq` builds have the patched and unpatched SASS of amendment 7's registration.
  - Note: amendment 7's G3 self-test rebuilt `build_7`'s n16k64_wA_e64_t0 after that registration. Its SASS is the
    registered one, but its library sha256 differs. Amendment 7's later gates (schedule tuning, G4, G5) ran on the
    rebuilt library, which is the one registered here.
- **G2 (`check_sass.py --only-new`):** the census and no predicated OMMA, for `build_7freq`'s four builds and
  `build_7`'s eight (stock: E2M1 OMMAs only).
- **G3 pytest:**
  - `test_gemm.py` and `test_select.py` on `build_7freq`;
  - `test_select.py -k "mixed_ko or stock_ko"` on `build_7`.
  - Self-tests are not re-run, because `build.py --selftest` rebuilds the library. The same SASS passed them in
    amendment 7, and G0 ties the two.
- **G4 (`check_bitwise.py --family wA`):** 0 differences against n16k64_wA (`sm120/build`) for `build_7freq`'s four
  builds, and for set:mixed_ko with the adopted table.
- **G5:** whole-model logits bitwise equal on 4 models × 5 shapes:
  - set:mixed (paper) vs set:mixed_ko (`build_7freq`, adopted table);
  - set:stock vs set:stock_ko (`build_7`, adopted table).

**M1 (`bench_cum_isolated.py`): GEMM time.**
- **Method:** deviation-2, as in amendments 5 and 6: cold weights by rotation plus a 512 MiB read-flush, activations
  quantized after the flush, CUPTI, 3 rotated rounds × 30.
- **Scope:** all four models, every text-Linear shape, T ∈ {1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192},
  typical and worst tags.
- **Configurations:** `stock_paper`, `stock_ko`, and for each tag variant `paper_16x64`, `comb_16x64` (combined) and
  `ko_16x64` (the adopted default dispatch). The tables differ in width (4b); each call records its width and scheduler
  setting.
- **Checks (a failure stops the run):**
  - every timed path equals NativeLinear's forward (M1's);
  - on the timed operands, bitwise: stock_ko equals the paper stock, and the combined build and ko each equal the
    paper build.
- **Reported (`cum_report.py`):**
  - per model and T, the per-forward GEMM sum;
  - the ratios: combined vs paper (typical, worst), stock_ko vs paper stock, adopted default vs paper, and #2's vs the
    default dispatch;
  - each 16x64 path's gap to its own stock and to the paper stock;
  - over all cells: median, min, max, and the cells slower or faster in every round.

**End to end (`cum_e2e.py`).** The paper's per-process scripts run unchanged: `bench_prefill.py` (step 05) and
`bench_decode.py` (Experiment D).
- **Policies:**

  | policy | map | kernel | build directory | installed set, table |
  |---|---|---|---|---|
  | ours-16x64-paper | TC 16x64 | `paper_mixed` | `sm120/build` | `mixed`, paper table |
  | ours-16x64-combined | TC 16x64 | `auto_mixed` | `build_7freq` | `mixed_ko`, adopted table |
  | fo6-paper | FourOverSix | `paper_stock` | `sm120/build` | `stock`, paper table |
  | fo6-ko | FourOverSix | `auto_stock` | `build_7` | `stock_ko`, adopted table |

- **Prefill:** all four models at the paper's shapes (1x128 … 1x8192, 4x2048), 7 repetitions. 5 rounds, with the policy
  list rotated by r − 1 in round r. That is 80 processes, about 2.3 h.
- **Decode:** Llama-3.1-8B, Mistral-7B-v0.3 and Phi-4, with D's 6 settings (batch 1, 4, 16 × prompt 512, 2048) and 64
  generated tokens. 5 rounds, rotated the same way. That is 60 processes, about 50 min.
  - Qwen3.8-27B is out, as in Experiment D: the decode harness does not support its hybrid cache.
- **Checks (a failure stops the run):**
  - prefill: the graph's logits equal eager's bitwise at every shape;
  - decode: the graph's first 33 greedy tokens equal an eager StaticCache decode's;
  - every library a process loaded is a build of its policy's directory, by sha256;
  - `MIXFP4_DISPATCH_FREQ=1` is set on exactly the `build_7freq` builds;
  - the installed set is the policy's family, on its table.
- **Reported (`cum_e2e_report.py`):**
  - per model and shape (or setting), the median over rounds of each process's value: prefill in ms per forward,
    decode in tokens per second;
  - combined vs paper, fo6-ko vs fo6-paper, and each 16x64 policy vs FourOverSix;
  - ranges pairing round r of both policies.
  - No significance claims beyond the round range. Earlier prefill runs varied by about 1 % per process.

**Nothing is adopted or tuned from this run.** It is descriptive: what the combined build changes against the paper
builds, at the GEMM and end-to-end level. The paper numbers (tm-opt) stay untouched; whether to re-measure them is the
user's decision. Still on hold: #1b M3, the decisive-margin rule, E0M3 (i)–(iii).

**Disclosed, before registration: a smoke test of the new scripts, into a scratch directory.**
- **M1:** Llama-3.1-8B o_proj and down_proj at T = 16 and 512, 1 round × 2.
- **The e2e harness:** Llama-3.1-8B, all 4 policies, 1 round. Prefill at 1x128 and 1x2048 × 2 reps; decode at 1x512.
- **Result:**
  - the registered checks passed;
  - the build check's negative controls failed as they should: the wrong directory, the wrong family, a paper record
    against `build_7freq`.
- **One fix came from it:** `run_cum.sh` now creates its `gemm` directory.
- **G0 was dry-run on CPU** against this registration: it passed. Its negative control, a copy with one file hash and one
  build's defines altered, failed on exactly those two.
- Nothing from the smoke test is used.

### Amendment 9: the run

`run_cum.sh` ran on 2026-10-01 from 05:55 to 09:13 UTC (registration baec35e), with no deviation.
- **Every gate passed:**
  - G0: 41/41 files, 20/20 builds, 12/12 with amendment 7's SASS;
  - G2;
  - G3 pytest: 174 passed on `build_7freq`, 17 on `build_7`;
  - G4: 33,600 comparisons, 0 differences;
  - G5: 16x64 and FourOverSix.
- **Every registered check passed:**
  - M1: 4,680 bitwise comparisons on the timed operands;
  - 80 prefill processes and 60 decode processes.
- **Results** are in `results/kernel_opt/cum/REPORT.md`. Headline medians for the combined build against the paper
  build:
  - GEMM −1.96 %; the gap to its own stock goes from +2.90 % to +1.45 %;
  - prefill −0.89 %;
  - decode +0.10 % tokens per second.
- Nothing is adopted or tuned from the run.

## Note 2 to amendment 1 (2026-10-01): optimization 1b's M3 resumed

Written before any process of the resumed M3 started. The user gave the start, relayed by the coordinator on
2026-10-01 at about 12:20 UTC, with the new 8x64 focus: "Resume #1b's paused M3 decode as registered: decode-only, same
settings, the #1b builds. Close #1b with its report, noting the pause/resume."

**The registered condition for M3 holds.** Amendment 1 runs M3 only if the re-tuned table changes a choice at the
decode buckets (T ≤ 16) of a shape of Llama, Mistral or Phi-4. That diff was not recorded before the pause, so it is
recorded here. `opt1/table_opt1.json` against `opt1b/table_opt1b.json`, `mixed_wB` rows:

| shape | bucket | optimization 1 | 1b | model |
|---|---|---|---|---|
| 35840x5120 | 4 | 16 | 32 | Phi-4 gate_up_proj |
| 5120x17920 | 16 | 32 | 16 | Phi-4 down_proj |
| 7680x5120 | 16 | 16 | 32 | Phi-4 qkv_proj |
| 10240x5120 | 1 | 32 | 16 | not a decode model's shape |

**What runs:** `ab_e2e.py --what decode --out /home/dev/n16k64_campaign/kernel_opt/opt1b/e2e`, exactly as registered.
- The policies are ours-8x64, ours-8x64-opt (`auto_wB` from the kernel-opt `build`, the 1b builds), fo6 and fo6-wB.
- Llama-3.1-8B, Mistral-7B-v0.3 and Phi-4, with Experiment D's 6 settings.
- 5 rounds, rotated by r − 1; the registered checks.
- It writes to the same output directory; no decode record exists there.

**Provenance, checked now:**
- `ab_e2e.py` and `experiments/paper_extra/bench_decode.py` have their registered sha256 (`registration_1b.json`).
- Every library that 1b's prefill records loaded equals its current manifest and file, by sha256: the 1b set's 5
  builds, n8k64_wB, the paper `stock` set and stock_wB.
- The tile table `sm120/configs/<gpu>.json` equals `opt1b/table_opt1b.json`, the table 1b's M1 and M2 ran with.

**Disclosed: code that changed since 1b's registration.**
- `sm120/mixfp4_sm120/model.py`, `lib.py`, `select.py` and `configs.py` changed in later amendments: routing names,
  #4's scheduler plumbing, and new families and configurations.
  - The configurations include amendment 10's two diagnostic builds, added before this run.
- **For these four policies the changes select the same builds and table:**
  - `'auto_stock'` falls back to the paper `stock` set, because `sm120/build` holds no `stock_ko` build. The record's
    routing note says so, and the build check requires `sm120/build`.
  - `'auto_wB'` and the single builds are unchanged.
  - Without scheduler rows, lib.py calls the GEMM through the same entry point as before.

## Amendment 10: the 8x64 baseline on the current best kernels

Written 2026-10-01, before any registered GPU run of this amendment. The hashes and time are in `registration_10.json`.

**The request**, relayed by the coordinator: "new focus is 8x64 optimization on kernel-opt. Goal, as for 16x64: make
the FlipQuant 8x64 GEMM reach stock latency at all T. Reference: stock_wA, i.e. FourOverSix on the deployment set
(auto_stock / stock_ko). Report vs stock_wB too, for placement."
- Step 1 b): "A baseline for 8x64 on the current best kernels: the #1b set (auto_wB / mixed_wB) with and without #2's
  freq dispatch, vs stock_wA (stock_ko and paper stock) and stock_wB." It asks for a deviation-2 M1 and a C2-style
  4096³ split.
- The baseline is descriptive. Nothing is tuned or adopted from it; it ranks the step 2 plan.

**What is measured.** No deployment build and no library source changes.
- **The 8x64 path:**
  - `mixed_wB` from the kernel-opt `build`: optimization 1b's five builds, default dispatch;
  - `mixed_wB` from `build_freq`: #2's builds of the same set (`MIXFP4_DISPATCH_FREQ=1`).
  - Both read the paper table `sm120/configs/<gpu>.json`, whose `mixed_wB` rows are 1b's (`opt1b/table_opt1b.json`).
- **The references:**
  - `stock_ko` from `build_7` on the adopted table: the deployment's stock, weights on A;
  - the paper `stock` set and `stock_wB` (one 128-wide build) from `sm120/build`.
- **New diagnostic builds, for C2w only.** These are n8k64_wB's no-dispatch ceiling: the same tile and 1x8 arrangement,
  with the format dispatch compiled out (`MIXFP4_NO_DISPATCH=1`, E2M1 only), as n16k64_wA_nodisp is for 16x64.
  - `n8k64_wB_nodisp` keeps the blob's site-0 prmt tags.
  - `n8k64_wB_nodisp_t0` drops them (`TAG0=0`).
  - Both are `patch=False`, in `configs.py` since 0105c7f, and were built CPU-only in `build_W` before registration.
  - The wB blob carries 32 identity PRMTs, against n16k64_wA_nodisp's 8 (SASS count); the t0 build has none. The pair
    sizes t0 for the wB family.

**Gates** (`run_w8.sh`; a failure stops the run):
- **G0 (`check_provenance.py`):** every file listed in the registration has its sha256. Every build used has its
  registered library sha256, SASS sha256 and defines, and its library file matches its manifest.
  - These builds are 1b's five, `build_freq`'s five, `build_7`'s four `stock_ko`, `sm120/build`'s paper `stock` set
    and `stock_wB`, and `build_W`'s two ceilings.
  - Their trail, checked at registration: 1b's five library sha256 equal those in 1b's M1 records. `build_freq`'s
    five equal those in #2's M1′ records, which #2's G4′ gated. `build_7`'s are amendment 9's G0.
- **G2 (`check_sass.py --only-new`):** the two ceilings have E2M1 OMMAs only, with none predicated.

**M1 (`bench_w8_isolated.py`): GEMM time.**
- **Method:** deviation-2, as in amendments 5–9: cold weights by rotation plus a 512 MiB read-flush, activations
  quantized after the flush, CUPTI, 3 rotated rounds × 30.
- **Scope:** all four models, every text-Linear shape, T ∈ {1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192},
  typical and worst tags of the TC 8x64 artifacts.
- **Configurations:** `stock_paper`, `stock_ko`, `stock_wB`, and `wB_{typical,worst}` and `wBfreq_{typical,worst}`.
- **Checks:** M1's (the isolated path equals NativeLinear's forward), plus wBfreq equal to wB bitwise on the timed
  operands.
- **Reported (`w8_report.py`):**
  - per model and T, the per-forward GEMM sums;
  - the ratios: 8x64 vs stock_ko (typical, worst; #2's), vs the paper stock, and vs stock_wB; #2's vs the default
    dispatch; stock_wB vs stock_ko; stock_ko vs the paper stock;
  - over all cells and by T band: median, min, max, and the cells above or below zero in every round;
  - the widths per projection and T.

**C2w (`c2_w8.py`): the 4096³ breakdown.**
- **Method:** C2′'s operands and modes: b2b, isolated, and sustained under NVML, 3 rotated rounds each.
- **Configurations:**
  - stock_wA (paper);
  - stock_ko (its build and scheduler row at this shape);
  - stock_wB;
  - the two ceilings;
  - n8k64_wB from 1b's `build` and from `build_freq`, each with all-E2M1, real (Llama-3.1-8B layer 0 o_proj, 8x64 map)
    and all-E0M3 tags.
- **Checks before timing:** nodisp_wB_t0 equals nodisp_wB bitwise, and each freq configuration equals its default.
- **The decomposition read off it:**
  - placement: stock_wB vs stock_wA;
  - arrangement and tags: the ceilings vs stock_wB;
  - dispatch: n8k64_wB all-E2M1 vs the tagged ceiling;
  - E0M3 tiles: the real and all-E0M3 rows vs all-E2M1.

**Disclosed, before registration: a smoke test of the new scripts, into a scratch directory.**
- M1: Llama-3.1-8B o_proj and down_proj at T = 16 and 512, 1 round × 2.
- C2w: 1 round.
- Every check passed.
- One fix came from it: the report skips an empty T band.
- Nothing from it is used.

### Amendment 10: the run

`run_w8.sh` ran on 2026-10-01 from 13:16 to 13:30 UTC (registration 0a57ff2), with no deviation.
- **Gates:** G0 (34/34 files, 21/21 builds) and G2 passed.
- **Checks:** M1's 3,240 bitwise comparisons and C2w's 4 were all equal.
- **Results:** `results/kernel_opt/w8/REPORT.md`.
  - The 8x64 path is +6.1 % per forward against `stock_ko` (median; +1.3 % at T ≤ 16, +10.2 % at T ≥ 2048), and #2's
    dispatch is −0.70 %.
  - At 4096³ the gap splits into the 1x8 arrangement (+2.3 … +3.3 %), the site-0 tags (+1.1 … +1.3 %), the dispatch
    (+3.8 … +4.4 %, or +2.5 … +3.6 % with #2's) and the real map's E0M3 tiles (+0.5 … +1.1 %).
- Nothing is adopted or tuned from the run.

## Amendment 11 (the 8x64 plan's P2, with P1): t0 for the weights-on-B family, and the 8x64 dispatch

Written 2026-10-02, before any registered GPU run of this amendment. The hashes and time are in `registration_11.json`.

**The request**, relayed by the coordinator: the user's go on the 8x64 plan (`results/kernel_opt/w8/PLAN.md`, a84ab6f),
in the order P2 (with P1) → P4 → P3 → P5 → P6 → P7.
- "P2: t0 for the wB family (TAG0=0 + --untagged-site0), default and freq variants; P1 (the freq default for 8x64) is
  chosen on P2's M1."
- The goal is unchanged: the FlipQuant 8x64 GEMM at `stock_ko` latency at all T, bitwise-safe. Each item is
  registered before its measured runs and has its own build directories.

**What changes** (the sources are commit c5bb523):
- **`configs.py`:** n8k64_wB{,_m64,_m32,_m16,_n64}_t0, each its base with `TAG0=0` and the same census. build.py
  patches them with `--untagged-site0`.
- **`select.py`:** the set `mixed_wB_t0` = {16: n8k64_wB_m16_t0, 32: …_m32_t0, 64: …_m64_t0, 128: n8k64_wB_t0,
  '128x64': n8k64_wB_n64_t0}, on the `mixed_wB` rows (`TABLE_FAMILY`).
- **Tests:** test_gemm.py's MIXED gets the five builds. test_select.py draws 8x64 maps for every weights-on-B family and
  checks `mixed_wB_t0`'s batch invariance.
- **Unchanged:** the library sources, the blob generator and the patcher (amendment 6b's).
  - The wB blob tags each MMA's scale word as the wA blob does, so `TAG0=0` applies as it is.
  - Its site-0 MMAs read the scale word directly. The E0M3-on-B site (2) keeps its tag.
- **New scripts:** `check_patcher_sites.py`, `check_same_sass.py`, `bench_w8p2_isolated.py`, `c2_w8p2.py`,
  `w8p2_report.py`, `run_w8p2.sh`.

**Builds**, CPU only with no GPU visible, from the clean main checkout at c5bb523 (`results/kernel_opt/w8/p2/build_P2.sh`):
- `build_P2`: all 50 configurations, 13 parallel `build.py --config` processes;
- `build_P2freq`: the five wB t0 builds with `MIXFP4_DISPATCH_FREQ=1` (#2's dispatch).
- All 55 passed build.py's checks. The five t0 builds have their bases' census, mainloop stages and register counts.
- The GEMM kernel loses 384 PRMTs at width 128 (4,976 → 4,592 SASS instructions) and 192 at '128x64' and width 64. It
  loses 96 at width 32 and 48 at width 16.

**Gates** (`run_w8p2.sh`; a failure stops P2):
- **G0 (`check_provenance.py`):** every registered file has its sha256. Every registered build has its library sha256,
  SASS sha256 and defines, and its library file matches its manifest.
  - These builds are amendment 10's (kernel-opt `build`, `build_freq`, `build_7`, `build_W`, `sm120/build`),
    `build_P2`'s 50 and `build_P2freq`'s 5.
- **G3, the self-tests:** `build.py --selftest` for the five t0 builds, with the default dispatch and with #2's, one
  process each.
  - The requirement is PASS patched and FAIL unpatched under random tagging of the weights (`randb`), on the three
    upstream shapes.
  - A self-test rebuilds its library first, so these run in the separate directories `build_P2st` and `build_P2stfreq`.
    The registered builds stay untouched.
  - `check_same_sass.py`: each rebuilt library carries the registered build's patched and unpatched SASS, patcher and
    census.
- **The patcher check (`check_patcher_sites.py`):** 6b's validation, extended to the wB family.
  - Every tagged weights-on-B library and self-test executable in sm120/build and the kernel-opt `build`, `build_freq`,
    `build_4`, `build_A1`, `build_e0m3`, `build_T`, `build_7` and `build_P2`: the reaching-definitions analysis gives
    every OMMA the strict parser's site. The self-test executables carry the BRX jump table.
  - Every t0 wB library and self-test executable (`build_P2`, `build_P2freq`, `build_P2st`, `build_P2stfreq` and
    `build_W`'s ceiling): its per-site counts equal its census and the patcher's record.
- **G1 / G2 (`check_sass.py`):**
  - G1 on `build_P2`: the 45 existing configurations keep their patched and unpatched SASS. The before roots are
    sm120/build, then kernel-opt `build`, `build_e0m3`, `build_A1`, `build_4`, `build_T`, `build_7`, `build_W` and
    `build_C3k`.
  - G2: the five new builds have their census, with nothing predicated; `build_P2freq`'s five likewise.
- **G3 pytest:** test_gemm.py and test_select.py on `build_P2`; the same with `-k wB` on `build_P2freq`.
- **G4 (`check_bitwise.py --family wB`):** 0 differences against n8k64_wB from sm120/build.
  - Candidates: the five t0 builds each alone, and set:mixed_wB_t0 on the paper table, on `build_P2` and on
    `build_P2freq`.
  - Inputs: the densest and the lower-median modules per shape of the 4 TC 8x64 artifacts, and synthetic all-E2M1,
    all-E0M3 and random maps.
  - Coverage: 15 token counts; the fused, reuse and unfused paths, and CUDA graphs.
- **G5 (`check_model_logits.py --unit 8x64`):** whole-model logits bitwise equal, on 4 models × 5 shapes:
  - set:mixed_wB (kernel-opt `build`) vs set:mixed_wB_t0 (`build_P2`);
  - with #2's dispatch, set:mixed_wB (`build_freq`) vs set:mixed_wB_t0 (`build_P2freq`).

**M1 (`bench_w8p2_isolated.py`):** amendment 10's method and scope.
- **Method:** deviation-2: cold weights by rotation plus a 512 MiB flush, activations quantized after the flush,
  CUPTI, 3 rotated rounds × 30.
- **Scope:** 4 models, every text-Linear shape, T ∈ {1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192},
  typical and worst tags.
- **Configurations:**
  - `stock_ko` (`build_7`, adopted table), the target; `stock_wB` (sm120/build), the same placement;
  - the 8x64 path in four builds, each with typical and worst tags:
    - 1b's `mixed_wB` (kernel-opt `build`);
    - #2's (`build_freq`);
    - t0 (`build_P2`);
    - t0 with #2's dispatch (`build_P2freq`).
  - All four 8x64 builds read the paper table's `mixed_wB` rows (1b's), so they run the same widths.
- **Checks:** M1's own (the isolated path equals NativeLinear's forward). #2's, t0 and t0 #2 each equal 1b bitwise on the
  timed operands.
- **Reported (`w8p2_report.py`):** per model and T, the per-forward GEMM sums and their ratios, with the range over the
  rounds; over all cells and by T band.

**C2w‴ (`c2_w8p2.py`):** amendment 10's C2w with the two t0 builds added.
- **Configurations:** stock_wA, stock_ko, stock_wB, the two ceilings (`build_W`), and n8k64_wB in four builds (1b, #2,
  t0, t0 #2) × {all-E2M1, real map (Llama-3.1-8B layer 0 o_proj), all-E0M3}.
- **Modes:** b2b, isolated and sustained, 3 rotated rounds.
- **Checks before timing:** the ceilings equal each other, and every build equals 1b's for each tag.

**Rules**, written before the run:
- **t0 adoption.** Adopt the t0 builds for 8x64, with both dispatch variants, if both hold:
  - every t0 unit (t0 vs 1b and t0 #2 vs #2, with typical and with worst tags) has a negative median over the 48
    (model, T) cells;
  - no cell is above zero in every round by more than 0.5 %.
  - Why not amendment 6's strict form (no cell above zero in every round): under a zero effect each cell has about a
    1 in 8 chance of that. On 16x64 the strict form failed on cells of +0.1 to +0.35 %, and the user adopted t0 as a
    disclosed deviation (amendment 7). The strict form is reported as well, with its cells listed.
  - Adoption is proposed to the coordinator, not automatic.
- **P1.** Recommend #2's dispatch as the 8x64 default if, on the t0 builds with typical tags:
  - its median over the cells is negative;
  - and more cells are below zero in every round than above.
  - The worst tags and C2w‴'s all-E0M3 row are reported alongside, as for 16x64.
  - The choice goes to the coordinator for the user.
- **In any case:** the residual 8x64 gap per T band against `stock_ko` and against `stock_wB`, before (1b, #2) and after
  (t0, t0 #2).

**Maps and tags.** The FlipQuant calibration will change later, so the inputs are stated here.
- The paper's TM-OPT+TC 8x64 artifacts in /home/dev/n16k64_campaign/paper/artifacts (`<model>_tc_8x64`); their
  artifact.json sha256 are registered.
- Typical is the lower-median module per projection and worst the densest, as in amendment 10.
- C2w‴'s real map is the Llama-3.1-8B 8x64 map's layer 0 o_proj. G4 also uses synthetic maps.

**Disclosed, before registration** (nothing from it is used):
- **CPU pre-checks, into a scratch directory:**
  - `check_sass.py` on `build_P2` and `build_P2freq`: G1 passed for the 45 existing configurations, G2 for the new five
    in both;
  - `check_patcher_sites.py` on the tagged roots above and the new libraries: 53 tagged binaries (8 with a BRX) and 12
    t0 binaries, 0 failures. A first dry run on 3 roots found the same.
- **A smoke test of the new M1, C2w‴ and report scripts, into a scratch directory:**
  - M1: Llama-3.1-8B o_proj and down_proj at T = 16 and 512, 1 round × 2. C2w‴: 1 round. Every bitwise check passed.
  - The report then failed on the partial projection set, and now sums the measured projections only. A full run
    measures all of them.
  - The t0 rule's tolerance was written after the smoke test. It rests on amendment 6's experience above, not on
    the smoke numbers.

### Amendment 11: the run

`run_w8p2.sh` ran on 2026-10-02 from 15:36 to 16:10 UTC (registration 022240a), with no deviation.
- **Gates:** all passed.
  - G0: 47 files and 74 builds.
  - Self-tests: 10 × PASS patched / FAIL unpatched, each rebuild with the registered SASS.
  - The patcher check: 53 tagged and 32 t0 binaries, 0 failures.
  - G1 / G2; pytest: 1,013 passed on `build_P2`, 211 on `build_P2freq`.
  - G4: 2 × 28,800 comparisons, 0 differences. G5: both equal.
- **Rules:** the t0 adoption rule and the P1 rule are both met; amendment 6's strict form is not met.
  - t0 per forward: −0.76 % vs 1b, −0.55 % vs #2 (medians); −1.3 to −2.2 % at T ≥ 256.
  - The strict form flags 41 cells at T ≤ 64, all ≤ +0.68 %.
  - P1: #2's dispatch on the t0 builds is faster in 48 of 48 typical-tag cells in every round (−0.53 %).
- **The residual gap** of t0 with #2's dispatch to `stock_ko` (typical, median) is +4.35 % (1b: +5.69 %):
  - +0.6 … +0.8 % at T ≤ 16;
  - +2.2 … +2.8 % at 32–128;
  - +5.4 / +9.3 / +6.3 % at 256 / 512 / 1024;
  - +7.0 … +7.5 % at T ≥ 2048.
- **Results:** `results/kernel_opt/w8/p2/REPORT.md`. Adoption of t0 and of #2's dispatch for 8x64 is proposed to the
  coordinator, not automatic.

## Amendment 12 (the 8x64 plan's P3, with P4): the adopted 8x64 path's widths and scheduler rows, stock_wB tuned alike

Written 2026-10-02, before any registered GPU run of this amendment. The hashes and time are in `registration_12.json`.

**The decisions**, relayed by the coordinator after amendment 11:
1. **8x64 adoption.** Adopt t0 for the 8x64 (wB) family, with #2's dispatch as the 8x64 default (`build_P2freq`), as
   registered in amendment 11, with the disclosed tolerance.
2. **P4 is folded into P3, as one amendment.** It covers:
   - stock_wB_e64, and a stock_wB set tuned alike;
   - the schedule tuner extended to weights on B, and the wB schedule bitwise tests;
   - one act-warm tuning of the adopted wB set's widths and scheduler rows, and of stock_wB_e64's rows;
   - gates and M1.
3. **The P3 rules.**
   - The widths are the fastest median, as 4b.
   - The scheduler rows follow amendment 7's decisive rule against (0, 1).
   - The decisive-margin rule on the widths is reported as a sensitivity only. Its default is the single 128-wide
     build, the rule's original proposal ("use a non-128 build only if it is decisively faster"). It is also reported
     against the fallback width.
4. **kernel-opt stays local** until the user authorizes a push.

**Why P4 has no mixed build** (disclosed exploration, in worktree `wtP4` with builds in `build_P4x`, CPU only; amendment
11's report):
- The 1x8 arrangement already names a 64 × 64 epilogue tile at width 128 (`MIXFP4_EPI_M/N`). n8k64_wB_t0 built with
  `MIXFP4_EPI_TILE_M/N=64` has n8k64_wB_t0's exact SASS.
- The '128x64' build takes the builder's auto tile, 64 × 32: an explicit 64 × 32 build has its SASS. With 64 × 64 it
  drops from 6 mainloop stages to 5, which is the plan's condition for skipping it.
- stock_wB takes the auto 64 × 32 tile; stock_wB_e64 keeps its 4 stages.

**What changes** (the sources are commit 097d1e5):
- **`configs.py`:** stock_wB_e64, stock_wB with `MIXFP4_EPI_TILE_M/N=64`.
- **`select.py`:**
  - `mixed_wB_ko`: the five t0 builds. It is deployed with #2's dispatch: its build directory holds them built with
    `MIXFP4_DISPATCH_FREQ=1`.
  - `stock_wB_ko`: {128: stock_wB_e64}.
  - Both read the adopted table `<gpu>.ko.json` (`TABLE_FILE`). `mixed_wB_ko` takes that table's 'mixed_wB' width rows
    and its own scheduler rows. The paper sets and the paper table are unchanged.
- **`tune_tiles.py`:**
  - `--schedule` accepts weights-on-B sets; M is the tokens there.
  - A family on the 'mixed_wB' rows takes the `--maps8` tags in both modes. Every other family is tuned as before.
- **Tests:** `test_schedules_bitwise_equal` for `mixed_wB_ko` and `stock_wB_ko` (all 12 settings, every width), and
  batch invariance for `mixed_wB_ko`.
- **Scripts:**
  - `p3_tables.py`: composes the adopted table, and computes the sensitivity;
  - `bench_w8p3_isolated.py`, `w8p3_report.py`, `run_w8p3.sh`;
  - `check_same_sass.py --no-selftest`.

**Builds**, CPU only with no GPU visible, from the clean main checkout at 097d1e5 (`results/kernel_opt/w8/p3/build_P3.sh`):
- `build_P3`: all 51 configurations;
- `build_P3freq`: the five t0 builds with `MIXFP4_DISPATCH_FREQ=1`, and stock_wB_e64 without it (it has no dispatch).
  The tuning and M1 run from it, and it is the adopted 8x64 path's directory.

**The tuning** (`run_w8p3.sh`, on `build_P3freq`):
- **Widths** (`tune_tiles.py --families mixed_wB_ko --cold --act-warm --rounds 3 --iters 30`): the 4b method.
  - Conditions: cold weights; activations quantized after the flush; 3 rotated rounds × 30 isolated launches per width.
  - Coverage: every Linear shape of the four models, buckets 1 … 8192, all five widths, '128x64' included.
  - The fastest median wins.
  - Tags: the tuner's rule, per projection the module with the most E0M3 tiles in the TC 8x64 map, as 1b's tuning
    used.
- **The adopted table** (`p3_tables.py compose`): the tracked `<gpu>.ko.json` with those rows added as 'mixed_wB'.
  Every other key is unchanged.
- **Scheduler rows** (`tune_tiles.py --schedule --families mixed_wB_ko,stock_wB_ko`), at those widths, with amendment 7's
  method and rule. A setting replaces (0, 1) only if its median is at least 0.5 % below and every round is below every
  round of (0, 1).
- **The sensitivity** (`p3_tables.py sensitivity`), from the width tuning's per-round times. The table the
  decisive-margin rule would choose (amendment 7's rule, applied to the widths) is computed with two defaults:
  - `dm128`: the 128-wide build;
  - `dmfb`: the fallback width.
  - Recorded: every cell whose width would change, with the tuning's times.

**Gates** (`run_w8p3.sh`; a failure stops P3):
- **G0 (`check_provenance.py`):** the registered files and builds.
- **G1 / G2 (`check_sass.py`):**
  - G1: the 50 existing configurations keep their SASS (before roots as in amendment 11, plus `build_P2`).
  - G2: stock_wB_e64 is E2M1-only with nothing predicated; `build_P3freq`'s six likewise, with the five t0 builds'
    census.
- **The same device code (`check_same_sass.py --no-selftest`):**
  - `build_P3freq`'s five t0 builds carry `build_P2freq`'s SASS, the 8x64 path amendment 11 gated and the coordinator
    adopted;
  - its stock_wB_e64 carries `build_P3`'s.
- **G3 pytest:** test_gemm.py and test_select.py on `build_P3`, and `-k wB` on `build_P3freq`.
- **G4 (`check_bitwise.py --family wB`):** on `build_P3freq`, 0 differences against n8k64_wB from sm120/build for the five
  builds and set:mixed_wB_ko with the P3 table, so every scheduler row runs.
- **G5 (`check_model_logits.py`):** logits bitwise equal on 4 models × 5 shapes:
  - 8x64: set:mixed_wB_t0 (`build_P2freq`, paper table) vs set:mixed_wB_ko (`build_P3freq`, P3 table);
  - FourOverSix: stock_wB (sm120/build) vs set:stock_wB_ko (`build_P3freq`, P3 table).

**M1 (`bench_w8p3_isolated.py`):** amendment 10's method and scope (deviation-2; 4 models; T ∈ {1, …, 8192}; typical and
worst tags). Every 8x64 configuration runs the same libraries (`build_P3freq`'s t0 builds with #2's dispatch), so only
the table differs.
- **The references:**
  - stock_ko (`build_7`, tracked adopted table);
  - stock_wB (sm120/build);
  - `stock_wB_ko`: stock_wB_e64 with its P3 scheduler rows.
- **The 8x64 path:**
  - before: `wBp2_{typical,worst}`, P2's adopted path on 1b's rows (paper table), with no scheduler rows;
  - after: `wBko_{typical,worst}`, the P3 table;
  - `wBw_typical`: the P3 widths without scheduler rows;
  - `wBdm128_typical` and `wBdmfb_typical`: the decisive-margin widths, without scheduler rows.
- **Checks:** M1's own; every 8x64 configuration equals wBp2 on the same tags, and stock_wB_ko equals stock_wB, bitwise on
  the timed operands.
- **Reported (`w8p3_report.py`):**
  - after vs before;
  - the widths alone, and the rows on top of them;
  - stock tuned alike;
  - the sensitivity: its per-forward effect and the tuning's cells;
  - the gap to stock_ko and to stock_wB tuned alike, by T band and per T;
  - every (model, projection, T) whose width or scheduler setting changed, with its time change.

**The rule**, written before the run. Adopt the P3 table's 8x64 rows and scheduler rows (as the tracked
`<gpu>.ko.json`), and stock_wB_e64 with its rows, unit by unit, if:
- the 8x64 units (after vs before, typical and worst) have negative medians over the 48 (model, T) cells, and no cell
  is above zero in every round by more than 0.5 % (amendment 11's form);
- stock_wB tuned alike meets the same rule for its unit.
Adoption is proposed to the coordinator, not automatic. The sensitivity is reported for the user to decide; it is not
a rule.

**Maps and tags**, as in amendment 11:
- the paper's TM-OPT+TC 8x64 artifacts and their 8x64 maps (sha256 registered);
- typical is the lower-median module, worst the densest;
- the tuner takes, per projection, the module with the most E0M3 tiles.
- The FlipQuant calibration will change later, and the tuning depends on the map's E0M3 share through the tags.

**Disclosed, before registration** (nothing from it is used):
- **The P4 exploration builds** above.
- **CPU pre-checks** on `build_P3` and `build_P3freq`: G1 for 50 configurations, G2, and both same-SASS checks passed.
- **A smoke test, into a scratch directory, on Llama-3.1-8B's shapes only:**
  - the width tuning (2 rounds × 3), the composition, the schedule tuning and the sensitivity;
  - the new tests (17 passed) and the FourOverSix logits gate for stock_wB_ko on Llama-3.1-8B (equal);
  - M1 on o_proj and down_proj at T = 16 and 512 (1 round × 2; all 64 bitwise checks equal), and the report.
  - The first M1 call stopped because the scratch directory did not exist. The chain creates its own, so nothing
    was changed.

### Amendment 12: the run

`run_w8p3.sh` ran on 2026-10-02 from 16:31 to 17:19 UTC (registration 6215946), with no deviation.
- **Gates:** all passed.
  - G0: 48 files and 68 builds.
  - G1 / G2, and both same-SASS checks.
  - pytest: 1,030 passed on `build_P3`, 228 on `build_P3freq`.
  - G4: 28,800 comparisons, every scheduler row exercised, 0 differences.
  - G5: 8x64 and FourOverSix (stock_wB) equal.
- **The tuning:**
  - 22 of 224 width cells changed against 1b's rows.
  - Non-default scheduler rows: 42 cells for `mixed_wB_ko`, 24 for `stock_wB_ko`.
  - The decisive-margin rule would change 1 width cell (default 128) or 4 (default the fallback width).
- **The rule:**
  - The 8x64 rule is NOT met. The after-vs-before medians are −0.01 % (typical) and −0.08 % (worst), but Phi-4 at T = 64
    (typical tags) is +0.84 %, every round above +0.67 %.
  - The stock_wB rule is met: −0.52 %.
  - The re-tune's only clear gain is at T = 128 (−1.0 %). The sensitivity's per-forward effect is +0.03 … +0.04 %.
- **The residual gap** to `stock_ko` (typical, median): +3.97 % on the P3 table, +4.35 % on P2's path.
- **Post hoc, not part of the registered analysis:** an A/A check over the 1,654 M1 cells where two configurations
  run identical computations.
  - The median |Δ| is 0.15 % and the max 3.0 %. 15 % of the pairs are separated across all rounds, against about
    10 % under pure noise.
  - So configuration-level offsets of up to about 2 % on single GEMMs escape the round range.
- **Results:** `results/kernel_opt/w8/p3/REPORT.md`. Adoption is for the coordinator to decide.

## Amendment 12b (the 8x64 plan's P3b, option C): a fresh confirmation of a reduced 8x64 table

Written 2026-10-02, after amendment 12's result and before any registered GPU run of this amendment. The hashes and
time are in `registration_12b.json`.

**The decisions**, relayed by the coordinator after amendment 12's result:
- **Option C.** Register a reduced 8x64 table before running it: 1b's rows plus only the P3 width changes that were
  faster in every round on both tag sets, with no scheduler rows for 8x64. Confirm it with a fresh M1 under the same
  rule.
- **The fallback.** If P3b fails the rule, adopt option A without asking again: P2's adopted path, on 1b's rows.
- **stock_wB_ko** (stock_wB_e64 with its P3 scheduler rows) is adopted, since it met its rule.
- **The post-hoc A/A analysis** stays in amendment 12's report, labelled as such.

**The selection criterion** (`p3b_tables.py select`; written before any run of this amendment, applied to amendment 12's
registered M1 records):
- **Candidates.** A P3 width change at (shape, bucket) is a P3 'mixed_wB' width that differs from 1b's paper row.
- **Kept only if** it was faster in every round on both tag sets, for every (model, projection) of that shape at
  T = bucket. Faster in every round means every round's per-GEMM median of the change lies below every round's of
  1b's row.
  - **Typical tags:** `wBw_typical` (P3's widths, no scheduler rows) against `wBp2_typical` (1b's rows).
  - **Worst tags:** `wBko_worst` (the P3 table) against `wBp2_worst`. M1 had no widths-only worst configuration, so the
    worst-tag evidence includes P3's scheduler row where the cell has one. The fresh M1 tests the width without it.
- **Not kept:** a bucket M1 did not measure (2 or 8) gives no evidence.
- **The result:** 11 of the 22 changes are kept (`results/kernel_opt/w8/p3b/cells.json` lists all 22 with their evidence):

| shape (out x in) | bucket | 1b → P3b | used by | amendment 12's per-GEMM evidence, medians of the round medians (typical / worst) |
|---|---:|---|---|---|
| 1024x4096 | 128 | 64 → 32 | Llama-3.1-8B, Mistral-7B k/v_proj | −10.9 … −12.0 % / −11.7 … −12.9 % |
| 1024x5120 | 64 | 32 → 16 | Qwen3.8-27B k/v_proj | −4.8, −5.5 % / −6.5, −8.1 % |
| 1024x5120 | 256 | 64 → 32 | Qwen3.8-27B k/v_proj | −1.0, −1.9 % / −5.3, −6.4 % |
| 4096x14336 | 64 | 32 → 64 | Llama-3.1-8B, Mistral-7B down_proj | −1.4, −0.8 % / −0.7, −0.7 % |
| 4096x4096 | 1024 | 128 → 128x64 | Llama-3.1-8B, Mistral-7B q/o_proj | −0.7 … −1.6 % / −3.1 … −4.3 % |
| 48x5120 | 1024 | 16 → 32 | Qwen3.8-27B in_proj_a/b | −1.4, −3.6 % / −2.0, −2.9 % |
| 48x5120 | 8192 | 64 → 128x64 | Qwen3.8-27B in_proj_a/b | −12.5, −12.5 % / −11.9, −11.8 % |
| 5120x17408 | 32 | 16 → 32 | Qwen3.8-27B down_proj | −0.7 % / −0.8 % |
| 5120x5120 | 128 | 64 → 128x64 | Phi-4 o_proj | −4.5 % / −4.0 % |
| 5120x6144 | 128 | 64 → 128x64 | Qwen3.8-27B o_proj, out_proj | −2.9, −2.9 % / −2.9, −2.6 % |
| 7680x5120 | 16 | 32 → 16 | Phi-4 qkv_proj | −0.7 % / −1.0 % |

**The tables** (`p3b_tables.py compose`, CPU, before registration). Each is the tracked `<gpu>.ko.json` with 'mixed_wB'
rows and `stock_wB_ko`'s P3 scheduler rows added; every other key is unchanged.
- `table_p3b/`, the candidate: 1b's rows with the 11 cells above. No scheduler rows for `mixed_wB_ko`.
- `table_a/`, the fallback (option A): 1b's rows. No scheduler rows for `mixed_wB_ko`.

**The run** (`run_w8p3b.sh`). There is no build and no source change: the libraries are `build_P3freq`'s.
- **G0:** the registered files and builds.
- **G4 (`check_bitwise.py --family wB`):** set:mixed_wB_ko on `table_p3b`, 0 differences against n8k64_wB from sm120/build.
- **G5:** logits bitwise equal on 4 models × 5 shapes, set:mixed_wB_t0 (`build_P2freq`, paper table) vs
  set:mixed_wB_ko (`build_P3freq`, `table_p3b`).
- **M1 (`bench_w8p3b_isolated.py`):** amendment 10's method and scope; a fresh run.
  - Configurations:
    - stock_ko (`build_7`, tracked table);
    - stock_wB (sm120/build);
    - stock_wB_ko (`table_p3b`);
    - before: `wBp2_{typical,worst}`, P2's path on 1b's rows;
    - after: `wBp3b_{typical,worst}`, the reduced table.
  - Both 8x64 configurations run the same libraries.
  - Checks: M1's own; wBp3b equals wBp2 on the same tags, and stock_wB_ko equals stock_wB, bitwise on the timed operands.
- **Reported (`w8p3b_report.py`):** after vs before, the gaps to stock_ko and to stock_wB tuned alike (by band and per
  T), and every (model, projection, T) whose width changed.
  - The report also evaluates stock_wB tuned alike against stock_wB. That is for information only; it was adopted on
    amendment 12's result.

**The rule** (amendment 12's form):
- The reduced table is adopted if the 8x64 units (after vs before, with typical and with worst tags) have negative
  medians over the 48 (model, T) cells, and no cell is above zero in every round by more than 0.5 %.
- **Otherwise the fallback is adopted.**
- Either way, the adopted table becomes the tracked `sm120/configs/<gpu>.ko.json` in a separate adoption commit.

**Disclosed, before registration:** a smoke test of the new M1 and report scripts into a scratch directory, on Llama-3.1-8B
k_proj and down_proj at T = 128 and 512, 1 round × 2. Every check passed. Nothing from it is used.

### Amendment 12b: the run, and the 8x64 adoption

`run_w8p3b.sh` ran on 2026-10-02 from 17:27 to 17:43 UTC (registration 3ef1699), with no deviation.
- **Gates:** all passed (G0: 39 files and 17 builds; G4: 28,800 comparisons, 0 differences; G5: equal).
- **The rule is met.**
  - After vs before: −0.06 % (typical) and −0.09 % (worst), medians.
  - No cell is above zero in every round by more than 0.5 %; the largest every-round minimum is +0.20 %.
  - All 23 GEMMs whose width changed are faster, by −0.4 to −12.6 %. Per forward the gain is −1.1 % at T = 128.
- **The adoption.** The reduced table (`results/kernel_opt/w8/p3b/table_p3b/`) becomes the tracked
  `sm120/configs/<gpu>.ko.json`. Its other keys are unchanged.
- **The adopted 8x64 path (`mixed_wB_ko`):**
  - t0 builds, deployed with #2's dispatch (`build_P2freq`; the same SASS as `build_P3freq`);
  - 1b's widths with the 11 cells;
  - no scheduler rows.
- **The same-placement stock reference (`stock_wB_ko`):** stock_wB_e64 with its P3 scheduler rows.
- **Its residual gap** to `stock_ko` (typical, median over the models):
  - +0.3 … +1.0 % at T ≤ 16;
  - +1.2 … +2.0 % at 32–128;
  - +5.7 / +9.0 / +6.0 % at 256 / 512 / 1024;
  - +7.0 … +7.4 % at T ≥ 2048.
- **Results:** `results/kernel_opt/w8/p3b/REPORT.md`.

## Amendment 13 (the 8x64 plan's P5, decision a): the adopted 8x64 path's no-dispatch ceiling at every width

Written 2026-10-02, before any registered GPU run of this amendment. The hashes and time are in `registration_13.json`.

**The request.** Decision (a), approved by the user and relayed by the coordinator: the same-placement reference is the
wB family's own no-dispatch t0 ceiling at every width, with no narrow stock_wB. It is descriptive; nothing is tuned
or adopted from it.

**What changes** (the sources are commit 09f67e4, which also carries amendment 14's routing):
- **`configs.py`:** n8k64_wB_{m16,m32,m64,n64}_nodisp_t0. Each is its width's dispatch build with
  `MIXFP4_NO_DISPATCH=1`, `MIXFP4_PIPE_FLAGS=0` and `TAG0=0`: E2M1 only, `patch=False`. This is what
  n8k64_wB_nodisp_t0 is for width 128 (amendment 10).
- **`select.py`:** `nodisp_wB_ko` = those four plus n8k64_wB_nodisp_t0. It reads the adopted table's 'mixed_wB' rows
  (`TABLE_FAMILY`) and `mixed_wB_ko`'s scheduler rows, of which there are none (`SCHEDULE_FAMILY`), as `nodisp_ko` does
  for 16x64 (amendment 8).
- **Scripts:** `bench_w8p5_isolated.py`, `w8p5_report.py`, `run_w8p5.sh`.

**Builds.** `build_P5` holds all 55 configurations, built CPU-only from the clean checkout at 09f67e4
(`results/kernel_opt/w8/p5/build_P5.sh`).
- The four ceilings keep their dispatch builds' mainloop stages: 13, 10, 7, 6.
- n8k64_wB_nodisp_t0 has amendment 10's SASS.

**Gates** (`run_w8p5.sh`):
- **G0:** the registered files and builds.
- **G1 / G2 (`check_sass.py`):** the 51 existing configurations keep their SASS (before roots as in amendment 12, plus
  `build_P3`). The four new ceilings are E2M1-only, with nothing predicated.
- **pytest:** `test_select.py -k nodisp_wB_ko` on `build_P5`: the ceiling's widths are bitwise interchangeable.

**M1 (`bench_w8p5_isolated.py`):** amendment 10's method and scope (deviation-2; 4 models; T ∈ {1, …, 8192}).
- **Table:** every set reads the tracked adopted table (0c8dbcb). The ceiling runs the 8x64 path's width and scheduler
  setting in every cell, and the report asserts it.
- **Configurations:**
  - stock_ko (`build_7`);
  - stock_wB_ko (`build_P3freq`);
  - the ceiling, `nodisp_wB_ko` from `build_P5` on the FourOverSix artifact (E2M1 only);
  - the adopted 8x64 path, `mixed_wB_ko` from `build_P3freq` (t0 with #2's dispatch; the same SASS as `build_P2freq`),
    with typical and worst tags.
- **Checks:** M1's own (each configuration's isolated path equals NativeLinear's forward). The ceiling computes E2M1
  only, so it has no bitwise counterpart.
- **Reported (`w8p5_report.py`):** by T band and per T:
  - 8x64 vs ceiling: what dispatch work could still recover at each width;
  - the ceiling vs stock_ko and vs stock_wB tuned alike: what the 1x8 / 1x4 tiles cost;
  - the 8x64 gap to stock_ko.

**Maps and tags:** as in amendment 11.

**Disclosed, before registration:**
- a CPU pre-check of G1 / G2 on `build_P5`, which passed;
- a smoke test of the new M1 and report scripts into a scratch directory: Llama-3.1-8B k_proj and down_proj at
  T = 16, 128 and 512, 1 round × 2. Nothing from it is used.

### Amendment 13: the run

`run_w8p5.sh` ran on 2026-10-02 from 17:53 to 18:03 UTC (registration 80cac11), with no deviation.
- **Gates:** all passed (G0, G1 / G2, pytest).
- **The ceiling** ran the 8x64 path's width and scheduler setting in every cell.
- **The split** of the adopted 8x64 path's gap to `stock_ko` (+3.96 % per forward, typical, median):
  - the dispatch (8x64 vs its no-dispatch ceiling): +2.10 %, from +0.9 % at T ≤ 16 to +3.1 % at 256–1024;
  - the same-placement tiles (ceiling vs `stock_ko`): +1.29 %, about 0 at T ≤ 128 and +3.4 … +4.2 % at T ≥ 256;
  - at T = 512: +5.7 % tiles and +3.4 % dispatch.
- **Results:** `results/kernel_opt/w8/p5/REPORT.md`.

## Amendment 14 (the 8x64 plan's P6, decision b): 'auto' routes 8x64 maps to the adopted 8x64 set

Written 2026-10-02, after the 8x64 adoption (0c8dbcb) and before any registered GPU run of this amendment. The hashes and
time are in `registration_14.json`.

**The request.** Decision (b), approved by the user and relayed by the coordinator: after P2–P4 land, `'auto'` routes
8x64 artifacts to the adopted wB set on kernel-opt, with `'paper_wB'` = n8k64_wB.

**What changes** (`sm120/mixfp4_sm120/model.py`, commit 09f67e4):
- **`resolve_kernel('auto', meta)`** for a map whose unit rows are not a multiple of 16 (the 8x64 maps): the adopted
  `mixed_wB_ko` if its builds are in the build directory, else the paper kernel n8k64_wB. The two give the same outputs
  bit for bit.
  - The note names the set and its dispatch variant. That variant is a property of the build directory: the adopted
    one, #2's, is `build_P2freq`'s.
  - 16x64 and 256x64 maps keep their routing.
- **`'auto_wB'`** follows `'auto'` for 8x64. It was 1b's `mixed_wB` set.
- **`'paper_wB'`** is n8k64_wB.
- **Test:** `test_select.py::test_auto_routes_8x64_maps`.
- **`check_model_logits.py --after auto:<name>`** installs with `kernel='<name>'`, resolved per artifact from the default
  build directory and table, as deployed. `--expect-family` requires the resolved set, with every library from that
  directory. The routing note, set, table and extra defines are recorded. The existing uses are unchanged; only
  `--after-root` became optional for the routed form.

**Gates** (`run_w8p6.sh`; no build, no timing):
- **G0:** the registered files and builds.
- **The routing test** in three build directories:
  - `build_P2freq`: the adopted path, routed to `mixed_wB_ko`, the note saying #2's dispatch;
  - sm120/build: the fallback to n8k64_wB, and `'paper_wB'`;
  - `build_P3`: the default-dispatch builds, routed to `mixed_wB_ko` with its note. There `test_g32.py`'s
    16x64 / 256x64 routing test runs as well.
- **G5 routed:** with `SM120_BUILD_DIR` = `build_P2freq`, `install(kernel='auto')` on each model's TC 8x64 artifact gives
  logits bitwise equal to n8k64_wB from sm120/build, on 4 models × 5 shapes. The routed set must be `mixed_wB_ko`, every
  library from `build_P2freq`, on the tracked adopted table.

**Disclosed, before registration:** a smoke test.
- The routing tests in the three directories passed. In `build_P2freq` the 256x64 test cannot run, because that
  directory has no 16x64 builds; the chain does not run it there.
- The routed G5 on Llama-3.1-8B gave equal logits at all 5 shapes, routed to `mixed_wB_ko` from `build_P2freq` with #2's
  dispatch. One fix came from it: `--after-root` is not required for the routed form.

### Amendment 14: the run

`run_w8p6.sh` ran on 2026-10-02 from 18:05 to 18:08 UTC (registration 5d9db9d), with no deviation. Every gate passed.
- **G0:** 26 files and 6 builds.
- **The routing tests:**
  - `build_P2freq`: 1 passed;
  - sm120/build: 1 passed;
  - `build_P3`: 2 passed, with the 16x64 / 256x64 routing.
- **G5 routed:** `install(kernel='auto')` on the four TC 8x64 artifacts gave logits bitwise equal to n8k64_wB on all
  5 shapes.
  - Every model was routed to `mixed_wB_ko` from `build_P2freq`, with #2's dispatch and on the tracked adopted table.
  - The note read: "auto: 8x64 map -> mixed_wB_ko (kernel-opt 8x64 adoption, amendments 11-12b; #2's dispatch)".
- **The 8x64 default** on kernel-opt is now the adopted set, through `'auto'`. `'paper_wB'` keeps the paper kernel.

## Amendment 15 (the 8x64 plan's P7): the cumulative registered 8x64 run

Written 2026-10-02, before any registered GPU run of this amendment. The hashes and time are in `registration_15.json`.

**The request.** The plan's P7: a cumulative registered 8x64 e2e, with prefill on 4 models and decode on 3, the Phi-4
1x512 check, and D4's in-graph clock.
- The coordinator's change: the same-placement FourOverSix reference is `stock_wB_ko` (stock_wB_e64 with its tuned rows,
  adopted in amendment 12b) instead of the paper stock_wB. So both stock references are the tuned "ko" sets, like the
  adopted 8x64 path.
- **Descriptive:** nothing is tuned or adopted from it.

**What changes** (`sm120/mixfp4_sm120/model.py`):
- `'auto_stock_wB'` → `stock_wB_ko` when built, else the paper stock_wB.
- `test_auto_routes_8x64_maps` checks it.
- New scripts: `cum8_e2e.py`, `cum8_e2e_report.py`, `p7_split.py`, `p7_split_report.py`, `run_w8p7.sh`.
- No build. The libraries are:
  - sm120/build's n8k64_wB;
  - `build_7`'s `stock_ko`;
  - `build_P3freq`'s adopted 8x64 path (`mixed_wB_ko`, t0 with #2's dispatch) and `stock_wB_ko`. `build_P3freq`'s
    8x64 builds carry `build_P2freq`'s SASS (amendment 12's gate), so the two wB policies share a directory.

**The policies** (`cum8_e2e.py`; the paper's per-process scripts, unchanged):

| policy | artifact | kernel | build directory | the install must report |
|---|---|---|---|---|
| ours-8x64-paper | TC 8x64 | `'paper_wB'` | sm120/build | the single build n8k64_wB |
| ours-8x64-adopted | TC 8x64 | `'auto'` | `build_P3freq` | `mixed_wB_ko` on the adopted table |
| fo6-ko | FourOverSix | `'auto_stock'` | `build_7` | `stock_ko` on the adopted table |
| fo6-wB-ko | FourOverSix | `'auto_stock_wB'` | `build_P3freq` | `stock_wB_ko` on the adopted table |

**Gates** (`run_w8p7.sh`):
- **G0:** the registered files and builds.
- **The routing test** on `build_P3freq`: `'auto'` → `mixed_wB_ko`, and `'auto_stock_wB'` → `stock_wB_ko`.

**The in-graph split** (`p7_split.py`, D4's method):
- **Procedure:** one loaded model, the four policies installed in turn. Each capture is replayed 20 times under the
  profiler, with NVML sampled every 5 ms.
- **Recorded:** wall, GEMM, quantizer, other kernels, idle time, the SM clock (median), power and the power-cap share.
- **The clock:** Llama-3.1-8B at 1x2048 and 1x4096, one ABCD-DCBA pass, so 2 captures per policy.
- **The Phi-4 1x512 check:** Phi-4 at 1x512, three passes, so 6 captures per policy. This shows whether 1b's open +1.5-point
  item is the GEMM or D4's per-capture idle bimodality.

**The end-to-end run** (`cum8_e2e.py`, amendment 9's harness):
- **Prefill:** 4 models, shapes 1x128 … 1x8192 and 4x2048, 7 repetitions per process.
- **Decode:** Llama-3.1-8B, Mistral-7B and Phi-4, at Experiment D's settings.
- **Rounds:** 5, the policy order rotated per round; one process per (model, policy, round).
- **Registered checks** (amendment 9's):
  - the graph's logits equal eager's, and the decode tokens equal eager's;
  - every library loaded is a build of the policy's directory (by sha256);
  - #2's define is on exactly `build_P3freq`'s 8x64 builds;
  - the install reports the policy's set on the adopted table, or its single build.

**Reported:**
- **E2E** (`cum8_e2e_report.py`): per model and shape or setting, the policies' medians and these ratios with their
  ranges over the rounds:
  - adopted vs paper (8x64);
  - adopted vs fo6-ko;
  - adopted vs fo6-wB-ko;
  - paper 8x64 vs fo6-ko;
  - fo6-wB-ko vs fo6-ko.
  The ratios are also summarized over all cells.
- **Split** (`p7_split_report.py`): per policy, the capture medians, SM clock and power, the comparison against fo6-ko,
  and the per-capture idle times.

**Maps and tags:** the paper's TM-OPT+TC 8x64 and FourOverSix artifacts, whose sha256 are registered. The FlipQuant
calibration will change later.

**Disclosed, before registration:** a smoke test into a scratch directory. Nothing from it is used.
- the routing test on `build_P3freq` (passed);
- `cum8_e2e.py` on Llama-3.1-8B, 1 round, prefill 1x512 with 2 repetitions and decode 1x512, all four policies; every
  registered check passed;
- `p7_split.py` on Llama-3.1-8B 1x512, 1 pass × 3 replays;
- both reports.

### Amendment 15: the run

`run_w8p7.sh` ran on 2026-10-02 from 18:15 to 21:24 UTC (registration 417283e), with no deviation. The gates and every
registered e2e check passed.
- **Prefill** (32 cells): the adopted 8x64 path is +3.80 % against fo6-ko (median) and −1.27 % against the paper 8x64
  kernel (24 cells better in every round, none worse). By shape, against fo6-ko:
  - +0.5 % at 1x128 and +2.0 % at 1x256;
  - +7.4 % at 1x512 and +5.7 % at 1x1024;
  - +3.7 … +4.8 % at 1x2048 … 1x8192.
- **Decode** (18 cells): −0.32 % tokens per second against fo6-ko (range −1.5 … +0.1 %), +18.1 % against the paper 8x64
  kernel.
- **The in-graph split:** at Llama 1x2048 / 1x4096 the adopted GEMMs run at a 3.8–4.9 % lower SM clock than fo6-ko's.
  Their in-graph gap is +9.1 / +11.0 %, against M1's +7.2 / +7.6 %.
- **Phi-4 1x512:** the adopted path is +0.77 % against the paper kernel, with rounds −0.6 … +1.1 %.
  - It is not D4's idle artifact (23 of 24 captures are about 2.3 ms), not the clock, and not the tags.
  - In-graph, its GEMMs are −2.2 % and the adjacent non-GEMM kernels +2.7 %. The item stays open.
- **Results:** `results/kernel_opt/w8/p7/REPORT.md`. This closes the 8x64 plan (`results/kernel_opt/w8/CLOSING.md`).

## Amendment 16: the E0M3-fraction sweep on the adopted 8x64 path (C3w)

Written 2026-10-03, before any registered GPU run of the sweep. The protocol is `results/kernel_opt/c3w/PROTOCOL.md`
and the hashes are in `results/kernel_opt/c3w/registration.json`.
- **The request:** the user, through the coordinator: C3k (amendment 8) for the adopted 8x64 path, so the 16x64 and
  8x64 curves are directly comparable. GEMM only.
- **Kernels:**
  - the adopted 8x64 path with #2's dispatch (`'auto'`) and with the default dispatch;
  - its no-dispatch ceiling (P5);
  - the paper n8k64_wB.
- **References:** `stock_ko`, `stock_wB_ko` and the paper stock.
- **Descriptive only.** There are no new builds.
- **Disclosed:** a smoke test of the new scripts.

### Amendment 16: the run

`run_c3w.sh` ran on 2026-10-03 from 06:27 to 06:30 UTC (registration 91db345), with no deviation.
- G0 passed, and all 306 bitwise checks were equal.
- **At f = 0 the 8x64 gap is mostly the tiles** (the ceiling vs `stock_ko`):
  - +2.4 … +5.7 % at T ≥ 512;
  - +10.3 / +14.0 % in two single-wave cells;
  - about 0 … +2 % at T = 128.
- **The dispatch** adds +0.3 … +3.0 % (#2's), about as on 16x64.
- **#2's dispatch wins at f ≤ 2–5 %,** where the real maps lie (1.6–2.9 %); the default wins from about 10–25 %.
- **Results:** `results/kernel_opt/c3w/REPORT.md`.
