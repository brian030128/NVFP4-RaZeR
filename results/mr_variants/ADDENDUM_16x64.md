# MR-OPT variants — addendum: the 16x64 unit

Written 2026-09-26, before any 16x64 test or run. The hash and registration time are in
`registration_16x64.json`. Deviations are appended in the last section, never edited in place.
[PROTOCOL.md](PROTOCOL.md) and its decision stand; this addendum adds a third unit.

**Why (user request, relayed by nvfp4-razer-c9).** The SM120 deployment kernel
(`origin/SM120-kernel`, config `n16k64_wA`) uses a 16x64 weight granule.
- **Models:** Llama-3.1-8B, Mistral-7B-v0.3 and Phi-4 only.
- **The Qwen3.8-27B gate stays closed.**

## Code changes

Nothing else changes:
- **`run_multiround.py`:**
  - `UNITS` gains `'16x64': (16, 64)`.
  - An 8x64 evaluation process accepts 16x64 maps the way it accepts 256x64 maps. Each 16x64
    tile becomes its two 8x64 tiles, and the element masks are checked equal.
- **`repro_local/realquant/native_dev.py`:** `NativeDev` accepts 16x64, a union of two 8x64 granules
  of the b8x64 build.
- **`repro_local/realquant/tile_score.py`:** B1 accepts 16-row tiles, four per 64-row program
  block. The kernel code itself is unchanged; it was already written for any tile height that
  divides 64.
- **`repro_local/realquant/test_tile_score.py`:** new options `--rows` and `--models`. The defaults
  are Phase 2's test.
- **New scripts:** `check_16x64.py` (pre-run checks) and `analyze_16x64.py` (tables).

## Development numerics vs the deployment path

**What is established:**
- **Native evaluation of 16x64 maps.** They are evaluated on the b8x64 build as unions of 8x64
  granules. Every map's native weight equals, bitwise, the weight `apply()` installs. The start,
  random-mixed and restored maps are checked at the start of every run.
- **The repro study** (`repro_local/REPORT.md`) found that the `wt_as_A` build (weights on A, 16x64
  granule) and the `b8x64` build give bit-identical per-window NLL after the contiguity fix. That
  result is limited to:
  - FourOverSix (all-E2M1 weights), on all five models;
  - the repro's own evaluator (`rq.RealLinear`, per-token activation scales).

**What is not established:**
- **Maps with E0M3 tiles:** the two builds were never compared on them, since their granules
  differ.
- **The current development evaluator (`native_dev.py`)** supports only b8x64 and was never run
  with `wt_as_A`.
- **The SM120 deployment kernel** (`origin/SM120-kernel`, `n16k64_wA`), per its `sm120/NUMERICS.md`:
  - it uses per-token activation global scales and a fused one-rounding epilogue;
  - that document states that the repro harness's two-rounding path is not bit-identical to it.

  This study's evaluator uses per-document (in evaluation, per-window) activation global scales,
  convention (a). It applies the activation scale after a bf16 GEMM output, which rounds twice.

**So the development numerics are not those of the deployment path.** The 16x64 maps have the
deployment kernel's granule, but they are selected and evaluated under this study's conventions.

## Pre-run checks

All run before any calibration run; the queue stops on any failure.

1. **B1 unit tests at 16x64**
   (`test_tile_score.py --rows 16 --models llama8b mistral7b phi4`):
   - **Cases:** one real matrix of every distinct layer-0 shape of the three models, and a random
     matrix of each shape.
   - **Setting:** a random 30 % E0M3 map; 16 batches of 8 × 512 tokens.
   - **Criterion:** normwise relative error ≤ 1e-6 for g, μ and SE in every case (Phase 2's
     tolerance).
   - **Also reported:** the max elementwise relative error, candidate-set agreement and time.
2. **Round-0 candidate sets at 16x64, legacy hook vs B1**, on Llama-3.1-8B, Mistral-7B-v0.3 and
   Phi-4:
   - **Runs:** two per model, `--stop-after-scoring --dump-round0-scores` with the MR-OPT settings,
     without and with `--tile-score-kernel`.
   - **Reported:** the candidates under each hook, their symmetric difference, the max normwise
     differences of μ and SE, and how far disagreeing tiles lie from the threshold.
   - **Criterion:** every disagreeing tile lies within noise of the threshold:
     |μ + 2 SE| ≤ 1e-5 · max|μ| of its matrix (10× the unit-test tolerance).
3. **Native map build at 16x64:** in each of those runs, the native weights of the start,
   random-mixed and restored maps must equal `apply()`'s bitwise. The run asserts this.
4. **The existing code path is unchanged.** A Llama-3.1-8B 8x64 run
   (`--stop-after-scoring --dump-round0-scores`, MR-OPT settings) must reproduce, bitwise:
   - Phase 2's round-0 score file (sha256);
   - the MR-OPT 8x64 run's initial development values, fake and native.

## Runs

- **Settings:** those of MR-OPT and its variants at the other units ([PROTOCOL.md](PROTOCOL.md)), with
  `--unit 16x64`.
- **Order:**
  1. MR-OPT 16x64 on Llama-3.1-8B, then Mistral-7B-v0.3, then Phi-4.
  2. The evaluations of the MR-OPT 16x64 maps (below). Commit and report.
  3. MR-OPT+SIG, +WS and +SIG+WS at 16x64: Llama, then Mistral, then Phi-4, each model followed by
     its variant evaluation. The user may cut the variants; the queue checks a stop file before
     each variant run.
  4. Commit, report, and stop at the Qwen3.8-27B gate.
- **Evaluation settings:** convention (a); 2,048-token windows, one per forward; lean; an 8x64
  process with `--fused-act-quant --single-pass-epilogue`. The 16x64 and 256x64 maps are expanded
  exactly to 8x64 tiles.
- **Evaluations:**
  - **Native, after MR-OPT:** FourOverSix and MR-OPT 256x64, 8x64 and 16x64, in one process per
    model.
    - **Reported:** PPL; paired ΔNLL ± 2 SE vs FourOverSix; MR-OPT 16x64 minus MR-OPT 8x64, and
      minus MR-OPT 256x64.
    - **Repeat check:** the FourOverSix, 8x64 and 256x64 window NLLs must repeat the earlier
      evaluation bitwise.
  - **Fake:** FourOverSix and MR-OPT 16x64. The FourOverSix NLLs must repeat the earlier fake
    evaluation.
  - **Native, after the variants:** FourOverSix, MR-OPT 16x64 and the three variants, in one
    process per model. Reported: PPL, paired ΔNLL vs FourOverSix and vs MR-OPT 16x64, and tile
    overlap with MR-OPT 16x64.
- **Calibration metrics per run:** rounds, development evaluations, tiles, development KL,
  optimization and setup time, peak memory.

## Decision

- **The registered recommendation stands.** Over 8x64 and 256x64 it is MR-OPT.
- **16x64 acceptability is additional information.** It is reported per variant under the official
  rule (mean − 2 SE ≤ 0 vs MR-OPT 16x64, on both corpora), with the secondary reading too. It
  cannot reverse the recommendation.

## Rules

- **Commit and push:** commit as chenjiaj109550158 <chenjiaj.cs13@nycu.edu.tw> twice: after the
  MR-OPT 16x64 runs with their evaluations, and after the variants. Try once to push
  `repro/n16k64-rtx-pro-6000` (never main, no force). If the push is blocked, report it and do not
  retry.
- **No selection or tuning** on WikiText-2, C4 or zero-shot.
- **Scope:** the Qwen3.8-27B gate stays closed. Part C, QAT and zero-shot stay paused.

## Deviations (append-only)

1. **2026-09-26 06:44 UTC, during the MR-OPT 16x64 evaluations: the variants at 16x64 were cut by
   user decision** (relayed by nvfp4-razer-c9, who created the stop file).
   - **Queue:** it stopped before the first variant run, at 06:50:37 UTC, and logged the gate.
   - **Not run:** no variant run at 16x64 was started.
   - **Complete:** the MR-OPT 16x64 runs and their evaluations.
