# Native-decision calibration: a safe-transfer test — protocol

Written 2026-09-24, before the verification measurements and before the two calibration runs.
The criteria below are fixed. Deviations are appended in the last section with their reason.

**Question.** The shadow study (`results/native_dev_shadow/`) found that the native kernel
would have changed 11 of 72 backtracking decisions. Can multi-round calibration still safely
make its development decisions on the native kernel? "Safely" is judged by the quality of the
final map on the native kernel: it must not be significantly worse than the fake-decided maps.

## Setup

- Same machine, environment, data root and flags as the cost study: 1× RTX PRO 6000, env
  `n16k64`, `--data-root /home/dev/n16k64_campaign/cost_comparison/data
  --transformers-deviation`.
- Native kernel: `libb8x64.so` (sha256 0e237ada…).
- **Native development evaluator:** the verified shadow evaluator,
  `repro_local/realquant/native_dev.py`, eval batch 16. Each document gets its own
  FourOverSix activation scale.
- **Native final evaluator:**
  - The same WikiText-2 windows and C4 crops as the fake final evaluation: `data(tok, prior,
    2048)` plus `validate_evaluation_data` against the published report.
  - One 2,048-token window per forward, with the window's tensor-wide FourOverSix activation
    scale (as `per_document_act` at batch 1). The weights are the packed map.
  - `lm_head` stays BF16. WikiText runs with `use_cache=True`, as in fake.
- **New flags in `run_multiround.py`**, all defaulting to today's behavior:
  - `--dev-backend fake|native`: which evaluator's development KL decides every step. With
    native, the fake development KL is logged once, at the start.
  - `--eval-backend fake|native`: the final-evaluation backend.
  - `--evaluate-map LABEL=PATH`: evaluates saved maps without calibrating. PATH may also be
    `fourover6` or `bf16`.
  - `--deterministic`: `torch.use_deterministic_algorithms(True)`, with
    `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Verification (before any measurement; a failure stops the study until it is fixed)

1. **Bitwise:**
   - the packed candidates decode to `decode_base` / `decode_alt` for all 224 matrices;
   - the packed map weight decodes to the weight `apply()` installs, at the start map, at a random
     mixed map, at the restored start map, and at every evaluated map;
   - the first 64 native activation calls of every evaluated map equal `quant_per_document`.
2. **The fake `--evaluate-map` path reproduces the recorded fake final evaluations bitwise, per
   window:** FourOverSix against the cost study's arm A, and the B-256-opt map against the
   B-256-opt run. The evaluation path is then the calibration path's.
3. **Native vs fake WikiText-2 / C4 NLL at the FourOverSix and B-256-opt maps.** A small
   systematic offset is expected: the development pre-check showed +1.1–1.6% in KL. **Bug
   threshold, fixed now:** a mean per-window NLL difference larger than 0.01 nats (about 1% PPL)
   on either corpus counts as an implementation bug.

## Calibration runs (both deterministic)

The deterministic setting was verified in the cost study
(`runs/determinism_check_deterministic.json`): FlashAttention-2 runs its deterministic
backward, and the KL weight gradients of a scoring batch are bitwise identical across repeats.
So the two runs differ only in the development backend.

| run | flags |
|---|---|
| DET-FAKE | `--unit 256x64 --objective kl --skip-ce-backward --eval-batch 16 --score-batch 8 --budget-hours 12 --deterministic --dev-backend fake` |
| DET-NATIVE | the same with `--dev-backend native` |

Logged for both runs:
- per-round trajectories (candidates, tries, accepted sizes, tiles, development KL);
- time (setup, optimization, per development evaluation);
- per-phase GPU and host memory.

Reported:
- the first point where the two paths diverge (round, try);
- the optimization-time speed-up of DET-NATIVE over DET-FAKE.

## Evaluation (after both runs)

- **Maps:**
  - FourOverSix (all E2M1);
  - DET-FAKE and DET-NATIVE;
  - the existing fake-decided 256×64 maps: B-256-opt, B'-256-opt, SHADOW (the shadow run's map)
    and B-256-ref.
- **Backends:** every map on the native kernel (primary) and with fake (secondary), plus BF16
  once (fake; used in the Task 2 report).
- **How:** all evaluations go through `--evaluate-map` in the default (non-deterministic) mode, one
  process per backend, so every map is evaluated by the same code and settings. The DET runs' own
  final evaluations (deterministic mode) are recorded, but the tables do not use them.
- **Statistics:** paired per-window ΔNLL, mean ± 2 SE (SE = sd(ddof = 1)/√n; 141 WikiText / 256
  C4 windows), against FourOverSix and against DET-FAKE, plus PPL. A difference is "worse" if
  mean − 2 SE > 0, "better" if mean + 2 SE < 0, and inconclusive otherwise.

## Criterion (judged on the native evaluation)

**SAFE** requires both:
1. DET-NATIVE − DET-FAKE is not significantly worse on WikiText-2 or on C4. It may be
   inconclusive or significantly better.
2. For every fake-decided map M in {DET-FAKE, B-256-opt, B'-256-opt, SHADOW, B-256-ref},
   DET-NATIVE − M is not significantly worse on either corpus. This paired difference equals the
   difference of the two maps' ΔNLL vs FourOverSix, window by window.

If DET-NATIVE is significantly worse on either corpus against any of these, the result is **NOT
SAFE**, reported as measured.

**Caveats, stated in the report:**
- There is one selection per backend. B vs B' in the cost study showed that different paths
  reach maps of statistically equal quality, and the paired ±2 SE does not include this
  selection variance.
- The 192 development documents decide every step and are not held out.
- Nothing is selected or tuned on WikiText, C4 or zero-shot results.

## Deviations (append-only)

1. **2026-09-24 11:10 UTC: `native_dev.py` gained two opt-in options after registration.**
   Both are for Task 2's zero-shot runs:
   - `alt_signed_zero`, for bitwise checks against `quant_mix_4_6`'s signed zeros;
   - a configurable activation-check `reference`.

   The defaults are unchanged, and every Task 1 run uses the defaults.
   B'-256-opt carries the label `Bprime-256-opt` in the run files.
