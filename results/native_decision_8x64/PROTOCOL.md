# Native-decision calibration at 8x64 (Task 1b) — protocol

Written 2026-09-24, before either run. This repeats Task 1
(`results/native_decision/PROTOCOL.md`, registered 10:59 UTC with sha256 ad688841…) at the 8x64
unit.

Everything is reused unchanged: the implementation, the verification, the evaluation protocol and
the PPL-only criterion. This covers `run_multiround.py` as committed in bb997ea and
`repro_local/realquant/native_dev.py`; no code changes are made for this task. Deviations are
appended in the last section.

## Granule

The unit is set by `--unit` only. The native kernel `libb8x64.so` (sha256 0e237ada…) puts the
weights on operand B with a format granule of 8 output rows × 64 K, so one 8x64 map tile is
exactly one kernel granule.

## Runs (deterministic, same data and settings as Task 1 except the unit)

| run | flags |
|---|---|
| DET-FAKE-8x64 | `--unit 8x64 --objective kl --skip-ce-backward --eval-batch 16 --score-batch 8 --budget-hours 12 --deterministic --dev-backend fake` |
| DET-NATIVE-8x64 | the same with `--dev-backend native` |

Both run with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `--data-root
/home/dev/n16k64_campaign/cost_comparison/data` and `--transformers-deviation`.

Reported for both runs:
- the first point of divergence (round, step size, fake vs native ΔKL);
- rounds, development evaluations and final E0M3 tiles;
- setup, optimization and per-try development-evaluation time;
- peak GPU and host memory.

## Verification (as Task 1)

- Each native-backend run checks, bitwise at startup:
  - the packed candidates of all 224 matrices;
  - the start map, a random mixed 8x64 map and the restored start map;
  - the first 64 activation calls.
- Each evaluation checks every evaluated map and its first 64 activation calls.
- The fake `--evaluate-map` evaluation of each DET map must reproduce that run's built-in fake
  final evaluation bitwise, per window.

## Evaluation

- **Maps:** FourOverSix, DET-FAKE-8x64 and DET-NATIVE-8x64. B_8x64_opt (the cost study's
  B-8x64-opt, stopped after 2 rounds) is incomplete and is excluded.
- **Backends:** every map on the native kernel (primary) and with fake (secondary), through
  `--evaluate-map`, in the default mode, one process per backend.
- **Statistics:** paired per-window ΔNLL, mean ± 2 SE (141 WikiText / 256 C4 windows), against
  FourOverSix and between the two DET maps, plus PPL.

## Criterion (judged on the native evaluation)

- **SAFE:** DET-NATIVE-8x64 − DET-FAKE-8x64 is within ±2 SE or significantly better, on both
  WikiText-2 and C4.
- **NOT SAFE:** significantly worse (mean − 2 SE > 0) on either corpus. Reported as measured.

## Context only (not a criterion)

- Both 8x64 maps are compared with Task 1's 256x64 DET maps (DET-FAKE, DET-NATIVE) under native
  evaluation, by paired per-window ΔNLL.
- The two unit sizes cannot share one `--evaluate-map` process, because the map shapes differ.
  The pairing therefore uses Task 1's native per-window NLLs (`results/native_decision/runs/
  eval_native`).
- This cross-process pairing is valid only if this task's native FourOverSix evaluation
  reproduces Task 1's bitwise, per window. That is checked. If it fails, the context comparison
  is dropped and the failure reported.

## Rules

- The uncommitted zero-shot work (Task 2, on hold) is not touched and not committed.
- Nothing is selected or tuned on WikiText or C4.
- Only this task's files are committed on `repro/n16k64-rtx-pro-6000` after REPORT.md is
  complete. Nothing is pushed.

## Deviations (append-only)

(none yet)
