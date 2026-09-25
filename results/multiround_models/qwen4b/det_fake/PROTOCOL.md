# Qwen3-4B native-decision safety test (addendum to Part C) — protocol

Written 2026-09-25, while Mistral-7B's calibration runs and before any run of this addendum. The
hash and registration time are in `registration.json`. Deviations are appended in the last
section.

## Why

Qwen3-4B's MixFP4 256x64 map, whose backtracking decisions were all made on the native kernel
(DET-NATIVE), is significantly worse than FourOverSix on C4 under native evaluation
(+0.0074 ± 0.0021 nats per window; `../REPORT.md`). Native-decision safety was tested only on
Llama-3.1-8B before (`results/native_decision`, `results/native_decision_8x64`). This test asks
whether the C4 loss comes from native decisions, or from the model and unit themselves.

## Runs (after Mistral-7B's Part C queue; before Phi-4)

**DET-FAKE-256x64 first, then DET-FAKE-8x64.** Each uses exactly the flags of its DET-NATIVE run
in `../runs`, except `--dev-backend fake`:

```
run_multiround.py --model qwen4b --objective kl --data-root /home/dev/n16k64_campaign/multimodel/data
  --memory-mode lean --transformers-deviation --unit U --dev-backend fake --skip-ce-backward
  --deterministic --eval-batch 1 --score-batch 1 --budget-hours 12 --record-dev-values
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The code is Part C's (commit 6587a5b). The data is the same, and the same checks run at start.

**Evaluation.** One process per backend per unit:
- native (primary) and fake (secondary);
- `--memory-mode lean --unit U`;
- maps FourOverSix, DET-NATIVE-U (Part C's `calib_U/map.pt`) and DET-FAKE-U.

The 256x64 evaluation runs right after DET-FAKE-256x64 and before DET-FAKE-8x64, so its result
can be reported first. FourOverSix's per-window NLL must equal Part C's evaluation bitwise. That
makes pairing across processes valid; if it fails, cross-process comparisons are dropped and the
failure reported.

## Criterion (per unit, native evaluation), the Llama Task 1/1b rule

Paired per-window ΔNLL of DET-NATIVE minus DET-FAKE, mean ± 2 SE (SD with ddof = 1, over √n
windows; 146 WikiText-2 and 256 C4 windows).

- **SAFE:** within ±2 SE, or significantly better, on both WikiText-2 and C4.
- **NOT SAFE:** significantly worse (mean − 2 SE > 0) on either corpus. It is reported as
  measured.

## Reported per unit

- **Where the paths first diverge:** round, try and step size, with the fake and native
  development KL change.
- **Run statistics:** rounds, development evaluations, tiles, time (setup, optimization, per
  pass, per try) and peak GPU and host memory, for both runs.
- **PPL:** native (primary) and fake (secondary) WikiText-2 and C4.
- **Paired ΔNLL:** against FourOverSix, and DET-NATIVE minus DET-FAKE.
- **For 256x64:** whether DET-FAKE-256x64 is also significantly worse than FourOverSix on C4.

## Rules

- **Commit:** on `repro/n16k64-rtx-pro-6000` with the usual identity. Nothing is pushed.
- **Out of scope:** zero-shot work stays on hold and out of the commit.
- **No tuning:** nothing is selected or tuned on WikiText-2 or C4.
- **Afterwards:** Part C resumes unchanged with Phi-4, then Qwen3.8-27B.

## Deviations (append-only)

(none yet)
