# TM-OPT verification (Llama-3.1-8B) — status report

**Status (2026-09-26 07:30 UTC): STOPPED at the unit tests, as the protocol requires.** One of the
registered unit tests failed.
- **What failed:** the B1 tile-sum test.
- **Not run:** group 1 (bitwise, B1 off) and group 2 (B1, end to end).
- **Waiting:** for the user's decision on B1.

The protocol is [PROTOCOL.md](PROTOCOL.md), registered 2026-09-26 07:06:31 UTC (sha256 bc3702b3…),
before any test or run. See deviation 1 there.

## What is built (branch `tm-opt`)

`run_train_map.py` (main's trained-map method, unchanged) gains the MR-OPT optimizations behind
flags:
- `--memory-mode lean`;
- `--fused-act-quant`;
- `--tile-grad-kernel` (B1);
- `--chunked-loss`;
- `--deterministic`;
- `--dev-backend native` and `--eval-backend native`, with `--single-pass-epilogue`.

Every default is the legacy behaviour. `--tm-opt` turns all of them on (the TM-OPT configuration);
`--no-<flag>` turns one off. It also runs locally with `--data-root` and `--transformers-deviation`,
without Slurm.

Two supporting functions were added:
- `chunked_loss.train_kl_gradient`: the chunked version of the training loss.
- `tile_score.tile_sums`: B1 for the batch tile sum. It is the Phase 2 kernel with the batch as one
  sequence and sign +1.

## Unit tests (`runs/unit_tests.json`)

| test | cases | result |
|---|---:|---|
| 1(a) chunked training loss vs the legacy step: logits gradient and per-sequence KL | 24 (vocabularies of 32,768, 100,352 and 128,256 × batches 1–8) | **PASS**, all bitwise |
| 1(b) B1 tile sums vs the legacy hook: normwise error ≤ 1e-6 | 72 (three models' layer-0 shapes, real + random, × 8x64 / 16x64 / 256x64) | **FAIL**, 2 cases over: 1.41e-6 and 1.39e-6; the rest ≤ 9.6e-7 |

## Diagnostic after the failure (not a criterion; `diagnose_b1.py`, `runs/diagnose_b1.json`)

Both paths against an FP64 reference, on the three models' layer-0 shapes at 8x64 and 16x64, over 4
batches of 8 × 512 tokens:

| | max normwise error vs FP64 | closer to FP64 |
|---|---:|---:|
| legacy hook (one FP32 cuBLAS GEMM per batch) | 1.09e-6 | 19 of 24 cases |
| B1 (`tile_sums`) | 1.32e-6 | 5 of 24 cases |

- **The tolerance is at noise level.** Both paths are about 1e-6 from the exact sums of
  4,096-token batches. The legacy hook itself would fail a 1e-6 tolerance against FP64. That
  tolerance was set in Phase 2 for 512-token per-sequence sums.
- **B1 is also slower here:**

  | path | one batch, five largest shapes |
  |---|---:|
  | legacy hook | 56 ms |
  | B1, default launch | 76 ms |
  | B1, best of 7 launch configurations | 76 ms |

  In MR-OPT, B1 replaced per-sequence GEMMs and FP64 statistics. The training step already needs
  only one GEMM per module per batch, so there is nothing for B1 to fuse away.

## Decision needed

- **(A) Recommended: take B1 out of TM-OPT.**
  - `--tile-grad-kernel` stays available (opt-in), but the `--tm-opt` preset leaves it off.
  - Every remaining TM-OPT optimization is designed to be bitwise-neutral: lean store, fused
    quantizers, chunked loss, deterministic mode. The native evaluator only changes the monitor
    and the final evaluation.
  - Verification: group 1 as registered. Group 2 becomes one full STE 8x64 TM-OPT run, for the
    per-epoch time and memory against legacy and main's H200 reference. There is no B1
    end-to-end comparison.
- **(B) Keep B1:** relax 1(b) to the measured noise level (for example 2e-6, or "no worse than 2×
  the legacy hook's error vs FP64"). Then run groups 1 and 2 as registered; group 2's end-to-end
  ΔNLL criterion decides. TM-OPT would then be slower than without B1.
- **(C) Stop here.**

Nothing else runs until the user decides.
