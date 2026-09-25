# Probe: Qwen3.8-27B with development evaluation at batch 16 and scoring at batch 2 (option B')

A feasibility and timing probe (user request, 2026-09-25); not a calibration.

**Setup.**
- **Settings:** Qwen3.8-27B, 8x64, `--objective kl --dev-backend native --skip-ce-backward
  --deterministic --memory-mode lean`, items 1, 3 and B1 on (`--fused-act-quant
  --single-pass-epilogue --tile-score-kernel`), `--chunked-loss`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Data:** Part C's Qwen3.8-27B data.
- **Scope:** `--stop-after-scoring`, so each run does one full round-0 scoring pass (128
  sequences).
- **Records:** `probe_qwen27b/`, the report.json of both runs.

**Result: score batch 2 FITS, with about 2 GiB of the 95 GiB GPU to spare. Score batch 1 fits
easily.**

| phase | batch 16 / 2: peak allocated / reserved / host | time | batch 16 / 1: peak allocated / reserved / host | time |
|---|---|---:|---|---:|
| model load | 51.0 / 51.0 / 51.4 GiB | 4 s | 51.0 / 51.0 / 52.1 GiB | 4 s |
| teacher precompute | 52.2 / 52.4 / 78.2 GiB | 202 s | 52.2 / 52.4 / 78.2 GiB | 199 s |
| candidate packing | 56.2 / 57.1 / 78.8 GiB | 139 s | 56.2 / 57.2 / 78.8 GiB | 139 s |
| initial fake development evaluation, batch 16 | 40.0 / 54.8 / 79.0 GiB | 54 s* | 40.0 / 55.7 / 78.9 GiB | 54 s* |
| initial native development evaluation, batch 16 | 53.1 / 54.8 / 79.3 GiB | **43 s** | 53.1 / 55.7 / 79.3 GiB | 43 s |
| **round-0 scoring pass** | **91.2 / 92.7** / 80.0 GiB | **209 s** | **62.3 / 62.9** / 80.0 GiB | **360 s** |

\* This phase also counts a few seconds of bookkeeping around the native evaluation.

- **Scoring batch 2 is 1.72× faster than batch 1** (209 vs 360 s per pass).
- **The speed-ups at batch 1:** Phase 1's batch-1 run (items 1 and 3 only, whole-batch loss) took
  478 s per pass; B1 plus the chunked loss bring it to 360 s.
- **Memory per sequence:** each additional 512-token scoring sequence stores about 29 GiB of
  activations. That is why scoring batch 4 does not fit (`REPORT_CHUNKED.md`).
- **Determinism check:** the per-document development values are bitwise identical across the two
  probes, as expected (same evaluation batch; deterministic mode).
