# Calibration speed-ups — Phase 2 protocol: item 2, the fused tile-score kernel B1

Written 2026-09-25, during Phase 1 and before any Phase 2 test or run. The hash and registration
time are in `registration_phase2.json`. Deviations are appended in the last section.

## What changes (`--tile-score-kernel`, off by default; lean mode only)

`repro_local/realquant/tile_score.py` replaces the per-sequence loop of the scoring backward hook:
`G_s = dy_sᵀ x_s` (FP32), then `G_s ⊙ D`, then the tile reduction, then FP64 accumulation.

- **One launch per matrix and backward call covers the whole batch.** Each program owns a block of
  rows and one 64-column tile:
  - it decodes D from the packed candidates (bitwise the legacy D);
  - it forms G's block with IEEE-FP32 `tl.dot` (no TF32 or BF16 math) and multiplies by D;
  - it reduces each tile, and adds each sequence's g into the FP64 sum and sum of squares, in
    sequence order.
- **Nothing is materialized:** no full-size G, D or G⊙D.
- **The only numeric difference allowed** is the FP32 summation order inside G and the tile
  reduction. Scores can therefore move by a few ULPs, and the search path and map can change at
  noise level.
- Items 1 and 3 (Phase 1) stay on in every Phase 2 run.

## Tests and criteria

1. **Unit numeric tests** (`repro_local/realquant/test_tile_score.py`), against the legacy hook.
   - **Layers:** one real matrix of every distinct shape from layer 0 (Qwen3.8-27B: layers 0 and
     3) of the four final models, and a random matrix of each shape.
   - **Units:** 8x64 and 256x64.
   - **Setting:** a random map with 30 % E0M3 tiles, and 16 batches of 8 sequences × 512 tokens
     (128 sequences).
   - **Reported:** max absolute, normwise relative (max|Δ| / max|ref|) and elementwise relative
     error (over tiles with |ref| ≥ 10⁻³·max|ref|), for g (first batch, per sequence and tile), μ
     and SE.
   - **Registered tolerance:** normwise relative error ≤ 1e-6 for g, μ and SE on every layer and
     unit. Exceeding it is reported. B1 is then not adopted unless the excess is explained as
     summation order and the end-to-end criterion (4) passes.
2. **Timing:** one batch, legacy hook vs B1, per layer.
3. **Round-0 candidate-set agreement.** Two runs per model at batch 16/8, lean, 8x64,
   `--stop-after-scoring --dump-round0-scores`: items 1 and 3 on, B1 off vs on. For Llama you
   reuse Phase 1's runs and add one B1 run.
   - **Models:** Llama-3.1-8B, Phi-4 and Qwen3.8-27B.
   - **Reported:** the tiles with μ + 2 SE < 0 under each (the flip candidates), their
     symmetric difference, and how far the disagreeing tiles lie from the threshold.
   - **Qwen3.8-27B:** if it did not fit at batch 16/8 in Phase 1, it is skipped here too, with the
     memory recorded.
4. **End-to-end, Llama-3.1-8B.** DET-NATIVE-8x64 and DET-NATIVE-256x64 with
   `--fused-act-quant --single-pass-epilogue --tile-score-kernel`, otherwise the Phase 1 runs.
   - **Reported:**
     - rounds, tries and tiles;
     - the tile overlap with the committed maps (intersection, and tiles only in either);
     - time and memory against Phase 1;
     - native evaluation (convention (a), per-document scales, `--evaluate-map`, one process for
       both units, the 256x64 maps expanded exactly): PPL and paired per-window ΔNLL ± 2 SE,
       B1 map minus committed map.
   - **Adoption criterion:** B1 is adopted if its map is within ±2 SE of the committed map, or
     significantly better, on both WikiText-2 and C4, at both units. If it is significantly worse
     on either corpus at either unit, it is not adopted, and the report says why.

## Rules

As Phase 1. Commit after Phase 2 with its report, as chenjiaj109550158
<chenjiaj.cs13@nycu.edu.tw>, then push `repro/n16k64-rtx-pro-6000` (never main, never force). Part
C, QAT and zero-shot stay paused/untouched. Nothing is tuned on WikiText-2 or C4.

## Deviations (append-only)

(none yet)
