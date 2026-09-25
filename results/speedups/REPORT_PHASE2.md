# Calibration speed-ups — Phase 2: the fused tile-score kernel B1

**Verdict: B1 is adopted.** It is judged under [PROTOCOL_PHASE2.md](PROTOCOL_PHASE2.md), registered
2026-09-25 14:39:51 UTC (sha256 fe595eca…), before any Phase 2 test or run.

- **End to end** on Llama-3.1-8B, with items 1, 3 and B1 on, DET-NATIVE reaches **exactly the
  committed maps** at both units (identical `map.pt` sha256).
- **PPL** is therefore identical on both backends: ΔNLL = 0 on every WikiText-2 and C4 window.
- **Speed:** B1 cuts the scoring pass by another 26–30 %.

Against Part B's lean code paths, the three items together make Llama calibration **21–23 % faster**:

| Llama DET-NATIVE, lean, batch 16/8 | 256x64 | 8x64 |
|---|---:|---:|
| scoring per pass: Part B → items 1+3 → + B1 | 60.2 → 48.4 → **35.6 s (−41 %)** | 60.2 → 48.7 → **34.1 s (−43 %)** |
| native development evaluation per try | 12.97 → 11.44 → 11.40 s | 13.02 → 11.43 → 11.40 s |
| optimization | 21.6 → 18.5 → **16.7 min (−23 %)** | 33.8 → 29.0 → **26.8 min (−21 %)** |
| peak GPU allocated / reserved (+ B1) | 48.1 / 55.8 GiB | 49.2 / 55.4 GiB |
| peak host RSS (+ B1) | 44.0 GiB | 44.2 GiB |

The flag stays opt-in: `--tile-score-kernel`, lean mode.

## What B1 is

`repro_local/realquant/tile_score.py` computes each tile's score in one fused step.

- **Batching:** one launch per matrix per backward call covers every sequence of the batch.
- **Per program:** each program owns a block of rows and one 64-column tile. It:
  - decodes D from the packed candidates (bitwise the legacy D);
  - forms G's block with IEEE-FP32 `tl.dot` over the 512 tokens (products of bf16 values are exact
    in FP32; no TF32 or BF16 math);
  - multiplies by D and reduces each tile;
  - adds each sequence's g into the FP64 sum and sum of squares, in sequence order as before.
- **Never materialized:** G, D and G⊙D.
- **The only numeric difference** is the FP32 summation order inside G and the tile reduction.

## 1. Unit numeric tests (`phase2_unit_tests.json`): within the registered tolerance

- **Cases:** 80 layer/unit cases. That is one real matrix of every distinct shape from the four
  final models (layer 0; Qwen3.8-27B layers 0 and 3) and a random matrix of each, at 8x64 and
  256x64.
- **Setting:** a 30 % E0M3 map and 128 sequences in 16 batches of 8.

| error vs the legacy hook | g (per sequence, tile) | μ | SE |
|---|---:|---:|---:|
| max normwise relative (tolerance ≤ 1e-6) | 3.6e-7 | 2.8e-7 | 6.6e-8 |
| max elementwise relative (tiles with \|ref\| ≥ 10⁻³·max) | 9.7e-5 | 1.3e-4 | 9.5e-8 |

- **Candidate sets** {μ + 2 SE < 0}: 0 disagreements out of 3,946,336 tiles.
- **Where the larger elementwise errors are:** on tiles whose sums nearly cancel.
- **Speed:** summed over all cases, one batch costs 1,048 ms in the legacy hook and 579 ms in B1
  (1.8× faster).

## 2. Round-0 candidate-set agreement on real data (batch 16/8, lean)

| model, unit | tiles | candidates (legacy = B1) | disagreeing tiles | max normwise rel. error μ / SE |
|---|---:|---:|---:|---|
| Llama-3.1-8B, 256x64 | 425,984 | 15,136 | 0 | 1.1e-6 / 5.5e-7 |
| Llama-3.1-8B, 8x64 | 13,631,488 | 379,759 | 0 | 6.0e-7 / 8.0e-7 |
| Phi-4, 8x64 | 26,624,000 | 700,279 | 0 | 2.2e-7 / 2.3e-7 |

- **Llama 256x64:** μ's error of 1.1e-6 is summation-order noise over a 256x64 tile's 16,384
  products per token. It is reported here, not a criterion; the registered tolerance applies to
  the unit tests.
- **Qwen3.8-27B:** skipped, as registered. It does not fit at batch 16/8 in lean mode (Phase 1).
  The chunked-loss change that addresses that is a separate task.

## 3. End-to-end, Llama-3.1-8B (items 1, 3 and B1 on)

| unit | rounds / dev evaluations | tiles | tile overlap with committed | map |
|---|---:|---:|---|---|
| 256x64 | 8 / 64 | 8,385 | 8,385 shared, 0 only committed, 0 only B1 | `map.pt` sha256 equal to committed |
| 8x64 | 9 / 115 | 3,801 | 3,801 shared, 0 / 0 | `map.pt` sha256 equal to committed |

| native evaluation (convention (a); criterion) | WikiText-2 | C4 | B1 − committed ΔWiki | ΔC4 |
|---|---:|---:|---|---|
| 256x64 | 6.836861 | 9.759441 | 0 (every window equal) | 0 |
| 8x64 | 6.813365 | 9.764371 | 0 | 0 |

- **Fake evaluation:** also identical (6.831542 / 9.773669 and 6.817782 / 9.760603).
- **Adoption criterion:** met at both units, on both corpora (within ±2 SE, trivially).

## Notes

- **Registered code:** every Phase 2 run recorded source hashes equal to `registration_phase2.json`.
- **Peak memory:** B1 does not move the peak. The peak sits in the forward and backward activations
  of scoring, not in the hook.
- **B1's exact 256x64 reproduction is not guaranteed in general.** At 256x64 the search took the
  same path because no decision was near enough to a threshold for a ~1e-7 relative score change to
  flip it. On other models or data a noise-level decision could flip, as the protocol allowed.

## Reproduction

```
python repro_local/realquant/test_tile_score.py results/speedups/phase2_unit_tests.json
/home/dev/n16k64_campaign/speedups/queue_phase2.sh           # copy in runs_phase2/
python results/speedups/analyze_phase2.py                    # summary_phase2.json, tables_phase2.md
```
