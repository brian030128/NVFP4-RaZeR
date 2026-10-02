# Kernel-opt 8x64 plan, P3 (with P4): the adopted 8x64 path's widths and scheduler rows, stock_wB tuned alike — report

2026-10-02, branch `kernel-opt`. The protocol is amendment 12 of `results/kernel_opt/PROTOCOL.md`: sources at 097d1e5,
registered at 6215946. `run_w8p3.sh` ran 16:31–17:19 UTC with no deviation. Every number is in `w8p3_tables.md` and
`w8p3.json`; the tables are in `table/` and `widths/`, the sensitivity in `sensitivity.json`.

## In brief

- **Every gate passed.** The re-tuned 8x64 set computes bitwise what P2's adopted path computes, at every width and
  scheduler setting; stock_wB tuned alike computes what stock_wB computes.
- **The registered 8x64 rule is NOT met.** One cell exceeds the 0.5 % tolerance.
  - After vs before, per forward: median −0.01 % (typical tags) and −0.08 % (worst tags) over the 48 (model, T) cells.
  - The cell over tolerance is Phi-4 at T = 64 with typical tags: +0.84 %, every round between +0.67 and +0.89 %.
- **The re-tune changes almost nothing.**
  - The only clear gain is at T = 128: −0.99 % typical, −1.22 % worst. It comes from two choices:
    - Llama-3.1-8B and Mistral-7B's k/v_proj (1024x4096) now run width 32 with raster along N: −11 to −12 % per GEMM;
    - Phi-4 and Qwen3.8-27B's o_proj run the '128x64' tile: −3 to −5 %.
  - Elsewhere the effect is −0.3 … +0.1 % by T.
  - 46 widths and 64 scheduler settings changed (110 cells).
- **stock_wB tuned alike meets its rule.** stock_wB_e64 with its rows is −0.52 % (median) against stock_wB, −0.8 % at
  T ≤ 128.
- **The decisive-margin sensitivity is negligible.**
  - Against the 128-wide default the rule would change 1 tuning cell, against the fallback width 4.
  - Per forward the effect is +0.04 % and +0.03 % (median).
- **The residual gap to stock_ko** (typical tags, median over the 4 models): +3.97 % with the P3 table, +4.35 % without.

| T | before (P2's path) | after (P3 table) | after, worst tags | after vs stock_wB tuned alike |
|---:|---:|---:|---:|---:|
| 1 | +0.24 % | +0.43 % | +1.05 % | −38.5 % |
| 4 | +0.34 % | +0.58 % | +1.07 % | −38.6 % |
| 16 | +0.94 % | +1.15 % | +1.63 % | −36.5 % |
| 32 | +2.06 % | +2.28 % | +2.68 % | −33.1 % |
| 64 | +1.95 % | +2.26 % | +2.42 % | −27.8 % |
| 128 | +2.69 % | +1.66 % | +2.38 % | −24.9 % |
| 256 | +5.27 % | +5.11 % | +6.54 % | −15.8 % |
| 512 | +9.45 % | +9.35 % | +10.82 % | +4.9 % |
| 1024 | +6.26 % | +5.93 % | +7.19 % | +3.8 % |
| 2048 | +7.00 % | +6.78 % | +7.89 % | +6.7 % |
| 4096 | +7.16 % | +7.21 % | +8.21 % | +7.2 % |
| 8192 | +7.29 % | +7.27 % | +8.49 % | +7.4 % |

These are per-T medians of each ratio separately, so the two columns need not differ by the "after vs before" medians.

**Run-to-run variation.** The same "before" path measured in amendment 11's run gave +0.76 / +0.55 / +0.82 % at
T = 1 / 4 / 16, against +0.24 / +0.34 / +0.94 % here. Small-T gaps move by about 0.5 % between runs.

## The gates

| gate | result |
|---|---|
| G0 provenance | 48 files, 68 builds as registered |
| G1 / G2 | the 50 existing configurations keep their SASS; stock_wB_e64 is E2M1-only, unpredicated; `build_P3freq`'s six likewise, the five t0 builds with their census |
| Same device code | `build_P3freq`'s five 8x64 builds carry `build_P2freq`'s SASS; its stock_wB_e64 carries `build_P3`'s |
| G3 pytest | 1,030 passed on `build_P3`; 228 passed on `build_P3freq` (211 skipped: builds not in that directory) |
| G4 bitwise vs n8k64_wB (sm120/build) | 28,800 comparisons with set:mixed_wB_ko on the P3 table (every scheduler row exercised), 0 differences |
| G5 logits, 4 models × 5 shapes | 8x64: set:mixed_wB_t0 (paper table) → set:mixed_wB_ko (P3 table) equal; FourOverSix: stock_wB → set:stock_wB_ko equal |
| M1 checks | every bitwise check equal (each 8x64 configuration against P2's path; stock_wB_ko against stock_wB) |

## What the tuning chose

- **Widths:** 22 of 224 (shape, bucket) cells differ from 1b's rows.
  - 18 are at buckets ≤ 256 and 4 at buckets 1024–8192. Bucket 512 is unchanged everywhere: the 128 × 128 tile stays
    fastest at T = 512.
  - The '128x64' tile is newly chosen in 7 cells, e.g. 5120x5120, 5120x6144 and 5120x17408 at T = 128.
- **Scheduler rows:**
  - `mixed_wB_ko`: non-default in 42 of 224 cells. Mostly raster along N, (2, 1), in 26 cells; raster along M, (1, 1),
    in 13.
  - `stock_wB_ko`: non-default in 24 cells.
- **The sensitivity** (`sensitivity.json`), with amendment 7's rule applied to the widths:
  - With the 128-wide default it would change one tuning cell: 17408x5120 at 256 uses 128 instead of '128x64', +0.06 %
    in the tuning.
  - With the fallback width it would change four, at most +0.44 % in the tuning.
  - Per forward in M1: +0.04 % (dm128) and +0.03 % (dmfb), medians. The decisive-margin rule and the fastest median
    give nearly the same table.

## Why the rule failed, and what M1 can resolve (post hoc, not part of the registered analysis)

**The flagged Phi-4 T = 64 cell** combines two changes:
- o_proj 32 → 64: +0.8 % typical, +2.0 % worst;
- a scheduler row (2, 1) on down_proj: +2.4 % typical, +1.1 % worst.

**But the same down_proj cell also measures +1.5 % under "widths only".** That configuration is an identical
computation to "before" (same width, default scheduler).

**An A/A check.** Over every M1 cell where two configurations run identical computations (same width and scheduler
setting; 1,654 pairs from the before / widths-only / sensitivity configurations):
- single-GEMM differences have a median |Δ| of 0.15 %, p90 0.51 %, max 3.0 % (Llama-3.1-8B v_proj at T = 8192);
- 15 % of the pairs are "separated", every round of one below every round of the other. Pure noise would give about
  10 %.
- The largest separated A/A differences, about 1.6–1.9 %, are down_proj at T ≤ 64.

So there are configuration-level offsets of up to about 2 % on single GEMMs that the round range does not capture, and
the flagged cell lies within them. The registered rule still decides; this only says the failure is not evidence of a
real regression.

## Maps and tags used

- The paper's TM-OPT+TC 8x64 artifacts and their maps, with sha256 in `registration_12.json`, plus the FourOverSix
  artifacts for the stock references.
- M1: typical is the lower-median module per projection, worst the densest.
- The tuning takes, per projection, the module with the most E0M3 tiles: the tuner's rule, as 1b's tuning.
- The FlipQuant calibration will change later. The bitwise results do not depend on the map; the tuning and the
  timings depend on its E0M3 share.
