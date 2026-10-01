# Kernel-opt amendment 10: the 8x64 baseline on the current best kernels

Protocol: `results/kernel_opt/PROTOCOL.md`, amendment 10 (`registration_10.json`, 0a57ff2). The chain `run_w8.sh` ran
on 2026-10-01 from 13:16 to 13:30 UTC, with no deviation.

**What was measured:**
- **The 8x64 path:** optimization 1b's `mixed_wB` set (default dispatch), and #2's build of the same set
  (`MIXFP4_DISPATCH_FREQ=1`), on 1b's table.
- **The references:**
  - stock with weights on A: the deployment's `stock_ko` and the paper `stock`;
  - stock with the same placement as ours: `stock_wB`, a single 128-wide build.
- C2w also times n8k64_wB's no-dispatch ceilings, with and without the site-0 prmt tags.

**Files:**
- Tables: `w8_tables.md`. Data: `w8.json`.
- Gates: `g0_provenance.json`, `g2_sass_ceilings.json`.
- Raw records: `/home/dev/n16k64_campaign/kernel_opt/w8`.

**Checks:**
- G0: 34/34 files and 21/21 builds as registered.
- G2: both ceilings are E2M1-only, with none predicated.
- M1: 3,240 bitwise comparisons, all equal, including #2's builds against the default ones; launch counts ok; no
  other processes.
- C2w: the t0 ceiling equals the tagged one, and #2's equals the default on all three tag patterns.

## In short

- **Gap at the GEMM level.** The 8x64 path is +6.1 % slower than `stock_ko` per forward: the median over 48
  (model, T) cells with typical tags, slower in every round in all 48. It grows with T:

  | T | 8x64 vs stock_ko | with #2's dispatch |
  |---|---:|---:|
  | ≤ 16 | +1.3 % | +0.7 % |
  | 32–128 | +2.7 % | +2.5 % |
  | 256–1024 | +8.9 % | +7.6 % |
  | ≥ 2048 | +10.2 % | +8.9 % |

  - The worst cells are Llama and Mistral at T = 512, +15.4 % and +15.3 %. There both sets run the 128 × 128 tile.
- **#2's dispatch is worth −0.70 % on typical tags** (faster in every round in 47 of 48 cells) and −1.0 to −1.1 % at
  T ≥ 256.
  - On the worst tags it is −0.13 %. 11 cells are slower in every round, all at T ≤ 128 and by up to +0.75 % (Llama,
    Mistral, one Qwen cell).
  - On all-E0M3 tags it loses (+1.8 % at 4096³), as C3k found for 16x64.
- **Placement itself costs nothing at large T.** At T ≥ 2048, `stock_wB` is within +0.2 % of `stock_ko` (median); at
  4096³ it is −0.1 to +0.7 % against the paper stock_wA.
  - At small T `stock_wB` is +64 % slower, because it has no narrow tiles.
  - So the same-placement reference only matters for T ≤ 1024.

## The 4096³ breakdown (C2w)

In %, b2b / isolated / sustained (2 s at the power cap):

| step | change | b2b | isolated | sustained |
|---|---|---:|---:|---:|
| placement | stock_wB vs stock_wA | −0.1 | −0.4 | +0.7 |
| #4 tuning | stock_ko vs stock_wA | −0.9 | −1.6 | −0.4 |
| **1x8 arrangement** | the t0 ceiling vs stock_wB | **+3.3** | **+2.9** | **+2.3** |
| **site-0 prmt tags** | the tagged ceiling vs the t0 ceiling | **+1.2** | **+1.1** | **+1.3** |
| **dispatch**, default | n8k64_wB all-E2M1 vs the tagged ceiling | **+3.9** | **+3.8** | **+4.4** |
| dispatch, #2's | the same, with #2's dispatch | +2.7 | +2.5 | +3.6 |
| E0M3 tiles, real map (Llama o_proj) | real vs all-E2M1, default | +0.8 | +0.5 | +1.1 |
| E0M3 tiles, all-E0M3 | all-E0M3 vs all-E2M1, default (#2's) | +2.0 (+5.1) | +1.8 (+5.3) | +2.6 (+4.8) |
| **total**, real map | n8k64_wB vs stock_ko, default (#2's) | **+10.4 (+9.5)** | **+9.9 (+8.3)** | **+10.6 (+10.4)** |

- **Tags.** The wB blob carries 32 identity PRMTs (SASS count), four times n16k64_wA's 8. Dropping them (t0) is worth
  about 1.2 % on the ceiling. On 16x64 the ceiling gained about 1 % and the real kernel 0.2–0.4 %.
- **Arrangement.** The 1x8 arrangement's own cost here, +2.3 … +3.3 %, matches the kernel report's 1x8 ceiling reading
  for this card. A warp reads 1.5× the shared-memory data of the stock arrangement.
- **Dispatch.** The 16-arm dispatch is the largest single part: +3.8 … +4.4 %, or +2.5 … +3.6 % with #2's.

## Per model (M1, typical tags; 8x64 vs stock_ko, default / #2's dispatch, in %)

| T | Llama-3.1-8B | Mistral-7B-v0.3 | Phi-4 | Qwen3.8-27B |
|---|---|---|---|---|
| 1 | +1.8 / +1.5 | +1.8 / +1.4 | +0.8 / +0.1 | +0.5 / +0.3 |
| 16 | +2.0 / +1.5 | +2.2 / +1.9 | +0.8 / +0.3 | +0.6 / +0.3 |
| 128 | +5.0 / +4.7 | +4.6 / +4.4 | +2.3 / +1.6 | +2.3 / +2.0 |
| 256 | +7.4 / +6.3 | +7.3 / +6.2 | +8.6 / +7.0 | +9.1 / +7.6 |
| 512 | **+15.4 / +13.9** | **+15.3 / +13.7** | +10.4 / +9.4 | +8.7 / +7.5 |
| 1024 | +7.9 / +7.3 | +7.8 / +7.0 | +9.7 / +8.6 | +9.5 / +8.0 |
| 2048 | +10.2 / +8.9 | +10.5 / +9.4 | +9.6 / +8.5 | +8.3 / +6.9 |
| 8192 | +11.1 / +10.0 | +11.0 / +10.1 | +10.1 / +8.9 | +9.8 / +8.6 |

**Telemetry:**
- busy SM clock median 2,647–2,662 MHz per model, minimum 2,280 MHz (Phi-4);
- power median 134–139 W, maximum 489 W (500 W limit);
- the software power cap was active in 1–5 % of the samples.

## What it implies for the plan

At large T the gap of about 10 % splits as follows:
- dispatch, about 4 points (about 2.5–3.6 with #2's);
- the 1x8 arrangement, about 3;
- the tags, about 1.2;
- E0M3 tiles on real maps, about 1;
- `stock_ko`'s #4 advantage, about 1.

| part | lever | bitwise-safe? |
|---|---|---|
| tags | t0 | yes |
| #4's advantage | #4 applied to the wB family | yes |
| dispatch | #2's dispatch, already built and gated | yes |
| arrangement | none known | no |

The arrangement is structural: a wider per-warp N extent multiplies the dispatch arms. Mid T (256–1024) adds a
width/tile question; T = 512 on Llama and Mistral is the worst cell. The plan is in `PLAN.md`.
