# Kernel-opt C3w: GEMM latency against the E0M3 tile share, on the adopted 8x64 path

The protocol is `PROTOCOL.md` here (amendment 16, registration 91db345). The chain ran on 2026-10-03 from 06:27 to 06:30
UTC, with no deviation. All 306 bitwise checks were equal: the adopted 8x64 path with #2's dispatch, with the default
dispatch, and the paper kernel n8k64_wB, on every map, shape and T. The method and grid are C3k's (amendment 8), so the
16x64 and 8x64 curves are directly comparable.

**Files:**
- **Tables:** `C3w.md`. **Data:** `C3w.csv`, `C3w_raw.json`, `C3w_summary.json`.
- **Figures** (overhead vs own stock against the E0M3 share, on C3k's symlog axis; one panel per T; random placement
  solid, contiguous dashed, a star for the real map):
  - `C3w_overhead.{png,pdf}`: one row per shape.
  - `C3w_overhead_mean.{png,pdf}`: the mean of the 3 shapes, C3k's layout.
  - The panels are scaled to the adopted path and its ceiling. At T ≤ 128 the paper kernel is mostly printed as off
    scale: it is a single 128-wide build, up to +114 % over its stock there.
  - `C3w_vs_C3k_mean.{png,pdf}`: the deployed 16x64 (C3k, run 2026-10-01) and 8x64 paths against `stock_ko`, with their
    ceilings. These are two runs, so differences below about 1 point are not meaningful.

**Real-map points:** the typical (lower-median) module of Llama-3.1-8B's TM-OPT+TC 8x64 map per projection.
- 4096x4096: layers.28 o_proj, f = 2.9 %;
- 14336x4096: layers.21 gate_proj, f = 1.6 %;
- 4096x14336: layers.16 down_proj, f = 1.8 %.

**References:** the adopted kernels and the ceiling are measured against `stock_ko` (the target, weights on A). The
paper kernel is measured against the paper stock.

## Findings

**1. At f = 0 the gap is mostly the 8x64 tiles, not the dispatch.** The decomposition (`C3w.md`) has three parts.
- **Tiles** (the no-dispatch ceiling vs `stock_ko`):
  - +2.4 … +5.7 % at T ≥ 512, apart from the two outliers below;
  - +0.1 … +1.7 % at T = 128;
  - −3.4 … +4.8 % at T ≤ 16. At 4096x14336 the 8x64 narrow tiles are about 3 % faster than stock's.
  - The two outlier cells:

    | cell | tiles | stock_wB_ko vs stock_ko |
    |---|---:|---:|
    | 4096x4096, T = 2048 | +14.0 % | −0.2 % |
    | 14336x4096, T = 512 | +10.3 % | −0.3 % |

    Both run width 128 with the default scheduler, like stock. stock_wB_ko is within 0.3 % of `stock_ko` there, so
    the cost is the 1x8 arrangement itself, not the placement.
- **Dispatch** (adopted vs ceiling): +0.3 … +3.0 % with #2's dispatch, and +0.7 … +3.8 % with the default.
- **Placement** (`stock_wB_ko` vs `stock_ko`): about 0 at T ≥ 512 (−0.6 … +0.2 %).
  - At T ≤ 128 `stock_wB_ko` has no narrow tiles: +39 … +84 %.
  - The exception is 14336x4096 at T = 128, where `stock_ko` also runs width 128: +0.1 %.

**Compared with 16x64 (C3k)**, mean of the 3 shapes, against `stock_ko`:

| T | 8x64: deployed at f = 0 | ceiling | real map | 16x64: deployed at f = 0 | ceiling | real map |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | +1.9 % | +1.2 % | +2.7 % | +0.7 % | +0.3 % | +1.3 % |
| 16 | +2.2 % | +1.3 % | +2.9 % | +0.6 % | +0.4 % | +1.5 % |
| 128 | +2.1 % | +0.7 % | +2.7 % | +2.8 % | +2.3 % | +3.3 % |
| 512 | +9.0 % | +6.2 % | +10.6 % | +2.2 % | +0.2 % | +2.9 % |
| 2048 | +8.9 % | +6.7 % | +10.0 % | +2.8 % | +0.5 % | +3.1 % |
| 8192 | +7.0 % | +4.7 % | +7.7 % | +1.7 % | +0.0 % | +1.9 % |

- **The dispatch costs about the same in both families:** deployed vs ceiling (means of the shapes) is 0.7–2.8 points
  for 8x64 and 0.2–2.3 for 16x64.
- **The difference is the ceiling.** The 16x64 tiles match stock (about 0 … +0.5 % outside T = 128, where the widths
  differ). The 8x64 tiles cost +4.7 … +6.7 % at T ≥ 512.
- **At T = 128 the 8x64 path is the closer of the two to stock.** Its narrow tiles fit there, while 16x64 pays a width
  split.

**2. The cost of the E0M3 share.** The least-squares slope, in points per 100 % E0M3, random placement:

| T | adopted, default dispatch | adopted, #2's dispatch | paper |
|---|---|---|---|
| ≤ 16 | +1.2 … +2.3 | +2.4 … +6.5 | +0.2 … +17.1 |
| 128 | +0.6 … +3.3 | +0.8 … +9.1 | +1.0 … +12.4 |
| ≥ 512 | +1.8 … +10.6 | +7.2 … +16.1 | −0.2 … +7.3 |

- **At T ≥ 512** the slopes are close to C3k's: 16x64 was +3.4 … +10.9 with the default dispatch and +5.9 … +14.1 with
  #2's.
- **At T ≤ 16** #2's dispatch is somewhat steeper for 8x64 than for 16x64.
- **The paper kernel's large small-T slopes** come from its single 128-wide tile.

**3. Which dispatch wins depends on f; at real maps it is #2's.**
- **#2's pattern-0-first dispatch is faster at f ≤ 2–5 %:** −0.1 … −1.5 % against the default. At f = 0 it is −0.3 …
  −1.3 % in 17 of 18 cells.
- **The default dispatch is faster from about f = 10–25 %,** by up to 4.8 % at f = 100 %.
- **On the real maps** (f = 1.6–2.9 %), #2's dispatch is faster or equal in 14 of 18 cells (−1.2 … +0.7 %). The other 4
  cells are within +0.7 %. This agrees with amendment 11's P1 choice of #2's dispatch for 8x64.

**4. Placement matters when the GEMM is one wave,** as in C3k. At T = 512, the adopted path with #2's dispatch, against
`stock_ko`:

| shape | contiguous, f = 1–10 % | random, same f |
|---|---|---|
| 4096x4096 | +13.8 … +16.6 % | +5.8 … +9.2 % |
| 4096x14336 | +27.2 … +28.9 % | +10.0 … +14.6 % |

- 14336x4096 shows no such effect (448 tiles), and neither does T = 2048.
- The real maps behave like random placement: their stars sit on the random curves.

**5. The adopted 8x64 path against the paper 8x64 kernel, in absolute time:** −5.5 % at f = 0 (median; −52.4 … −0.7 %)
and −4.7 % on the real maps (−50.8 … −0.5 %). The large values are at small T, from the narrow tiles.

**Real-map overhead vs own stock, in %** (adopted with #2's dispatch / with the default dispatch / the paper kernel):

| shape (module, f) | T = 1 | 16 | 128 | 512 | 2048 | 8192 |
|---|---|---|---|---|---|---|
| 4096x4096 (layers.28 o_proj, 2.9 %) | +4.2 / +3.4 / +106 | +4.2 / +4.6 / +106 | +5.3 / +5.0 / +85 | +7.0 / +7.0 / +12.9 | +18.5 / +18.3 / +20.2 | +7.3 / +8.7 / +9.3 |
| 14336x4096 (layers.21 gate_proj, 1.6 %) | +5.7 / +6.8 / +41 | +5.5 / +6.4 / +41 | +0.9 / +1.0 / +0.9 | +14.3 / +15.3 / +17.0 | +6.6 / +7.9 / +8.7 | +7.4 / +7.9 / +9.7 |
| 4096x14336 (layers.16 down_proj, 1.8 %) | −1.8 / −1.5 / +99 | −1.1 / −0.8 / +100 | +1.9 / +2.3 / +85 | +10.7 / +9.9 / +13.3 | +4.8 / +5.8 / +7.0 | +8.3 / +8.8 / +9.3 |

**Telemetry:**
- busy SM clock median 2,685 MHz, minimum 2,415 MHz;
- power median 170 W, maximum 427 W (500 W limit);
- the software power cap was active in 9 % of the samples.

## Reading

- **The 8x64 path's remaining gap to stock is the tiles,** as P5 found per forward. The dispatch costs about what it
  costs on 16x64. At T ≥ 512 the 1x8 / 1x4 arrangement is +5 … +7 % on its own (mean of the shapes), and up to +14 % in
  the single-wave cells.
- **The E0M3 share costs about as much as on 16x64,** and the real maps sit at f ≈ 2–3 %, where #2's dispatch (the
  deployed variant) is the faster one.
- **Closing the gap at T ≥ 512 needs the tiles changed.** That means a map-format change (e.g. 16-column granules on B),
  which is not bitwise-reachable and needs the user's approval.
