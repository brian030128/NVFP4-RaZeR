# Kernel-opt C3v: GEMM latency against the E0M3 tile share, on the adopted paths of the three units

The protocol is `PROTOCOL.md` here (amendment 19, registration ae90a74). The chain ran on 2026-10-03 from 16:17 to 16:26
UTC, with no deviation.
- **Gates and checks:** G0 passed. All 3 × 306 bitwise checks were equal: per unit, the adopted path, the path deployed
  before it and the paper kernel, on every map, shape and T.
- **Grid and method:** C3k's (amendment 8). The 16x64 and 8x64 maps, weights and activations are C3k's and C3w's,
  identical by construction and checked on CPU.
- **One session:** the three units ran one after another (256x64, 16x64, 8x64), within 9 minutes.

**Files:**
- **Tables:** `C3v.md`: the summary, then per unit the decomposition at f = 0, the overhead tables and the slopes, and
  the cross-session check.
- **Data:** `C3v.csv`, `C3v_summary.json`, `C3v_<unit>_raw.json`.
- **Figures** (overhead against the E0M3 share, on C3k's symlog axis; one panel per T; random placement solid,
  contiguous dashed, a star for the real map):
  - `C3v_<unit>_overhead.{png,pdf}`: one row per shape.
  - `C3v_<unit>_overhead_mean.{png,pdf}`: the mean of the 3 shapes, C3k's layout.
  - `C3_units_mean.{png,pdf}`: **the three-unit figure**, with each unit's adopted path against `stock_ko` and its
    no-dispatch ceiling. All three curves are from this run. The old C3k and C3w records are unchanged.

**The kernels per unit:**

| | adopted (`build_V`, what `'auto'` gives) | previous (the path deployed before) | paper kernel |
|---|---|---|---|
| 256x64 | `mixed256_ko` (amendment 18) | A′ (`mixed256`, the paper table) | n16k64_wA, one 128-wide build |
| 16x64 | `mixed_ko` with #2's dispatch and amendment 17's uniform-branch dispatch at widths 64/128 | #2's dispatch alone (`build_7freq`; C3k's deployed path) | the paper set `mixed` |
| 8x64 | `mixed_wB_ko` with #2's dispatch and amendment 17's pipelined flag read at widths 64/'128x64'/128 | #2's dispatch alone (`build_P3freq`; C3w's deployed path) | n8k64_wB, one 128-wide build |

- **Ceilings:** each adopted path's tiles with the dispatch compiled out, on its widths and scheduler rows.
- **References:** the adopted paths, their ceilings and the previous 16x64 / 8x64 paths are measured against
  `stock_ko`, the target. A′ and the paper kernels run on the paper table and are measured against the paper stock.

**Real-map points:** the typical (lower-median) module of Llama-3.1-8B's TM-OPT+TC map of the unit, per projection.

| unit | 4096x4096 (o_proj) | 14336x4096 (gate_proj) | 4096x14336 (down_proj) | mean |
|---|---|---|---|---:|
| 256x64 | layers.27, 10.8 % | layers.22, 6.1 % | layers.14, 7.1 % | 8.0 % |
| 16x64 | layers.28, 3.8 % | layers.19, 2.1 % | layers.22, 2.4 % | 2.8 % |
| 8x64 | layers.28, 2.9 % | layers.21, 1.6 % | layers.16, 1.8 % | 2.1 % |

## Findings

**1. With amendment 18, 256x64 is as close to stock as 16x64.** It is closer at T ≤ 128 and at T = 2048. At T = 512 and
8192, 16x64 is as close or closer. Overhead against `stock_ko`, in %, mean of the 3 shapes:

| T | 256x64: f = 0 | real map | 100 % | 16x64: f = 0 | real map | 100 % | 8x64: f = 0 | real map | 100 % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.4 | +0.6 | +0.5 | +1.0 | +1.5 | +2.5 | +2.0 | +2.6 | +5.4 |
| 16 | +0.5 | +0.8 | +0.6 | +0.7 | +1.3 | +2.4 | +2.1 | +2.8 | +5.8 |
| 128 | +0.8 | +1.3 | +0.7 | +2.8 | +3.5 | +5.1 | +1.9 | +2.4 | +5.1 |
| 512 | +1.7 | +2.1 | +1.8 | +1.1 | +2.1 | +10.5 | +8.6 | +10.2 | +20.5 |
| 2048 | +1.2 | +1.5 | +2.5 | +2.2 | +2.7 | +7.2 | +8.9 | +9.3 | +15.8 |
| 8192 | +1.6 | +1.6 | +1.4 | +0.9 | +1.1 | +5.9 | +7.1 | +7.6 | +14.4 |

- **256x64 is this close at its real maps too,** although they hold 3–4 times the E0M3 share of the other two units
  (8 % against 2–3 %).
- **The decomposition at f = 0** (means of the shapes; tiles = the ceiling vs `stock_ko`, dispatch = the adopted path vs
  the ceiling):

  | | tiles | dispatch |
  |---|---|---|
  | 256x64 | +0.1 … +0.9 % | +0.1 … +1.5 % |
  | 16x64 | 0.0 … +2.3 % | +0.4 … +0.9 % |
  | 8x64 | +0.8 … +1.3 % at T ≤ 128; +4.8 … +6.5 % at T ≥ 512 | +0.8 … +2.2 % |

- **16x64's +2.3 % tiles at T = 128** come from the table's width 32 for two shapes.
  - At width 32 the ceiling is +5.7 % (4096x4096) and +1.2 % (4096x14336).
  - The same no-dispatch builds at width 64, as on the 256x64 rows, are +0.2 % and +0.7 %.
  - Moving the adopted 16x64 path to width 64 did not recover this. Amendment 17 part B moved 4096x4096 @ 128 to width
    64. Its two affected cells (Llama and Mistral at T = 128) gave −0.13 % (typical) and +0.21 % (worst) per forward,
    and its rule failed.
- **8x64 is still bound by its tiles at T ≥ 512,** as C3w found.

**2. The E0M3 share costs 256x64 at most 1.6 points per 100 %.** The least-squares slope, in points per 100 % E0M3,
random placement:

| T | 256x64 adopted | A′ | n16k64_wA on 256x64 maps | 16x64 adopted | 8x64 adopted |
|---|---|---|---|---|---|
| ≤ 16 | −0.1 … +0.5 | +0.0 … +0.5 | +0.3 … +21.9 | +0.4 … +3.5 | +2.2 … +6.3 |
| 128 | +0.1 … +0.4 | +0.1 … +0.4 | +0.7 … +20.5 | +0.7 … +3.7 | +0.8 … +5.3 |
| ≥ 512 | −0.2 … +1.6 | −1.1 … +0.0 | +4.1 … +11.5 | +5.1 … +14.2 | +5.4 … +16.7 |

- **The flatness belongs to A′'s 4-arm, 32-row-granule design.** A′ is just as flat. The paper's 16-arm kernel on the
  same 256x64 maps is not.
- **Placement:**
  - On 16x64 and 8x64, contiguous placement still costs much more than random in the single-wave cells at T = 512, as
    in C3k and C3w:

    | cell | 16x64, contiguous | 16x64, random | 8x64, contiguous | 8x64, random |
    |---|---:|---:|---:|---:|
    | 4096x4096 | +13.5 … +15.8 % | −0.5 … +1.8 % | +13.8 … +14.9 % | +5.8 … +9.7 % |
    | 4096x14336 | +24.5 … +28.2 % | −0.4 … +2.6 % | +25.6 … +30.8 % | +8.7 … +13.6 % |

    Each range is over f = 1–10 %, against `stock_ko`.
  - On 256x64 the two placements differ by at most 4.0 points (4096x4096 at T = 2048), and there is no such effect at
    T = 512.
- **The real maps lie on the random curves.** At its share, the adopted path's real-map point is within −1.9 … +1.3
  points of the random curve (median +0.1), in every unit. The curve is interpolated linearly between the neighbouring
  shares.

**3. The adopted paths against the previous ones, at every f** (adopted / previous − 1, random placement, mean of the
shapes and of the T in the band):

| unit | T | f = 0 | 1–5 % | 10–25 % | 50–75 % | 100 % | real maps (median of the 18 cells) |
|---|---|---:|---:|---:|---:|---:|---|
| 256x64 (amendment 18) | ≤ 16 | −0.2 | −0.0 … −0.3 | −0.2 … −0.3 | −0.3 | −0.3 | −1.43 % (−7.7 … +0.6; 17 of 18 cells below zero) |
| | 128 | −5.1 | −4.9 … −5.0 | −4.8 | −4.7 … −4.8 | −5.2 | |
| | ≥ 512 | −2.5 | −2.0 … −2.4 | −1.8 … −2.1 | −1.5 … −1.7 | −1.5 | |
| 16x64 (amendment 17) | ≤ 16 | +0.2 | −0.2 … −0.0 | −0.1 … +0.1 | −0.0 … +0.1 | +0.1 | −0.25 % (−1.8 … +0.4; 14 of 18 below zero) |
| | 128 | −0.0 | −0.1 … +0.1 | −0.0 … +0.0 | −0.2 … −0.0 | −0.0 | |
| | ≥ 512 | −0.8 | −0.9 … −1.1 | −1.4 … −2.1 | −2.9 … −3.2 | +1.0 | |
| 8x64 (amendment 17) | ≤ 16 | +0.1 | −0.1 … +0.1 | −0.0 … +0.0 | −0.2 … −0.0 | −0.2 | −0.08 % (−0.6 … +0.6; 11 of 18 below zero) |
| | 128 | −0.1 | −0.5 … −0.7 | −0.7 … −1.2 | −1.5 … −2.3 | −2.5 | |
| | ≥ 512 | −0.1 | −0.1 … +0.0 | +0.0 … +0.1 | −0.2 … −0.3 | −0.0 | |

- **256x64:** the gain over A′ hardly depends on the share. It is largest at T = 128 (about −5 %). This is consistent
  with amendment 18's M1, where Llama's per-forward gain was −3.7 % at T = 128 and −7.1 % at T = 256.
- **16x64 at T ≥ 512:** the uniform-branch dispatch gains more as the share grows, up to −3.2 % at 75 %, and costs +1.0 %
  at all-E0M3 (+3.1 % at most). Amendment 17's C2 (4096³) also found it costing 0.4–0.8 % at all-E0M3. At T ≤ 128 it is
  neutral.
- **8x64:** the pipelined flag read is neutral, apart from T = 128, where it gains up to −2.5 % at high shares.
- **At the real maps,** the medians are −1.43 % (256x64), −0.25 % (16x64) and −0.08 % (8x64).

**4. The cross-session check: C3k's and C3w's curves agree with this run's to about half a point in the median.** This
run's
previous paths, paper kernels, ceilings and stocks were compared with the same builds in C3k (16x64, 2026-10-01) and C3w
(8x64, 2026-10-03 06:27). Every matched cell had the same width and scheduler row: 306 of 306 per tagged kernel, and 18
of 18 per reference.

| | median | range |
|---|---:|---|
| 16x64: previous / C3k's `ko_freq` | +0.34 % | −0.37 … +3.93 % |
| 16x64: `stock_ko` | +0.13 % | −0.26 … +1.64 % |
| 16x64: ceiling | +0.21 % | −0.44 … +3.73 % |
| 8x64: previous / C3w's `adopted_freq` | +0.30 % | −1.93 … +2.13 % |
| 8x64: `stock_ko` | +0.42 % | −0.17 … +1.79 % |
| 8x64: ceiling | +0.38 % | −0.14 … +1.49 % |

- Differences between the old and the new curves below about half a point are not meaningful. Single cells can differ by
  up to 4 %.

**5. The adopted paths against the paper kernels, in absolute time** (median over shapes and T, with the range):

| | f = 0 | real maps |
|---|---|---|
| 256x64 | −6.1 % (−47.3 … −1.1 %) | −5.9 % |
| 16x64 | −2.9 % (−4.9 … 0.0 %) | −3.1 % |
| 8x64 | −5.3 % (−52.3 … −0.1 %) | −5.0 % |

The large values (256x64 and 8x64) are at small T, where the paper's single 128-wide build competes with narrow tiles.

**Telemetry** (the 256x64 / 16x64 / 8x64 sweeps):
- **Clock:** busy SM clock median 2,677 / 2,670 / 2,670 MHz, minimum 2,437 / 2,422 / 2,415 MHz.
- **Power:** median 173 / 178 / 178 W, maximum 409 / 434 / 422 W (500 W limit).
- **Power cap:** the software power cap was active in 6 / 9 / 9 % of the samples.

## Reading

- **256x64:**
  - After amendment 18 it is within +0.6 … +2.1 % of `stock_ko` at its real maps (mean of the shapes). Against 16x64
    that is lower at four of the six T, equal at T = 512 and higher at T = 8192.
  - It barely depends on the E0M3 share, so a calibration that selects more 256x64 E0M3 tiles costs little GEMM time.
  - What remains at T ≥ 512 is +1.2 … +1.7 % at f = 0: the dispatch +0.3 … +1.5 % and the tiles +0.1 … +0.9 %.
- **16x64:**
  - Within +1.1 … +3.5 % at its real maps.
  - The share matters at T ≥ 512 (+5 … +14 points per 100 %), but the real maps sit at 2–4 %.
  - At T ≥ 512 amendment 17 helps at every share below 100 %.
- **8x64:** still +7.6 … +10.2 % at T ≥ 512 at its real maps, mostly from the tiles. This is unchanged by amendment 17,
  and closing it needs a map-format change (C3w).
- **The maps** are the paper's TM-OPT+TC artifacts. The FlipQuant calibration will change the real shares, but the curves
  give the cost at any share. Descriptive only: nothing is tuned or adopted from this sweep.
