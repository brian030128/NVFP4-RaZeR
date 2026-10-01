# Kernel-opt C3k: GEMM latency against the E0M3 tile share, on the adopted path

Protocol: `PROTOCOL.md` here (amendment 8, registration 881fbe4). The chain ran on 2026-10-01 from 05:29 to 05:32 UTC,
with no deviation. All 306 bitwise checks were equal: the adopted default dispatch, #2's dispatch and the paper kernel
on every map, shape and T.

Files:
- Tables: `C3k.md`. Data: `C3k.csv`, `C3k_raw.json`, `C3k_summary.json`.
- Figures: `C3k_overhead.{png,pdf}` (one row per shape) and `C3k_overhead_mean.{png,pdf}` (the mean of the 3 shapes).
  - Overhead vs own stock against the E0M3 share, on a symlog axis: linear below 1 %, logarithmic above.
  - One panel per T. Random placement is solid, contiguous is dashed, and a star marks the real map.

Each kernel is measured against its own stock:
- the adopted path (`mixed_ko`; default dispatch, and #2's) and its no-dispatch ceiling (`nodisp_ko`) against
  `stock_ko`;
- the paper kernel against the paper stock.

## Findings

**1. At f = 0 the ceiling equals stock, and the dispatch is what remains.**
- Where the ceiling and stock run the same width, the matched no-dispatch ceiling is within about ±1 % of `stock_ko`.
- The adopted default dispatch is +0.4…+2.5 % over its stock in most cells. That is the dispatch cost: adopted vs
  ceiling, typically +0.4…+2.6 %.
- Two cells stand out, both at width 128:

  | cell | adopted vs ceiling | paper kernel vs paper stock |
  |---|---|---|
  | 4096x4096, T = 2048 | +4.7 % | +6.5 % |
  | 14336x4096, T = 512 | +5.8 % | +5.9 % |

  The paper kernel has the same overhead there, so the adoption did not cause them.

**2. The width choice is part of the gap at T = 128.**
- For 4096x4096 and 4096x14336 the mixed sets run width 32 and stock runs 64; each family's own 4b widths differ.
- There the ceiling is +5.6 % and +1.2 % over stock, and the adopted kernel +7.6 % and +2.9 %.
- The paper kernel, at the same widths, is +10.5 % and +7.1 % over its stock.

**3. The cost of the E0M3 share depends on T.** Least-squares slope, in points per 100 % E0M3, random placement:

| T | adopted, default dispatch | adopted, #2's dispatch | paper |
|---|---|---|---|
| ≤ 128 | −0.0 … +1.8 | +0.6 … +3.9 | +0.1 … +1.2 |
| ≥ 512 | +3.4 … +10.9 | +5.9 … +14.1 | +4.4 … +11.0 |

**4. Which dispatch variant wins depends on f; on real maps it is #2's.**
- #2's pattern-0-first dispatch is faster at f ≤ 5–10 %: −0.1…−2.2 % against the default dispatch, the most at
  4096x14336 with T = 512–2048.
- The default dispatch is faster from about f = 25 %, by up to 4.3 %.
- The typical real modules have f = 3.8 % (Llama o_proj), 2.1 % (gate_proj) and 2.4 % (down_proj). On them #2's
  dispatch is 0.0–0.9 points closer to stock in every cell (equal in one).

**5. Placement matters when the GEMM is one wave.**
- At T = 512, 4096x4096 and 4096x14336 are 128 tiles at width 128: a single wave on 188 SMs.
- A contiguous E0M3 prefix there costs far more than the same share placed at random:

  | shape | contiguous, f = 1–10 % | random, same f |
  |---|---|---|
  | 4096x4096 | +14…+17 % | +0.6…+3.1 % |
  | 4096x14336 | +26…+30 % | +2.6…+5.7 % |

  The CTAs that hold the E0M3 rows set the makespan. From f = 25 % on, the contiguous cost falls back to +4…+8 %.
- 14336x4096 (448 tiles) shows no such effect.
- The real maps behave like random placement: their stars sit on the random curves.

**6. The adopted path vs the paper path.**
- **Absolute time:** at f = 0 the adopted default dispatch is −1.22 % against the paper kernel (median; −4.15 to
  +0.42 %). On the real maps it is −1.55 % (median; −4.25 to +1.73 %), and #2's dispatch −1.91 % (−4.52 to +1.64 %).
- **Relative to each one's own stock:** at f = 0 the adopted path's overhead is lower in 10 of 18 cells, equal in 1,
  and higher in 7, mostly at T ≤ 16. There, stock gained more from the adoption (#4 and 4b) than the mixed kernel did:
  `stock_ko` is −0.78 % against the paper stock (median; −1.62 to +0.14 %).
- On the real maps the adopted default dispatch is −0.1…+9.4 % over `stock_ko`, and #2's −0.3…+9.3 %. The large
  values are the two outlier cells of finding 1 and the width-split cells at T = 128.

**Real-map overhead vs own stock, in %:**

| shape (module, f) | T | adopted, default | adopted, #2's | paper |
|---|---:|---:|---:|---:|
| 4096x4096 (layers.28 o_proj, 3.8 %) | 1 / 16 / 128 / 512 / 2048 / 8192 | +2.7 / +2.7 / +7.9 / +1.2 / +9.4 / +2.7 | +1.9 / +2.3 / +7.3 / +0.8 / +9.3 / +1.9 | +1.1 / +1.5 / +10.5 / +3.0 / +6.6 / +3.6 |
| 14336x4096 (layers.19 gate_proj, 2.1 %) | same | +0.4 / +0.4 / −0.1 / +6.6 / +1.1 / +2.6 | +0.3 / +0.3 / −0.1 / +6.5 / +0.5 / +1.9 | +0.7 / +0.4 / +0.6 / +6.6 / +2.9 / +3.6 |
| 4096x14336 (layers.22 down_proj, 2.4 %) | same | +2.0 / +2.1 / +3.1 / +2.3 / +0.3 / +2.1 | +1.8 / +1.8 / +2.8 / +1.4 / −0.3 / +1.9 | +1.8 / +1.8 / +7.4 / +5.2 / +2.0 / +3.2 |

**Telemetry:**
- busy SM clock median 2,677 MHz, minimum 2,430 MHz;
- power median 170 W, maximum 408 W (500 W limit);
- the software power cap was active in 10 % of the samples.

## For B′

Real maps sit at f ≈ 2–4 %. There, #2's pattern-0-first dispatch already beats the default dispatch, and what remains
over the ceiling is the dispatch's fixed cost per k_tile. That is exactly what B′, the per-warp all-E2M1 bit, removes.
B′ should therefore be compared against #2's dispatch (the better current path at real f) as well as against stock.
