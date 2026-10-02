# Kernel-opt 8x64 plan, P5 (decision a): the adopted 8x64 path against its no-dispatch ceiling at every width — report

2026-10-02, branch `kernel-opt`. The protocol is amendment 13 of `results/kernel_opt/PROTOCOL.md`, registered at
80cac11. `run_w8p5.sh` ran 17:53–18:03 UTC with no deviation. Every number is in `w8p5_tables.md` and `w8p5.json`.
Descriptive: nothing is tuned or adopted from it.

## In brief

- **The gates passed.**
  - G0: 37 files and 65 builds.
  - G1: the 51 existing configurations keep their SASS.
  - G2: the four new ceilings are E2M1-only, with nothing predicated.
  - pytest: the ceiling's widths are bitwise interchangeable.
  - In every (projection, T) cell the ceiling ran the 8x64 path's width and scheduler setting (the report asserts it).
- **The split.** The adopted 8x64 path's gap to `stock_ko` (+3.96 % per forward, typical tags, median) splits into two
  parts.
  - **The dispatch** (8x64 vs its own no-dispatch ceiling, same tiles): +2.10 % (median; worst tags +3.01 %). This is
    what format-dispatch work could still recover at each width.
  - **The same-placement tiles** (the ceiling vs `stock_ko`): +1.29 %. This is what the weights-on-B 1x8 / 1x4
    arrangement costs against stock with the weights on A. It is about 0 at T ≤ 128 and +3.4 … +4.2 % at T ≥ 256.

| T band | 8x64 vs ceiling (typical / worst) | ceiling vs stock_ko | 8x64 vs stock_ko (typical) |
|---|---:|---:|---:|
| T ≤ 16 | +0.93 / +1.60 % | −0.14 % | +0.64 % |
| 32–128 | +1.41 / +1.96 % | +0.33 % | +1.83 % |
| 256–1024 | +3.08 / +4.48 % | +3.44 % | +5.98 % |
| ≥ 2048 | +2.70 / +4.03 % | +4.22 % | +7.12 % |

Per T (medians over the four models):

| T | 8x64 vs ceiling, typical | ceiling vs stock_ko | 8x64 vs stock_ko, typical |
|---:|---:|---:|---:|
| 1 | +0.96 % | −0.25 % | +0.69 % |
| 4 | +0.88 % | −0.36 % | +0.53 % |
| 16 | +0.86 % | −0.03 % | +0.74 % |
| 32 | +1.22 % | +0.74 % | +1.99 % |
| 64 | +1.31 % | +0.73 % | +2.08 % |
| 128 | +1.71 % | +0.05 % | +1.82 % |
| 256 | +3.65 % | +2.07 % | +5.52 % |
| 512 | +3.35 % | +5.70 % | +9.25 % |
| 1024 | +2.75 % | +3.09 % | +5.89 % |
| 2048 | +2.68 % | +4.19 % | +7.02 % |
| 4096 | +2.64 % | +4.29 % | +7.05 % |
| 8192 | +2.87 % | +4.54 % | +7.54 % |

## Reading

- **Small T (≤ 128):** the 8x64 path's own tiles are as fast as stock's (ceiling vs stock_ko −0.4 … +0.7 %). The gap
  there, +0.5 … +2.1 %, is the dispatch.
- **T = 512, the worst cell:** most of the +9.3 % is the tiles. The ceiling is +5.7 % against stock_ko and the dispatch
  +3.4 %. That matches the plan's reading: at T = 512 the 128 × 128 tile leaves q/o/down_proj at a single partial wave.
- **Large T (≥ 2048):** the tiles are +4.2 % and the dispatch +2.7 %. This is consistent with the 4096³ splits:
  - amendments 10–11 put the 1x8 arrangement at +2.3 … +4.3 %;
  - amendment 11 put the dispatch with t0 and #2's at +0.3 … +2.2 %, plus +0.6 … +2.1 % for the real map's E0M3 tiles.
- **What remains bitwise-reachable** is at most the dispatch part, and even a perfect dispatch would leave the tile
  part:
  - ≈ 0 at T ≤ 128;
  - +2 … +6 % at 256–1024;
  - ≈ +4 % at T ≥ 2048.
  As the plan's "Structural" section says, the tile part is not bitwise-reachable without a map-format change.

## Maps and tags used

- The paper's TM-OPT+TC 8x64 artifacts, typical (lower-median) and worst (densest) modules, with sha256 in
  `registration_13.json`.
- The ceiling and the stock references take the FourOverSix artifacts.
- The FlipQuant calibration will change later. The dispatch part depends on the map's E0M3 share (worst tags cost about
  1 point more than typical).
