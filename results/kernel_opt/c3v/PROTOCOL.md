# Kernel-opt C3v: GEMM latency against the E0M3 tile share on the adopted paths of the three units — protocol

Registered before any registered GPU run of the sweep, as amendment 19 of `results/kernel_opt/PROTOCOL.md`. The hashes
are in `registration.json`.

**The request.** The user, through the coordinator, 2026-10-03:
- **256x64** (item 7 of the 256x64 work): after amendment 18's adoption, a C3k-style E0M3-fraction sweep for 256x64, so
  that all three units have comparable curves.
- **16x64 and 8x64** (an addition, before registration): C3k (2026-10-01) and C3w (2026-10-03) ran before amendment 17
  part A was adopted. Re-run both sweeps on `build_V`'s adopted paths with C3k's and C3w's grid and method, together
  with their ceilings and `stock_ko`. The three-unit figure uses these curves, and the old C3k and C3w records stay
  unchanged.

GEMM only. Descriptive only: nothing is tuned or adopted from it.

**What is timed** (`experiments/kernel_opt/c3v_fraction.py --unit 256x64|16x64|8x64`):
- **Shapes and tokens**, as C3k and C3w: Llama-3.1-8B's 4096x4096 (o_proj), 14336x4096 (gate_proj) and 4096x14336
  (down_proj); T ∈ {1, 16, 128, 512, 2048, 8192}.
- **Maps,** per unit, on the unit's tiles (256x64, 16x64 or 8x64):
  - E0M3 tile share f ∈ {0, 1, 2, 5, 10, 25, 50, 75, 100} %, random (a seeded Bernoulli per tile) and contiguous (a
    row-major prefix of tiles), with C3k's and C3w's seeds. f = 0 and f = 100 are one map each. The 256x64 maps are
    stored as the 16x64 granules the artifacts use.
  - The real map: the typical module (lower median of the E0M3 count) of each projection in Llama-3.1-8B's TM-OPT+TC map
    of the unit (`llama8b_tc_<unit>.mixfp4map`).
  - The 16x64 and 8x64 maps are C3k's and C3w's. The real-map modules are the same, and all 17 maps per shape are
    identical, checked on CPU against their scripts' functions.
- **Weights:** seeded N(0, 0.02) per shape, quantized with each map: E0M3 with alpha = 1 on tagged tiles, FourOverSix
  elsewhere. Seeds and activations as C3k and C3w.
- **Kernels.** Each is a KernelSet that picks its width per T from its table and uses its scheduler rows:

  | kernel | 256x64 | 16x64 | 8x64 |
  |---|---|---|---|
  | **adopted** — the deployed path, what `'auto'` gives, from `build_V` on the adopted table | `mixed256_ko` (amendment 18) | `mixed_ko`: #2's dispatch, plus the uniform-branch dispatch at widths 64/128 (amendment 17) | `mixed_wB_ko`: #2's dispatch, plus the pipelined flag read at widths 64/'128x64'/128 (amendment 17) |
  | **previous** — the path deployed before | A′: `mixed256` from `build_A1` on the paper table | `mixed_ko` with #2's dispatch alone, from `build_7freq` (C3k's `ko_freq`) | `mixed_wB_ko` with #2's dispatch alone, from `build_P3freq` (C3w's `adopted_freq`) |
  | **paper** — the paper kernel, from `sm120/build` | n16k64_wA (one 128-wide build) | the paper set `mixed` on the paper table (as C3k) | n8k64_wB (one 128-wide build; as C3w) |
  | **ceiling** — the adopted tiles with the dispatch compiled out, on the adopted table | `nodisp256_ko` (`build_C3k`) | `nodisp_ko` (`build_C3k`) | `nodisp_wB_ko` (`build_P5`) |
  | **references** | `stock_ko` (`build_7`), the paper `stock` | the same | the same, plus `stock_wB_ko` (`build_V`), as C3w |

- **The ceiling and the references** do not depend on the tags. They are timed once per shape and T, on all-E2M1
  (FourOverSix) weights.
- **Before timing,** the script checks:
  - every adopted build carries its registered defines (`results/kernel_opt/V/build_V.sh`), and every previous-path
    build its own;
  - the paper and ceiling builds carry no defines;
  - the ceiling uses the adopted path's table and scheduler rows;
  - for 16x64 and 8x64, the previous path uses the adopted path's table and scheduler rows, so only the builds differ.
- **Method:** C3k's deviation-2, unchanged.
  - Cold weights, by rotation and a 512 MiB flush; the activation quantizer after the flush, untimed.
  - Isolated CUPTI GEMM launches, 3 rounds × 30, in a rotated order per (shape, T); the median of all launches.
  - Telemetry sampled.
  - The three units run one after another in one session: 256x64, 16x64, 8x64.
- **Checks:** the adopted path, the previous path and the paper kernel give bitwise-equal outputs on every map, shape
  and T, on the timed operands. A failure stops the sweep.

**Output** (`experiments/kernel_opt/c3v_analyze.py`, into this directory), as C3k and C3w:
- **`C3v.md`:**
  - A summary per unit and T, the mean of the 3 shapes:
    - the adopted path against `stock_ko` at 0 %, at the real map and at 100 %;
    - the tiles (the ceiling vs `stock_ko`) and the dispatch (the adopted path vs the ceiling) at 0 %;
    - the adopted path against the previous one.
  - Per unit:
    - the decomposition at f = 0 per shape and T;
    - per kernel, three tables per shape × T × pattern: the overhead vs its own stock, kernel(f) / kernel(0), and the
      overhead vs `stock_ko`;
    - the slope, and the adopted path against the previous one at each f.
  - **The cross-session check:** this run's previous path, paper kernel, ceiling and stocks against the same builds in
    C3k (16x64) and C3w (8x64). Per matched cell, the ratio, and whether the width and the scheduler row match. It
    tells how far the old and the new curves can be compared.
- **Files:** `C3v.csv`, `C3v_summary.json`, and the raw `C3v_<unit>_raw.json`.
- **Figures:**
  - `C3v_<unit>_overhead{,_mean}.{png,pdf}`, in C3k's layout.
    - Panels are scaled to the adopted path, the previous one and the ceiling.
    - A paper kernel that does not fit is printed as off scale.
  - `C3_units_mean.{png,pdf}`: the three units' adopted paths against `stock_ko`, with their ceilings and the real maps.
    All three are from this run.
- **`REPORT.md`** summarizes them.

**Builds:** none new. All are registered earlier:

| amendment | directory | builds |
|---|---|---|
| 3 | `build_A1` | A′ |
| 7 | `build_7` | `stock_ko` |
| 7 | `build_7freq` | the 16x64 path with #2's dispatch |
| 8 | `build_C3k` | the 16x64 and 256x64 ceilings |
| 12 | `build_P3freq` | the 8x64 path with #2's dispatch |
| 13 | `build_P5` | the 8x64 ceiling |
| 18 | `build_V` | the adopted paths and `stock_wB_ko` |
| the paper | `sm120/build` | the paper builds |

**Disclosed, before registration** (nothing from it is used):
- **A smoke test of the 256x64 sweep** (one shape, T = 16 and 512, 1 round × 3) and of the analysis. It used the
  candidate table `V/table_v/table_v.json`, which the tracked table now equals.
- **A smoke test of the three-unit script** (one shape and T per unit, 1 round × 3).
- **The CPU map check above.**
- **The post-adoption routed-logits checks** (`V/adoption/`).
