# Kernel-opt C3w: GEMM latency against the E0M3 tile share on the adopted 8x64 path — protocol

Registered before any registered GPU run of the sweep, as amendment 16 of `results/kernel_opt/PROTOCOL.md`. The hashes
are in `registration.json`.

**The request.** The user, through the coordinator: a GEMM latency vs E0M3 tile fraction sweep for the adopted 8x64
path, done exactly like C3k (amendment 8), so that the 16x64 and 8x64 curves are directly comparable. GEMM only, no
e2e. Descriptive only: nothing is tuned or adopted from it.

**What is timed** (`experiments/kernel_opt/c3w_fraction.py`, C3k's script on the weights-on-B family):
- **Shapes and tokens**, as C3k: Llama-3.1-8B's 4096x4096 (o_proj), 14336x4096 (gate_proj) and 4096x14336
  (down_proj); T ∈ {1, 16, 128, 512, 2048, 8192}.
- **Maps** (8x64 tiles):
  - E0M3 tile share f ∈ {0, 1, 2, 5, 10, 25, 50, 75, 100} %, in two patterns: random (a seeded Bernoulli per 8x64
    tile) and contiguous (a row-major prefix). f = 0 and f = 100 are one map each.
  - The real map: the typical module (lower median of the E0M3 count) of each projection in Llama-3.1-8B's
    TM-OPT+TC 8x64 map (`llama8b_tc_8x64.mixfp4map`).
- **Kernels**, each a KernelSet picking its width per T from its table and using its scheduler rows:

  | kernel | set | builds | table | what it is |
  |---|---|---|---|---|
  | adopted, #2's dispatch | `mixed_wB_ko` | `build_P3freq` | adopted | the deployed 8x64 path: what `'auto'` gives (t0, #2's dispatch, the P3b widths) |
  | adopted, default dispatch | `mixed_wB_ko` | `build_P3` | adopted | the same builds with the default dispatch |
  | paper | n8k64_wB | `sm120/build` | — (one 128-wide build) | the paper's 8x64 kernel |
  | ceiling | `nodisp_wB_ko` | `build_P5` | adopted | the adopted tiles with the dispatch compiled out (P5) |
  | references | `stock_ko`; `stock_wB_ko`; the paper `stock` | `build_7`; `build_P3freq`; `sm120/build` | adopted; adopted; paper | the target; the same placement; the paper kernel's own stock |

- **Timing of the ceiling and the references:** they do not depend on the tags. They are timed once per shape and T,
  on all-E2M1 (FourOverSix) weights.
- **Weights:** seeded N(0, 0.02) per shape, quantized with each map: E0M3 with alpha = 1 on tagged tiles, FourOverSix
  elsewhere.
- **Method, C3k's deviation-2:**
  - cold weights, by rotation and a 512 MiB flush;
  - the activation quantizer after the flush, untimed;
  - isolated CUPTI GEMM launches, 3 rounds × 30, in a rotated order per (shape, T);
  - the median of all launches; telemetry sampled.
- **Checks:** the two adopted variants and the paper kernel give bitwise-equal outputs on every map, shape and T, on the
  timed operands. A failure stops the sweep.

**Output** (`experiments/kernel_opt/c3w_analyze.py`, into this directory), as C3k:
- `C3w.md`, which contains:
  - the decomposition at f = 0: tiles (the ceiling vs `stock_ko`), dispatch (adopted vs the ceiling) and placement
    (`stock_wB_ko` vs `stock_ko`);
  - the overhead vs own stock and kernel(f) / kernel(0), per shape × T × pattern × kernel;
  - the adopted path against `stock_wB_ko`;
  - the slope, and where the two dispatch variants cross.
- `C3w.csv`, `C3w_summary.json`, `C3w_raw.json`.
- **Figures:**
  - `C3w_overhead{,_mean}.{png,pdf}`, in C3k's layout;
  - `C3w_vs_C3k_mean.{png,pdf}`: the deployed 16x64 and 8x64 paths together. It compares two runs (C3k ran 2026-10-01).
  - Each panel is scaled to the adopted path and the ceiling. The paper 8x64 kernel, a single 128-wide build, is printed
    as off scale at small T.
- `REPORT.md` summarizes them.

**Builds:** none new. All are registered earlier (amendments 7, 11–13, and the paper builds).

**Disclosed, before registration:**
- a smoke test into a scratch directory: 4096x4096 at T = 16 and 512, 1 round × 3; 34 of 34 bitwise checks equal.
- One change came from it: the figure panels are scaled to the adopted path, because the paper kernel at T = 16 is
  about +114 % over its stock.
- Nothing from it is used.
