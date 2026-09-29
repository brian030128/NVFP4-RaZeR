# Experiment C3: E0M3 tile share against GEMM latency, at fixed matrix size — protocol

Written 2026-09-29 on branch `tm-opt`, before any C3 GPU run; the hashes and time are in `registration.json`. Deviations
are appended at the end. Requested by the user (relayed by nvfp4-razer-c9), approved with the proposals below. It runs
after the paper runs and their deviation-1 GEMM rerun (C1), with nothing else on the GPU.

## Question

How does the mixed kernel's GEMM time depend on the share of E0M3 tiles, at fixed matrix size? The answer is split
into:
- **a fixed cost:** the mixed kernel with no E0M3 tile against stock;
- **a dispatch cost:** against the same kernel with the format dispatch compiled out;
- **an E0M3-share cost:** the curve.

The curve is related to item #3 (`results/tm_opt/REPORT_ITEMS.md` §#3): real-map E0M3 density was worth at most 0.7 % of
the prefill.

## Design (`experiments/paper_extra/C3_fraction.py`)

- **Shapes (out x in):** 4096x4096, 14336x4096, 4096x14336 (Llama-3.1-8B's q/o, gate/up and down projections).
  **T** ∈ {128, 512, 2048, 8192}.
- **E0M3 tile share:** f ∈ {0, 1, 2, 5, 10, 25, 50, 75, 100} %, in two patterns:
  - **random:** a seeded Bernoulli(f) per tile, the worst case for mixing at 50 %;
  - **contiguous:** the first round(f × tiles) tiles in row-major tile order.
  - Tiles are 16x64 for the weights-on-A kernel and 8x64 for n8k64_wB.
- **Operands:** `sm120/bench/kernel.py operands`: seeded random normal weights × 0.02, quantized with the tags (E0M3
  alpha = 1 tiles, FourOverSix elsewhere), and FourOverSix per-token activations. GEMM only.
- **Kernels:**
  - **mixed@table:** n16k64_wA at the RTX PRO 6000 tile table's width, the deployment path;
  - **mixed@128:** n16k64_wA forced to width 128;
  - **n8k64_wB:** its single build.
- **References** (all-E2M1 weights):
  - stock_wA at the table's width and at 128;
  - stock_wB;
  - n16k64_wA_nodisp at 128: same tile and arrangement, format dispatch compiled out. Built on 2026-09-29 with
    `sm120/build.py --config n16k64_wA_nodisp` (repository f239539, CUDA 13.1), without a self-test: it is a latency
    ceiling, never a deployment kernel. Its census is 64 E2M1 OMMAs, as for stock, and its hashes are in
    registration.json.
- **Timing:** CUPTI kernel time, the median of 20 calls, in each of 3 rounds.
  - Within a (shape, T), all configurations are timed in a rotated order: round r starts at position r × len / 3, the
    method of deviation 1 in `results/paper/PROTOCOL.md`.
  - The reported value is the median of the per-round medians.
  - The GPU must be idle (the script refuses otherwise).

## Analysis (`experiments/paper_extra/C3_analyze.py`)

- **CSV:** shape, T, kernel, pattern, fraction, time_us, overhead vs stock. The stock reference is stock@table for
  mixed@table, stock@128 for mixed@128 and nodisp@128, and stock_wB for n8k64_wB.
- **Decomposition at f = 0:**
  - the fixed cost (mixed vs stock at the same width; n8k64_wB vs stock_wB);
  - the dispatch cost (mixed@128 vs nodisp@128);
  - the arrangement (nodisp@128 vs stock@128).
- **The curve:** mixed(f) / mixed(0) − 1 per pattern, shape and T.
- **The real maps:** the E0M3 tile shares of the committed 16x64 and 8x64 maps are marked. The text relates the curve's
  value at those shares to item #3's ≤ 0.7 %.

## Deviations (append-only)

(none yet)
