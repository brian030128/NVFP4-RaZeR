# Experiment C2-lite: the mixed kernel at M = N = K = 4096 (paper §4.3) — protocol

Written 2026-09-29 on branch `tm-opt`, before any C2 GPU run; the hashes and time are in `registration.json`. Deviations
are appended at the end.

The user approved the reduced form (relayed by nvfp4-razer-c9), on the condition that the report and every C2 table
carry this caveat:

> **Caveat.**
> - ncu is unavailable on this machine: it is not installed, and the counters are blocked (ERR_NVGPUCTRPERM).
> - Clocks were not locked (no root); the rotated 3-round method stands in for that.
> - Tensor-pipe counts are static SASS census counts (per-path OMMA count × loop trips), not measured counters.
> - There are no stall reasons and no bank-conflict data.
> - Per-MMA-branch kernel numbers are historical (sm120/kernel/docs/mixed_nvfp4_report.md, RTX 5090, ncu), cited but
>   not re-measured on this GPU.

## Timing (`experiments/paper_extra/C2_time.py`)

- **Size:** M = N = K = 4096. Weights 4096 × 4096 (seeded normal × 0.02); 4096 tokens with FourOverSix per-token
  activations. GEMM only, at the 128-wide CTA tile.
- **Kernels:**
  - **stock_wA:** NVFP4 weights;
  - **n16k64_wA:** with every tile E2M1, every tile E0M3, and the tags of a real layer. The real layer is
    `model.layers.0.self_attn.o_proj` of the committed Llama-3.1-8B TM-OPT+TC 16x64 map; its E0M3 share is recorded;
  - **n16k64_wA_nodisp:** the same tile and arrangement, format dispatch compiled out. Built for C3 with
    `sm120/build.py --config n16k64_wA_nodisp`.
- **The old per-MMA-branch kernel** (~504 TFLOP/s) cannot be built from the current sources: the generator emits only
  the per-k_block dispatch. It is cited from the report, not re-measured.
- **Timing:** CUPTI kernel time, the median of 20 calls, in each of 3 rounds with the kernels in a rotated order; the
  median of the per-round medians. TFLOP/s = 2 · 4096³ / time. Default clocks; the GPU must be idle.

## Static SASS census (`experiments/paper_extra/C2_sass.py`)

- **Recorded per build** (stock_wA, n16k64_wA, n16k64_wA_nodisp, n8k64_wB, stock_wB): the GEMM function's instruction
  count, OMMAs by format, predicated OMMAs, WARPSYNC, BRX, the opcode mix.
- **Per path through the steady-state k-loop:** the minimum and the maximum OMMA count, from a basic-block graph of
  the disassembly.
- **The tensor-pipe estimate at 4096³:** CTAs × 8 warps × (K / 128) × OMMAs per iteration. It holds only when every
  path issues the same count; it is labelled an estimate.

## Report (`experiments/paper_extra/C2_report.py`)

`C2.md` has three tables, each preceded by the caveat: kernel time, the census, and the historical numbers
(stock 1207 TFLOP/s and 8,388,608 tensor instructions; branch per MMA 504 TFLOP/s and 16,777,216; brx.idx per MMA
296; one brx.idx per k_block 865; all RTX 5090, ncu).

## Deviations (append-only)

(none yet)
