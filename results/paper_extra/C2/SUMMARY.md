# C2-lite summary: the mixed kernel at M = N = K = 4096 on the RTX PRO 6000

> **Caveat.**
> - ncu is unavailable on this machine: it is not installed, and the counters are blocked (ERR_NVGPUCTRPERM).
> - Clocks were not locked (no root); the rotated 3-round method stands in for that.
> - Tensor-pipe counts are static SASS census counts (per-path OMMA count × loop trips), not measured counters.
> - There are no stall reasons and no bank-conflict data.
> - Per-MMA-branch kernel numbers are historical (sm120/kernel/docs/mixed_nvfp4_report.md, RTX 5090, ncu), cited but
>   not re-measured on this GPU.

**Run:** registered protocol (`PROTOCOL.md`, 08d361a).
- The static SASS census ran at 17:47 UTC on the CPU. The timing ran at 19:22 UTC on the idle GPU and exited 0.
- The five libraries were built on 2026-09-27 (n16k64_wA_nodisp on 2026-09-29 17:26, for C3), and none was rebuilt
  after the census.
- The board's power limit was 500 W (default 600 W), as in every earlier GPU record of this campaign.
- **Deviation 1:** a wording fix in the report script. n8k64_wB's warp arrangement is 1 x 8, not 4 x 2. No number
  changed.
- **Tables** in `C2.md`, data in `C2.json` (the timing and the census).

**Kernel time at 4096³** (CUPTI, 128-wide CTA tile, median of 3 rotated rounds of 20 calls):

| kernel | µs | TFLOP/s | vs stock_wA |
|---|---:|---:|---:|
| stock_wA (NVFP4 weights) | 101.74 | 1350.8 | — |
| n16k64_wA, every tile E2M1 | 104.78 | 1311.6 | +3.0 % |
| n16k64_wA, the tags of a real layer (4.44 % E0M3 tiles) | 105.50 | 1302.7 | +3.7 % |
| n16k64_wA, every tile E0M3 | 109.09 | 1259.9 | +7.2 % |
| n16k64_wA_nodisp (format dispatch compiled out) | 102.72 | 1338.0 | +1.0 % |

- **The real layer** is `model.layers.0.self_attn.o_proj` of the committed FlipQuant (ours) Llama-3.1-8B 16x64 map
  (TM-OPT+TC).
- **The three rounds agree** within 0.8 % for every kernel.
- **The fixed cost splits** into the format dispatch (all-E2M1 vs nodisp) at +2.0 % and the rest (nodisp vs stock) at
  +1.0 %. C3 measured the same shape at T = 2048 / 8192: dispatch +2.3 / +2.0 %, the rest +1.0 / +1.2 %.
- **All E0M3 costs +4.1 % over all E2M1.** C3's random tags at 100 % cost +3.4 to +4.8 % at T ≥ 2048.
- **Against history** (a different GPU, so as ratios only):
  - the mixed kernel now reaches 97 % of stock with every tile E2M1, 96 % with the real layer's tags, and 93 % with
    every tile E0M3;
  - on the RTX 5090, the per-MMA-branch kernel reached 42 % of stock (504 vs 1207 TFLOP/s), and one brx.idx per
    k_block 72 % (865).

**Static SASS census of the GEMM function:**
- **Code size.** The stock builds have 1624–1640 instructions and 64 static OMMAs. The mixed builds have 4264
  (n16k64_wA) and 4976 (n8k64_wB), and 1024 static OMMAs each, 512 per format.
- **No wasted tensor work in the code.** No build has a predicated OMMA or a BRX.
- **Every path issues the same OMMAs.** Through the steady-state k-loop, every path issues 32 OMMAs per iteration
  (minimum = maximum = 32), in all five builds.
- **The tensor-pipe estimate at 4096³** is therefore 8,388,608 for stock and mixed alike. That equals the historical
  ncu count of the stock kernel. The historical per-MMA-branch kernel executed 16,777,216 (2×, half of them
  predicated off).
- **What this does and does not show.** The E0M3 tags change which OMMA encoding a path issues, not how many OMMAs it
  issues. The remaining +3.0 to +7.2 % is outside the tensor-instruction count: dispatch, the E0M3 path and the
  arrangement. Without counters, this run cannot say where inside the kernel it goes.
