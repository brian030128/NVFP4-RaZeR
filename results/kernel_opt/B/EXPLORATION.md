# Kernel-opt B: per-warp run tables — exploration only, not registered

2026-09-30, branch kernel-opt, step 2 of the user's plan ("B, per-warp run tables for 16x64 maps … Explore first
(disclosed), then register"). These are disclosed exploratory timings, not results.

The exploration shows that B as specified loses on the real maps of both units, so no B amendment is registered. The
prototype stays out of kernel-opt. Its full diff is `exploration/B_runs_prototype.patch` (worktree `wtB` on
9d45342): the collective hook, the C ABI, `mixfp4_sm120/runs.py`, the wrappers, NativeLinear, configurations, tests
and the gate and measurement scripts.

## What was built

- **Collective hook `MIXFP4_RUNS=1`:**
  - `Arguments` / `Params` gain `ptr_runs` / `runs_stride`.
  - Given a per-warp table, the pipelined joint-K-A mainloop enters each run's arm once and loops the unchanged k_tile
    body inside it; the tail body closes the last run.
  - Without a table it runs the per-k_tile dispatch.
  - Unset, the collective is upstream.
- **Supporting code:**
  - C ABI: `sm120_gemm_runs`, `sm120_linear_runs`, and a `runs` flag in `sm120_describe`.
  - `runs.py`: builds the table from the tags, vectorized, with a plain-loop reference.
  - `Kernel.gemm` / `gemm_ptr` / `KernelSet` / NativeLinear pass the table on every path.
- **Builds:** eight in `build_B`, all with the expected census (twice the plain build's, since both paths are
  compiled) and 0 predicated OMMAs, and no spills:
  - n16k64_wA_rt{,_n64,_n32,_n16} (set `mixed_rt`);
  - n16k64_wA_g32_rt{,_n64,_n32,_n16} (set `mixed256_rt`: the A′ builds; B applies to them unchanged).
- **Correctness:** `pytest test_runs.py test_select.py` on `build_B` gave 183 passed (23 skipped: families not built
  there).
  - With a table, every width of both sets is bitwise equal to its own per-k_tile path and to the reference build
    (n16k64_wA from `sm120/build`, n16k64_wA_g32 from `build_A1`).
  - The tests cover the decode probe, random maps with every joint pattern, partial panels, K not a multiple of 128,
    and NativeLinear's fused, reuse and unfused paths.
  - Every timing below was also checked bitwise.

## Timings

**Real 16x64 maps** (`exploration/quick_B.py`, `quick_B.out`; Llama-3.1-8B layer 0), relative to n16k64_wA:

| case | freq (#2) | run tables |
|---|---|---|
| 4096³ width 128 (warm), all-E2M1 | −0.33 % | −0.76 % (+0.6 % over nodisp) |
| … the real map (o_proj, 10.0 runs per warp over 32 k_tiles) | −0.76 % | **+1.80 %** |
| … the densest o_proj (14.4 runs per warp) | −0.00 % | **+3.08 %** |
| … patterns 3 / 5 / all-E0M3 everywhere (1 run per warp) | +0.2 / +2.2 / +1.5 % | −1.4 / −3.0 / −5.9 % |
| q_proj 4096x4096, T = 1 / 16 / 64 (width 16 / 16 / 32), cold | −0.2 / −0.9 / +1.4 % | **+9.3 / +11.5 / +27.0 %** |
| down_proj 4096x14336, T = 1 / 16 / 64, cold | −0.5 / −0.4 / −0.5 % | −0.2 / −0.1 / +3.0 % |
| gate_proj 14336x4096, T = 1 / 16 / 64, cold | −0.3 / −0.1 / −0.1 % | +0.2 / +0.1 / +2.2 % |

"Cold" means weights rotated and flushed, with the activations quantized after the flush (M1's condition).

**Real 256x64 maps on the g32 builds** (`exploration/quick_B3.py`, `quick_B3.out`):
- q_proj (10.6 runs per warp): +0.7 to +9.2 % cold, +6 to +13 % warm.
- down_proj (31 runs over 112 k_tiles): −0.1 to +1.5 % cold.
- gate_proj (9 runs): +0.1 to +3.5 % cold.

**Why: synthetic probes** (`exploration/quick_B2.py`, `quick_B2.out`; 4096x4096, warm, width 128 and 16):

| map | runs per warp | width 128 | width 16 |
|---|---:|---:|---:|
| all-E2M1 | 1 | −0.3 % | **−11.7 %** |
| alternating every k_tile, all warps together | 32 | +5.7 % | +13.3 % |
| alternating every 2 k_tiles, together | 16 | +0.9 % | −0.4 % |
| alternating every 4 k_tiles, together | 8 | +1.0 % | −5.6 % |
| every 2 k_tiles, warps staggered | 17 | +2.5 % | +9.2 % |
| every 4 k_tiles, warps staggered | 9 | +0.2 % | +8.4 % |

## Reading

- **Removing the dispatch is worth a lot where it is expensive.** With one run per warp, width 16 is 11.7 % faster.
- **A run boundary costs about twice a per-k_tile dispatch.**
  - Runs of one k_tile are +13 % at width 16.
  - A boundary is the run loop's control, the entry decode, the arm tree and the inner loop's entry and exit.
- **The CTA pays for its slowest warp.** The CTA's warps meet at a barrier every k_tile.
  - When the warps' boundaries are staggered, some warp is at a boundary on most k_tiles: +8 to +9 % at width 16 with
    runs of 2 to 4.
- **Real maps have short runs** (2 to 3 k_tiles on average).
  - 16x64 maps: per-warp boundaries are scattered.
  - 256x64 maps: the boundaries are shared by the panel's warps but still frequent.
  - So per-warp run tables lose on both: +2 to +27 % on the cells that matter.
- **B does not supersede #2's pattern-0-first dispatch.**

## A refinement that keeps what works (not built; proposed)

Keep the per-k_tile dispatch, but give each warp one bit per k_tile ("all-E2M1"), built from the tags at load and read
from a register word.
- On a zero bit (88–94 % of k_tiles), run arm 0 directly: a shift and a test instead of the flag reads and the tree.
- On a one bit, run the flag reads and the tree as today.
- There are no run loops and no run boundaries, and every warp does the same kind of work every k_tile, so the barrier
  coupling does not bite.
- This is what "dispatch-free all-E2M1 k_tiles" needs. The probe's all-E2M1 case (−11.7 % at width 16, −0.3 % at 128)
  bounds its gain from above.
