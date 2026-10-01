# Kernel-opt B′: per-warp all-E2M1 bits — exploration only, not registered

The user approved B′ through the coordinator: "B′: go. Per-warp all-E2M1 bit per k_tile, built at load and held in
registers. A zero bit runs arm 0 with no flag reads or tree. Build it on the t0 (tag-free) sources, for the 16x64
family … Explore first, then register … State whether it supersedes freq." These are disclosed exploratory timings,
not results.

**Outcome: B′ does not supersede #2's dispatch (freq), so no B′ amendment is registered.**
- **All-E2M1 maps**, where every bit is zero (B′'s best case): the best variant only matches today's default dispatch
  (−0.4 … +1.8 %). #2's dispatch is 0.7–1.3 % faster than both.
- **Maps with E0M3 tiles:** B′ is slower than both. A set bit reads the flags unpipelined.

## What was built

`Bprime_zerobit_prototype.patch` here is on top of amendment 7's sources (0ccb0a9).

**Host side:**
- `sm120/mixfp4_sm120/zbits.py` builds, at load, one bit per (CTA row panel, warp row, k_tile). Bit kt is 0 iff the
  warp's pattern there is 0, i.e. every granule it reads is E2M1. Rows are 32-bit words, padded by one for a read-ahead.
- `NativeLinear` and `Kernel.gemm` / `gemm_ptr` pass the bits. csrc gains `sm120_gemm_ex2` / `sm120_linear_ex2`
  (scheduler setting plus bits), and the library describes itself with `"zbits"`.

**The collective's hook `MIXFP4_ZEROBIT`**, in the pipelined joint-K-A mainloop:
- the warp loads its bit row; per k_tile, a zero bit runs arm 0's body (`k_tile_body(C<0>)`) with no flag read and no
  tree;
- a set bit reads the flags at the top of the k_tile and dispatches among arms 1–15;
- without bits, the unchanged pipelined dispatch runs.
- Three loop shapes:
  1. a flat loop with a funnel shift and a predicated window update;
  2. a loop over 32-k_tile windows, one shift per k_tile, the next word read a window ahead;
  3. as 2, but with arm 0 on the fall-through path.

**Builds** of n16k64_wA_e64_t0_zb and n16k64_wA_n16_t0_zb went into `build_Z` / `build_Z2` / `build_Z3`. All had the
expected census (twice their bases', because the fallback path stays), no predication and 168 registers.

**Correctness.** Every timed output was bitwise equal to the t0 build's. `tests/test_zbits.py` (in the patch) holds
the reference test of the bit packing, which passed on CPU, and the GEMM and negative-control tests, which were not
run.

## SASS (CPU)

| path | per-k_tile cost |
|---|---|
| bit-zero k_tile, variant 1 | about 16 overhead instructions around arm 0's 86, including 5 predicated window-update instructions that issue every k_tile |
| bit-zero k_tile, variant 2 | about 10 overhead instructions |
| today's pipelined dispatch, arm 0 | about 26: the 4-level tree, the flag reads inside the body, `BSSY`/`BSYNC`/`WARPSYNC` |

Variants 1 and 2 reach arm 0 through a taken branch (`@!P0 BRA` to a far block), so a bit-zero k_tile takes two
branches, counting the back-edge. Today's dispatch reaches arm 0 entirely by fall-through. Variant 3 restores the
fall-through (checked in the SASS).

## Timing

Method: the M1 condition (cold weights by rotation and a 512 MiB flush, activations quantized after the flush),
CUPTI, 3 rotated rounds × 30. Maps: all-E2M1; "real" (Llama o_proj layer 0 for 4096x4096, otherwise random 12 %);
all-E0M3.

Change vs today's adopted default dispatch (t0, `build_7`), cold, from `quick_Z.out` and `quick_Z3.out`:

| cell | map | #2's dispatch | B′ v1 | B′ v2 | B′ v3 | stock |
|---|---|---|---|---|---|---|
| width 128, 4096³ | all-E2M1 | −0.6 / −0.7 | +2.6 | +1.4 | −0.0 | −1.7 |
| | real | −0.5 / −0.7 | +4.2 | +2.4 / +3.0 | +1.8 | −2.0 / −2.3 |
| | all-E0M3 | +1.2 | +5.6 | +4.1 / +4.3 | +4.7 | −3.8 / −4.0 |
| width 128, 4096x4096, T = 1024 | all-E2M1 | −1.0 / −1.1 | +5.9 | +4.5 / +4.8 | +1.8 | −1.2 |
| | real | −0.8 / −0.9 | +9.3 | +6.3 / +7.3 | +4.5 | −2.0 / −2.3 |
| width 128, 14336x4096, T = 2048 | all-E2M1 | −0.9 | +1.9 | +1.1 / +1.3 | −0.3 | −1.4 / −1.6 |
| | real | −0.2 / −0.5 | +3.7 | +2.2 / +2.3 | +1.6 | −3.5 |
| width 128, 4096x14336, T = 512 | all-E2M1 | −0.9 / −1.3 | +4.5 | +2.6 / +2.8 | −0.4 | −1.5 / −1.7 |
| | real | +0.3 / +0.5 | +8.2 | +4.8 / +5.0 | +3.6 | −5.3 / −5.8 |
| width 16, T = 16 (3 shapes) | all-E2M1 | −0.2 … −0.9 | −0.1 … +0.4 | −0.6 … +0.6 | −0.0 … −0.3 | −0.4 … −1.3 |
| | real | −0.9 … +0.4 | +0.4 … +4.5 | +0.6 … +2.4 | +0.6 … +2.2 | −0.4 … −1.7 |

Where a cell shows two values, they come from the two runs, `quick_Z` and `quick_Z3`.

**Reading:**
- **The taken branch was the dominant cost of variants 1 and 2.** Moving arm 0 to the fall-through (variant 3) gains
  1.4–3.0 points on all-E2M1 maps at width 128, and about nothing at width 16.
- **Even then B′ only matches the default dispatch, not #2's.** #2's dispatch tests pattern 0 first, with its arm on
  the fall-through path. On all-E2M1 maps it already recovers 0.6–1.3 points of the ≈1.5 % between the default
  dispatch and stock in these cells.
- **What is left looks like the per-k_tile branch itself.** It is the reconvergence `BSSY`/`BSYNC`/`WARPSYNC`, and B′
  keeps it, as C2′ suggested ("it comes with having the dispatch at all").
- **With E0M3 tiles B′ is slower.** A set bit reads the k_tile's flags at its top, unpipelined, with the shared-memory
  load on the critical path.

## Recommendation

B′ does not supersede #2's dispatch. C3k showed #2's dispatch is the better variant at real map densities (f ≈ 2–4 %),
and B′ is not faster than it in any measured cell.

For the cumulative e2e measurement, the 16x64 dispatch would be #2's: freq, together with t0, e64, the scheduler rows
and the 4b widths, i.e. the `mixed_ko` builds built with `MIXFP4_DISPATCH_FREQ=1`.
