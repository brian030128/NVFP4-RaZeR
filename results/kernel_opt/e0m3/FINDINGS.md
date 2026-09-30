# Why is E0M3 slower? — findings

Protocol `PROTOCOL.md` (registered in ef31dc6). Runs on 2026-09-30 on the idle RTX PRO 6000 (500 W power limit):
- test A (`test_a.json`, and the host-bound first run `test_a_iters64_hostbound.json`; see deviation 1);
- tests B and C (`test_bc.json`).

Every check passed:
- **Hook inert:** n16k64_wA and n8k64_wB rebuilt from the sources with the hook have the paper builds' patched and
  unpatched SASS.
- **Permuted builds bitwise:** the `*_xor` builds' outputs equal the defaults' bitwise (5 of 5).
- **Test A streams:** test A's four unpatched instruction streams are identical apart from the PRMT selector, and every
  patch census is complete (232 OMMAs per kernel).

**The answer:** E0M3 is not slower; its arm is. The 16-arm dispatch is a balanced binary tree, and ptxas lays out each
`if (p <= mid)` with the lower half as fall-through:
- the all-E2M1 arm (pattern 0) is reached with 0 taken branches;
- the all-E0M3 arm (pattern 15) needs 4 taken branches, per k_tile, per warp.

Permuting the arms flips the sign of the "E0M3 cost". The E0M3 MMA itself runs as fast as E2M1 and draws no more power.

## Findings table

| hypothesis | verdict | evidence |
|---|---|---|
| **H1** E0M3 MMA throughput | **ruled out** | Test A (identical instruction streams; only the patched format bits differ), random data, isolated, vs E2M1×E2M1: E0M3×E2M1 −0.4 %, E2M1×E0M3 −0.6 %, E0M3×E0M3 −0.6 %. Constant and zero data: 0.0 %. |
| **H2** dispatch path / branch layout | **supported** | Test B at 4096³. n16k64_wA, all-E0M3 vs all-E2M1: **+4.2 %** isolated / **+4.0 %** sustained. The same arms permuted (`MIXFP4_ARM_XOR=15`, all-E0M3 on the fall-through path): **−4.1 % / −4.0 %**. Moving all-E2M1 onto the 4-taken-branch leaf costs it +4.3 %; moving all-E0M3 off it saves 4.0 %. n8k64_wB: **+1.3 % / +1.4 %** → **−2.1 % / −2.6 %**. |
| **H3** data-dependent power at the 500 W cap | **not supported** (the registered "supported" condition fails. The literal "ruled out" clause, "neither clocks nor power differ", does not hold either, but only because E0M3's clock is higher: the opposite of H3's mechanism.) | Test A: at the cap, the E0M3 MMA loops run at a higher SM clock than E2M1 (2430–2445 vs 2407 MHz), i.e. less power per MMA. Test C, sustained (every 4096³ configuration at the cap): all-E0M3 runs at a higher clock than all-E2M1 (1837 vs 1725 MHz) and is still slower, so it draws less power. The E0M3 cost is the same isolated and sustained (+4.2 / +4.0 %). With constant data it persists (+6.1 / +5.5 %), not smaller than with real data (+4.2 / +4.0 %), so there is no data-dependent part. The permutation reverses it under the same power conditions. |
| **H4** other | nothing further needed | The same bits, tagged E0M3 instead of quantized as E0M3 (`e0m3_samebits`), cost +3.9 % (isolated), against +4.2 % for real E0M3 data: the quantized values play no part. Consistent with the RTX 5090 ncu record at base clocks (+7.5 %, no_instruction stalls ×2.6, branch_resolving +21 %): instruction-fetch redirects after taken branches. |

**Power governs absolute speed, but equally for both formats.**
- In sustained runs every 4096³ GEMM holds the 500 W cap, at 1687–2010 MHz.
- Constant data runs 6–10 % faster than random data (1912 vs 1725 MHz).
- This is why the isolated and sustained times differ, and why C2's back-to-back times depend on what ran before.

## Numbers (test B/C, n16k64_wA and n8k64_wB at 4096³, µs, medians of 3 rotated rounds)

| configuration | isolated | sustained | SM MHz (sust.) |
|---|---:|---:|---:|
| stock_wA | 116.1 | 139.9 | 1732 |
| nodisp | 116.8 | 140.9 | 1747 |
| e2m1 | 119.6 | 144.9 | 1725 |
| e0m3 | 124.6 | 150.7 | 1837 |
| real (4.44 % E0M3 tiles) | 120.0 | 145.0 | 1725 |
| e0m3_samebits | 124.4 | 149.1 | 1852 |
| e2m1_const | 112.3 | 130.7 | 1912 |
| e0m3_const | 119.1 | 137.9 | 2010 |
| xor_e2m1 | 124.7 | 149.1 | 1871 |
| xor_e0m3 | 119.6 | 143.1 | 1740 |
| xor_real | 124.7 | 149.1 | 1852 |
| wB_stock | 115.6 | 138.9 | 1732 |
| wB_e2m1 | 125.7 | 153.2 | 1687 |
| wB_e0m3 | 127.4 | 155.3 | 1740 |
| wBxor_e2m1 | 128.4 | 157.3 | 1762 |
| wBxor_e0m3 | 125.8 | 153.2 | 1702 |

Isolated ratios repeat C2's: e0m3 vs e2m1 +4.2 % (C2: +4.1 %), e2m1 vs nodisp +2.5 %, nodisp vs stock +0.6 %. Test A
per launch (4096 iterations, µs): random 766 / 763 / 762 / 762 (sites 0–3, isolated); constant and zero 747 for every
site.

## What it means

- **For the paper:** the "all-E0M3 costs +4.1 %" of C2 and the E0M3-share curves of C3 are the dispatch tree's code
  layout, not a property of the E0M3 format or the hardware. With the arms reversed, the all-E0M3 GEMM is faster than the
  all-E2M1 one.
- **For real maps:** about 90 % of a warp's k_tiles are all-E2M1 at their 1.3–3.3 % E0M3 shares. The default layout
  already gives that pattern the fall-through path, which is why C3 found the real maps' E0M3 cost within about ±1 %.
  The remaining taken branches belong to the mixed patterns (1–3 E0M3 granules).
- **A possible follow-up (not run; bitwise-safe, the same arms):** a frequency-aware tree. For example, test pattern 0
  first (1 compare instead of 4), then pattern 15, then the rest, or order the tree by the measured pattern
  frequencies. It could recover part of the dispatch cost (e2m1 vs nodisp +2.5 %) and the mixed patterns' taken
  branches. This is direction #2/#3, pending the user's decision.
- **ncu is not needed** to settle H1–H3. Counters cannot be enabled on this node (test D cancelled), and a run on the
  RTX 5090 is not required for these conclusions.
