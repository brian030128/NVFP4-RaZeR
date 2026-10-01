# Kernel-opt t0: why n16k64_wA_nodisp was slower than stock_wA, and the bitwise-safe fix

Protocol: `results/kernel_opt/PROTOCOL.md`, amendments 6 and 6b (`registration_6.json`, 75dade0;
`registration_6b.json`, c7d5880).
- The first chain stopped at its first gate (`stop_g3/STOP.md`, df73edf).
- After amendment 6b it ran from the start on 2026-10-01, 03:17–04:18 UTC, with no deviation.

Files:
- Tables: `abT_tables.md`; data: `abT.json`.
- Gates: `g1_g2_sass.json`, `g2_sass_freq.json`, `g4_bitwise_*.json`, `g5_logits_*.json`.
- Exploration (disclosed, before registration): `exploration/`. The 6b validation: `6b/`.

**In short:**
- **The cause.** The whole nodisp-vs-stock gap is 4 identity PRMT instructions per k_tile and MMA warp. They are the
  "site tags" the MMA blob emits so that the SASS patcher can find the OMMAs to switch to E0M3. Everything else in the
  k_tile loop is stock's, opcode for opcode.
- **The fix.** Site 0 (E2M1) is never patched, so it no longer gets a tag. The kernels are bitwise identical before
  and after (all gates passed).
- **nodisp now equals stock.** At 4096³, nodisp vs stock_wA goes from +0.9 / +0.7 / +1.1 % to +0.05 / −1.0 / +0.02 %
  (back-to-back / isolated / sustained).
- **The real kernels gain less.** The 16x64 per-forward GEMM time drops by 0.2–0.4 % at the median. The gap to stock
  shrinks by 0.86 points at the median at T ≥ 256 (range −0.04 to −1.36) and by about 0–0.2 below that. 256x64 is
  unchanged (+0.01 %).
- **What remains of the 16x64 gap is the dispatch itself:** about 2.1 % over the tag-free ceiling at 4096³, plus the
  real map's E0M3 tiles.
- **Adoption.** The registered criterion holds only for nodisp and 16x64 typical with the default dispatch. Small
  "slower beyond the round range" cells (+0.1 to +0.35 %) fail it for the other units; see Adoption.

## The cause (disclosed exploration)

- **The k_tile loop.** stock_wA and n16k64_wA_nodisp were built from the same sources with the same epilogue tile.
  The MMA warps' k_tile loop is 91 vs 95 instructions. The only difference is `PRMT Rd, Rs, 0x3210, Rs` ×4: identity
  copies of the A scale-factor register (`exploration/stock_loop6.txt`, `nodisp_loop6.txt`).
- **Where they come from.** The blob emits `prmt.b32 sf, x, x, SEL` before every MMA's scale operand; its selector
  tells the patcher the MMA's format site. ptxas keeps one per (m-atom, k_block).
- **The real kernel pays the same.** Every dispatch arm carries 4 of them per k_tile, the all-E2M1 arm (88–95 % of
  k_tiles) included.
- **The MMA order** (the blob issues m-major; CUTLASS alternates the m-atoms, with operand reuse) has no measurable
  effect.
- **Timing, 7 cells:**

  | variant | vs stock |
  |---|---|
  | nodisp | +0.9 to +2.0 % |
  | tags dropped | −0.6 to +0.3 % |
  | CUTLASS order with tags | +0.5 to +1.8 % |
  | both | −0.4 to +0.4 % |

## The fix

**`gen_mixed_mma_blob.py TAG0=0`:** site-0 MMAs read their scale word directly. The E0M3 sites keep their tags. With
defaults the generated output is byte-identical.

**`patch_mixed_nvfp4_gemm.py --untagged-site0`:** each OMMA's site is read from the reaching definitions of its scale
register over the kernel's control-flow graph.
- All reaching definitions are tags of site s: site s. None is an E0M3 tag: site 0. Anything else is an error.
- An indirect branch (`BRX`) is resolved from the kernel's `.nv.constant2` jump table (amendment 6b). An unresolvable
  one is an error.
- Validation, by comparison with today's strict patcher:
  - all 74 existing tagged libraries get the same site for every OMMA;
  - so do all 21 existing tagged self-test executables, which do contain the BRX.
- Without the flag, the patcher is unchanged.

**Builds:**
- 16x64: n16k64_wA{,_n64,_n32,_n16}_t0, with the default dispatch (`build_T`) and with #2's (`build_Tfreq`);
- 256x64: n16k64_wA_g32{,_n64,_n32,_n16}_t0;
- the ceiling: n16k64_wA_nodisp_t0;
- sets `mixed_t0` and `mixed256_t0`.

LOCAL_CHANGES covers the two vendored scripts.

## Gates (all passed)

- **G3 self-tests:** 8 of 8 PASS (the upstream driver: PASS patched, FAIL unpatched under random tagging).
- **G1:** the 28 existing configurations keep their patched and unpatched SASS.
- **G2:** the 9 new builds in `build_T` and the 4 in `build_Tfreq` have their census; nothing is predicated.
- **pytest:** 1,054 passed on `build_T`, including all t0 builds, the 256x64 probes and the width / batch-invariance
  tests of the t0 sets. 6 passed on `build_Tfreq`.
- **G4:** 0 differences in 91,200 comparisons against n16k64_wA (`sm120/build`):
  - 33,600 for 16x64;
  - 24,000 for 256x64;
  - 33,600 for 16x64 with #2's dispatch.
- **G5:** whole-model logits bitwise equal on 4 models × 5 shapes, for 16x64, 16x64 with #2's dispatch, and 256x64.
- **M1's own checks:** every after call equals its before call bitwise, and nodisp_t0 equals stock_wA.

## M1: per-forward GEMM time, after vs before (deviation-2, 4 models × 12 T)

| unit | median | min | max | cells slower beyond the round range |
|---|---:|---:|---:|---:|
| nodisp → nodisp_t0 | −1.01 % | −1.46 % | −0.36 % | 0 of 48 |
| 16x64 typical, default dispatch | −0.42 % | −1.32 % | −0.00 % | 0 of 48 |
| 16x64 worst, default dispatch | −0.35 % | −1.09 % | +0.32 % | 5 of 48 |
| 16x64 typical, #2's dispatch | −0.27 % | −1.05 % | +0.19 % | 6 of 48 |
| 16x64 worst, #2's dispatch | −0.21 % | −1.04 % | +0.34 % | 9 of 48 |
| 256x64 typical | +0.01 % | −0.67 % | +0.26 % | 15 of 48 |
| 256x64 worst | −0.10 % | −0.53 % | +0.35 % | 9 of 48 |

Where the flagged cells are (each +0.0 to +0.35 % at the median):
- 16x64 worst, default dispatch: Phi-4 at T ≤ 64.
- 16x64 with #2's dispatch: mostly T ≤ 16 (Llama, Mistral, Phi-4), plus Qwen at T = 128 and Mistral at T = 512.
- 256x64: Qwen at T ≤ 64, and T ≥ 512 on all models.

A note on the criterion: if the true effect is zero, each cell has about a 1 in 8 chance that all 3 rounds land on
the slower side. So about 6 of 48 flags are expected for a unit with no change. The 16x64 units' flags are at or near
that level. 256x64 (15) shows a real but tiny slowdown in those cells.

### Residual gap vs stock_wA per T (typical tags; range over the 4 models), before → after

| T | 16x64, default dispatch | 16x64, #2's dispatch | 256x64 (A′) |
|---|---|---|---|
| 1 | +0.7…+1.1 → +0.6…+1.0 % | +0.3…+0.9 → +0.5…+1.0 % | +0.1…+0.4 → −0.0…+0.3 % |
| 16 | +0.9…+1.1 → +0.7…+0.9 % | +0.4…+0.8 → +0.4…+0.9 % | +0.1…+0.2 → +0.1…+0.2 % |
| 64 | +1.2…+1.8 → +0.9…+1.3 % | +0.8…+1.5 → +0.7…+1.3 % | +0.4…+0.5 → +0.3…+0.5 % |
| 128 | +1.9…+4.5 → +1.8…+4.1 % | +1.0…+4.3 → +1.1…+4.1 % | +0.5…+3.8 → +0.6…+3.7 % |
| 256 | +3.4…+4.4 → +2.4…+3.6 % | +2.8…+3.5 → +1.7…+2.6 % | +1.3…+2.5 → +1.2…+2.1 % |
| 512 | +4.0…+5.9 → +3.0…+5.3 % | +3.5…+5.3 → +2.6…+5.3 % | +2.3…+4.5 → +2.5…+3.9 % |
| 1024 | +3.0…+3.9 → +2.2…+2.7 % | +2.6…+3.1 → +1.9…+2.1 % | +1.9…+2.4 → +1.4…+2.4 % |
| 2048 | +2.9…+4.4 → +2.1…+3.7 % | +2.3…+3.4 → +1.3…+2.7 % | +1.6…+2.6 → +1.8…+2.8 % |
| 4096 | +3.0…+3.3 → +2.0…+2.5 % | +2.3…+2.7 → +1.5…+1.9 % | +1.4…+2.0 → +1.6…+2.1 % |
| 8192 | +3.4…+3.9 → +2.6…+3.1 % | +2.8…+3.4 → +1.8…+2.6 % | +2.0…+2.6 → +2.2…+2.6 % |

`abT_tables.md` has every T and model. The nodisp rows there compare a width-128-only build against stock's
width-selected kernels, so nodisp vs stock is meaningful only where both run width 128 (T ≥ 2048):
- T = 2048: +0.8…+1.6 → +0.0…+0.9 %;
- T = 4096: +0.9…+1.1 → −0.2…+0.2 %;
- T = 8192: +1.0…+1.1 → +0.2…+0.3 %.

## C2‴: 4096³ (CUPTI; medians of 3 rotated rounds)

| comparison | back-to-back | isolated | sustained (500 W cap) |
|---|---:|---:|---:|
| nodisp vs stock_wA | +0.88 % | +0.72 % | +1.07 % |
| **nodisp_t0 vs stock_wA** | **+0.05 %** | **−1.00 %** | **+0.02 %** |
| t0 vs tagged, default dispatch: all-E2M1 / real map / all-E0M3 | −0.44 / −0.54 / −2.80 % | −0.49 / −0.89 / −2.83 % | −1.05 / −0.35 / −2.36 % |
| t0 vs tagged, #2's dispatch: all-E2M1 / real map / all-E0M3 | −0.71 / −0.75 / −2.88 % | −1.38 / −1.61 / −3.50 % | −0.38 / −0.18 / −2.59 % |
| dispatch cost, all-E2M1, #2's dispatch, vs its own ceiling: before → after | +1.95 → +2.07 % | +1.76 → +2.10 % | +0.34 → +1.00 % |
| 16x64 real map vs stock_wA, #2's dispatch: before → after | +3.35 → +2.57 % | +2.99 → +1.33 % | +2.70 → +2.52 % |

The all-E0M3 arm keeps its 4 tags, yet it gains 2.4–3.5 %. That is a side effect of the other arms shrinking (code
layout; the E0M3 investigation found that branch layout drives that arm's cost). It is not the tag removal itself.

## Adoption

The registered criterion: adopt the t0 builds for 16x64 and 256x64, with both dispatch variants, if no unit is
slower beyond its round range at any (model, T).
- It holds for nodisp and for 16x64 typical with the default dispatch.
- It fails for 16x64 worst (5 cells), 16x64 with #2's dispatch (6 and 9 cells) and 256x64 (15 and 9).
- **So t0 is not adopted under the registered rule.** The coordinator and the user decide among:
  1. **Adopt t0 for the 16x64 family, both dispatch variants (recommended).** This needs a disclosed deviation from
     the criterion.
     - The medians improve on every 16x64 unit, and the gap shrinks by 0.86 points at the median at T ≥ 256.
     - No cell is worse than +0.35 %, and the flag counts are at the chance level for a 3-round test.
  2. **Do not adopt for 256x64.** It is neutral, with a small consistent cost in 15 cells.
  3. Keep everything as is. t0 then stands as the diagnosis, and the builds stay available.

## What this means for the gap

With the tags gone, the no-dispatch ceiling equals stock. Everything left of the 16x64 gap belongs to the dispatch
(4096³, #2's dispatch, isolated):
- **the per-k_tile pattern dispatch itself:** +2.1 % over the tag-free ceiling on an all-E2M1 map;
- **the real map's E0M3 tiles:** +0.25 % (isolated) to +1.5 % (sustained) on top.

B′, a per-warp all-E2M1 bit per k_tile, is aimed at the first term, which is now the dominant one.

## Adoption (2026-10-01)

The user adopted t0 for the 16x64 family, for both dispatch variants. This is a disclosed deviation from the
registered criterion: every 16x64 median improves, and the flagged cells are ≤ +0.35 %, about the chance level for 3
rounds.

It is deployed in `mixed_ko` (amendment 7): n16k64_wA_n16_t0 / _n32_t0 / _n64_t0, and n16k64_wA_e64_t0 (t0 with #4's
epilogue tile) at width 128.

t0 is not adopted for 256x64, where it is neutral; 256x64 keeps its A′ builds.
