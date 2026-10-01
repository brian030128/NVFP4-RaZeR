# Kernel-opt #4: epilogue tile and tile-scheduler order — both families faster, gap to stock unchanged

Protocol: `results/kernel_opt/PROTOCOL.md`, amendment 5 (registration `registration_5.json`, 663f8d9). The chain
`run_4.sh` ran on 2026-10-01 from 00:09 to 00:57 UTC with no deviation.

Files:
- Tables: `ab4_tables.md`; data: `ab4.json`.
- Tuned table: `table/`; its non-default rows: `schedule_rows.md`.
- Gates: `g1_g2_sass.json`, `g4_bitwise.json`, `g5_logits_*.json`.
- Exploration (disclosed, before registration): `exploration/`.

**In short:**
- #4 makes stock_wA and the 16x64 kernels faster by about the same amount: −0.6 % median per-forward GEMM time, with
  no (model, T) slower.
- The registered adoption criterion is met.
- Because #4 helps both families equally, the 16x64 gap to stock is unchanged (median −0.05 points).
- The 256x64 path, which #4 does not touch, falls about 0.6 points further behind the faster stock.

## What #4 is

Two changes that leave every output bit unchanged, applied identically to the mixed 16x64 family and to stock_wA. The
width rows are not touched, so every call runs at today's width.

- **Epilogue tile 64 × 64** instead of the automatic 64 × 32, at width 128 only (builds n16k64_wA_e64, stock_wA_e64).
  - At width 128 the 64 × 64 tile keeps the 4 mainloop stages.
  - At width 64 it costs a stage (6 → 5) and was +4.9 % / +3.6 % slower on Qwen down at T = 128 in the disclosed
    exploration. So width 64 keeps today's build there. The width-64 e64 builds were built and gated, but no set
    uses them.
- **Persistent tile-scheduler raster order and swizzle**, per (shape, T bucket).
  - These are host-side settings passed per call (`sm120_gemm_ex` / `sm120_linear_ex`); the device code is
    unchanged.
  - `tune_tiles.py --schedule` tuned them with the deviation-2 method: cold weights, activations quantized after the
    flush, 3 rotated rounds × 30.
  - A setting replaced the default only with a decisive margin: median ≥ 0.5 % better, and every round separated from
    the default's rounds.
- **Sets:**
  - `mixed_e` = {16: n16k64_wA_n16, 32: n16k64_wA_n32, 64: n16k64_wA_n64, 128: n16k64_wA_e64};
  - `stock_e` is the same with the stock builds;
  - both use the `mixed` / `stock` width rows plus the tuned `schedule` rows.

## Gates (all passed)

- **G3 self-tests:** n16k64_wA_e64 and n16k64_wA_n64_e64 PASS.
  - The step rebuilt both from the commit. Patched and unpatched SASS equal the registered `build_4` hashes; the
    library files differ only in non-code bytes.
- **G1:** all 24 other configurations keep their before builds' patched and unpatched SASS.
- **G2:** the census is correct and nothing is predicated:
  - n16k64_wA_e64 {0: 512, 1: 512};
  - n16k64_wA_n64_e64 {0: 256, 1: 256};
  - stock_wA_e64 and stock_wA_n64_e64 are E2M1-only.
- **Schedule tuning:** 19 minutes, 16 shapes × 14 buckets × 12 settings × 2 families.
  - `mixed_e`: 69 of 224 rows changed. 63 of them are "along M, swizzle 1".
  - `stock_e`: 81 of 224 rows changed. 76 of them are "along M, swizzle 1".
  - Per row, the tuner measured −0.5 to −3.9 % (`schedule_rows.md`).
- **pytest on `build_4`:** 563 passed.
  - This includes the width-equality tests of `mixed_e` / `stock_e`.
  - It also includes `test_schedules_bitwise_equal`: all 12 settings at every width, bitwise equal to the default.
- **G4:** 20,160 comparisons against n16k64_wA from `sm120/build`, 0 differences.
  - It covered n16k64_wA_e64, n16k64_wA_n64_e64 and set:mixed_e with the tuned table, on the 16x64 and 256x64 maps.
- **G5:** whole-model logits bitwise equal on 4 models × 5 shapes:
  - 16x64: set:mixed vs set:mixed_e;
  - FourOverSix: set:stock vs set:stock_e.
- **M1's own checks:** every after call equals its before call bitwise on the timed operands (1,260 / 1,260 / 720 /
  2,160 checks), with the same widths and no other processes.

## M1: per-forward GEMM time, after vs before (deviation-2, 4 models × 12 T)

| unit | median over the 48 (model, T) | min | max | cells slower beyond the round range |
|---|---:|---:|---:|---:|
| stock_wA → stock_e | −0.62 % | −1.15 % | −0.09 % | 0 |
| 16x64 typical: mixed → mixed_e | −0.61 % | −2.10 % | +0.02 % | 0 |
| 16x64 worst: mixed → mixed_e | −0.55 % | −2.00 % | −0.03 % | 0 |

- **Only one cell has a positive median:** Llama 16x64 typical at T = 1, +0.02 %, with rounds from −0.08 to
  +0.06 %.
- **16x64 gains most at:**
  - T = 128 on Llama / Mistral: −2.1 / −2.0 %. This is raster along M at width 32, as the exploration found.
  - T = 256 / 512 on Llama / Mistral: −1.2 to −1.4 %.
  - Qwen at T = 2048 / 4096: −1.5 / −1.2 %.
- **Stock gains most at** T = 32 to 256 and T = 4096 on Llama / Mistral: −0.8 to −1.2 %.
- **Phi-4 gains least:** −0.2 to −1.0 %.

**Adoption criterion** (no unit's per-forward sum slower beyond its round range at a (model, T) where anything
changed): met.

## Residual gap vs stock per T (typical tags; range over the 4 models)

| T | 16x64 vs stock: before (both today) | after (both with #4) | 256x64 (A′, unchanged) vs stock: before | vs stock with #4 | 16x64 on #2's freq builds vs stock today |
|---|---|---|---|---|---|
| 1 | +1.0 … +1.2 % | +0.9 … +1.7 % | −0.2 … +0.4 % | +0.0 … +0.8 % | +0.6 … +0.8 % |
| 4 | +1.0 … +1.3 % | +0.9 … +1.4 % | −0.2 … +0.2 % | −0.0 … +0.6 % | +0.7 … +0.9 % |
| 16 | +1.0 … +1.3 % | +1.0 … +1.5 % | −0.3 … +0.3 % | +0.1 … +0.7 % | +0.5 … +0.8 % |
| 32 | +0.8 … +1.2 % | +0.6 … +1.3 % | −0.5 … +0.1 % | −0.3 … +1.0 % | +0.4 … +1.0 % |
| 64 | +1.4 … +1.7 % | +1.2 … +2.3 % | +0.2 … +0.6 % | +0.3 … +1.7 % | +1.0 … +1.5 % |
| 128 | +2.1 … +4.4 % | +1.9 … +3.2 % | +0.8 … +3.6 % | +1.4 … +4.5 % | +1.3 … +4.1 % |
| 256 | +3.0 … +4.3 % | +2.8 … +4.1 % | +1.4 … +2.4 % | +2.6 … +3.0 % | +2.4 … +3.5 % |
| 512 | +4.2 … +5.9 % | +3.8 … +5.1 % | +2.6 … +4.2 % | +2.8 … +5.0 % | +3.6 … +5.1 % |
| 1024 | +3.1 … +3.9 % | +2.5 … +4.0 % | +1.7 … +2.4 % | +1.9 … +3.3 % | +2.6 … +3.2 % |
| 2048 | +3.2 … +4.1 % | +2.3 … +3.7 % | +1.8 … +2.6 % | +2.5 … +3.4 % | +2.5 … +3.6 % |
| 4096 | +3.1 … +3.3 % | +2.8 … +3.8 % | +1.5 … +2.1 % | +2.5 … +3.0 % | +2.5 … +2.8 % |
| 8192 | +3.3 … +3.9 % | +3.3 … +3.8 % | +2.1 … +2.6 % | +2.7 … +3.5 % | +2.9 … +3.4 % |

- **16x64:** over the 48 cells the gap changes by −0.05 points at the median (−1.28 to +0.71) and narrows in 28 of
  them.
  - It narrows most at T = 128 on Llama / Mistral (−1.3 / −1.2 points). There mixed runs width 32 and stock width 64,
    and raster along M helps the narrow width most, as in the exploration.
  - It also narrows at T = 512 / 1024 on Llama / Mistral (−0.5 to −0.8) and on Qwen at T = 2048 (−0.9).
  - On Llama / Mistral it widens at T ≤ 64 (+0.1 to +0.6), where stock gains more; the largest widening is +0.7 (Llama,
    T = 4096). On Phi-4 and Qwen it moves by ±0.2 at T ≤ 64.
- **256x64:** the gap grows by stock's gain (+0.1 to +1.2 points), because #4 was not applied to the g32 builds.
- **The "after" columns are measured on the default-dispatch builds.** #2's frequency-aware dispatch is not combined
  with #4 here: `build_freq` predates `sm120_gemm_ex` and the e64 builds. The freq column shows #2 alone vs today's
  stock.

## Assessment and proposal

1. **Adopt #4 for both families** (criterion met). It lowers absolute GEMM time by about 0.6 % for FlipQuant and for
   FourOverSix alike. Adopting it for one family only would bias the comparison, so the proposal is both or neither.
2. **Extend #4 to the 256x64 path** for the same reason. This is small: a g32 e64 build at width 128 plus schedule rows
   for `mixed256`, with the same gates. Without it, 256x64 falls 0.6 points further behind the tuned stock.
3. **What this says about continuing.** The tile table (4b), the epilogue tile and the scheduler order are all
   family-neutral: they speed up stock as much as FlipQuant, so none of them closes the gap.
   - C2′ (`results/kernel_opt/opt2/REPORT.md`, 4096³, isolated) splits FlipQuant's gap to stock (#2's freq build, real
     map: +3.3 %) into three parts:
     - the no-dispatch build is +1.2 % over stock;
     - the dispatch's fixed cost after #2 is +1.1 %;
     - the real map's E0M3 tiles add +0.9 %.
   - B′ targets only the middle part, so it cannot reach parity on its own. The gap at T ≥ 128 after #4 is +1.9 to
     +5.1 %.
   - Larger moves would need non-bitwise changes, such as K-tile reordering or map-structure constraints, which need
     the user's approval.
4. **The cumulative e2e measurement can be proposed now.** It needs:
   - the user's decisions on the 4b table, #4 and B′;
   - one combined build from the #4 sources with `MIXFP4_DISPATCH_FREQ=1` for the mixed families (n16k64_wA*, the
     e64 build, the g32 builds);
   - the schedule rows re-tuned at the 4b widths, if 4b is adopted. They were tuned at today's widths, and 4b changes
     some of them.

## Adoption (2026-10-01)

The user adopted #4 for both families. It is deployed in the adopted sets `mixed_ko` (with t0) and `stock_ko`, with
the scheduler rows re-tuned at the 4b widths (amendment 7, `sm120/configs/<gpu>.ko.json`).

#4 on the 256x64 path was cancelled by the user's later decision: no 256x64 work for now. n16k64_wA_g32_e64 stays
built and gated, but no set uses it.
