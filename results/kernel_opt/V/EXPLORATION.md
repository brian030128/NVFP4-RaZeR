# Kernel-opt V: the 256x64 path (A′) brought to the 16x64 state — exploration (disclosed, before registration)

2026-10-03, branch `kernel-opt` at 52aa8db (worktree `wtV`: one t0 configuration, n16k64_wA_g32_e64_t0, and two
exploration families, `mixed256_e` / `mixed256_e_t0`). Nothing here is registered. Amendment 18 registers the design
chosen from it, and its fresh 4-model M1 is the test.

- **Maps and tags:** the paper's TM-OPT+TC 256x64 artifacts (stored as 16x64 granules, uniform over 256x64 tiles).
  Typical = the lower-median module per projection; worst = the densest. The FlipQuant calibration will change later.
- **Scripts and outputs:** `exploration/`. The raw records are in `/home/dev/n16k64_campaign/kernel_opt/V/explore`,
  with sha256 in `exploration/raw_sha256.txt`.

**The request**, relayed by the coordinator (the user's go): bring the 256x64 path (`mixed256`, A′'s g32 builds) up to
the current 16x64 state.
1. The uniform dispatch on the g32 builds; check the SASS first; a per-width design is allowed, disclosed.
2. #4: the 64×64 epilogue at width 128; check the stage count; per-call scheduler rows.
3. #2's frequency-aware dispatch: adopt it only if it helps with 4 arms.
4. t0: re-test it only if it is cheap in combination, and adopt it only if the rule is met.
5. An act-warm re-tune of the widths and scheduler rows on the final builds; the decisive-margin rule as a sensitivity
   only.
6. M1 and a C2-style 4096³ split; 7. if adopted, a C3k-style E0M3-fraction sweep.

**Today's path:** `mixed256` = n16k64_wA_g32{,_n64,_n32,_n16} (`build_A1`). It has the default dispatch and the site-0
tags, reads the PAPER table's 'mixed' rows, and has no scheduler rows.

## 1. SASS, CPU only (`exploration/sass_V.json`)

The 4-arm g32 builds still carry reconvergence code in the k_tile loop. Between the first and the last OMMA:

| | BSSY / BSYNC / WARPSYNC | WARPSYNC in the kernel |
|---|---|---|
| today's builds, with or without #2's dispatch or t0 | 3–5 / 5–6 / 11 | 14–15 |
| with `MIXFP4_UNIFORM_DISPATCH=1` | 0 / 0 / 0, 4 REDUX there | 1 |

- **For comparison:** the 16-arm 16x64 builds have 47 WARPSYNC there.
- **The uniform builds** have 16–48 fewer instructions, no predicated OMMA, and the census unchanged.
- **#4 at width 128 (n16k64_wA_g32_e64):** 4 mainloop stages, as today's n16k64_wA_g32. The shared storage goes from
  88,064 to 100,352 B.
- **The default-dispatch g32 builds from `wtV` reproduce `build_A1`'s SASS exactly** (n16, n32, n64).

## 2. Per T at today's widths, M1's method (`exploration/quick_V1.out`)

Llama-3.1-8B and Phi-4, typical and worst modules. Per-forward sums, % against today's path; median over the 12 T.
Every set reads the paper table's rows, so only the builds differ. All 4,752 bitwise checks against today were equal.

| variant (all with #4's e64 tile at width 128) | Llama typical | Llama worst | Phi-4 typical | Phi-4 worst |
|---|---:|---:|---:|---:|
| e64 alone | −0.55 % | −0.65 % | −0.42 % | −0.48 % |
| + #2's dispatch | −0.58 % | −0.50 % | −0.66 % | −0.41 % |
| + uniform index (REDUX) | −0.82 % | −0.86 % | −0.61 % | −0.68 % |
| + #2 + REDUX | −0.84 % | −0.88 % | −0.55 % | −0.59 % |
| + t0 | −0.38 % | −0.46 % | −0.44 % | −0.48 % |
| + #2 + t0 | −0.64 % | −0.63 % | −0.44 % | −0.61 % |
| **+ REDUX + t0** | **−0.86 %** | **−0.93 %** | **−0.86 %** | **−0.74 %** |
| + #2 + REDUX + t0 | −0.84 % | −0.95 % | −0.75 % | −1.01 % |

- **At T ≥ 128 every variant gains.** REDUX + t0 gains −0.6 … −3.4 %, below zero in every round in 27 of 28 cells.
- **At T ≤ 64 (widths 16 and 32) the differences are small.**
  - On Phi-4 every variant is within ±0.4 %, with mixed signs. On Llama the REDUX variants gain −0.2 … −0.8 %.
  - The "e64 alone" builds are identical to today's at those widths, yet show +0.15 … +0.29 % in every round on Phi-4:
    the A/A noise floor.
  - Unlike 16x64, REDUX does not cost the E0M3-heavy modules on the narrow tiles here: with 4 arms the non-zero
    patterns' paths are short.
- **#2's dispatch does not help with 4 arms.** With or without REDUX, its effect has mixed signs within the noise
  (e.g. REDUX vs #2 + REDUX: −0.82 / −0.84 % on Llama, −0.61 / −0.55 % on Phi-4).
- **t0 helps only in combination with REDUX:** a further −0.3 … −0.5 points at T ≥ 128 on both models. Alone it is
  neutral, as amendment 6 found.

**The per-width breakdown** (module-weighted, typical): REDUX + t0 gains at every width on both models.
- Llama: −0.33 / −0.94 / −0.59 / −1.45 % at widths 16 / 32 / 64 / 128.
- Phi-4: −0.37 / −0.64 / −1.17 / −1.22 %.

## 3. Fixed widths, cold (`exploration/quick_V2.out`)

Against today's build at the same width:

| cells | REDUX + t0 + e64, all-E2M1 | real map | all-E0M3 |
|---|---:|---:|---:|
| width 128, 4096-wide shapes (4096³, 4096x4096 T=1024, 14336x4096 T=2048) | −1.7 … −2.7 % | −1.3 … −2.1 % | −0.8 … −1.1 % |
| width 128, 4096x14336 T=512 | −0.2 % | +0.3 % | +0.2 % |
| width 16, T=16 (three shapes) | −0.4 … −0.8 % | −0.3 … 0.0 % | −0.2 … −0.8 % |

- **Without REDUX, t0 is slower on all-E0M3:** +0.3 … +2.9 % at width 128.
- **The ceiling** (n16k64_wA_nodisp_*_t0) and stock are 1.3–3.3 % below today at width 128.

## 4. The design registered as amendment 18

- **The family `mixed256_ko`:** n16k64_wA_g32_n16_t0, _n32_t0, _n64_t0 and n16k64_wA_g32_e64_t0 (new: #4's tile + t0),
  built with `MIXFP4_UNIFORM_DISPATCH=1` and the default 4-arm tree.
  - #2's dispatch is not used.
  - The same build at every width: no width cut, since no width showed a consistent loss.
- **Then the act-warm re-tune** of its widths and scheduler rows (items 2 and 5), M1 against today's path and
  `stock_ko`, and C2.
- **Today's path vs stock_ko** is +4.4 … +8.2 % at T = 128–512 on Llama and +2.1 … +2.7 % on Phi-4. Much of that is
  the paper table's widths: the ceiling on the same widths is +3.6 / +6.0 % at T = 128 / 256 on Llama.
