# Kernel-opt 8x64 plan, P2 (with P1): t0 for the weights-on-B family — report

2026-10-02, branch `kernel-opt`. The protocol is amendment 11 of `results/kernel_opt/PROTOCOL.md`: sources at c5bb523,
registered at 022240a. `run_w8p2.sh` ran 15:36–16:10 UTC with no deviation. Every number is in `w8p2_tables.md` and
`w8p2.json`.

## In brief

- **Every gate passed.** The P2 builds compute bitwise what 1b's do, with the default dispatch and with #2's.
- **The t0 adoption rule is met.** The t0 builds are −0.76 % per forward against 1b's (median over 48 cells), and
  −0.55 % against #2's.
  - The gain is at T ≥ 256: −1.3 to −2.2 %.
  - At T ≤ 16 they are +0.06 to +0.35 % slower, within the rule's 0.5 % tolerance.
- **The P1 rule is met.** On the t0 builds, #2's dispatch is faster than the default in all 48 cells with typical tags,
  every round (median −0.53 %). With worst tags it is a wash (+0.03 %).
- **Recommendation:** adopt t0 for the 8x64 family, with #2's dispatch as the 8x64 default (`build_P2freq`).
- **The residual gap of that path to `stock_ko`** (typical tags) is +4.35 % per forward (median), down from +5.69 % for
  1b. By T:

| T | 1b (before) | t0 + #2 (after) | t0 + #2, worst tags | t0 + #2 vs stock_wB |
|---:|---:|---:|---:|---:|
| 1 | +1.06 % | +0.76 % | +1.41 % | −38.9 % |
| 4 | +1.05 % | +0.55 % | +1.28 % | −39.1 % |
| 16 | +1.21 % | +0.82 % | +1.63 % | −37.2 % |
| 32 | +2.56 % | +2.23 % | +3.01 % | −33.7 % |
| 64 | +2.71 % | +2.43 % | +3.13 % | −28.2 % |
| 128 | +3.47 % | +2.84 % | +3.49 % | −24.6 % |
| 256 | +7.48 % | +5.37 % | +6.65 % | −16.4 % |
| 512 | +13.02 % | +9.28 % | +10.98 % | +4.6 % |
| 1024 | +8.71 % | +6.34 % | +7.62 % | +3.7 % |
| 2048 | +9.93 % | +7.12 % | +8.01 % | +7.0 % |
| 4096 | +9.92 % | +6.98 % | +8.13 % | +6.7 % |
| 8192 | +10.51 % | +7.49 % | +8.81 % | +7.5 % |

Medians over the four models of the per-forward GEMM sums' ratios against `stock_ko`, the deployment's stock with the
weights on A. The last column compares against stock_wB, the paper's single 128-wide build.
- **The worst cell** is still Llama-3.1-8B / Mistral-7B at T = 512: +15.6 % → +11.0 % / +11.4 %. P3's re-tune targets it.
- **stock_wB** is far slower than `stock_ko` at small T (+65 % at T = 1), because it has no narrow tiles. Same placement
  costs about 0 at T ≥ 2048 (+0.1 to +0.4 %).

## The gates

| gate | result |
|---|---|
| G0 provenance | 47 files, 74 builds as registered |
| G3 self-tests, five t0 builds × {default, #2} | 10 × PASS patched / FAIL unpatched (`randb`, 3 shapes); each rebuilt library has the registered SASS (`g3_selftest_sass*.json`) |
| Patcher check (6b's, extended to wB) | 53 tagged binaries: the reaching-definitions sites equal the strict parser's; 8 carry a BRX. 32 t0 binaries, the 10 self-tests among them: census = per-site counts = patcher record |
| G1 / G2 | the 45 existing configurations keep their SASS; the 5 new (and `build_P2freq`'s 5) have their census, nothing predicated |
| G3 pytest | 1,013 passed on `build_P2`; 211 passed on `build_P2freq` (211 skipped: builds not in that directory) |
| G4 bitwise vs n8k64_wB (sm120/build) | 28,800 comparisons on `build_P2`, 28,800 on `build_P2freq`, 0 differences |
| G5 logits, 4 models × 5 shapes | set:mixed_wB → set:mixed_wB_t0 equal, default and #2's dispatch |
| M1 / C2w‴ checks | every bitwise check equal |

## M1, all models (per-forward GEMM, median over the 48 (model, T) cells)

| ratio | median | T ≤ 16 | 32–128 | 256–1024 | ≥ 2048 | cells above / below 0 in every round |
|---|---:|---:|---:|---:|---:|---:|
| t0 vs 1b, typical | −0.76 % | +0.15 % | +0.03 % | −1.80 % | −2.18 % | 11 / 29 |
| t0 vs 1b, worst | −0.76 % | +0.35 % | +0.02 % | −1.80 % | −2.13 % | 14 / 32 |
| t0 #2 vs #2, typical | −0.55 % | +0.06 % | −0.15 % | −1.48 % | −1.62 % | 4 / 33 |
| t0 #2 vs #2, worst | −0.45 % | +0.13 % | −0.08 % | −1.30 % | −1.78 % | 12 / 29 |
| P1: #2's vs default on t0, typical | −0.53 % | −0.60 % | −0.39 % | −0.54 % | −0.50 % | 0 / 48 |
| P1: #2's vs default on t0, worst | +0.03 % | +0.15 % | +0.20 % | +0.09 % | −0.23 % | 20 / 18 |
| 1b vs stock_ko, typical | +5.69 % | +1.05 % | +2.79 % | +8.82 % | +10.12 % | 48 / 0 |
| t0 #2 vs stock_ko, typical | +4.35 % | +0.64 % | +2.37 % | +6.31 % | +7.15 % | 48 / 0 |
| t0 #2 vs stock_ko, worst | +5.85 % | +1.34 % | +3.13 % | +7.62 % | +8.36 % | 48 / 0 |

- **The strict form of the t0 rule** (amendment 6's: no cell above zero in every round) is not met. 41 cells over the four
  units are above zero in every round, all at T ≤ 64 and all ≤ +0.68 %; their every-round minimum is ≤ +0.43 %.
  - They concentrate on worst tags and on the default dispatch, where the t0 builds lose 0.1–0.7 % at decode sizes.
  - With #2's dispatch, the variant recommended, 4 typical-tag cells are flagged, ≤ +0.20 %. They are listed in
    `w8p2_tables.md`.
- **Worst tags with #2's dispatch** are slightly slower than with the default at T ≤ 512 (+0.05 to +0.29 %), about
  equal at T = 1024, and faster at T ≥ 2048 (−0.2 %). This matches 16x64.

## C2w‴: the 4096³ split after P2 (real map; b2b / isolated / sustained)

| step | before (1b, this run) | after (t0 + #2) |
|---|---|---|
| total vs stock_ko | +10.2 / +8.7 / +10.3 % | +7.7 / +6.0 / +8.2 % |
| placement and stock_ko's tuning (stock_wB vs stock_ko) | | +0.9 / +0.5 / +1.2 % |
| 1x8 arrangement (tag-free ceiling vs stock_wB) | | +3.5 / +2.9 / +4.3 % |
| dispatch (all-E2M1 vs the tag-free ceiling) | | +2.2 / +2.0 / +0.3 % |
| the real map's E0M3 tiles (real vs all-E2M1) | | +0.8 / +0.6 / +2.1 % |

- **What t0 removed.** On the dispatch kernels, t0 is −1.4 to −3.0 % on all-E2M1 and real tags. On all-E0M3, whose
  sites keep their tags, it is −0.0 to +0.7 %.
  - That is more than the ceilings' −0.7 to −1.3 %, so the tags cost the dispatch kernel more than they cost the
    ceiling.
  - SASS: the width-128 GEMM kernel loses 384 identity PRMTs (4,976 → 4,592 instructions), and the narrower builds
    proportionally fewer.
- **#2's dispatch on the t0 builds** is −0.3 to −1.1 % on the real map. On all-E0M3 it costs +1.8 to +2.3 %, the
  added jump, as on 16x64.

## A finding for P4: the 8x64 path already has #4's epilogue tile

This comes from a disclosed exploration: CPU builds in the worktree `wtP4` into `build_P4x`, with no timing.
- **Width 128 already has it.** The 1x8 arrangement names its own epilogue tile, and it is 64 × 64 at width 128
  (`MIXFP4_EPI_M/N` in `mixed_nvfp4_gemm.cu`).
  - n8k64_wB_t0 built with `MIXFP4_EPI_TILE_M/N=64` has n8k64_wB_t0's exact SASS.
  - Its shared storage is 100,352 bytes, the 64 × 64 tile's; it is 88,064 bytes with the auto tile.
- **'128x64' would lose a stage.** That build uses the auto tile, which is 64 × 32: an explicit 64 × 32 build has its
  SASS. With 64 × 64 it drops from 6 mainloop stages to 5, the condition under which the plan skips it, as at 16x64's
  width 64.
- **stock_wB** uses the auto 64 × 32 tile. stock_wB_e64 keeps its 4 stages.
- **So for the 8x64 path,** P4's epilogue part has nothing to change. Its only remaining parts are:
  - stock_wB_e64, which tunes the same-placement reference alike;
  - the per-call scheduler rows. The 8x64 sets have none today, while `stock_ko` uses non-default rows in 92 of 224
    cells.
  - The plan tunes the scheduler rows once, in P3, on the final builds. The schedule tuner needs a small extension to
    weights-on-B sets; it is drafted in `wtP4`.

## Maps and tags used

- The paper's TM-OPT+TC 8x64 artifacts, `/home/dev/n16k64_campaign/paper/artifacts/<model>_tc_8x64`, with sha256 in
  `registration_11.json`; FourOverSix artifacts for the stock references.
- Typical tags are the lower-median module per projection, worst the densest, as in amendment 10.
- C2w‴'s real map is the Llama-3.1-8B 8x64 map's layer 0 o_proj. G4 also uses synthetic maps (all-E2M1, all-E0M3,
  random 30 %).
- The FlipQuant calibration will change later. The kernels' bitwise results do not depend on the map; the timings
  depend on its E0M3 share.
