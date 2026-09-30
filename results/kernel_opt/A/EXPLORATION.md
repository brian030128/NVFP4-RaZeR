# Kernel-opt A: arm-subset dispatch for 16x64 maps — exploration only, not registered

2026-09-30, branch kernel-opt, after A′ (amendment 3). These are disclosed exploratory timings, not results. By the
coordinator's plan ("explore arm-set variants (5+fallback, 8, 16) in a disclosed exploratory timing first, then
register the best one"), the best variant is the existing 16 arms with #2's pattern-0-first dispatch, so no A
amendment is registered. The prototype stays out of kernel-opt. Its full diff, including the SASS-patcher change it
needed, is `exploration/A_armset_prototype.patch` (worktree `wtA2` on 74dfce1).

## Design that was tried

- **Hook** `MIXFP4_ARM_SET=<mask>`, in the collective's pipelined joint-K-A mainloop:
  - only the 4-bit joint patterns in the mask get a specialized k_tile arm;
  - pattern 0 is tested first, then membership (one shift and test), then a balanced tree over the other members;
  - every other pattern runs ONE exact fallback arm. It dispatches each k_block on that k_block's own 2 flags among
    the same per-k_block blobs, so every MMA keeps its format, operands and order.
  - A `bar.warp.sync` in each k_block arm stops ptxas from if-converting them into predicated OMMAs. It worked: 0
    predicated OMMAs in every build.
- **SASS patcher:** its check that all present sites have equal OMMA counts cannot hold for an arm subset. The optional
  `--expect-sites` replaces it with exact equality to the configuration's expected census. `build.py` passes the option
  only for such configurations. The census matched the arithmetic exactly in every build.
- **Pattern shares** (the kernel's own bit layout, FLOP-weighted over the 16x64 maps of the four models):
  - pattern 0 87.7–93.9 %;
  - single-granule patterns 1/2/4/8 5.9–11.5 %;
  - everything else 0.2–0.8 %, spread evenly over the 11 patterns at 0.03–0.13 % each.
  - So arms {0,1,2,4,8} cover 99.2–99.8 % of (warp, k_tile)s. An 8-arm set can add at most about 0.3 %.

| build (build_A2) | arms | census E2M1 / E0M3 | SASS |
|---|---|---|---|
| n16k64_wA_as2 | {0} + fallback | 192 / 128 | 390acf92… |
| n16k64_wA_as6 | {0,1,2,4,8} + fallback | 384 / 192 | 486dc922… |
| n16k64_wA_as8 | {0,1,2,3,4,8,12} + fallback | 448 / 256 | 95442c62… |
| n16k64_wA_as6_n64 / _n32 / _n16 | {0,1,2,4,8} + fallback | 192 / 96, 96 / 48, 48 / 24 | 30f7a9b3…, 49dcb0d6…, ed427394… |

Every variant's output was bitwise equal to n16k64_wA's on every tag set timed below, fallback-forcing maps included.

## Exploratory timings

**Method:** isolated single calls, CUPTI, 3 rotated rounds × 15, WARM weights. Reference: today's builds
(`sm120/build`); freq = #2's builds (`build_freq`). Values are relative to the default build.

**4096³, width 128** (`exploration/quick_A.py`, `quick_A.out`):

| tags | freq (16 arms, 0 first) | {0}+FB | {0,1,2,4,8}+FB | {0,1,2,3,4,8,12}+FB |
|---|---|---|---|---|
| all-E2M1 | −0.19 % | −0.24 % | −0.19 % | −0.27 % |
| real map (Llama layer-0 o_proj) | −0.49 % | +3.87 % | +0.08 % | +0.03 % |
| all-E0M3 (pattern 15: the fallback) | +1.50 % | +8.51 % | +10.21 % | +11.24 % |
| pattern 3 everywhere | −0.54 % | +11.56 % | +13.09 % | −0.30 % (own arm) |
| pattern 5 everywhere | +3.03 % | +11.02 % | +12.08 % | +13.49 % |

**Every width, real shapes** (`exploration/quick_A2.py`, `quick_A2.out`): Llama-3.1-8B layer-0 projections with their
own 16x64 tags, at the `mixed` table's width.

| shape (fallback share) | T = 1 (16) | T = 16 (16) | T = 64 | T = 256 |
|---|---|---|---|---|
| q_proj 4096x4096 (1.4 %) | freq +2.0 %, as6 +8.3 % | freq +2.4 %, as6 +9.5 % | (32) freq +1.7 %, as6 +8.0 % | (64) freq +0.7 %, as6 +8.7 % |
| down_proj 4096x14336 (0.4 %) | freq −5.9 %, as6 −6.0 % | freq −6.3 %, as6 −6.4 % | (32) freq −5.7 %, as6 −5.0 % | (64) freq −2.5 %, as6 −2.6 % |
| gate_proj 14336x4096 (0.3 %) | freq −1.2 %, as6 −1.2 % | freq −1.3 %, as6 −1.0 % | (64) freq −3.7 %, as6 −4.2 % | (128) freq −0.5 %, as6 +1.4 % |

## Reading

- **On all-E2M1 k_tiles every subset equals freq.** Both test pattern 0 first. Having 6 or 8 arm bodies instead of 16
  gains nothing measurable, so code size is not the dispatch's cost.
- **The exact fallback is expensive:** +11–13 % on every k_tile that takes it at width 128. Its per-k_block branch sits
  inside the register pipeline. The CTA's warps meet at a barrier every k_tile, so presumably one warp's fallback
  delays all of them.
  - At the narrow widths a k_block holds only 2–8 MMAs per warp, so even 1.4 % fallback k_tiles (q_proj) cost 8–9.5 %.
  - None of the builds spills (0 stack, 0 local, the same registers as the paper builds).
- **So no subset beats the 16-arm dispatch with pattern 0 first (#2).** It loses badly wherever the fallback fires.
  - "The best one" to register is therefore #2's, which is already registered and measured.
- **The lever is per-k_tile instructions, not arm count.** A′ (amendment 3) halved the flag reads and the tree depth for
  256x64 maps and halved the fixed cost. For 16x64 maps the granule cannot grow, so the remaining lever is to stop
  dispatching per k_tile: run tables, "B", in the coordinator's list.
