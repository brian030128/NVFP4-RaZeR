
### llama-3.1-8b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                         | HW |  wikitext | dwikitext |        c4 |       dc4 |
|--------------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_clipdense9e0_h1.5_1x16 | -  |    6.5509 |   -0.0475 |    9.3492 |   -0.0748 |
| mix_4_6_clipheade0_h1.5_8x64   | y  |    6.5719 |   -0.0265 |    9.4158 |   -0.0081 |
| mix_4_6_clipdense9_h1.5_8x64   | y  |    6.5824 |   -0.0160 |    9.3994 |   -0.0245 |
| mix_4_6_clipdense9_h3_8x64     | y  |    6.5826 |   -0.0157 |    9.4013 |   -0.0226 |
| mix_4_6_clipdense9e0_h1.5_8x64 | y  |    6.5834 |   -0.0150 |    9.3987 |   -0.0253 |
| mix_4_6_clipdense9e0_h3_8x64   | y  |    6.5834 |   -0.0150 |    9.4003 |   -0.0237 |
| mix_4_6_clipdense9_e2m1_8x64   | y  |    6.5837 |   -0.0146 |    9.4022 |   -0.0218 |
| mix_4_6_clipdense5_h1.5_8x64   | y  |    6.5857 |   -0.0126 |    9.4041 |   -0.0199 |
| mix_4_6_clipdense5_e2m1_8x64   | y  |    6.5892 |   -0.0091 |    9.4053 |   -0.0187 |
| mix_4_6_clipheadx_e2m1_8x64    | y  |    6.5902 |   -0.0082 |    9.4189 |   -0.0050 |
| nvfp4_4over6                   |    |    6.5984 |   +0.0000 |    9.4239 |   +0.0000 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 15 settles: a DENSE alpha grid dominates, and it needs no E0M3

The alpha-choice statistics (recorded in `DECIDE_SUMMARY.md`) said the coarse `headx` set wastes its
candidates: `alpha = 2` and `3` are never selected on real weights, and the whole gain over
FourOverSix comes from the single extra point at `1.25`. Subdividing `[1, 1.5]` instead is worth far
more. Ranked by mean delta over both datasets:

| config @ 8x64 | dwikitext | dc4 | mean |
|---|---|---|---|
| `clipdense9_h1.5` | -0.0160 | -0.0245 | **-0.0203** |
| `clipdense9e0_h1.5` | -0.0150 | -0.0253 | -0.0202 |
| `clipdense9e0_h3` | -0.0150 | -0.0237 | -0.0194 |
| **`clipdense9_h3`** | -0.0157 | -0.0226 | **-0.0192** |
| **`clipdense9_e2m1`** (no E0M3 at all) | -0.0146 | -0.0218 | **-0.0182** |
| `clipheade0_h1.5` (best of rounds 7/10) | -0.0265 | -0.0081 | -0.0173 |
| `clipdense5_e2m1` | -0.0091 | -0.0187 | -0.0139 |
| `clipheadx_e2m1` | -0.0082 | -0.0050 | -0.0066 |

`dense9` is `alpha in {1, 1.0625, 1.125, ..., 1.5}`, nine points at 6.25% spacing — about the finest
that survives the ue4m3 scale's 3-bit mantissa.

Three things change because of this.

**1. The alpha search now dominates the element-type decision.** `clipdense9_e2m1` switches E0M3
**off entirely** and still reaches -0.0182 mean, better than every configuration from rounds 1-14
including the ones built around the E0M3 type block. The practical answer to "E2M1 or E0M3" on
weights is turning out to be *mostly E2M1, with a properly searched block scale*.

**2. Dense alpha makes the election rule much less critical.** `clipdense9_h3` (-0.0192) is within
0.001 of `clipdense9_h1.5` (-0.0203), where on the coarse set `h3` gave up most of `h1.5`'s gain
(-0.0106 against -0.0174). So the safe, cross-model-validated `kappa^2 = 3` can be used without
giving much up — which removes the main reason the round 8/9 recommendation had to be hedged.

**3. E0M3 headroom is now redundant.** `clipdense9e0_*` matches `clipdense9_*` to within 0.001 on
both datasets. Once E2M1 has a fine scale search it covers what the E0M3 alpha candidates were
adding.

Note this is the *opposite* outcome from `headxx`, which also lowered MSE and raised wikitext. The
difference is where the candidates sit: `headxx` added coarse uniform grids (`alpha` out to 4) that
real blocks never choose, while `dense9` subdivides the interval blocks actually use. More
candidates is not automatically better; more candidates *in the right place* is.

The `1x16` upper bound also improves again, to -0.0475 / -0.0748 (mean -0.0612), the best measured.
So the realizable 8x64 configuration now captures about a third of it.

**Cross-model validation is required before this becomes the recommendation** -- rounds 8 and 9
showed twice that a Llama-3.1-8B result need not transfer. Queued as `decide_r17a` (Llama-2-7B) and
`decide_r17b` (Llama-3.2-3B).
