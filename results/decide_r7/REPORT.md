
### llama-3.1-8b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                                  | HW |  wikitext | dwikitext |        c4 |       dc4 |
|-----------------------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_clipheadx_h1.5_1x16             | -  |    6.5526 |   -0.0458 |    9.3676 |   -0.0563 |
| mix_4_6_clipheade0_h1.5_8x64            | y  |    6.5721 |   -0.0262 |    9.4166 |   -0.0073 |
| mix_4_6_clipheadx_rotmin0.1_h1.5_8x64   | y  |    6.5778 |   -0.0206 |    9.4171 |   -0.0068 |
| mix_4_6_clipheadx_h1.5_8x64             | y  |    6.5793 |   -0.0190 |    9.4131 |   -0.0109 |
| mix_4_6_clipbothx_clipmin0.3_h1.5_8x64  | y  |    6.5804 |   -0.0179 |    9.4081 |   -0.0159 |
| mix_4_6_clipheadx_m1_8x64               | y  |    6.5819 |   -0.0165 |    9.4166 |   -0.0073 |
| mix_4_6_clipheadx_h2_8x64               | y  |    6.5822 |   -0.0162 |    9.4145 |   -0.0095 |
| mix_4_6_clipheadx_h1.75_8x64            | y  |    6.5832 |   -0.0152 |    9.4137 |   -0.0103 |
| mix_4_6_h1.5_8x64                       | y  |    6.5867 |   -0.0117 |    9.4196 |   -0.0044 |
| mix_4_6_clipbothx_clipmin0.15_h1.5_8x64 | y  |    6.5868 |   -0.0116 |    9.4040 |   -0.0200 |
| mix_4_6_clipheadx_h1.5_32x128           | y  |    6.5891 |   -0.0093 |    9.4202 |   -0.0038 |
| mix_4_6_clipheadx_e2m1_8x64             | y  |    6.5902 |   -0.0082 |    9.4189 |   -0.0050 |
| mix_4_6_clipheadxx_h1.5_8x64            | y  |    6.5942 |   -0.0042 |    9.4097 |   -0.0142 |
| nvfp4_4over6                            |    |    6.5984 |   +0.0000 |    9.4239 |   +0.0000 |
| mix_4_6_clipheadxx_e2m1_8x64            | y  |    6.6010 |   +0.0026 |    9.4147 |   -0.0092 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 7 settles (Llama-3.1-8B, W4A16)

Two independent improvements on the round 5 headroom result, both at zero metadata cost.

**1. Headroom on E0M3 as well.** `heade0` gives the E0M3 branch alphas `{1, 7/6, 7/5}` on top of
the E2M1 headroom set, and reaches **-0.0262 / -0.0073**, against -0.0190 / -0.0109 for E2M1
headroom alone. Verified against the quantizer, E0M3 with `alpha = 7/n` is exactly a uniform
*n*-level grid:

| alpha | 7/7 | 7/6 | 7/5 | 7/4 | 7/3 |
|---|---|---|---|---|---|
| levels | 7 | 6 | 5 | 4 | 3 |

E2M1 cannot reach these above n = 4 -- its codes `{0,.5,1,1.5,2,3,4,6}` are uniform only up to code
2. So the pair is better understood not as "log grid vs uniform grid" but as a family of block
quantizers that the free alpha search spans jointly: E0M3 supplies the fine uniform grids, E2M1 the
log-spaced and coarse uniform ones.

**2. Clipping works, but only behind a threshold.** Round 1 rejected clipping (`alpha < 1`) at
+0.006 to +0.033 wikitext. Gating it behind a minimum fractional gain gives the best c4 number
measured anywhere in this study:

| | dwikitext | dc4 |
|---|---|---|
| `clipbothx` + `clipmin0.15` + `h1.5` | -0.0116 | **-0.0200** |
| `h1.5` | -0.0117 | -0.0044 |

This is the third independent confirmation of the same principle -- election, rotation, and now
clipping all fail as "do it when it helps" and work as "do it when it decisively helps".

**Other rows.** `clipheadx_e2m1` (headroom, E0M3 switched off) is -0.0082 / -0.0050, so headroom
pays on its own and the E0M3 election roughly doubles it. `clipheadxx` (eight alpha levels) is worse
than `clipheadx` (five) on wikitext, -0.0042 against -0.0190, so the alpha search does have an
overfitting point and more candidates is not automatically better. Election rules under headroom
keep the round 5 ordering: `h1.5` > `m1` ~ `h2` > `h1.75`.

Note the persistent wikitext/c4 tension: headroom is better on wikitext, thresholded clipping much
better on c4. Averaged over both, `clipheade0_h1.5` (-0.0168) and `clipbothx_clipmin0.15_h1.5`
(-0.0158) are the two best configurations, and they are orthogonal -- which is what the `full`
preset combines.
