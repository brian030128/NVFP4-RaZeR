
### llama-2-7b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                                 | HW |  wikitext | dwikitext |        c4 |       dc4 |
|----------------------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_clipheadx_e2m1_8x64            | y  |    5.6043 |   -0.0044 |    7.1328 |   -0.0050 |
| mix_4_6_clipheade0_e2m1_8x64           | y  |    5.6043 |   -0.0044 |    7.1328 |   -0.0051 |
| mix_4_6_clipheadx_h3_8x64              | y  |    5.6057 |   -0.0030 |    7.1361 |   -0.0018 |
| mix_4_6_h3_8x64                        | y  |    5.6063 |   -0.0024 |    7.1357 |   -0.0022 |
| mix_4_6_e2m1_8x64                      | y  |    5.6071 |   -0.0016 |    7.1377 |   -0.0001 |
| mix_4_6_clipbothx_clipmin0.3_h3_8x64   | y  |    5.6082 |   -0.0005 |    7.1320 |   -0.0059 |
| nvfp4_4over6                           |    |    5.6087 |   +0.0000 |    7.1378 |   +0.0000 |
| mix_4_6_clipheade0_h3_8x64             | y  |    5.6110 |   +0.0023 |    7.1383 |   +0.0004 |
| mix_4_6_clipheade0_h1.5_1x16           | -  |    5.6138 |   +0.0051 |    7.1310 |   -0.0068 |
| mix_4_6_clipbothx_clipmin0.3_h1.5_8x64 | y  |    5.6180 |   +0.0093 |    7.1343 |   -0.0035 |
| mix_4_6_clipheade0x_h1.5_8x64          | y  |    5.6244 |   +0.0157 |    7.1440 |   +0.0061 |
| mix_4_6_clipheade0_h1.5_8x64           | y  |    5.6252 |   +0.0165 |    7.1439 |   +0.0061 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 8 settles: headroom generalizes, the E0M3 election does not

The same configurations, on the model where E0M3 has no signal:

| config (8x64) | Llama-3.1-8B | Llama-2-7B |
|---|---|---|
| `clipheadx_e2m1` -- headroom, **no E0M3 at all** | -0.0082 / -0.0050 | **-0.0044 / -0.0050** |
| `clipheadx_h3` -- headroom + strict election | (not measured) | -0.0030 / -0.0018 |
| `h3` -- 4over6 + strict election | -0.0014 / -0.0025 | -0.0024 / -0.0022 |
| `clipheade0_h3` | -0.0122 / -0.0089 | +0.0023 / +0.0004 |
| `clipheade0_h1.5` -- best on 3.1-8B | **-0.0265 / -0.0081** | **+0.0165 / +0.0061** |
| `h1.5` | -0.0117 / -0.0044 | +0.0128 / +0.0056 |

The champion of every Llama-3.1-8B round is a **loss** here, and by more than it wins there on c4.
Note also `clipheade0_h1.5` at 1x16 -- the finest possible election -- is +0.0051 on wikitext, so
this is not the type block's fault: on Llama-2-7B, electing E0M3 hurts at *every* granularity.

**Headroom is the part that survives.** `clipheadx_e2m1` is negative on both datasets of both
models, and it is the strongest configuration on Llama-2-7B outright. It also needs neither a type
block nor the E0M3 hardware path: it is a pure per-scale-block scale search on E2M1, i.e. plain
NVFP4 with a wider version of the FourOverSix choice, deployable on the existing kernel.

So the recommendation splits in two:

1. **Robust, deployable, no type block required** -- extend the FourOverSix alpha set from `{1, 1.5}`
   to `{1, 1.25, 1.5, 2, 3}`. Worth -0.004 to -0.008 wikitext and about -0.005 c4 on both models, at
   zero metadata cost. This is the result to keep.
2. **The E0M3 type block is worth a further -0.018 wikitext on Llama-3.1-8B and nothing on
   Llama-2-7B**, with kappa^2 = 1.5 where it works and kappa^2 = 3 needed to merely avoid harm where
   it does not. Round 6 established that no calibration-free weight statistic distinguishes the two
   cases, so this half cannot be turned on safely without measuring the model.
