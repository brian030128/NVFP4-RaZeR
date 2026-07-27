
### llama-2-7b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                       | HW |  wikitext | dwikitext |        c4 |       dc4 |
|------------------------------|----|-----------|-----------|-----------|-----------|
| fp16                         |    |    5.4738 |   -0.1349 |    6.9749 |   -0.1630 |
| nvfp4_razer_e3m3             |    |    5.5676 |   -0.0412 |    7.0928 |   -0.0451 |
| mix_4_6_e2m1_8x64            | y  |    5.6070 |   -0.0017 |    7.1367 |   -0.0012 |
| nvfp4_4over6                 |    |    5.6087 |   +0.0000 |    7.1378 |   +0.0000 |
| mix_4_6_h2_8x64              | y  |    5.6121 |   +0.0034 |    7.1395 |   +0.0016 |
| mix_4_6_h1.5_16x64           | y  |    5.6144 |   +0.0057 |    7.1417 |   +0.0039 |
| mix_4_6_1x16                 | -  |    5.6145 |   +0.0058 |    7.1285 |   -0.0093 |
| mix_4_6_rotmin0.15_h1.5_8x64 | y  |    5.6211 |   +0.0124 |    7.1438 |   +0.0060 |
| mix_4_6_h1.5_8x64            | y  |    5.6215 |   +0.0128 |    7.1435 |   +0.0056 |
| mix_4_6_rotmin0.1_h1.5_8x64  | y  |    5.6215 |   +0.0128 |    7.1452 |   +0.0074 |
| mix_4_6_m1_8x64              | y  |    5.6219 |   +0.0132 |    7.1412 |   +0.0033 |
| nvif4                        |    |    5.6222 |   +0.0135 |    7.1378 |   -0.0000 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 6 settles: kappa is model-dependent and CANNOT be chosen from the weights

`h1.5`, the clear winner on Llama-3.1-8B (-0.0117 / -0.0044), is **+0.0128 / +0.0056 on
Llama-2-7B** -- worse than 4over6 and worse than the E2M1-only control. The two models want opposite
ends of the range:

| kappa^2 | Llama-3.1-8B dwikitext | Llama-2-7B dwikitext |
|---|---|---|
| 1.5 | **-0.0117** | +0.0128 |
| 2 | -0.0063 | +0.0034 |
| 3 | -0.0014 | **-0.0023** (round 1) |

This is not noise, it is the same fact round 1 and round 2 already reported from opposite sides:
Llama-2-7B has no E0M3 signal to recover. Its `nvif4` -- per-scale-block choice, the finest possible
and an upper bound on any tile rule -- is itself +0.0135 wikitext, and `mix_4_6_1x16` is +0.0058. If
the best possible election loses on this model, every permissive rule must lose too, and the only
safe rule is one that elects almost nothing.

**And no calibration-free weight statistic separates the two cases.** The obvious candidate is how
much E0M3 wins on the weights at 1x16, as a fraction of the E2M1 error. Measured over the first 28
linear layers of each model, that fraction is:

| | median | mean | max |
|---|---|---|---|
| Llama-2-7B | 0.2050 | 0.1911 | 0.2095 |
| Llama-3.1-8B | 0.1990 | 0.1897 | 0.2063 |

Identical. The weight-side case for E0M3 is the same in both models; what differs is how that error
lands in the layer output, which lives in the activation statistics and is exactly what a
calibration-free method may not look at. So `kappa` cannot be picked per model from the weights, and
any single recommended value has to be one that is *safe* rather than optimal.

On the evidence so far that value is **kappa^2 = 3**: -0.0014 on Llama-3.1-8B and -0.0023 on
Llama-2-7B, positive on both but worth little. `kappa^2 = 1.5` is worth 8x more where it works and
is a real loss where it does not.
