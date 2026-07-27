
### llama-2-7b — W4A4   (baseline: nvfp4_4over6, sorted by wikitext)

| config                                 | HW |  wikitext | dwikitext |        c4 |       dc4 |
|----------------------------------------|----|-----------|-----------|-----------|-----------|
| fp16                                   |    |    5.4738 |   -0.2418 |    6.9749 |   -0.2789 |
| razer                                  |    |    5.6715 |   -0.0441 |    7.2045 |   -0.0493 |
| mix_4_6_1x16                           | -  |    5.7037 |   -0.0119 |    7.2288 |   -0.0249 |
| mix_4_6_e2m1_8x64                      | y  |    5.7152 |   -0.0004 |    7.2535 |   -0.0002 |
| nvfp4_4over6                           |    |    5.7156 |   +0.0000 |    7.2537 |   +0.0000 |
| mix_4_6_h3_8x64                        | y  |    5.7157 |   +0.0001 |    7.2536 |   -0.0001 |
| nvif4                                  |    |    5.7158 |   +0.0002 |    7.2412 |   -0.0125 |
| mix_4_6_clipbothx_clipmin0.3_h3_8x64   | y  |    5.7167 |   +0.0012 |    7.2462 |   -0.0076 |
| mix_4_6_clipbothx_clipmin0.3_e2m1_8x64 | y  |    5.7177 |   +0.0021 |    7.2526 |   -0.0011 |
| mix_4_6_clipdense9_amin0.1_e2m1_8x64   | y  |    5.7204 |   +0.0048 |    7.2373 |   -0.0165 |
| mix_4_6_clipbothx_clipmin0.3_h1.5_8x64 | y  |    5.7260 |   +0.0104 |    7.2483 |   -0.0055 |
| mix_4_6_clipdense9_e2m1_8x64           | y  |    5.7261 |   +0.0105 |    7.2354 |   -0.0184 |
| mix_4_6_clipdense9_h3_8x64             | y  |    5.7265 |   +0.0109 |    7.2406 |   -0.0131 |
| mix_4_6_clipdense9_h1.5_8x64           | y  |    5.7265 |   +0.0109 |    7.2431 |   -0.0107 |
| mix_4_6_h1.5_8x64                      | y  |    5.7282 |   +0.0126 |    7.2589 |   +0.0052 |
| mix_4_6_clipdense9_amin0.1_h1.5_8x64   | y  |    5.7311 |   +0.0155 |    7.2427 |   -0.0110 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## W4A4, Llama-2-7B: E0M3's value is a function of how coarse the alpha search is

No rotation. Mean of wikitext and c4 against `nvfp4_4over6`.

| alpha candidate set | E0M3 off | E0M3 on (`h3`) | E0M3 contributes |
|---|---|---|---|
| **dense, 9 points across [1, 1.5]** | **-0.0040** | -0.0011 | **+0.0029 (hurts)** |
| clipping (`bothx` + `clipmin0.3`) | +0.0005 | -0.0032 | -0.0037 (helps) |
| plain FourOverSix `{1, 1.5}` | -0.0003 | 0.0000 | +0.0003 (nothing) |

The same inversion Llama-3.2-3B showed, on a second model. With a coarse alpha set the E0M3
election earns its keep; with a dense one it is a small net loss. **E0M3 was compensating for a scale
search that was too coarse**, and it stops paying once the search is done properly.

Cross-model, W4A4, E0M3's marginal contribution under the dense alpha set:

| | Llama-3.2-3B | Llama-2-7B |
|---|---|---|
| E0M3 off | -0.0449 | -0.0040 |
| E0M3 on | -0.0437 | -0.0011 |
| contribution | **+0.0012** | **+0.0029** |

Best realizable here is `clipdense9_amin0.1_e2m1` at -0.0059 — no E0M3, no rotation, no type block —
which is 32% of this model's `1x16` bound (-0.0184). Note how small that bound is: Llama-2-7B simply
has little to gain at W4A4, against -0.1175 on Llama-3.2-3B.

**The alpha gate flips sign between models.** It helps here (-0.0059 gated against -0.0040 ungated)
and hurts on Llama-3.2-3B (-0.0395 against -0.0449). Same knob, opposite sign, so it is not a safe
default at W4A4 the way it was on weights.
