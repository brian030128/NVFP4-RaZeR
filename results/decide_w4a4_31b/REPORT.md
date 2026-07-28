
### llama-3.1-8b — W4A4   (baseline: nvfp4_4over6, sorted by wikitext)

| config                                 | HW |  wikitext | dwikitext |        c4 |       dc4 |
|----------------------------------------|----|-----------|-----------|-----------|-----------|
| fp16                                   |    |    6.2398 |   -0.6375 |    8.9580 |   -0.8597 |
| razer                                  |    |    6.7604 |   -0.1169 |    9.6720 |   -0.1457 |
| mix_4_6_1x16                           | -  |    6.7963 |   -0.0810 |    9.7171 |   -0.1006 |
| nvif4                                  |    |    6.8084 |   -0.0689 |    9.7309 |   -0.0868 |
| mix_4_6_clipdense9_amin0.1_h1.5_8x64   | y  |    6.8412 |   -0.0361 |    9.7790 |   -0.0387 |
| mix_4_6_clipdense9_amin0.1_e2m1_8x64   | y  |    6.8430 |   -0.0343 |    9.7782 |   -0.0395 |
| mix_4_6_clipdense9_e2m1_8x64           | y  |    6.8432 |   -0.0341 |    9.7826 |   -0.0351 |
| mix_4_6_clipbothx_clipmin0.3_h1.5_8x64 | y  |    6.8460 |   -0.0313 |    9.7948 |   -0.0229 |
| mix_4_6_clipdense9_h3_8x64             | y  |    6.8505 |   -0.0267 |    9.7738 |   -0.0439 |
| mix_4_6_clipdense9_h1.5_8x64           | y  |    6.8538 |   -0.0235 |    9.7746 |   -0.0431 |
| mix_4_6_clipbothx_clipmin0.3_e2m1_8x64 | y  |    6.8624 |   -0.0149 |    9.7910 |   -0.0267 |
| mix_4_6_h1.5_8x64                      | y  |    6.8651 |   -0.0122 |    9.8137 |   -0.0040 |
| mix_4_6_clipbothx_clipmin0.3_h3_8x64   | y  |    6.8665 |   -0.0107 |    9.7944 |   -0.0233 |
| nvfp4_4over6                           |    |    6.8773 |   +0.0000 |    9.8177 |   +0.0000 |
| mix_4_6_h3_8x64                        | y  |    6.8780 |   +0.0007 |    9.8149 |   -0.0029 |
| mix_4_6_e2m1_8x64                      | y  |    6.8792 |   +0.0019 |    9.8248 |   +0.0071 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## W4A4, Llama-3.1-8B: E0M3's value decays to zero as the alpha search improves

No rotation. Mean of wikitext and c4 against `nvfp4_4over6`. This is the third model to show the
same pattern, and it has the most W4A4 headroom of the three (`1x16` bound -0.0908).

**E0M3's marginal contribution, by how rich the alpha candidate set is:**

| alpha set | E0M3 off | E0M3 on | E0M3 adds |
|---|---|---|---|
| plain FourOverSix `{1, 1.5}` | +0.0045 | -0.0081 (`h1.5`) | **-0.0126** |
| clipping (`bothx` + `clipmin0.3`) | -0.0208 | -0.0271 (`h1.5`) | **-0.0063** |
| dense, 9 points across [1, 1.5] | -0.0346 | -0.0333 (`h1.5`) / -0.0353 (`h3`) | **+0.0013 / -0.0007** |
| dense + `amin0.1` | -0.0369 | -0.0374 (`h1.5`) | **-0.0005** |

Monotone: -0.0126, then -0.0063, then ~0. Across all three models at W4A4, under the dense set,
E0M3 contributes -0.0005 (Llama-3.1-8B), +0.0012 (Llama-3.2-3B), +0.0029 (Llama-2-7B) — nothing,
within noise, and negative on two of three.

**Best realizable: `clipdense9_amin0.1_h1.5` at -0.0374**, i.e. 41% of the `1x16` bound. Its
E0M3-free twin `clipdense9_amin0.1_e2m1` is -0.0369, a 0.0005 difference. So the deployable
recommendation does not need the E0M3 operand at all.

The alpha gate helps here (-0.0369 gated against -0.0346 ungated) and on Llama-2-7B, but hurts on
Llama-3.2-3B. It is worth about -0.002 where it works and costs about +0.005 where it does not, so
it stays optional rather than default.

## The W4A4 answer

| config | 3.1-8B | 3.2-3B | 2-7B | needs E0M3? | needs rotation? | needs type block? |
|---|---|---|---|---|---|---|
| **`clipdense9_e2m1`** | -0.0346 | **-0.0449** | -0.0040 | no | no | no |
| `clipdense9_amin0.1_e2m1` | **-0.0369** | -0.0395 | **-0.0059** | no | no | no |
| `clipbothx_clipmin0.3_h1.5` | -0.0271 | -0.0105 | +0.0025 | yes | no | yes |
| `h1.5` (4over6 set + E0M3) | -0.0081 | ~+0.006 | +0.0089 | yes | no | yes |
| `nvfp4_4over6` | 0 | 0 | 0 | — | — | — |

A dense per-scale-block scale search on plain E2M1 beats every configuration that uses the E0M3
type block, on all three models, at W4A4.
