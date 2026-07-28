
### llama-3.1-8b — W4A4   (baseline: nvfp4_4over6, sorted by wikitext)

| config                                  | HW |  wikitext | dwikitext |        c4 |       dc4 |
|-----------------------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_clipdense9_m1_8x64              | y  |    6.8353 |   -0.0420 |    9.7795 |   -0.0382 |
| mix_4_6_clipdense9_amin0.1_h1.5_8x64    | y  |    6.8412 |   -0.0361 |    9.7790 |   -0.0387 |
| mix_4_6_clipbothx_clipmin0.2_h1.5_8x64  | y  |    6.8460 |   -0.0313 |    9.7807 |   -0.0370 |
| mix_4_6_clipbothx_clipmin0.3_h1.5_8x64  | y  |    6.8460 |   -0.0313 |    9.7948 |   -0.0229 |
| mix_4_6_clipbothx_clipmin0.15_h1.5_8x64 | y  |    6.8464 |   -0.0309 |    9.7822 |   -0.0355 |
| mix_4_6_clipdense9_h2_8x64              | y  |    6.8513 |   -0.0260 |    9.7867 |   -0.0310 |
| mix_4_6_clipdense9_amin0.05_h1.5_8x64   | y  |    6.8544 |   -0.0229 |    9.7765 |   -0.0412 |
| mix_4_6_clipdense9_amin0.2_h1.5_8x64    | y  |    6.8554 |   -0.0219 |    9.7908 |   -0.0270 |
| mix_4_6_clipbothx_clipmin0.5_h1.5_8x64  | y  |    6.8663 |   -0.0110 |    9.8107 |   -0.0070 |
| nvfp4_4over6                            |    |    6.8773 |   +0.0000 |    9.8177 |   +0.0000 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## W4A4 knob retuning: the margin rule beats the harm rule, and the W4A16 tuning does not carry

Llama-3.1-8B W4A4, mean of wikitext and c4 against `nvfp4_4over6`, no rotation.

| config | mean |
|---|---|
| **`clipdense9_m1`** (margin, z = 1) | **-0.0401** |
| `clipdense9_amin0.1_h1.5` | -0.0374 |
| `clipbothx_clipmin0.2_h1.5` | -0.0342 |
| `clipbothx_clipmin0.15_h1.5` | -0.0332 |
| `clipdense9_amin0.05_h1.5` | -0.0321 |
| `clipdense9_h2` | -0.0285 |
| `clipbothx_clipmin0.3_h1.5` | -0.0271 |
| `clipdense9_amin0.2_h1.5` | -0.0245 |
| `clipbothx_clipmin0.5_h1.5` | -0.0090 |

Three corrections to conclusions drawn from W4A16:

1. **The margin rule `m<z>` beats the robust/harm rule `h<lambda>` at W4A4.** Every earlier round
   preferred `h`, and the whole robust-optimization derivation was built around it. At W4A4 `m1`
   (-0.0401) beats `h1.5` (-0.0333) and `h2` (-0.0285) under the same alpha set. The derivation is
   still correct about what `h` optimizes; it is simply not the better estimator here.
2. **The clipping threshold retunes down**: 0.2 (-0.0342) beats 0.3 (-0.0271) beats 0.5 (-0.0090).
   W4A16 preferred 0.3.
3. **The alpha gate has an interior optimum at 0.1** (-0.0374), with 0.05 (-0.0321) and 0.2 (-0.0245)
   both worse -- the same non-monotone shape seen at W4A16, where weak thresholds were the worst of
   both worlds.
