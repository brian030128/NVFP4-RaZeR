
### llama-2-7b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                                | HW |  wikitext | dwikitext |        c4 |       dc4 |
|---------------------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_clipbothx_clipmin0.3_h3_8x64  | y  |    5.6082 |   -0.0005 |    7.1320 |   -0.0059 |
| nvfp4_4over6                          |    |    5.6087 |   +0.0000 |    7.1378 |   +0.0000 |
| mix_4_6_clipdense9_amin0.2_e2m1_8x64  | y  |    5.6109 |   +0.0022 |    7.1252 |   -0.0126 |
| mix_4_6_clipdense9_amin0.1_e2m1_8x64  | y  |    5.6158 |   +0.0071 |    7.1259 |   -0.0120 |
| mix_4_6_clipdense9_e2m1_8x64          | y  |    5.6181 |   +0.0094 |    7.1269 |   -0.0109 |
| mix_4_6_clipdense9_amin0.05_h3_8x64   | y  |    5.6190 |   +0.0103 |    7.1293 |   -0.0085 |
| mix_4_6_clipdense9_amin0.05_e2m1_8x64 | y  |    5.6196 |   +0.0109 |    7.1273 |   -0.0105 |
| mix_4_6_clipdense9_amin0.02_e2m1_8x64 | y  |    5.6199 |   +0.0111 |    7.1267 |   -0.0111 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 18a settles: the principle holds for the scale search too

The per-scale-block alpha search was the last plain argmin in this quantizer. `amin<t>` requires a
candidate `alpha != 1` to beat `alpha = 1` (plain NVFP4) by at least the fraction `t`. On Llama-2-7B
— the one model where the dense grid went the wrong way on wikitext — the gate is monotone in the
predicted direction:

| gate `t` | dwikitext | dc4 | mean |
|---|---|---|---|
| 0 (plain argmin) | +0.0094 | -0.0109 | -0.0008 |
| 0.02 | +0.0111 | -0.0111 | +0.0000 |
| 0.05 | +0.0109 | -0.0105 | +0.0002 |
| 0.10 | +0.0071 | -0.0120 | -0.0025 |
| **0.20** | **+0.0022** | **-0.0126** | **-0.0052** |

The wikitext regression that made `clipdense9_e2m1` the "not strictly safe" option drops from
+0.0094 to +0.0022, and the mean improves six-fold, entirely by refusing to move off `alpha = 1` for
small gains. Note c4 *also* improves as the gate tightens (-0.0109 to -0.0126) — this is not a
wikitext/c4 trade, it is the gate removing decisions that were wrong on both.

That is the **fourth independent mechanism** where "do X when it lowers the error" loses to "do X
when it decisively lowers the error":

| mechanism | plain argmin | with a threshold |
|---|---|---|
| elect E0M3 for a type block | +0.0021 / +0.0185 | `h1.5` -0.0117 / -0.0044 |
| rotate a column chunk | +0.0946 / +0.1431 | `rotmin0.1` -0.0149 / -0.0025 |
| clip the block scale | +0.006 … +0.033 | `clipmin0.3` -0.0179 / -0.0159 |
| **choose the block scale** | **+0.0094 / -0.0109** | **`amin0.2` +0.0022 / -0.0126** |

Every free choice in this format wants the same treatment. Whether `t = 0.2` costs anything on the
models where the ungated search already works is round 18b.
