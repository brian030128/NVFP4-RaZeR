
### llama-2-7b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                       | HW |  wikitext | dwikitext |        c4 |       dc4 |
|------------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_clipheadx_e2m1_8x64  | y  |    5.6043 |   -0.0044 |    7.1328 |   -0.0050 |
| nvfp4_4over6                 |    |    5.6087 |   +0.0000 |    7.1378 |   +0.0000 |
| mix_4_6_clipdense9_e2m1_8x64 | y  |    5.6181 |   +0.0094 |    7.1269 |   -0.0109 |
| mix_4_6_clipdense9_h2.5_8x64 | y  |    5.6183 |   +0.0096 |    7.1293 |   -0.0086 |
| mix_4_6_clipdense9_h3_8x64   | y  |    5.6186 |   +0.0099 |    7.1293 |   -0.0086 |
| mix_4_6_clipdense5_e2m1_8x64 | y  |    5.6217 |   +0.0130 |    7.1291 |   -0.0088 |
| mix_4_6_clipdense9_h1.5_8x64 | y  |    5.6235 |   +0.0148 |    7.1327 |   -0.0052 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 17a settles: the dense alpha grid does NOT generalize

Round 15 found a nine-point alpha grid across `[1, 1.5]` to be the best weight-side result on
Llama-3.1-8B (-0.0182 mean with E0M3 switched off entirely, -0.0203 with `h1.5`). On Llama-2-7B it
is a wash, and worse than the coarse headroom set it was supposed to replace:

| config @ 8x64 | Llama-3.1-8B | Llama-2-7B |
|---|---|---|
| `clipheadx_e2m1` (coarse headroom, 5 points) | -0.0082 / -0.0050 (mean -0.0066) | **-0.0044 / -0.0050 (mean -0.0047)** |
| `clipdense9_e2m1` (dense, 9 points) | -0.0146 / -0.0218 (mean **-0.0182**) | +0.0094 / -0.0109 (mean -0.0008) |
| `clipdense9_h3` | -0.0157 / -0.0226 (mean -0.0192) | +0.0099 / -0.0086 (mean +0.0007) |
| `clipdense9_h1.5` | -0.0160 / -0.0245 (mean -0.0203) | +0.0148 / -0.0052 (mean +0.0048) |
| `clipdense5_e2m1` | -0.0091 / -0.0187 (mean -0.0139) | +0.0130 / -0.0088 (mean +0.0021) |

The dense grid trades wikitext for c4 on this model: every dense row is *better* than baseline on c4
and *worse* on wikitext. That is the same signature every over-aggressive MSE optimization in this
study has shown — clipping in round 1, `headxx` in round 7 — and it is why the coarse five-point set
survives here while the nine-point set does not.

So the two leading ideas separate cleanly:

| | Llama-3.1-8B | Llama-3.2-3B | Llama-2-7B | generalizes? |
|---|---|---|---|---|
| **`clipbothx_clipmin0.3_h3`** (gated clipping) | **-0.0123** | **-0.0074** | **-0.0032** | **yes** |
| `clipdense9_*` (dense alpha) | -0.0182 … -0.0203 | (round 17b) | -0.0008 … +0.0048 | no |
| `clipheadx_e2m1` (coarse headroom) | -0.0066 | +0.0003 | -0.0047 | neutral, never harmful |

Gated clipping is the recommendation. The dense alpha grid is a larger win where it works and should
be treated as a Llama-3.1-8B-specific result, not a default.
