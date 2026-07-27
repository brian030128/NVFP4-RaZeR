
### llama-3.1-8b — W4A4   (baseline: nvfp4_4over6, sorted by wikitext)

| config                                 | HW |  wikitext | dwikitext |        c4 |       dc4 |
|----------------------------------------|----|-----------|-----------|-----------|-----------|
| fp16                                   |    |    6.2398 |   -0.6375 |    8.9580 |   -0.8597 |
| razer                                  |    |    6.7604 |   -0.1169 |    9.6720 |   -0.1457 |
| mix_4_6_1x16                           | -  |    6.7963 |   -0.0810 |    9.7171 |   -0.1006 |
| nvif4                                  |    |    6.8084 |   -0.0689 |    9.7309 |   -0.0868 |
| mix_4_6_clipheade0_h1.5_8x64           | y  |    6.8457 |   -0.0316 |    9.8145 |   -0.0032 |
| mix_4_6_clipbothx_clipmin0.3_h1.5_8x64 | y  |    6.8476 |   -0.0297 |    9.7982 |   -0.0195 |
| mix_4_6_clipheade0_h3_8x64             | y  |    6.8655 |   -0.0118 |    9.8041 |   -0.0136 |
| mix_4_6_clipheade0_e2m1_8x64           | y  |    6.8678 |   -0.0095 |    9.8067 |   -0.0110 |
| mix_4_6_h1.5_8x64                      | y  |    6.8694 |   -0.0079 |    9.8092 |   -0.0085 |
| nvfp4_4over6                           |    |    6.8773 |   +0.0000 |    9.8177 |   +0.0000 |
| mix_4_6_h3_8x64                        | y  |    6.8780 |   +0.0007 |    9.8149 |   -0.0029 |
| mix_4_6_e2m1_8x64                      | y  |    6.8792 |   +0.0019 |    9.8248 |   +0.0071 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 12 settles (Llama-3.1-8B, W4A4): the A operand is where this format pays

**The prize is roughly twice as large on activations.** The unrealizable `1x16` upper bound is
-0.0810 / -0.1006 here, against -0.0468 / -0.0623 at W4A16, and `nvif4` is -0.0689 / -0.0868. The
E0M3 element type is worth more on the A operand than on the B operand -- which is the opposite of
where the MixFP4 study started, since the type block was introduced for weights.

**The alpha search matters more than the election, and it matters twice over.**

| | dwikitext | dc4 | mean |
|---|---|---|---|
| `clipheade0_h1.5` (headroom + election) | **-0.0316** | -0.0032 | -0.0174 |
| `clipbothx_clipmin0.3_h1.5` (gated clipping + election) | -0.0297 | **-0.0195** | **-0.0246** |
| `clipheade0_h3` | -0.0118 | -0.0136 | -0.0127 |
| `clipheade0_e2m1` (headroom, **no E0M3**) | -0.0095 | -0.0110 | -0.0103 |
| `h1.5` (election alone) | -0.0079 | -0.0085 | -0.0082 |
| `h3` | +0.0007 | -0.0029 | -0.0011 |
| E2M1 only | +0.0019 | +0.0071 | +0.0045 |

The election alone is -0.0082; the widened alpha search alone is -0.0103; the two together are
-0.0174 to -0.0246. Both halves are needed, and each is worth about as much as the other -- the same
structure as W4A16, at roughly double the magnitude.

**Gated clipping is the best configuration by mean delta.** `clipbothx_clipmin0.3` wins c4 by a wide
margin (-0.0195 against -0.0032 for headroom) and is close on wikitext. This is the clearest
statement of the operand dependence: a 16-element activation block routinely contains a genuine
outlier that the other fifteen elements are paying for, so saturating it is cheap -- **provided the
gain is decisive**, since ungated clipping loses in every setting measured.

Caveat on the E2M1-only control: at W4A4 it is +0.0019 / +0.0071, i.e. slightly *worse* than the
reference `nvfp4_4over6`, where at W4A16 it was slightly better. The two differ only in E2M1
rounding-tie convention, and on activations that convention costs a little. Deltas quoted against
`nvfp4_4over6` are therefore mildly conservative for every mix_4_6 row in this table.
