
### llama-3.1-8b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                      | HW |  wikitext | dwikitext |        c4 |       dc4 |
|-----------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_clipheadx_h1.5_8x64 | y  |    6.5798 |   -0.0186 |    9.4141 |   -0.0098 |
| mix_4_6_h1.75_8x64          | y  |    6.5866 |   -0.0117 |    9.4143 |   -0.0097 |
| mix_4_6_h1.6_8x64           | y  |    6.5868 |   -0.0116 |    9.4180 |   -0.0059 |
| mix_4_6_cliphead_h1.5_8x64  | y  |    6.5872 |   -0.0112 |    9.4186 |   -0.0053 |
| mix_4_6_h1.4_8x64           | y  |    6.5895 |   -0.0089 |    9.4233 |   -0.0006 |
| mix_4_6_h1.3_8x64           | y  |    6.5903 |   -0.0080 |    9.4265 |   +0.0026 |
| mix_4_6_m0.75_8x64          | y  |    6.5914 |   -0.0070 |    9.4229 |   -0.0011 |
| mix_4_6_m1.25_8x64          | y  |    6.5920 |   -0.0064 |    9.4149 |   -0.0090 |
| mix_4_6_h1.2_8x64           | y  |    6.5929 |   -0.0055 |    9.4304 |   +0.0065 |
| mix_4_6_m0.5_8x64           | y  |    6.5943 |   -0.0040 |    9.4330 |   +0.0091 |
| mix_4_6_h1.5_16x64          | y  |    6.5948 |   -0.0036 |    9.4244 |   +0.0005 |
| mix_4_6_h1.1_8x64           | y  |    6.5973 |   -0.0010 |    9.4358 |   +0.0118 |
| nvfp4_4over6                |    |    6.5984 |   +0.0000 |    9.4239 |   +0.0000 |
| mix_4_6_cliphead_e2m1_8x64  | y  |    6.5984 |   +0.0000 |    9.4221 |   -0.0018 |
| mix_4_6_h1.5_32x128         | y  |    6.5999 |   +0.0015 |    9.4238 |   -0.0001 |
| mix_4_6_h1.5_32x64          | y  |    6.6003 |   +0.0019 |    9.4246 |   +0.0007 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 5 settles (Llama-3.1-8B, W4A16, 16 configs)

**Headroom is the biggest single win in this study, and it is the OPPOSITE of clipping.**

The block scale is `alpha * block_max / grid_max`. Round 1 tested `alpha < 1` (clipping, which
saturates the block maximum) and rejected it at +0.006 to +0.033 wikitext. Round 5 tests `alpha > 1`
(headroom, which wastes the top codes and puts the bulk on a more uniform part of the E2M1 grid).
FourOverSix is exactly the `alpha = 1.5` member -- block max to code 4 instead of code 6 -- and
there is nothing special about 4:

| E2M1 alpha candidates | block max maps to code | dwikitext | dc4 |
|---|---|---|---|
| `{1, 1.5}` = 4over6 (`base`) | 6, 4 | -0.0117 | -0.0044 |
| `{1, 1.5, 2}` (`head`) | 6, 4, 3 | -0.0112 | -0.0053 |
| **`{1, 1.25, 1.5, 2, 3}` (`headx`)** | 6, 4.8, 4, 3, 2 | **-0.0186** | **-0.0098** |

all with the same `h1.5` election. The finer steps are what pays: three levels is no better than
two, five is much better. Like the 4/6 choice it costs no metadata, because `alpha` only changes the
value written into the ue4m3 scale field that already exists.

**But headroom needs the E0M3 election to pay off.** `cliphead_e2m1` -- headroom with the E0M3
branch switched off -- is +0.0000 / -0.0018, i.e. nothing. The two are synergistic rather than
independent, which makes sense: a wider alpha search lets each scale block sit better on whichever
grid its tile elected, so it is worth more when the tile is genuinely choosing.

**The robust rule's kappa^2 has a broad optimum in [1.5, 2].**

| kappa^2 | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 | 1.6 | 1.75 | 2 (round 2) | 3 (round 2) |
|---|---|---|---|---|---|---|---|---|---|
| dwikitext | -0.0010 | -0.0055 | -0.0080 | -0.0089 | -0.0117 | -0.0116 | -0.0117 | -0.0063 | -0.0014 |
| dc4 | +0.0118 | +0.0065 | +0.0026 | -0.0006 | -0.0044 | -0.0059 | **-0.0097** | -0.0066 | -0.0025 |

Flat from 1.5 to 1.75 on wikitext, with c4 still improving out to 1.75. The margin rule `m<z>` is
consistently a little worse than the robust rule at every setting tried.

The type-block ordering is unchanged: 8x64 (-0.0117) > 16x64 (-0.0036) > 32x128 (+0.0015). The
smallest hardware-realizable tile is always the right one.
