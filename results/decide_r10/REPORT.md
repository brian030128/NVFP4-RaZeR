
### llama-3.1-8b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                                 | HW |  wikitext | dwikitext |        c4 |       dc4 |
|----------------------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_clipheade0_h1.5_1x16           | -  |    6.5516 |   -0.0468 |    9.3616 |   -0.0623 |
| mix_4_6_clipheade0_h1.5_8x64           | y  |    6.5719 |   -0.0265 |    9.4158 |   -0.0081 |
| mix_4_6_clipheade0x_h1.5_8x64          | y  |    6.5720 |   -0.0263 |    9.4159 |   -0.0081 |
| mix_4_6_clipheade0_rotmin0.1_h1.5_8x64 | y  |    6.5728 |   -0.0255 |    9.4186 |   -0.0053 |
| mix_4_6_clipheade0_m1_8x64             | y  |    6.5757 |   -0.0227 |    9.4192 |   -0.0048 |
| mix_4_6_clipheade0_h2_8x64             | y  |    6.5795 |   -0.0188 |    9.4147 |   -0.0092 |
| mix_4_6_clipheade0_h3_8x64             | y  |    6.5862 |   -0.0122 |    9.4150 |   -0.0089 |
| mix_4_6_clipheade0_e2m1_8x64           | y  |    6.5902 |   -0.0082 |    9.4189 |   -0.0050 |
| mix_4_6_clipheade0_h1.5_32x128         | y  |    6.5913 |   -0.0070 |    9.4186 |   -0.0053 |
| nvfp4_4over6                           |    |    6.5984 |   +0.0000 |    9.4239 |   +0.0000 |
| mix_4_6_clipheade0_e0m3_8x64           | y  |    6.6384 |   +0.0400 |    9.4910 |   +0.0671 |
| mix_4_6_clipheade0x_e0m3_8x64          | y  |    6.6387 |   +0.0403 |    9.4903 |   +0.0663 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 10 settles (Llama-3.1-8B, W4A16): the headroom family, fully controlled

`clipheade0` -- headroom on both grids, E2M1 `{1, 1.25, 1.5, 2, 3}` and E0M3 `{1, 7/6, 7/5}` -- at
8x64 with `h1.5` reaches **-0.0265 / -0.0081**, and the controls say where that comes from.

| | dwikitext | dc4 | reading |
|---|---|---|---|
| `_e0m3` (always E0M3) | +0.0400 | +0.0671 | the type block is not collapsing to INT4 |
| `_e2m1` (never E0M3) | -0.0082 | -0.0050 | headroom alone is worth about a third |
| `_h3` | -0.0122 | -0.0089 | |
| `_h2` | -0.0188 | -0.0092 | |
| `_m1` | -0.0227 | -0.0048 | |
| **`_h1.5`** | **-0.0265** | **-0.0081** | both halves together |

So neither half is close to sufficient. Always-E0M3 is a large loss, never-E0M3 gives a third of the
gain, and the robust election on top of the widened alpha family gives all of it. That is the
clearest statement of the answer to "how do you decide E2M1 or E0M3": **widen what each block can
do with its scale, then let the tile elect between the two grids under a robust rule.**

Two null results worth recording:

- `heade0x` (E0M3 headroom carried out to 3 levels, alphas `{1, 7/6, 7/5, 7/4, 7/3}`) is identical
  to `heade0` (-0.0263 vs -0.0265). The extra coarse uniform grids are never chosen -- E2M1 with
  headroom already covers 4 levels and below. Five candidates per grid is where this saturates.
- Adding threshold rotation on top (`_rotmin0.1`) is -0.0255 / -0.0053, i.e. very slightly worse
  than `clipheade0_h1.5` alone. Once the alpha family is wide enough, rotation has nothing left to
  contribute -- consistent with round 4, where its own contribution was already ~0.

Type block ordering is unchanged and now stark: 8x64 (-0.0265) vs 32x128 (-0.0070). And the 1x16
upper bound also improves under headroom, from -0.0450/-0.0580 to **-0.0468/-0.0623**, so the
realizable 8x64 configuration now captures about a third of a *better* ceiling.
