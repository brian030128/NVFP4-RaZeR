
### llama-3.1-8b — W4A4   (baseline: nvfp4_4over6, sorted by wikitext)

| config                           | HW |  wikitext | dwikitext |        c4 |       dc4 |
|----------------------------------|----|-----------|-----------|-----------|-----------|
| nvfp4_4over6                     |    |    6.8773 |   +0.0000 |    9.8177 |   +0.0000 |
| mix_4_6_e2m1_8x64                | y  |    6.8797 |   +0.0024 |    9.8179 |   +0.0002 |
| mix_4_6_clipdense9_rot_h1.5_8x64 | y  |    6.9062 |   +0.0289 |    9.8898 |   +0.0721 |
| mix_4_6_rot_8x64                 | y  |    6.9188 |   +0.0415 |    9.8898 |   +0.0721 |
| mix_4_6_clipdense9_rot_e2m1_8x64 | y  |    6.9342 |   +0.0569 |    9.9466 |   +0.1289 |
| mix_4_6_rot_h1.5_8x64            | y  |    6.9373 |   +0.0600 |    9.9071 |   +0.0894 |
| mix_4_6_rot_h3_8x64              | y  |    6.9563 |   +0.0791 |    9.9350 |   +0.1172 |
| mix_4_6_rot64_h1.5_8x64          | y  |    6.9789 |   +0.1016 |    9.9424 |   +0.1247 |
| mix_4_6_rot_e2m1_8x64            | y  |    6.9877 |   +0.1104 |   10.0147 |   +0.1970 |
| mix_4_6_rot64_e2m1_8x64          | y  |    7.0467 |   +0.1695 |   10.0330 |   +0.2153 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## W4A4 rotation x E0M3: the chemistry is real and 10x, and still not enough

At W4A4 `rotate="all"` is exact -- both operands rotate every chunk, so the two rotations cancel
inside the GEMM. (`rotmin`/"col" is NOT valid at W4A4: weights and activations would each pick their
own chunk pattern and the rotations would not cancel.)

**E0M3's marginal contribution, with and without rotation** (mean of wikitext and c4 vs
`nvfp4_4over6`, Llama-3.1-8B W4A4):

| setting | E0M3 off | E0M3 on | E0M3 adds |
|---|---|---|---|
| rotate everything, `argmin` | +0.1537 | +0.0568 | **-0.0969** |
| rotate everything, `h1.5` | +0.1537 | +0.0747 | -0.0790 |
| rotate 64-wide, `h1.5` | +0.1924 | +0.1132 | -0.0792 |
| dense alpha + rotate, `h1.5` | +0.0929 | +0.0505 | -0.0424 |
| **no rotation**, 4over6 alphas | +0.0013 | -0.0081 | **-0.0094** |
| no rotation, dense alphas | -0.0346 | -0.0333 | +0.0013 |

**Under rotation E0M3 is worth ten times what it is worth without it** (-0.097 against -0.009), and
this is a larger effect than the 4x measured at W4A16. The mechanism is the one round 3 identified:
rotation Gaussianizes each block, the uniform grid becomes the better fit, and E0M3 election jumps
from a few percent of tiles to most of them.

**And it is still not enough.** Rotation costs about +0.15 on its own and E0M3 gives back about
0.10, for a net loss of ~0.05. The best rotation configuration here is +0.0289/+0.0721; the best
non-rotation configuration is -0.0374. Every rotation row in this table is worse than plain
`nvfp4_4over6`.

So the honest statement is: **rotation makes E0M3 genuinely valuable, but only by first destroying
something worth more than E0M3 can repair.** The two are complementary in exactly the wrong way --
the thing that creates the chemistry is the same thing that costs the perplexity.

Note also `rot64` is worse than `rot` (16-wide) on both operands, as at W4A16: more mixing, more
damage.
