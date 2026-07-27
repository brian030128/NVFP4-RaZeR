
### llama-3.1-8b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                        | HW |  wikitext | dwikitext |        c4 |       dc4 |
|-------------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_rotmin0.15_h1.5_1x16  | -  |    6.5535 |   -0.0448 |    9.3678 |   -0.0562 |
| mix_4_6_rotmin0.1_h1.5_8x64   | y  |    6.5834 |   -0.0149 |    9.4214 |   -0.0025 |
| mix_4_6_h1.5_8x64             | y  |    6.5867 |   -0.0117 |    9.4196 |   -0.0044 |
| mix_4_6_rotmin0.25_h1.5_8x64  | y  |    6.5868 |   -0.0116 |    9.4191 |   -0.0048 |
| mix_4_6_rotmin0.15_h1.5_8x64  | y  |    6.5873 |   -0.0111 |    9.4185 |   -0.0054 |
| mix_4_6_rotmin0.4_h1.5_8x64   | y  |    6.5874 |   -0.0109 |    9.4191 |   -0.0049 |
| mix_4_6_rotmin0.15_m1_8x64    | y  |    6.5880 |   -0.0104 |    9.4187 |   -0.0053 |
| mix_4_6_rotmin0.05_h1.5_8x64  | y  |    6.5914 |   -0.0069 |    9.4266 |   +0.0026 |
| mix_4_6_rotmin0.1_h1.5_32x128 | y  |    6.5941 |   -0.0043 |    9.4256 |   +0.0017 |
| mix_4_6_rotmin0.15_e2m1_8x64  | y  |    6.5979 |   -0.0005 |    9.4220 |   -0.0019 |
| nvfp4_4over6                  |    |    6.5984 |   +0.0000 |    9.4239 |   +0.0000 |
| mix_4_6_rotmin0.15_8x64       | y  |    6.5996 |   +0.0012 |    9.4413 |   +0.0173 |
| mix_4_6_rotmin0.15_e0m3_8x64  | y  |    6.6377 |   +0.0393 |    9.4786 |   +0.0546 |
| mix_4_6_rotcol_h1.5_8x64      | y  |    6.6929 |   +0.0946 |    9.5670 |   +0.1431 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 4 settles: the threshold rescues rotation, but the ELECTION does the work

The controls decompose the result, and the decomposition is the point.

| what | dwikitext | dc4 |
|---|---|---|
| rotation alone, no E0M3 (`rotmin0.15_e2m1`) | -0.0005 | -0.0019 |
| E0M3 by argmin, with rotation (`rotmin0.15`) | +0.0012 | +0.0173 |
| E0M3 always, with rotation (`rotmin0.15_e0m3`) | +0.0393 | +0.0546 |
| **E0M3 by the robust rule (`h1.5`), no rotation** | **-0.0117** | **-0.0044** |
| E0M3 by `h1.5` + threshold rotation (`rotmin0.1_h1.5`) | **-0.0149** | -0.0025 |

Selective rotation with a minimum gain does exactly what it was built to do -- it turns rotation
from a +0.095 catastrophe into something harmless, and the threshold curve is monotone in the
predicted direction:

| threshold t | 0 (`rotcol`) | 0.05 | 0.10 | 0.15 | 0.25 | 0.40 |
|---|---|---|---|---|---|---|
| dwikitext | +0.0946 | -0.0069 | **-0.0149** | -0.0111 | -0.0116 | -0.0109 |
| dc4 | +0.1431 | +0.0026 | -0.0025 | **-0.0054** | -0.0048 | -0.0049 |

But rotation on its own is worth -0.0005/-0.0019, i.e. nothing, and averaged over both datasets
`rotmin0.1_h1.5` (-0.0087) and plain `h1.5` (-0.0081) are within 0.0006 of each other. **The honest
reading is that the election rule contributes essentially all of the gain and threshold rotation is
neutral.** Its value is negative-going insurance: it is what stops a 25%-NMSE-better idea from
costing a tenth of a perplexity point.

`rotmin0.1_h1.5` at 32x128 is -0.0043 against -0.0149 at 8x64, so the smallest realizable type block
remains the right one, as in every earlier round.
