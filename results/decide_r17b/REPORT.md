
### llama-3.2-3b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                       | HW |  wikitext | dwikitext |        c4 |       dc4 |
|------------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_clipdense9_e2m1_8x64 | y  |    8.1960 |   -0.0159 |   11.0133 |   -0.0417 |
| mix_4_6_clipdense9_h3_8x64   | y  |    8.1969 |   -0.0151 |   11.0145 |   -0.0405 |
| mix_4_6_clipdense9_h2.5_8x64 | y  |    8.1997 |   -0.0123 |   11.0141 |   -0.0408 |
| mix_4_6_clipdense9_h1.5_8x64 | y  |    8.2054 |   -0.0065 |   11.0203 |   -0.0347 |
| mix_4_6_clipdense5_e2m1_8x64 | y  |    8.2073 |   -0.0047 |   11.0328 |   -0.0221 |
| nvfp4_4over6                 |    |    8.2120 |   +0.0000 |   11.0549 |   +0.0000 |
| mix_4_6_clipheadx_e2m1_8x64  | y  |    8.2156 |   +0.0036 |   11.0520 |   -0.0030 |
| mix_4_6_clipheadx_h3_8x64    | y  |    8.2171 |   +0.0051 |   11.0493 |   -0.0056 |
| mix_4_6_clipheadx_h2.5_8x64  | y  |    8.2199 |   +0.0080 |   11.0500 |   -0.0050 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.
