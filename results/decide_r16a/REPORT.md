
### llama-3.2-3b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                                | HW |  wikitext | dwikitext |        c4 |       dc4 |
|---------------------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_clipbothx_clipmin0.3_h3_8x64  | y  |    8.2067 |   -0.0053 |   11.0454 |   -0.0095 |
| nvfp4_4over6                          |    |    8.2120 |   +0.0000 |   11.0549 |   +0.0000 |
| mix_4_6_clipfull_clipmin0.3_e2m1_8x64 | y  |    8.2123 |   +0.0004 |   11.0447 |   -0.0102 |
| mix_4_6_e2m1_8x64                     | y  |    8.2125 |   +0.0005 |   11.0565 |   +0.0015 |
| mix_4_6_clipfull_clipmin0.3_h3_8x64   | y  |    8.2162 |   +0.0043 |   11.0423 |   -0.0127 |
| mix_4_6_clipheade0_h3_8x64            | y  |    8.2191 |   +0.0071 |   11.0471 |   -0.0078 |
| mix_4_6_clipfull_clipmin0.3_h1.5_8x64 | y  |    8.2214 |   +0.0094 |   11.0541 |   -0.0008 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.
