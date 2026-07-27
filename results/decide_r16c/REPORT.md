
### llama-3.1-8b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                                | HW |  wikitext | dwikitext |        c4 |       dc4 |
|---------------------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_clipfull_clipmin0.3_h1.5_8x64 | y  |    6.5679 |   -0.0304 |    9.4071 |   -0.0169 |
| mix_4_6_clipfull_clipmin0.3_h3_8x64   | y  |    6.5856 |   -0.0128 |    9.4113 |   -0.0126 |
| mix_4_6_clipfull_clipmin0.3_e2m1_8x64 | y  |    6.5861 |   -0.0123 |    9.4125 |   -0.0115 |
| mix_4_6_clipheade0_h3_8x64            | y  |    6.5862 |   -0.0122 |    9.4150 |   -0.0089 |
| mix_4_6_clipbothx_clipmin0.3_h3_8x64  | y  |    6.5898 |   -0.0086 |    9.4081 |   -0.0159 |
| nvfp4_4over6                          |    |    6.5984 |   +0.0000 |    9.4239 |   +0.0000 |
| mix_4_6_e2m1_8x64                     | y  |    6.5985 |   +0.0001 |    9.4219 |   -0.0020 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.
