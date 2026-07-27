
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

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 17b settles: the dense alpha grid DOES generalize — round 17a's conclusion was premature

Round 17a saw the dense grid do nothing on Llama-2-7B and concluded it was a Llama-3.1-8B-specific
result. That was one model too few. On Llama-3.2-3B it is the **best configuration measured on that
model by a factor of four**, and it is the variant with **E0M3 switched off entirely**:

| config @ 8x64 | Llama-3.1-8B | Llama-3.2-3B | Llama-2-7B |
|---|---|---|---|
| `clipdense9_e2m1` (no E0M3) | -0.0146 / -0.0218 | **-0.0159 / -0.0417** | +0.0094 / -0.0109 |
| `clipdense9_h3` | -0.0157 / -0.0226 | -0.0151 / -0.0405 | +0.0099 / -0.0086 |
| `clipheadx_e2m1` (coarse, 5 points) | -0.0082 / -0.0050 | +0.0036 / -0.0030 | -0.0044 / -0.0050 |
| `clipbothx_clipmin0.3_h3` (gated clipping) | -0.0086 / -0.0159 | -0.0053 / -0.0095 | -0.0007 / -0.0057 |

Mean over both datasets, across the three models:

| config | 3.1-8B | 3.2-3B | 2-7B | 3-model mean | negative on all 6 measurements? |
|---|---|---|---|---|---|
| **`clipdense9_e2m1`** | -0.0182 | **-0.0288** | -0.0008 | **-0.0159** | no (2-7B wikitext +0.0094) |
| `clipbothx_clipmin0.3_h3` | -0.0123 | -0.0074 | -0.0032 | -0.0076 | **yes** |
| `clipheadx_e2m1` | -0.0066 | +0.0003 | -0.0047 | -0.0035 | no (3.2-3B wikitext +0.0036) |

So there are two defensible recommendations, and they trade off differently:

- **`clipdense9_e2m1` — twice the average gain, and needs nothing.** No type block, no E0M3 operand,
  no election rule: it is plain NVFP4 with a nine-point per-scale-block scale search. Its one
  weakness is Llama-2-7B wikitext (+0.0094), offset by -0.0109 on that model's c4.
- **`clipbothx_clipmin0.3_h3` — smaller but strictly safe.** The only configuration in the study
  negative on all six model x dataset measurements.

**Correction to round 17a.** Its report and commit state that the dense grid "does not generalize".
On the evidence of two models that was the wrong generalization to draw from one negative; the
correct statement is that it is a large win on two of three models, a wash on the third, and never
harmful in mean.
