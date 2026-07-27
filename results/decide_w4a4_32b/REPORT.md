
### llama-3.2-3b — W4A4   (baseline: nvfp4_4over6, sorted by wikitext)

| config                                      | HW |  wikitext | dwikitext |        c4 |       dc4 |
|---------------------------------------------|----|-----------|-----------|-----------|-----------|
| fp16                                        |    |    7.8170 |   -0.7099 |   10.4350 |   -1.0865 |
| razer                                       |    |    8.4041 |   -0.1227 |   11.3148 |   -0.2067 |
| mix_4_6_1x16                                | -  |    8.4483 |   -0.0785 |   11.3651 |   -0.1565 |
| nvif4                                       |    |    8.4598 |   -0.0671 |   11.4022 |   -0.1194 |
| mix_4_6_clipdense9_e2m1_8x64                | y  |    8.4967 |   -0.0301 |   11.4620 |   -0.0596 |
| mix_4_6_clipdense9_h3_8x64                  | y  |    8.4975 |   -0.0294 |   11.4635 |   -0.0580 |
| mix_4_6_clipdense9_amin0.1_e2m1_8x64        | y  |    8.5020 |   -0.0248 |   11.4675 |   -0.0541 |
| mix_4_6_clipdense9_amin0.2_e2m1_8x64        | y  |    8.5031 |   -0.0237 |   11.4734 |   -0.0481 |
| mix_4_6_clipbothx_clipmin0.3_h3_8x64        | y  |    8.5138 |   -0.0130 |   11.5046 |   -0.0170 |
| mix_4_6_clipfull_clipmin0.3_amin0.1_h3_8x64 | y  |    8.5214 |   -0.0054 |   11.4894 |   -0.0321 |
| mix_4_6_clipbothx_clipmin0.3_h1.5_8x64      | y  |    8.5219 |   -0.0049 |   11.5055 |   -0.0161 |
| nvfp4_4over6                                |    |    8.5268 |   +0.0000 |   11.5215 |   +0.0000 |
| mix_4_6_h3_8x64                             | y  |    8.5320 |   +0.0052 |   11.5278 |   +0.0062 |
| mix_4_6_e2m1_8x64                           | y  |    8.5458 |   +0.0190 |   11.5283 |   +0.0068 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## W4A4, Llama-3.2-3B: the alpha search is worth -0.058; E0M3 is worth nothing

No rotation anywhere. All deltas are the mean of wikitext and c4 against `nvfp4_4over6`.

| what changes | mean d |
|---|---|
| E2M1 only, plain 4over6 alpha set `{1, 1.5}` | **+0.0129** |
| E2M1 only, dense alpha set (9 points across [1, 1.5]) | **-0.0449** |
| ... plus the E0M3 election (`h3`) | -0.0437 |

**Switching E0M3 off is slightly better than leaving it on** (-0.0449 against -0.0437). So on this
model at W4A4, with a proper per-scale-block scale search, the E0M3 element type contributes
nothing — the entire -0.058 swing is the alpha search.

This also explains why the element type looked useful in earlier rounds: measured against the plain
FourOverSix candidate set, `h3` (+0.0057) does beat `e2m1` (+0.0129). E0M3 was compensating for a
scale search that was too coarse. Widen the search and the compensation is no longer needed.

Two more W4A4-specific results:

- **The alpha gate `amin<t>` hurts here**, the opposite of W4A16. Ungated is -0.0449; `amin0.1` is
  -0.0395 and `amin0.2` is -0.0359. At W4A16 the same gate was a pure win on two models. Activations
  apparently want the search to take small gains that weights do not.
- **Gated clipping is far behind the dense grid at W4A4 on this model** (-0.0150 against -0.0449),
  where at W4A16 the two were comparable.

For scale: `mix_4_6_1x16` (element type per 16-element block, not realizable) is -0.1175 and RaZeR
is -0.1647, so the best realizable configuration captures about 38% of the per-block bound.
