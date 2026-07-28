
### llama-3.1-8b — W4A4   (baseline: nvfp4_4over6, sorted by wikitext)

| config                                                            | HW |  wikitext | dwikitext |        c4 |       dc4 |
|-------------------------------------------------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_clipbothx_clipmin0.3_h1.5_8x64__a-mix_4_6_clipdense9_h1.5 | y  |    6.8423 |   -0.0350 |    9.7843 |   -0.0334 |
| mix_4_6_clipbothx_clipmin0.3_h1.5_8x64                            | y  |    6.8460 |   -0.0313 |    9.7948 |   -0.0229 |
| mix_4_6_clipdense9_h1.5_8x64__a-mix_4_6_clipdense9_e2m1           | y  |    6.8485 |   -0.0288 |    9.7761 |   -0.0416 |
| mix_4_6_clipdense9_h1.5_8x64__a-mix_4_6_clipbothx_clipmin0.3_h1.5 | y  |    6.8496 |   -0.0276 |    9.7825 |   -0.0352 |
| mix_4_6_clipdense9_e2m1_8x64__a-mix_4_6_clipdense9_h1.5           | y  |    6.8537 |   -0.0236 |    9.7760 |   -0.0417 |
| mix_4_6_clipdense9_h1.5_8x64                                      | y  |    6.8538 |   -0.0235 |    9.7746 |   -0.0431 |
| nvfp4_4over6                                                      |    |    6.8773 |   +0.0000 |    9.8177 |   +0.0000 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## W4A4 per-operand split: E0M3 pays more on WEIGHTS than on activations

The two operands can carry different data types and different alpha sets, so they were given
different configurations. Llama-3.1-8B W4A4, mean of wikitext and c4 against `nvfp4_4over6`.

| weights | activations | mean |
|---|---|---|
| dense alpha, **E0M3 on** | dense alpha, **E0M3 off** | **-0.0352** |
| gated clipping, E0M3 on | dense alpha, E0M3 on | -0.0342 |
| dense alpha, E0M3 on | dense alpha, E0M3 on | -0.0333 |
| dense alpha, E0M3 on | gated clipping, E0M3 on | -0.0276 |
| dense alpha, **E0M3 off** | dense alpha, **E0M3 on** | -0.0327 |
| gated clipping (both operands) | | -0.0271 |

**E0M3 on the weights and off on the activations (-0.0352) beats E0M3 on both (-0.0333), which beats
E0M3 on the activations only (-0.0327).** That is the opposite of what `nvif4` suggested: its W4A4
gain (-0.0779) is much larger than its W4A16 gain (-0.0457), which looked like evidence that the
element type matters more on the A operand. At the realizable type block the ordering reverses.

A plausible reading, consistent with the usage measurements in `results/e0m3_usage/`: E0M3 is never
elected for the q/k/v activation inputs at all (0.0%), so on the A operand the election has almost
nothing to work with once the tile is 16x64, while the B operand still elects it on ~5% of elements.

Giving the operands different alpha sets is also worth something on its own -- gated clipping on
weights with dense alpha on activations (-0.0342) beats gated clipping on both (-0.0271).
