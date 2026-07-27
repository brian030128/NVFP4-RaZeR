
### llama-2-7b — W4A4   (baseline: nvfp4_4over6, sorted by wikitext)

| config                                 | HW |  wikitext | dwikitext |        c4 |       dc4 |
|----------------------------------------|----|-----------|-----------|-----------|-----------|
| fp16                                   |    |    5.4738 |   -0.2418 |    6.9749 |   -0.2789 |
| razer                                  |    |    5.6715 |   -0.0441 |    7.2045 |   -0.0493 |
| mix_4_6_1x16                           | -  |    5.7037 |   -0.0119 |    7.2288 |   -0.0249 |
| mix_4_6_clipheade0_e2m1_8x64           | y  |    5.7148 |   -0.0008 |    7.2475 |   -0.0062 |
| mix_4_6_e2m1_8x64                      | y  |    5.7152 |   -0.0004 |    7.2535 |   -0.0002 |
| nvfp4_4over6                           |    |    5.7156 |   +0.0000 |    7.2537 |   +0.0000 |
| mix_4_6_h3_8x64                        | y  |    5.7157 |   +0.0001 |    7.2536 |   -0.0001 |
| nvif4                                  |    |    5.7158 |   +0.0002 |    7.2412 |   -0.0125 |
| mix_4_6_clipheade0_h3_8x64             | y  |    5.7188 |   +0.0032 |    7.2549 |   +0.0012 |
| mix_4_6_clipbothx_clipmin0.3_h1.5_8x64 | y  |    5.7217 |   +0.0061 |    7.2473 |   -0.0065 |
| mix_4_6_h1.5_8x64                      | y  |    5.7264 |   +0.0108 |    7.2585 |   +0.0048 |
| mix_4_6_clipheade0_h1.5_8x64           | y  |    5.7266 |   +0.0110 |    7.2569 |   +0.0031 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 13 settles (Llama-2-7B, W4A4): the W4A4 win does not generalize either

Round 12 found gated clipping to be the best W4A4 configuration on Llama-3.1-8B by a wide margin.
On Llama-2-7B the same configuration is a **wash**:

| config @ 8x64 | Llama-3.1-8B W4A4 | Llama-2-7B W4A4 |
|---|---|---|
| `clipbothx_clipmin0.3_h1.5` | -0.0297 / -0.0195 (mean **-0.0246**) | +0.0061 / -0.0065 (mean -0.0002) |
| `clipheade0_h1.5` | -0.0316 / -0.0032 (mean -0.0174) | +0.0110 / +0.0031 (mean +0.0071) |
| `clipheade0_e2m1` (no E0M3) | -0.0095 / -0.0110 (mean -0.0103) | -0.0008 / -0.0062 (mean **-0.0035**) |
| `h1.5` | -0.0079 / -0.0085 | +0.0108 / +0.0048 |
| `h3` | +0.0007 / -0.0029 | +0.0001 / -0.0001 |

Nothing here moves the model by more than 0.006 in either direction. The best row is the one with
the E0M3 branch **switched off** — the widened alpha search alone, at -0.0035 mean.

That completes the cross-model picture, and it is consistent in a way worth stating plainly:

> **Llama-3.1-8B benefits from this whole family; Llama-2-7B and Llama-3.2-3B benefit from
> essentially none of it.** Nothing is reliably *harmful* provided the election uses `kappa^2 = 3`
> or is switched off entirely, but the gains are model-specific and cannot be predicted
> calibration-free (rounds 6 and 9).

Note the `1x16` upper bound here is only -0.0119 / -0.0249, far smaller than Llama-3.1-8B's
-0.0810 / -0.1006, so on this model there was little to capture in the first place. That is *not*
the explanation for Llama-3.2-3B, whose `1x16` bound is large (-0.0416 / -0.0943) and which still
gains nothing realizable — see round 9.
