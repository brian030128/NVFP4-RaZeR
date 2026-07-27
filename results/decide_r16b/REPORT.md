
### llama-2-7b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                                | HW |  wikitext | dwikitext |        c4 |       dc4 |
|---------------------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_e2m1_8x64                     | y  |    5.6071 |   -0.0016 |    7.1377 |   -0.0001 |
| mix_4_6_clipbothx_clipmin0.3_h3_8x64  | y  |    5.6080 |   -0.0007 |    7.1322 |   -0.0057 |
| nvfp4_4over6                          |    |    5.6087 |   +0.0000 |    7.1378 |   +0.0000 |
| mix_4_6_clipfull_clipmin0.3_e2m1_8x64 | y  |    5.6106 |   +0.0018 |    7.1313 |   -0.0066 |
| mix_4_6_clipheade0_h3_8x64            | y  |    5.6110 |   +0.0023 |    7.1383 |   +0.0004 |
| mix_4_6_clipfull_clipmin0.3_h3_8x64   | y  |    5.6141 |   +0.0054 |    7.1356 |   -0.0023 |
| mix_4_6_clipfull_clipmin0.3_h1.5_8x64 | y  |    5.6242 |   +0.0155 |    7.1412 |   +0.0033 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 16 settles: gated clipping at kappa^2 = 3 is the first thing that works everywhere

`clipbothx` + `clipmin0.3` + `h3` — E2M1 and E0M3 both given clipping candidates, gated behind a
30% minimum error reduction, with the conservative election — is the only configuration in this
study that is **negative on every model and both datasets measured**:

| model / setting | dwikitext | dc4 | mean |
|---|---|---|---|
| Llama-3.1-8B W4A4 (`h1.5` variant, round 12) | -0.0297 | -0.0195 | **-0.0246** |
| Llama-3.1-8B W4A16 (`h1.5` variant, round 7) | -0.0179 | -0.0159 | **-0.0169** |
| Llama-3.2-3B W4A16 | -0.0053 | -0.0095 | **-0.0074** |
| Llama-2-7B W4A16 | -0.0007 | -0.0057 | **-0.0032** |

Compare the alternatives on the two models that resisted everything else:

| config @ 8x64 | Llama-2-7B | Llama-3.2-3B |
|---|---|---|
| **`clipbothx_clipmin0.3_h3`** | **-0.0032** | **-0.0074** |
| `clipheadx_e2m1` (headroom) | -0.0047 | +0.0003 |
| `h3` | -0.0023 | -0.0014 |
| `clipheade0_h3` | +0.0014 | -0.0004 |
| `clipfull_clipmin0.3_h1.5` | +0.0094 | +0.0043 |

The `clipfull` preset — headroom *and* clipping in one candidate set — is **worse** than `clipbothx`
on both models. Combining the two directions of alpha is not additive: giving the search both
options lets it pick headroom where clipping was the useful move. And `h1.5` remains harmful on
these models even with clipping, so the conservative election is doing real work.

**This is the direction round 1 rejected.** Clipping (`alpha < 1`) was measured at +0.006 to +0.033
wikitext there and dismissed. It is the best generalizing idea in the study once gated behind a
minimum gain — the third and strongest instance of the principle that a rule must fire only on
decisive error reductions.
