
### llama-3.1-8b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                       | HW |  wikitext | dwikitext |        c4 |       dc4 |
|------------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_clipheade0_h2.5_8x64 | y  |    6.5810 |   -0.0173 |    9.4123 |   -0.0116 |
| mix_4_6_clipheadx_h2.5_8x64  | y  |    6.5877 |   -0.0107 |    9.4169 |   -0.0071 |
| mix_4_6_clipheadx_h3_8x64    | y  |    6.5891 |   -0.0093 |    9.4173 |   -0.0066 |
| mix_4_6_clipheadx_e2m1_8x64  | y  |    6.5902 |   -0.0082 |    9.4189 |   -0.0050 |
| mix_4_6_clipheadx_e2m1_1x16  | -  |    6.5905 |   -0.0079 |    9.4188 |   -0.0051 |
| nvfp4_4over6                 |    |    6.5984 |   +0.0000 |    9.4239 |   +0.0000 |
| mix_4_6_clipheadxx_e2m1_1x16 | -  |    6.6010 |   +0.0026 |    9.4147 |   -0.0092 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 14 settles: kappa^2 = 2.5-3 with headroom is the robust operating point

`h1.5` is the best number on Llama-3.1-8B and a loss on the other two models (rounds 8, 9). This
round fills in the intermediate kappa and the headroom crossing, which changes the recommendation.

| config @ 8x64 | Llama-3.1-8B | Llama-2-7B | Llama-3.2-3B |
|---|---|---|---|
| `clipheadx_e2m1` (headroom, no E0M3) | -0.0082 / -0.0050 | -0.0044 / -0.0050 | +0.0036 / -0.0030 |
| **`clipheadx_h3`** | **-0.0093 / -0.0066** | **-0.0030 / -0.0018** | +0.0051 / -0.0056 |
| `clipheadx_h2.5` | -0.0107 / -0.0071 | (not run) | (not run) |
| `clipheade0_h2.5` | **-0.0173 / -0.0116** | (not run) | (not run) |
| `clipheade0_h3` | -0.0122 / -0.0089 | +0.0023 / +0.0004 | (not run) |
| `clipheadx_h1.5` | -0.0190 / -0.0109 | (not run) | +0.0221 / +0.0049 |
| `clipheade0_h1.5` | -0.0265 / -0.0081 | +0.0165 / +0.0061 | (not run) |

Two things fall out.

**`kappa^2 = 2.5` beats `kappa^2 = 3` without going as far as 1.5.** On Llama-3.1-8B,
`clipheade0_h2.5` is -0.0173 / -0.0116 (mean -0.0145) against -0.0106 for `h3` and -0.0174 for
`h1.5`, so it keeps most of the aggressive setting's gain. Whether it stays safe on the other two
models is untested and is the obvious next measurement.

**E0M3 headroom is the part that breaks generalization, not E2M1 headroom.** On Llama-2-7B,
`clipheadx_h3` is -0.0030 / -0.0018 but `clipheade0_h3` is +0.0023 / +0.0004 — the two differ only
in whether the E0M3 branch gets its own alpha candidates. Widening E0M3 makes the election fire more
often, which is exactly what this model does not want.

So the safest configuration that is non-harmful on all three models is **E2M1 headroom plus a strict
election**: `clipheadx_h3`, worth -0.0093 / -0.0066 where the model has E0M3 signal and about zero
where it does not.

Consistency check: `clipheadx_e2m1` at `1x16` (-0.0079 / -0.0051) matches the same configuration at
`8x64` (-0.0082 / -0.0050), as it must — with the election switched off the type block is inert.
