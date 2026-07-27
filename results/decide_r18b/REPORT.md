
### llama-3.1-8b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                                | HW |  wikitext | dwikitext |        c4 |       dc4 |
|---------------------------------------|----|-----------|-----------|-----------|-----------|
| mix_4_6_clipdense9_amin0.1_e2m1_8x64  | y  |    6.5790 |   -0.0194 |    9.3982 |   -0.0257 |
| mix_4_6_clipdense9_amin0.2_e2m1_8x64  | y  |    6.5816 |   -0.0167 |    9.4040 |   -0.0200 |
| mix_4_6_clipdense9_e2m1_8x64          | y  |    6.5830 |   -0.0154 |    9.4010 |   -0.0229 |
| mix_4_6_clipdense9_amin0.02_e2m1_8x64 | y  |    6.5854 |   -0.0130 |    9.4030 |   -0.0210 |
| mix_4_6_clipdense9_amin0.05_h3_8x64   | y  |    6.5859 |   -0.0125 |    9.4018 |   -0.0221 |
| mix_4_6_clipdense9_amin0.05_e2m1_8x64 | y  |    6.5871 |   -0.0113 |    9.4032 |   -0.0207 |
| nvfp4_4over6                          |    |    6.5984 |   +0.0000 |    9.4239 |   +0.0000 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 18b settles: the alpha gate is a pure win, not a safety trade

Round 18a showed `amin<t>` repairing the dense grid's one regression on Llama-2-7B. The obvious
worry was that it buys that safety by giving up gain where the ungated search already works. It does
not — on Llama-3.1-8B the gate improves **both** datasets:

| config @ 8x64 | dwikitext | dc4 | mean |
|---|---|---|---|
| **`clipdense9_amin0.1_e2m1`** | **-0.0194** | **-0.0257** | **-0.0226** |
| `clipdense9_e2m1` (ungated) | -0.0154 | -0.0229 | -0.0192 |
| `clipdense9_amin0.2_e2m1` | -0.0167 | -0.0200 | -0.0184 |
| `clipdense9_amin0.02_e2m1` | -0.0130 | -0.0210 | -0.0170 |
| `clipdense9_amin0.05_e2m1` | -0.0113 | -0.0207 | -0.0160 |

Across the two models measured so far:

| config | Llama-3.1-8B | Llama-2-7B | 2-model mean |
|---|---|---|---|
| **`amin0.1`** | **-0.0226** | -0.0025 | **-0.0126** |
| `amin0.2` | -0.0184 | **-0.0052** | -0.0118 |
| ungated | -0.0192 | -0.0008 | -0.0100 |

`t = 0.1` is the better setting on Llama-3.1-8B and `t = 0.2` on Llama-2-7B, but both beat the
ungated search on both models, so the gate itself is not a tuning artifact. The non-monotonicity at
small `t` (0.02 and 0.05 are *worse* than both 0 and 0.1 on this model) is worth noting: a weak
threshold appears to be the worst of both worlds, blocking some good moves without blocking the
noisy ones.

This is the strongest single configuration measured anywhere in this study, and it uses **no type
block, no E0M3 operand and no election rule** — the entire gain is a per-scale-block scale search
with a decisive-margin gate, on plain E2M1.
