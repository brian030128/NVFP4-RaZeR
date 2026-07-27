
### llama-3.2-3b — W4A16   (baseline: nvfp4_4over6, sorted by wikitext)

| config                      | HW |  wikitext | dwikitext |        c4 |       dc4 |
|-----------------------------|----|-----------|-----------|-----------|-----------|
| fp16                        |    |    7.8170 |   -0.3950 |   10.4350 |   -0.6199 |
| nvfp4_razer_e3m3            |    |    8.1032 |   -0.1088 |   10.8637 |   -0.1912 |
| mix_4_6_1x16                | -  |    8.1703 |   -0.0416 |   10.9606 |   -0.0943 |
| nvif4                       |    |    8.1774 |   -0.0346 |   10.9720 |   -0.0830 |
| mix_4_6_e2m1_8x64           | y  |    8.2114 |   -0.0006 |   11.0554 |   +0.0005 |
| nvfp4_4over6                |    |    8.2120 |   +0.0000 |   11.0549 |   +0.0000 |
| mix_4_6_h3_8x64             | y  |    8.2123 |   +0.0003 |   11.0518 |   -0.0032 |
| mix_4_6_clipheadx_e2m1_8x64 | y  |    8.2156 |   +0.0036 |   11.0520 |   -0.0030 |
| mix_4_6_clipheadx_h3_8x64   | y  |    8.2171 |   +0.0051 |   11.0493 |   -0.0056 |
| mix_4_6_h1.5_8x64           | y  |    8.2328 |   +0.0208 |   11.0621 |   +0.0071 |
| mix_4_6_clipheadx_h1.5_8x64 | y  |    8.2341 |   +0.0221 |   11.0599 |   +0.0049 |

HW: y = expressible by one mma.sync...kind::mxf4nvf4.m16n8k64 operand (K >= 64); - = accuracy upper bound only.

## What round 9 settles (Llama-3.2-3B, W4A16): a third regime, and a correction

Three models now, at 8x64, against `nvfp4_4over6`:

| config | Llama-3.1-8B | Llama-2-7B | Llama-3.2-3B |
|---|---|---|---|
| `h1.5` | **-0.0117 / -0.0044** | +0.0128 / +0.0056 | **+0.0208 / +0.0071** |
| `h3` | -0.0014 / -0.0025 | -0.0024 / -0.0022 | +0.0003 / -0.0032 |
| `clipheadx_e2m1` (headroom, no E0M3) | -0.0082 / -0.0050 | -0.0044 / -0.0050 | **+0.0036 / -0.0030** |
| `clipheadx_h3` | (not run) | -0.0030 / -0.0018 | +0.0051 / -0.0056 |
| `mix_4_6_1x16` (upper bound) | -0.0450 / -0.0580 | +0.0058 / -0.0093 | -0.0416 / -0.0943 |

**`h1.5` is harmful on two models out of three.** It is the best single number in this study and it
does not generalize. `kappa^2 = 3` is the only election setting that is non-harmful everywhere, and
it is worth roughly -0.002.

**Correction to what rounds 5-8 concluded about headroom.** Headroom was recorded as worth "-0.004
to -0.008 wikitext on every model measured", which was true of the two models measured then. On
Llama-3.2-3B it is **+0.0036 wikitext / -0.0030 c4**, i.e. a wash. The honest statement is that
headroom is **neutral-to-positive and never harmful** across the three models, worth about -0.005
mean on two of them and nothing on the third -- not a guaranteed win.

**Llama-3.2-3B is a third regime.** Llama-2-7B has no E0M3 signal at all (its `1x16` is +0.0058
wikitext). Llama-3.1-8B has signal that a coarse election can partly capture. Llama-3.2-3B has
plenty of signal -- `1x16` is -0.0416 / -0.0943, comparable to Llama-3.1-8B -- but **no realizable
rule captures any of it**: every 8x64 row is within 0.005 of the baseline or worse. So a large 1x16
gain is necessary but not sufficient for a tile election to work, which rules out the most obvious
remaining calibration-free predictor.

RaZeR is again far ahead: -0.1088 / -0.1912.
