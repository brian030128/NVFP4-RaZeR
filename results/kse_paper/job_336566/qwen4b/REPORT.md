# Search-free k-SE count rule: qwen4b

Every tile with max(mean CE + k SE, mean KL + k SE) < 0 is switched; there is no cap and no per-model search. k = 2 is the shipped score and reproduces it exactly; its 256-tile prefix reproduces the frozen fixed256_math_code128 map bitwise. Evaluation is the released 2048-token protocol.

| Policy | Tiles | % of type blocks | WikiText-2 | ΔWiki | C4 | ΔC4 |
|---|---:|---:|---:|---:|---:|---:|
| four_over_six | 0 | 0 | 14.269062 | +0.000000 | 17.326633 | +0.000000 |
| n256 | 256 | 0.003608% | 13.040957 | -1.228105 | 16.615953 | -0.710680 |
| k2 | 66,856 | 0.942122% | 10.850905 | -3.418157 | 15.162447 | -2.164186 |
| k3 | 7,912 | 0.111494% | 11.862908 | -2.406154 | 15.824034 | -1.502600 |
| k4 | 1,837 | 0.025887% | 12.715484 | -1.553578 | 16.358000 | -0.968634 |
| k5 | 576 | 0.008117% | 13.410134 | -0.858928 | 16.781254 | -0.545380 |
| k6 | 222 | 0.003128% | 13.822810 | -0.446252 | 17.037529 | -0.289104 |

Scope: 252 text linear matrices, 7,096,320 type blocks of 8x64.

The constant k is fixed in advance and identical for every model; it is not chosen from these results. Two-SE intervals in the machine-readable report are descriptive evaluation-window intervals.
