# Search-free k-SE count rule: qwen27b

Every tile with max(mean CE + k SE, mean KL + k SE) < 0 is switched; there is no cap and no per-model search. k = 2 is the shipped score and reproduces it exactly; its 256-tile prefix reproduces the frozen fixed256_math_code128 map bitwise. Evaluation is the released 2048-token protocol.

| Policy | Tiles | % of type blocks | WikiText-2 | ΔWiki | C4 | ΔC4 |
|---|---:|---:|---:|---:|---:|---:|
| four_over_six | 0 | 0 | 7.287076 | +0.000000 | 10.188365 | +0.000000 |
| n256 | 256 | 0.000538% | 7.292438 | +0.005361 | 10.167717 | -0.020648 |
| k2 | 149,033 | 0.313360% | 7.099872 | -0.187205 | 10.101456 | -0.086909 |
| k3 | 3,785 | 0.007958% | 7.214750 | -0.072327 | 10.149866 | -0.038499 |
| k4 | 593 | 0.001247% | 7.255247 | -0.031829 | 10.167598 | -0.020767 |
| k5 | 165 | 0.000347% | 7.282632 | -0.004445 | 10.174759 | -0.013606 |
| k6 | 47 | 0.000099% | 7.286022 | -0.001054 | 10.186415 | -0.001950 |

Scope: 496 text linear matrices, 47,559,680 type blocks of 8x64.

The constant k is fixed in advance and identical for every model; it is not chosen from these results. Two-SE intervals in the machine-readable report are descriptive evaluation-window intervals.
