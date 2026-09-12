# Search-free k-SE count rule: llama8b

Every tile with max(mean CE + k SE, mean KL + k SE) < 0 is switched; there is no cap and no per-model search. k = 2 is the shipped score and reproduces it exactly; its 256-tile prefix reproduces the frozen fixed256_math_code128 map bitwise. Evaluation is the released 2048-token protocol.

| Policy | Tiles | % of type blocks | WikiText-2 | ΔWiki | C4 | ΔC4 |
|---|---:|---:|---:|---:|---:|---:|
| four_over_six | 0 | 0 | 6.875525 | +0.000000 | 9.823733 | +0.000000 |
| k2 | 99,024 | 0.726436% | 6.907813 | +0.032289 | 9.842249 | +0.018516 |
| k2_kl | 385,903 | 2.830968% | 8.889927 | +2.014402 | 12.064046 | +2.240313 |
| k3 | 3,345 | 0.024539% | 6.849275 | -0.026249 | 9.773040 | -0.050694 |
| k3_kl | 32,774 | 0.240429% | 7.356566 | +0.481042 | 10.394302 | +0.570569 |
| k4 | 541 | 0.003969% | 6.852014 | -0.023511 | 9.777571 | -0.046163 |
| k4_kl | 6,591 | 0.048351% | 6.971170 | +0.095645 | 9.953737 | +0.130004 |
| k5 | 267 | 0.001959% | 6.861062 | -0.014463 | 9.795584 | -0.028150 |
| k5_kl | 2,950 | 0.021641% | 6.881298 | +0.005773 | 9.827334 | +0.003601 |
| k6 | 145 | 0.001064% | 6.860541 | -0.014984 | 9.800744 | -0.022989 |
| k6_kl | 1,653 | 0.012126% | 6.858618 | -0.016906 | 9.784282 | -0.039452 |

Scope: 224 text linear matrices, 13,631,488 type blocks of 8x64.

The constant k is fixed in advance and identical for every model; it is not chosen from these results. Two-SE intervals in the machine-readable report are descriptive evaluation-window intervals.
