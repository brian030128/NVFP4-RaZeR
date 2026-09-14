# Search-free k-SE count rule: llama8b

Every tile with max(mean CE + k SE, mean KL + k SE) < 0 is switched; there is no cap and no per-model search. k = 2 is the shipped score and reproduces it exactly; its 256-tile prefix reproduces the frozen fixed256_math_code128 map bitwise. Evaluation is the released 2048-token protocol.

| Policy | Tiles | % of type blocks | WikiText-2 | ΔWiki | C4 | ΔC4 |
|---|---:|---:|---:|---:|---:|---:|
| four_over_six | 0 | 0 | 6.875525 | +0.000000 | 9.823733 | +0.000000 |
| k2 | 99,024 | 0.726436% | 6.907813 | +0.032289 | 9.842249 | +0.018516 |
| k2_ce | 356,302 | 2.613816% | 6.938570 | +0.063045 | 9.876889 | +0.053156 |
| k3 | 3,345 | 0.024539% | 6.849275 | -0.026249 | 9.773040 | -0.050694 |
| k3_ce | 22,906 | 0.168037% | 6.836686 | -0.038839 | 9.768658 | -0.055076 |
| k4 | 541 | 0.003969% | 6.852014 | -0.023511 | 9.777571 | -0.046163 |
| k4_ce | 1,104 | 0.008099% | 6.853590 | -0.021935 | 9.781658 | -0.042075 |
| k5 | 267 | 0.001959% | 6.861062 | -0.014463 | 9.795584 | -0.028150 |
| k5_ce | 277 | 0.002032% | 6.862983 | -0.012541 | 9.793248 | -0.030485 |
| k6 | 145 | 0.001064% | 6.860541 | -0.014984 | 9.800744 | -0.022989 |
| k6_ce | 147 | 0.001078% | 6.858060 | -0.017465 | 9.791759 | -0.031975 |

Scope: 224 text linear matrices, 13,631,488 type blocks of 8x64.

The constant k is fixed in advance and identical for every model; it is not chosen from these results. Two-SE intervals in the machine-readable report are descriptive evaluation-window intervals.
