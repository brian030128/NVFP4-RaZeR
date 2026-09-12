# Search-free k-SE count rule: llama8b

Every tile with max(mean CE + k SE, mean KL + k SE) < 0 is switched; there is no cap and no per-model search. k = 2 is the shipped score and reproduces it exactly; its 256-tile prefix reproduces the frozen fixed256_math_code128 map bitwise. Evaluation is the released 2048-token protocol.

| Policy | Tiles | % of type blocks | WikiText-2 | ΔWiki | C4 | ΔC4 |
|---|---:|---:|---:|---:|---:|---:|
| four_over_six | 0 | 0 | 6.875525 | +0.000000 | 9.823733 | +0.000000 |
| k2 | 99,024 | 0.726436% | 6.907813 | +0.032289 | 9.842249 | +0.018516 |
| k2_kl | 385,903 | 2.830968% | 8.889927 | +2.014402 | 12.064046 | +2.240313 |
| k3 | 3,345 | 0.024539% | 6.849275 | -0.026249 | 9.773040 | -0.050694 |
| k3_kl | 32,774 | 0.240429% | 7.356566 | +0.481042 | 10.394302 | +0.570569 |

Scope: 224 text linear matrices, 13,631,488 type blocks of 8x64.

The constant k is fixed in advance and identical for every model; it is not chosen from these results. Two-SE intervals in the machine-readable report are descriptive evaluation-window intervals.
