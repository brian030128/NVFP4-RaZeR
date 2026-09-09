# Tile-count sweep on held-out C4: qwen4b

Re-election of the unchanged CE/KL two-SE rule from the saved 192-sequence score table at several counts. The 256 row reproduces the frozen pooled192 map exactly. 80978 of 7096320 tiles have a negative two-SE score.

ΔPPL and ΔNLL are relative to FourOverSix; negative is better. Two-SE intervals are descriptive evaluation-window intervals and do not account for calibration-draw variability or multiple comparisons across counts.

| Count | Selected | C4 PPL | ΔPPL | ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| baseline | 0 | 21.928243 | — | — |
| 16 | 16 | 21.723072 | -0.205171 | -0.009401 ±0.003048 |
| 64 | 64 | 21.400644 | -0.527599 | -0.024354 ±0.002932 |
| 256 | 256 | 20.902742 | -1.025501 | -0.047895 ±0.003392 |
| 1024 | 1024 | 20.223081 | -1.705162 | -0.080951 ±0.004184 |
| 4096 | 4096 | 19.574356 | -2.353887 | -0.113555 ±0.005325 |
| 16384 | 16384 | 19.258204 | -2.670039 | -0.129838 ±0.006384 |
| 65536 | 65536 | 19.065141 | -2.863102 | -0.139914 ±0.007585 |
| _all | 80978 | 19.064606 | -2.863637 | -0.139942 ±0.007806 |

Selection uses only the saved calibration score table; C4 evaluation documents have zero hash overlap with the 192 calibration documents. This sweep measures the shape of the count/gain curve. Choosing a count from this table would be selection on the evaluation set; any count rule derived here requires validation on domains untouched by this sweep.
