# Tile-count sweep on held-out C4: olmo1b

Re-election of the unchanged CE/KL two-SE rule from the saved 192-sequence score table at several counts. The 256 row reproduces the frozen pooled192 map exactly. 30485 of 2097152 tiles have a negative two-SE score.

ΔPPL and ΔNLL are relative to FourOverSix; negative is better. Two-SE intervals are descriptive evaluation-window intervals and do not account for calibration-draw variability or multiple comparisons across counts.

| Count | Selected | C4 PPL | ΔPPL | ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| baseline | 0 | 14.984432 | — | — |
| 16 | 16 | 14.946808 | -0.037624 | -0.002514 ±0.002172 |
| 64 | 64 | 14.918822 | -0.065610 | -0.004388 ±0.002285 |
| 256 | 256 | 14.883008 | -0.101424 | -0.006792 ±0.002574 |
| 1024 | 1024 | 14.868634 | -0.115798 | -0.007758 ±0.003034 |
| 4096 | 4096 | 14.844247 | -0.140184 | -0.009399 ±0.003269 |
| 16384 | 16384 | 14.863563 | -0.120868 | -0.008099 ±0.002979 |
| 65536 | 30485 | 14.854211 | -0.130221 | -0.008728 ±0.003658 |
| _all | 30485 | 14.854211 | -0.130221 | -0.008728 ±0.003658 |

Selection uses only the saved calibration score table; C4 evaluation documents have zero hash overlap with the 192 calibration documents. This sweep measures the shape of the count/gain curve. Choosing a count from this table would be selection on the evaluation set; any count rule derived here requires validation on domains untouched by this sweep.
