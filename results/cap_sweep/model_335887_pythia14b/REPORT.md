# Tile-count sweep on held-out C4: pythia14b

Re-election of the unchanged CE/KL two-SE rule from the saved 192-sequence score table at several counts. The 256 row reproduces the frozen pooled192 map exactly. 76665 of 2359296 tiles have a negative two-SE score.

ΔPPL and ΔNLL are relative to FourOverSix; negative is better. Two-SE intervals are descriptive evaluation-window intervals and do not account for calibration-draw variability or multiple comparisons across counts.

| Count | Selected | C4 PPL | ΔPPL | ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| baseline | 0 | 20.471159 | — | — |
| 16 | 16 | 20.418796 | -0.052363 | -0.002561 ±0.002857 |
| 64 | 64 | 20.328652 | -0.142507 | -0.006986 ±0.003037 |
| 256 | 256 | 20.162666 | -0.308493 | -0.015184 ±0.003167 |
| 1024 | 1024 | 19.850120 | -0.621039 | -0.030807 ±0.003188 |
| 4096 | 4096 | 19.890992 | -0.580167 | -0.028750 ±0.003516 |
| 16384 | 16384 | 20.652269 | +0.181110 | +0.008808 ±0.004875 |
| 65536 | 65536 | 22.421023 | +1.949864 | +0.090982 ±0.006580 |
| _all | 76665 | 22.677991 | +2.206832 | +0.102378 ±0.006664 |

Selection uses only the saved calibration score table; C4 evaluation documents have zero hash overlap with the 192 calibration documents. This sweep measures the shape of the count/gain curve. Choosing a count from this table would be selection on the evaluation set; any count rule derived here requires validation on domains untouched by this sweep.
