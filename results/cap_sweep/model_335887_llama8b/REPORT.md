# Tile-count sweep on held-out C4: llama8b

Re-election of the unchanged CE/KL two-SE rule from the saved 192-sequence score table at several counts. The 256 row reproduces the frozen pooled192 map exactly. 97908 of 13631488 tiles have a negative two-SE score.

ΔPPL and ΔNLL are relative to FourOverSix; negative is better. Two-SE intervals are descriptive evaluation-window intervals and do not account for calibration-draw variability or multiple comparisons across counts.

| Count | Selected | C4 PPL | ΔPPL | ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| baseline | 0 | 11.600109 | — | — |
| 16 | 16 | 11.571491 | -0.028618 | -0.002470 ±0.002631 |
| 64 | 64 | 11.566999 | -0.033109 | -0.002858 ±0.002578 |
| 256 | 256 | 11.509863 | -0.090246 | -0.007810 ±0.002113 |
| 1024 | 1024 | 11.509227 | -0.090881 | -0.007865 ±0.002663 |
| 4096 | 4096 | 11.528225 | -0.071884 | -0.006216 ±0.002806 |
| 16384 | 16384 | 11.567794 | -0.032314 | -0.002790 ±0.003410 |
| 65536 | 65536 | 11.599627 | -0.000482 | -0.000042 ±0.003299 |
| _all | 97908 | 11.647632 | +0.047524 | +0.004088 ±0.003532 |

Selection uses only the saved calibration score table; C4 evaluation documents have zero hash overlap with the 192 calibration documents. This sweep measures the shape of the count/gain curve. Choosing a count from this table would be selection on the evaluation set; any count rule derived here requires validation on domains untouched by this sweep.
