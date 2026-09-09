# Calibration-chosen tile count: qwen4b

Counts are chosen by the lowest actual next-token loss over calibration-source documents; the held-out C4 set is never consulted for selection. `sel` is 192 documents disjoint from the 192 scoring documents; `fit` is the scoring set itself. 80978 of 7096320 tiles are eligible.

| Count | Selected | fit NLL | sel NLL |
|---|---:|---:|---:|
| four_over_six | 0 | 2.179910 | 2.242726 |
| n16 | 16 | 2.172262 | 2.236124 |
| n64 | 64 | 2.160629 | 2.225878 |
| n256 | 256 | 2.138253 | 2.205116 |
| n1024 | 1024 | 2.108533 | 2.176753 |
| n4096 | 4096 | 2.076637 | 2.148148 |
| n16384 | 16384 | 2.055551 | 2.127507 |
| n65536 | 65536 | 2.030127 | 2.116758 |
| n_all | 80978 | 2.026145 | 2.112043 |

| Policy | Count | C4 PPL | ΔPPL vs FourOverSix | ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| FourOverSix | 0 | 21.928243 | — | — |
| n256 | 256 | 20.902742 | -1.025501 | -0.047895 ±0.003392 |
| n_all | 80978 | 19.064606 | -2.863637 | -0.139942 ±0.007806 |

Chosen on `sel`: n_all (80978 tiles). Against the fixed 256 map: ΔPPL -1.838136, ΔNLL -0.092047 ±0.006005.

Chosen on `fit`: n_all (80978 tiles). Against the fixed 256 map: ΔPPL -1.838136, ΔNLL -0.092047 ±0.006005.

The count is selected on calibration-source documents only. These held-out C4 numbers are a readout of that decision, not the criterion for it. Two-SE intervals are descriptive and do not account for calibration-draw variability. A count rule validated here still requires confirmation on the untouched literature, science, government and WikiText families.
