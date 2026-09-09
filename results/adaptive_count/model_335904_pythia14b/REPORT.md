# Calibration-chosen tile count: pythia14b

Counts are chosen by the lowest actual next-token loss over calibration-source documents; the held-out C4 set is never consulted for selection. `sel` is 192 documents disjoint from the 192 scoring documents; `fit` is the scoring set itself. 76665 of 2359296 tiles are eligible.

| Count | Selected | fit NLL | sel NLL |
|---|---:|---:|---:|
| four_over_six | 0 | 2.241906 | 2.323230 |
| n16 | 16 | 2.242456 | 2.321421 |
| n64 | 64 | 2.237567 | 2.318272 |
| n256 | 256 | 2.228961 | 2.307298 |
| n1024 | 1024 | 2.218643 | 2.300231 |
| n4096 | 4096 | 2.218370 | 2.296920 |
| n16384 | 16384 | 2.249563 | 2.334991 |
| n65536 | 65536 | 2.324243 | 2.423188 |
| n_all | 76665 | 2.332491 | 2.432618 |

| Policy | Count | C4 PPL | ΔPPL vs FourOverSix | ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| FourOverSix | 0 | 20.471159 | — | — |
| n256 | 256 | 20.162666 | -0.308493 | -0.015184 ±0.003167 |
| n4096 | 4096 | 19.890992 | -0.580167 | -0.028750 ±0.003516 |

Chosen on `sel`: n4096 (4096 tiles). Against the fixed 256 map: ΔPPL -0.271675, ΔNLL -0.013566 ±0.003427.

Chosen on `fit`: n4096 (4096 tiles). Against the fixed 256 map: ΔPPL -0.271675, ΔNLL -0.013566 ±0.003427.

The count is selected on calibration-source documents only. These held-out C4 numbers are a readout of that decision, not the criterion for it. Two-SE intervals are descriptive and do not account for calibration-draw variability. A count rule validated here still requires confirmation on the untouched literature, science, government and WikiText families.
