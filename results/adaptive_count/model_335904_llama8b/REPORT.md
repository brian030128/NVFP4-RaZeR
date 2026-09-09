# Calibration-chosen tile count: llama8b

Counts are chosen by the lowest actual next-token loss over calibration-source documents; the held-out C4 set is never consulted for selection. `sel` is 192 documents disjoint from the 192 scoring documents; `fit` is the scoring set itself. 97908 of 13631488 tiles are eligible.

| Count | Selected | fit NLL | sel NLL |
|---|---:|---:|---:|
| four_over_six | 0 | 1.822935 | 1.851211 |
| n16 | 16 | 1.821200 | 1.846167 |
| n64 | 64 | 1.816822 | 1.845296 |
| n256 | 256 | 1.814562 | 1.842657 |
| n1024 | 1024 | 1.811138 | 1.842193 |
| n4096 | 4096 | 1.811476 | 1.842593 |
| n16384 | 16384 | 1.808276 | 1.846824 |
| n65536 | 65536 | 1.799344 | 1.851774 |
| n_all | 97908 | 1.794580 | 1.855587 |

| Policy | Count | C4 PPL | ΔPPL vs FourOverSix | ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| FourOverSix | 0 | 11.600109 | — | — |
| n256 | 256 | 11.509863 | -0.090246 | -0.007810 ±0.002113 |
| n1024 | 1024 | 11.509227 | -0.090881 | -0.007865 ±0.002663 |
| n_all | 97908 | 11.647632 | +0.047524 | +0.004088 ±0.003532 |

Chosen on `sel`: n1024 (1024 tiles). Against the fixed 256 map: ΔPPL -0.000635, ΔNLL -0.000055 ±0.002404.

Chosen on `fit`: n_all (97908 tiles). Against the fixed 256 map: ΔPPL +0.137770, ΔNLL +0.011899 ±0.003280.

The count is selected on calibration-source documents only. These held-out C4 numbers are a readout of that decision, not the criterion for it. Two-SE intervals are descriptive and do not account for calibration-draw variability. A count rule validated here still requires confirmation on the untouched literature, science, government and WikiText families.
