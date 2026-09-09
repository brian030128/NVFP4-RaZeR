# Calibration-chosen tile count: olmo1b

Counts are chosen by the lowest actual next-token loss over calibration-source documents; the held-out C4 set is never consulted for selection. `sel` is 192 documents disjoint from the 192 scoring documents; `fit` is the scoring set itself. 30485 of 2097152 tiles are eligible.

| Count | Selected | fit NLL | sel NLL |
|---|---:|---:|---:|
| four_over_six | 0 | 2.168743 | 2.229398 |
| n16 | 16 | 2.166310 | 2.227722 |
| n64 | 64 | 2.164483 | 2.225820 |
| n256 | 256 | 2.162278 | 2.225050 |
| n1024 | 1024 | 2.159649 | 2.223598 |
| n4096 | 4096 | 2.154872 | 2.222157 |
| n16384 | 16384 | 2.141416 | 2.224059 |
| n65536 | 30485 | 2.133552 | 2.225029 |
| n_all | 30485 | 2.133552 | 2.225029 |

| Policy | Count | C4 PPL | ΔPPL vs FourOverSix | ΔNLL ±2SE |
|---|---:|---:|---:|---:|
| FourOverSix | 0 | 14.984432 | — | — |
| n256 | 256 | 14.883008 | -0.101424 | -0.006792 ±0.002574 |
| n4096 | 4096 | 14.844247 | -0.140184 | -0.009399 ±0.003269 |
| n65536 | 30485 | 14.854211 | -0.130221 | -0.008728 ±0.003658 |

Chosen on `sel`: n4096 (4096 tiles). Against the fixed 256 map: ΔPPL -0.038761, ΔNLL -0.002608 ±0.002394.

Chosen on `fit`: n65536 (30485 tiles). Against the fixed 256 map: ΔPPL -0.028797, ΔNLL -0.001937 ±0.002685.

The count is selected on calibration-source documents only. These held-out C4 numbers are a readout of that decision, not the criterion for it. Two-SE intervals are descriptive and do not account for calibration-draw variability. A count rule validated here still requires confirmation on the untouched literature, science, government and WikiText families.
