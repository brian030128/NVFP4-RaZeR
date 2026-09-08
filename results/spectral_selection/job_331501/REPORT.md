# Synthetic mechanism results

These are small synthetic matrices, not evidence of LLM or domain generalization.

| Family / seed | Spectral / baseline | MSE policy spectral / baseline | Solver / oracle − 1 | Frobenius / baseline |
|---|---:|---:|---:|---:|
| gaussian / 101 | 0.851576 | 0.874560 | 0.000000 | 0.952359 |
| heavy_tail / 101 | 0.997317 | 1.000000 | 0.000000 | 1.108892 |
| column_outliers / 101 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| row_correlated / 101 | 0.947079 | 0.990152 | 0.048276 | 0.979677 |
| gaussian / 102 | 0.870702 | 0.927638 | 0.000000 | 0.953919 |
| heavy_tail / 102 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| column_outliers / 102 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| row_correlated / 102 | 0.783176 | 1.000000 | 0.000000 | 1.065071 |
| gaussian / 103 | 0.784535 | 0.838202 | 0.026430 | 0.976563 |
| heavy_tail / 103 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| column_outliers / 103 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| row_correlated / 103 | 0.864224 | 0.967292 | 0.001583 | 0.986792 |

All maps, shifted-input output errors, hashes, and solver traces are in report.json.
The exhaustive optimum is an objective diagnostic; it does not select an exported policy.
