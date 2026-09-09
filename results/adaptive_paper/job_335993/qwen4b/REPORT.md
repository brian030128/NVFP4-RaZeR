# Paper-aligned adaptive tile count: qwen4b

Counts are chosen by the lowest actual loss on 128 OpenWebMath and CodeParrot documents disjoint from the 128 scoring documents. WikiText-2 and C4 are never consulted for the choice. Evaluation is the released 2048-token protocol. 66856 of 7096320 tiles are eligible; re-election at 256 reproduces the frozen fixed256_math_code128 map bitwise.

| Count | Selected | Selection NLL |
|---|---:|---:|
| n256 | 256 | 1.441248 |
| n1024 | 1,024 | 1.417720 |
| n4096 | 4,096 | 1.394362 |
| n16384 | 16,384 | 1.382328 |
| n65536 | 65,536 | 1.369785 |
| n_all | 66,856 | 1.370014 |

Chosen: **n65536** (65,536 tiles).

| Policy | E0M3 blocks | WikiText-2 | C4 |
|---|---:|---:|---:|
| FourOverSix baseline | 0 | 14.269062 | 17.326633 |
| MixFP4 fixed 256 | 256 | 13.040957 | 16.615953 |
| MixFP4 adaptive (n65536) | 65536 | 10.864750 | 15.161474 |

The count is selected on calibration-source documents only; these perplexities are a readout of that decision, not the criterion for it. Two-SE intervals in the machine-readable report are descriptive and do not cover calibration-draw or selection-draw variability.
