# Paper-aligned adaptive tile count: llama8b

Counts are chosen by the lowest actual loss on 128 OpenWebMath and CodeParrot documents disjoint from the 128 scoring documents. WikiText-2 and C4 are never consulted for the choice. Evaluation is the released 2048-token protocol. 99024 of 13631488 tiles are eligible; re-election at 256 reproduces the frozen fixed256_math_code128 map bitwise.

| Count | Selected | Selection NLL |
|---|---:|---:|
| n256 | 256 | 1.198041 |
| n1024 | 1,024 | 1.196060 |
| n4096 | 4,096 | 1.197289 |
| n16384 | 16,384 | 1.199099 |
| n65536 | 65,536 | 1.204372 |
| n_all | 99,024 | 1.206830 |

Chosen: **n1024** (1,024 tiles).

| Policy | E0M3 blocks | WikiText-2 | C4 |
|---|---:|---:|---:|
| FourOverSix baseline | 0 | 6.875525 | 9.823733 |
| MixFP4 fixed 256 | 256 | 6.848383 | 9.764387 |
| MixFP4 adaptive (n1024) | 1024 | 6.838759 | 9.773240 |

The count is selected on calibration-source documents only; these perplexities are a readout of that decision, not the criterion for it. Two-SE intervals in the machine-readable report are descriptive and do not cover calibration-draw or selection-draw variability.
