# Cross-domain E0M3/E2M1 selection experiments

All changes are relative to a matched FourOverSix W4A4 baseline. Negative paired NLL or relative perplexity change is better. All candidate maps were frozen before held-out evaluation. Rejected exports retain the baseline.

Completed experiment runs: 6/6 (three models; teacher follow-ups reuse two panel models/seeds). Target math/code stress complete: True.

The rules share candidate formats, type tile 8x64, 64 fitting windows, a 0.1 predicted-loss budget in nats/token, fit backtracking and independent NLL validation. Mixed pools 32 WikiText + 32 C4 windows; consensus requires stable negative scores and measured joint improvement in each domain separately.

Teacher follow-ups replace the scoring/fitting objective with full-precision teacher KL, while independent acceptance still checks actual NLL. Their fitting diagnostics motivated this prespecified follow-up; see TEACHER_PROTOCOL.md.

## Candidate performance

Absolute PPL change from mean example NLL. A reduction of 0.01 PPL is a worthwhile gain under the user-specified criterion. Brackets are paired mean ± two SE transformed with exp and scaled by baseline PPL; descriptive, not simultaneous confidence bounds. Practical magnitude and statistical uncertainty are reported separately; calibration gates remain as prespecified.

| Model / seed | Rule | Tiles | Export accepted | WikiText ΔPPL | C4 ΔPPL | Math text ΔPPL | Code text ΔPPL |
|---|---|---:|---|---:|---:|---:|---:|
| Qwen3.8-27B / 20260912 | wiki | 226 | True | -0.3684 [-0.4424, -0.2936] | +0.0028 [-0.0058, +0.0114] | -0.0489 [-0.0645, -0.0332] | -0.0166 [-0.0432, +0.0102] |
| Qwen3.8-27B / 20260912 | c4 | 905 | True | -0.1094 [-0.1479, -0.0707] | -0.0500 [-0.0589, -0.0410] | -0.0343 [-0.0488, -0.0197] | -0.0462 [-0.0700, -0.0223] |
| Qwen3.8-27B / 20260912 | mixed | 728 | True | -0.4509 [-0.5351, -0.3657] | -0.0026 [-0.0113, +0.0061] | -0.0665 [-0.0843, -0.0486] | -0.0398 [-0.0666, -0.0127] |
| Qwen3.8-27B / 20260912 | consensus | 1,227 | True | -0.1781 [-0.2120, -0.1442] | -0.0209 [-0.0298, -0.0120] | -0.0469 [-0.0642, -0.0295] | -0.0352 [-0.0633, -0.0069] |
| Qwen3.8-27B / 20260913 | wiki | 327 | True | -0.4088 [-0.4888, -0.3278] | +0.0064 [-0.0025, +0.0152] | -0.0308 [-0.0461, -0.0153] | -0.0417 [-0.0715, -0.0118] |
| Qwen3.8-27B / 20260913 | c4 | 864 | True | -0.0775 [-0.1192, -0.0355] | -0.0532 [-0.0623, -0.0441] | -0.0199 [-0.0356, -0.0041] | -0.0352 [-0.0643, -0.0060] |
| Qwen3.8-27B / 20260913 | mixed | 360 | True | -0.4183 [-0.4967, -0.3390] | +0.0004 [-0.0081, +0.0088] | -0.0280 [-0.0423, -0.0137] | -0.0304 [-0.0590, -0.0017] |
| Qwen3.8-27B / 20260913 | consensus | 1,247 | False | -0.2092 [-0.2478, -0.1705] | -0.0353 [-0.0446, -0.0261] | -0.0453 [-0.0626, -0.0280] | -0.0135 [-0.0407, +0.0139] |
| qwen3-4b / 20260918 | wiki | 23 | True | -1.0892 [-1.1656, -1.0123] | -0.4941 [-0.5308, -0.4574] | -0.3119 [-0.3938, -0.2286] | -0.1950 [-0.3137, -0.0747] |
| qwen3-4b / 20260918 | c4 | 125 | True | -1.9695 [-2.0632, -1.8750] | -1.1781 [-1.2369, -1.1191] | -0.4542 [-0.5353, -0.3718] | -0.5216 [-0.6624, -0.3787] |
| qwen3-4b / 20260918 | mixed | 53 | True | -1.5200 [-1.6036, -1.4358] | -0.7664 [-0.8130, -0.7197] | -0.3314 [-0.4166, -0.2447] | -0.2323 [-0.3703, -0.0922] |
| qwen3-4b / 20260918 | consensus | 143 | True | -2.0826 [-2.1762, -1.9884] | -1.1894 [-1.2482, -1.1303] | -0.4308 [-0.5124, -0.3479] | -0.3530 [-0.4992, -0.2045] |
| llama-3.1-8b-local / 20260918 | wiki | 2,239 | True | -0.1036 [-0.1164, -0.0908] | -0.0434 [-0.0635, -0.0232] | -0.0653 [-0.1054, -0.0249] | +0.1078 [+0.0404, +0.1760] |
| llama-3.1-8b-local / 20260918 | c4 | 2,924 | True | -0.0452 [-0.0582, -0.0321] | -0.0546 [-0.0753, -0.0339] | -0.0342 [-0.0728, +0.0046] | +0.0103 [-0.0565, +0.0778] |
| llama-3.1-8b-local / 20260918 | mixed | 1,188 | True | -0.0791 [-0.0903, -0.0679] | -0.0484 [-0.0647, -0.0321] | -0.0541 [-0.0924, -0.0157] | +0.0586 [-0.0125, +0.1306] |
| llama-3.1-8b-local / 20260918 | consensus | 17,835 | False | -0.0896 [-0.1005, -0.0786] | -0.0439 [-0.0595, -0.0282] | -0.0273 [-0.0623, +0.0079] | +0.0095 [-0.0572, +0.0770] |
| qwen3-4b / 20260918 | teacher_mixed | 347 | True | -1.4994 [-1.5717, -1.4268] | -0.6138 [-0.6591, -0.5684] | -0.2583 [-0.3431, -0.1721] | -0.1506 [-0.2902, -0.0089] |
| qwen3-4b / 20260918 | teacher_consensus | 191 | True | -0.6940 [-0.7647, -0.6231] | -0.2804 [-0.3152, -0.2456] | -0.0433 [-0.1361, +0.0511] | +0.1850 [+0.0298, +0.3426] |
| llama-3.1-8b-local / 20260918 | teacher_mixed | 26 | False | -0.0230 [-0.0344, -0.0115] | -0.0311 [-0.0445, -0.0177] | -0.0224 [-0.0648, +0.0203] | +0.0468 [-0.0200, +0.1144] |
| llama-3.1-8b-local / 20260918 | teacher_consensus | 170 | True | -0.0417 [-0.0536, -0.0298] | -0.0311 [-0.0459, -0.0162] | -0.0502 [-0.0805, -0.0197] | +0.0187 [-0.0542, +0.0925] |

## Cross-domain rule checks

Counts below describe measured model/seed/domain cells. Repeated seeds use the same held-out text and are not independent domain replications.

| Rule | Candidate cells | Gains ≥0.01 PPL | Supported gains | Supported harms | Accepted exports | Worst accepted ΔPPL |
|---|---:|---:|---:|---:|---:|---:|
| wiki | 16 | 13 | 12 | 1 | 4/4 | +0.1078 |
| c4 | 16 | 15 | 14 | 0 | 4/4 | +0.0103 |
| mixed | 16 | 13 | 13 | 0 | 4/4 | +0.0586 |
| consensus | 16 | 15 | 13 | 0 | 2/4 | -0.0209 |
| teacher_mixed | 8 | 7 | 6 | 0 | 1/2 | -0.1506 |
| teacher_consensus | 8 | 6 | 5 | 1 | 2/2 | +0.1850 |

## Fit and validation

| Model / seed | Rule | Attempts | Last budget | Validation WikiText ΔNLL ± 2SE | Validation C4 ΔNLL ± 2SE |
|---|---|---:|---:|---:|---:|
| Qwen/Qwen3.8-27B / 20260912 | wiki | 1 | 0.100000 | -0.039138 ± 0.023478 | +0.000578 ± 0.002368 |
| Qwen/Qwen3.8-27B / 20260912 | c4 | 4 | 0.012500 | -0.017404 ± 0.016944 | -0.007188 ± 0.003862 |
| Qwen/Qwen3.8-27B / 20260912 | mixed | 1 | 0.100000 | -0.067102 ± 0.049697 | -0.002752 ± 0.004560 |
| Qwen/Qwen3.8-27B / 20260912 | consensus | 4 | 0.012500 | -0.026758 ± 0.023195 | -0.006144 ± 0.004327 |
| Qwen/Qwen3.8-27B / 20260913 | wiki | 1 | 0.100000 | -0.074055 ± 0.040663 | +0.000781 ± 0.003448 |
| Qwen/Qwen3.8-27B / 20260913 | c4 | 4 | 0.012500 | +0.000459 ± 0.020428 | -0.005253 ± 0.002877 |
| Qwen/Qwen3.8-27B / 20260913 | mixed | 2 | 0.050000 | -0.115550 ± 0.080801 | -0.000556 ± 0.004082 |
| Qwen/Qwen3.8-27B / 20260913 | consensus | 4 | 0.012500 | -0.024090 ± 0.030626 | -0.004236 ± 0.004602 |
| qwen3-4b / 20260918 | wiki | 1 | 0.100000 | -0.064275 ± 0.011482 | -0.034794 ± 0.006338 |
| qwen3-4b / 20260918 | c4 | 1 | 0.100000 | -0.119527 ± 0.014865 | -0.077364 ± 0.006825 |
| qwen3-4b / 20260918 | mixed | 1 | 0.100000 | -0.094952 ± 0.017891 | -0.050103 ± 0.012092 |
| qwen3-4b / 20260918 | consensus | 1 | 0.100000 | -0.130374 ± 0.023954 | -0.088647 ± 0.017351 |
| llama-3.1-8b-local / 20260918 | wiki | 2 | 0.050000 | -0.014865 ± 0.005858 | -0.002426 ± 0.004704 |
| llama-3.1-8b-local / 20260918 | c4 | 3 | 0.025000 | -0.006198 ± 0.005984 | -0.003985 ± 0.003485 |
| llama-3.1-8b-local / 20260918 | mixed | 3 | 0.025000 | -0.008893 ± 0.008591 | -0.004001 ± 0.004815 |
| llama-3.1-8b-local / 20260918 | consensus | 1 | 0.100000 | -0.013234 ± 0.007507 | -0.005237 ± 0.006232 |
| qwen3-4b / 20260918 | teacher_mixed | 3 | 0.025000 | -0.091954 ± 0.012202 | -0.036789 ± 0.010171 |
| qwen3-4b / 20260918 | teacher_consensus | 4 | 0.012500 | -0.045823 ± 0.007391 | -0.021230 ± 0.009869 |
| llama-3.1-8b-local / 20260918 | teacher_mixed | 5 | 0.006250 | -0.004106 ± 0.008695 | -0.004645 ± 0.005951 |
| llama-3.1-8b-local / 20260918 | teacher_consensus | 5 | 0.006250 | -0.008452 ± 0.007210 | -0.006750 ± 0.005851 |

## Isolated tile diagnostic

Tiles were selected by prespecified score extremes/conflicts on fitting data. Measured effects use eight reserved training probes per domain. This selected sample cannot estimate all-tile error rates. “Stable wrong sign” requires predicted and measured two-SE intervals to exclude zero in opposite directions.

| Seed | Domain | Tiles | Point-sign agreement | Stable wrong sign |
|---|---|---:|---:|---:|
| 20260912 | wiki | 15 | 9/15 | 0 |
| 20260912 | c4 | 15 | 6/15 | 0 |
| 20260913 | wiki | 15 | 8/15 | 0 |
| 20260913 | c4 | 15 | 7/15 | 0 |

## Direct paired policy contrasts

Candidate A minus candidate B in NLL; negative favors A. These comparisons include rejected candidates and do not choose exports.

| Model / seed | A minus B | WikiText ΔNLL ± 2SE | C4 ΔNLL ± 2SE |
|---|---|---:|---:|
| Qwen/Qwen3.8-27B / 20260912 | c4 − wiki | +0.034767 ± 0.009043 | -0.005170 ± 0.000919 |
| Qwen/Qwen3.8-27B / 20260912 | mixed − wiki | -0.011334 ± 0.002318 | -0.000524 ± 0.000827 |
| Qwen/Qwen3.8-27B / 20260912 | consensus − mixed | +0.036990 ± 0.009138 | -0.001796 ± 0.000821 |
| Qwen/Qwen3.8-27B / 20260913 | c4 − wiki | +0.044502 ± 0.009756 | -0.005837 ± 0.000916 |
| Qwen/Qwen3.8-27B / 20260913 | mixed − wiki | -0.001307 ± 0.001429 | -0.000586 ± 0.000824 |
| Qwen/Qwen3.8-27B / 20260913 | consensus − mixed | +0.028351 ± 0.007364 | -0.003499 ± 0.000879 |
| qwen3-4b / 20260918 | c4 − wiki | -0.064342 ± 0.003612 | -0.040283 ± 0.002088 |
| qwen3-4b / 20260918 | mixed − wiki | -0.030972 ± 0.002577 | -0.015841 ± 0.001468 |
| qwen3-4b / 20260918 | consensus − mixed | -0.041950 ± 0.003215 | -0.025116 ± 0.001535 |
| llama-3.1-8b-local / 20260918 | c4 − wiki | +0.008295 ± 0.002071 | -0.001194 ± 0.001247 |
| llama-3.1-8b-local / 20260918 | mixed − wiki | +0.003487 ± 0.001733 | -0.000534 ± 0.001890 |
| llama-3.1-8b-local / 20260918 | consensus − mixed | -0.001484 ± 0.001730 | +0.000480 ± 0.001547 |
| qwen3-4b / 20260918 | teacher_consensus − teacher_mixed | +0.057056 ± 0.004026 | +0.019193 ± 0.001936 |
| qwen3-4b / 20260918 | teacher_mixed − mixed | +0.001502 ± 0.003844 | +0.008908 ± 0.001704 |
| qwen3-4b / 20260918 | teacher_consensus − consensus | +0.100507 ± 0.005717 | +0.053218 ± 0.002927 |
| llama-3.1-8b-local / 20260918 | teacher_consensus − teacher_mixed | -0.002641 ± 0.001633 | +0.000004 ± 0.001553 |
| llama-3.1-8b-local / 20260918 | teacher_mixed − mixed | +0.007942 ± 0.001756 | +0.001836 ± 0.001775 |
| llama-3.1-8b-local / 20260918 | teacher_consensus − consensus | +0.006785 ± 0.001646 | +0.001360 ± 0.001621 |

## Map overlap

| Model / seed | Maps | Shared tiles | Union tiles | Jaccard |
|---|---|---:|---:|---:|
| Qwen/Qwen3.8-27B / 20260912 | wiki / c4 | 10 | 1,121 | 0.0089 |
| Qwen/Qwen3.8-27B / 20260912 | mixed / wiki | 222 | 732 | 0.3033 |
| Qwen/Qwen3.8-27B / 20260912 | consensus / mixed | 25 | 1,930 | 0.0130 |
| Qwen/Qwen3.8-27B / 20260913 | wiki / c4 | 14 | 1,177 | 0.0119 |
| Qwen/Qwen3.8-27B / 20260913 | mixed / wiki | 275 | 412 | 0.6675 |
| Qwen/Qwen3.8-27B / 20260913 | consensus / mixed | 22 | 1,585 | 0.0139 |
| qwen3-4b / 20260918 | wiki / c4 | 22 | 126 | 0.1746 |
| qwen3-4b / 20260918 | mixed / wiki | 23 | 53 | 0.4340 |
| qwen3-4b / 20260918 | consensus / mixed | 52 | 144 | 0.3611 |
| llama-3.1-8b-local / 20260918 | wiki / c4 | 362 | 4,801 | 0.0754 |
| llama-3.1-8b-local / 20260918 | mixed / wiki | 929 | 2,498 | 0.3719 |
| llama-3.1-8b-local / 20260918 | consensus / mixed | 251 | 18,772 | 0.0134 |
| qwen3-4b / 20260918 | teacher_consensus / teacher_mixed | 179 | 359 | 0.4986 |
| llama-3.1-8b-local / 20260918 | teacher_consensus / teacher_mixed | 15 | 181 | 0.0829 |

## Agreement of fitted domain scores

CE rows use 64 windows per individual domain; teacher rows use 32. Consensus selection uses 32 per domain to keep its total budget at 64. The matching 32-window CE/teacher comparison is below. “Stable” means the descriptive two-SE sign check, without correction for millions of comparisons.

| Run | N/domain | Cross-domain cosine | C4 split cosine | Wiki noise ratio | C4 noise ratio | Both stable negative |
|---|---:|---:|---:|---:|---:|---:|
| seed20260912 | 64 | 0.03794 | 0.21933 | 0.2037 | 0.6462 | 259,112 |
| seed20260913 | 64 | 0.03334 | 0.21765 | 0.1610 | 0.6443 | 271,293 |
| panel/qwen3-4b | 64 | 0.79522 | 0.96072 | 0.0188 | 0.0231 | 514,654 |
| panel/llama-3.1-8b-local | 64 | 0.27648 | 0.20408 | 0.2578 | 0.6684 | 30,572 |
| teacher/qwen3-4b | 32 | 0.34968 | undefined | 0.2634 | 0.4957 | 45,825 |
| teacher/llama-3.1-8b-local | 32 | 0.31264 | undefined | 0.2599 | 0.7612 | 32,726 |

### Matching 32-window objective comparison

Noise ratio is sum(SE squared)/sum(mean squared), a descriptive estimate of noise relative to observed mean-score energy, not a guaranteed signal fraction.

| Model | Objective | Cross-domain cosine | Wiki noise ratio | C4 noise ratio |
|---|---|---:|---:|---:|
| qwen3-4b | observed_nll | 0.77479 | 0.0527 | 0.0507 |
| llama-3.1-8b-local | observed_nll | 0.15856 | 0.3936 | 0.8491 |
| qwen3-4b | teacher_kl | 0.34968 | 0.2634 | 0.4957 |
| llama-3.1-8b-local | teacher_kl | 0.31264 | 0.2599 | 0.7612 |

## Interpretation limits

- A successful common calibration algorithm is different from a fixed domain-independent tile map. The tested maps use WikiText/C4 task gradients.
- Worst-domain fit/validation is checked only on WikiText and C4. Math/code are uncalibrated stress tests, not guaranteed by those gates.
- Math/code report full-reference-text language-model loss, not reasoning accuracy, answer-only loss, pass@k, or causal decoding performance. Variable-length examples are weighted equally in the main table; raw reports also contain token-weighted perplexity.
- Windows and related models are not independent universal certification samples. Two-SE checks and sparse finite-step budgets are heuristics.
- C4 train/test documents are hash-disjoint; WikiText splits are official train/validation. Dataset revisions and token hashes are in raw reports.
- C4 length qualification and WikiText window boundaries depend on the tokenizer. Sampling seeds/splits are shared across models; exact C4 document selections can differ. Every within-model policy comparison uses identical examples.
- Native Transformers panel baselines must not be compared as if identical to older copied-model implementations. All numbers here use matched native baselines.

See PROTOCOL.md, ROBUST_EXTENSION.md, STRESS_PROTOCOL.md, PANEL_PROTOCOL.md TEACHER_PROTOCOL.md and RUN_NOTES.md for the frozen design and execution changes.
