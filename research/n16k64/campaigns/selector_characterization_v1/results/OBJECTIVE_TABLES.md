# Objective comparison: observed calibration-draw variation

N16K64, k=3; delta NLL versus the paired FourOverSix baseline (negative is favorable).
SD is the sample SD across available calibration draws, not a confidence interval.
Worst means the largest observed delta NLL; regression means delta NLL > 0.
Five draws and two corpora are not independent population tail-risk replications.
Pointwise paired-cluster intervals are in `quality_results.csv` and `quality_contrasts.csv`;
this table does not replace those intervals or imply simultaneous coverage.

## llama8b

| Corpus | Policy | Draws / 5 | Mean delta NLL | Sample SD | Minimum | Observed worst | Regressions |
|---|---|---:|---:|---:|---:|---:|---:|
| wiki | ce_natural | 5 | -0.005586 | 0.000394 | -0.006208 | -0.005240 | 0/5 |
| wiki | kl_natural | 5 | +0.013030 | 0.005053 | +0.008487 | +0.021219 | 5/5 |
| wiki | joint | 5 | -0.004713 | 0.000578 | -0.005276 | -0.003933 | 0/5 |
| wiki | ce_matched | 5 | -0.004978 | 0.000633 | -0.005869 | -0.004266 | 0/5 |
| wiki | kl_matched | 5 | -0.005210 | 0.000765 | -0.006521 | -0.004636 | 0/5 |
| c4 | ce_natural | 5 | -0.005684 | 0.000564 | -0.006563 | -0.005127 | 0/5 |
| c4 | kl_natural | 5 | +0.010613 | 0.004172 | +0.006283 | +0.017499 | 5/5 |
| c4 | joint | 5 | -0.005101 | 0.000582 | -0.006104 | -0.004593 | 0/5 |
| c4 | ce_matched | 5 | -0.004886 | 0.000756 | -0.005826 | -0.003998 | 0/5 |
| c4 | kl_matched | 5 | -0.005855 | 0.000348 | -0.006416 | -0.005520 | 0/5 |

## qwen4b

| Corpus | Policy | Draws / 5 | Mean delta NLL | Sample SD | Minimum | Observed worst | Regressions |
|---|---|---:|---:|---:|---:|---:|---:|
| wiki | ce_natural | 5 | -0.249439 | 0.009894 | -0.256944 | -0.232962 | 0/5 |
| wiki | kl_natural | 5 | -0.137575 | 0.023514 | -0.169276 | -0.107662 | 0/5 |
| wiki | joint | 5 | -0.144283 | 0.017708 | -0.167815 | -0.124670 | 0/5 |
| wiki | ce_matched | 5 | -0.237355 | 0.010777 | -0.246304 | -0.220412 | 0/5 |
| wiki | kl_matched | 5 | -0.123433 | 0.021371 | -0.152637 | -0.097847 | 0/5 |
| c4 | ce_natural | 5 | -0.102725 | 0.008277 | -0.111924 | -0.089739 | 0/5 |
| c4 | kl_natural | 5 | -0.057864 | 0.012708 | -0.073765 | -0.041804 | 0/5 |
| c4 | joint | 5 | -0.074033 | 0.008037 | -0.084134 | -0.063191 | 0/5 |
| c4 | ce_matched | 5 | -0.128428 | 0.004440 | -0.132020 | -0.121146 | 0/5 |
| c4 | kl_matched | 5 | -0.053965 | 0.011932 | -0.069121 | -0.039733 | 0/5 |

## mistral7b

| Corpus | Policy | Draws / 5 | Mean delta NLL | Sample SD | Minimum | Observed worst | Regressions |
|---|---|---:|---:|---:|---:|---:|---:|
| wiki | ce_natural | 5 | -0.003859 | 0.000592 | -0.004601 | -0.003106 | 0/5 |
| wiki | kl_natural | 5 | -0.005524 | 0.000553 | -0.006099 | -0.004962 | 0/5 |
| wiki | joint | 5 | -0.003791 | 0.000410 | -0.004403 | -0.003284 | 0/5 |
| wiki | ce_matched | 5 | -0.003875 | 0.000442 | -0.004645 | -0.003585 | 0/5 |
| wiki | kl_matched | 5 | -0.004838 | 0.000379 | -0.005404 | -0.004360 | 0/5 |
| c4 | ce_natural | 5 | -0.002391 | 0.000251 | -0.002690 | -0.002121 | 0/5 |
| c4 | kl_natural | 5 | -0.002876 | 0.000569 | -0.003300 | -0.001948 | 0/5 |
| c4 | joint | 5 | -0.002300 | 0.000257 | -0.002470 | -0.001844 | 0/5 |
| c4 | ce_matched | 5 | -0.002059 | 0.000089 | -0.002189 | -0.001956 | 0/5 |
| c4 | kl_matched | 5 | -0.002671 | 0.000348 | -0.003195 | -0.002274 | 0/5 |

