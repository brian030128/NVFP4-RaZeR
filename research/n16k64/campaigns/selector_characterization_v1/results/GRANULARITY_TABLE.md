# Measured granularity curve: frozen seed0, joint k=3

Same candidates/scales; floating-point fake-quant evaluation. ΔNLL is policy minus FourOverSix; relative PPL is exp(ΔNLL)−1. Intervals are 2,000 paired natural-cluster bootstrap pointwise 95% intervals, not simultaneous or multiplicity-adjusted. Missing is not zero.

## llama8b

| Corpus | N | Status | PPL | ΔNLL [pointwise 95% CI] | Relative PPL (%) | N8 gain retained (%) |
|---|---:|---|---:|---|---:|---:|
| wiki | 8 | validated_reuse | 6.841386 | -0.004960 [-0.006504, -0.003656] | -0.4947 | 100.00 |
| wiki | 16 | validated_reuse | 6.843237 | -0.004689 [-0.006312, -0.003183] | -0.4678 | 94.54 |
| wiki | 32 | blocked_retry_limit | missing | missing | missing | missing |
| wiki | 64 | blocked_retry_limit | missing | missing | missing | missing |
| wiki | 128 | blocked_retry_limit | missing | missing | missing | missing |
| wiki | 256 | blocked_retry_limit | missing | missing | missing | missing |
| c4 | 8 | validated_reuse | 9.776266 | -0.005010 [-0.007249, -0.003136] | -0.4997 | 100.00 |
| c4 | 16 | validated_reuse | 9.776162 | -0.005021 [-0.006687, -0.003489] | -0.5008 | 100.21 |
| c4 | 32 | blocked_retry_limit | missing | missing | missing | missing |
| c4 | 64 | blocked_retry_limit | missing | missing | missing | missing |
| c4 | 128 | blocked_retry_limit | missing | missing | missing | missing |
| c4 | 256 | blocked_retry_limit | missing | missing | missing | missing |

## qwen4b

| Corpus | N | Status | PPL | ΔNLL [pointwise 95% CI] | Relative PPL (%) | N8 gain retained (%) |
|---|---:|---|---:|---|---:|---:|
| wiki | 8 | validated_reuse | 11.884062 | -0.178481 [-0.188471, -0.169181] | -16.3460 | 100.00 |
| wiki | 16 | validated_reuse | 12.196773 | -0.152508 [-0.162206, -0.143525] | -14.1448 | 85.45 |
| wiki | 32 | validated_new | 12.499136 | -0.128020 [-0.135324, -0.121180] | -12.0164 | 71.73 |
| wiki | 64 | validated_new | 12.757726 | -0.107542 [-0.114166, -0.101239] | -10.1961 | 60.25 |
| wiki | 128 | validated_new | 12.810207 | -0.103437 [-0.110370, -0.096850] | -9.8267 | 57.95 |
| wiki | 256 | validated_new | 13.240423 | -0.070405 [-0.076425, -0.064775] | -6.7984 | 39.45 |
| c4 | 8 | validated_reuse | 15.848281 | -0.087784 [-0.091609, -0.083981] | -8.4042 | 100.00 |
| c4 | 16 | validated_reuse | 16.048518 | -0.075229 [-0.078348, -0.072119] | -7.2469 | 85.70 |
| c4 | 32 | validated_new | 16.236142 | -0.063606 [-0.066556, -0.060760] | -6.1625 | 72.46 |
| c4 | 64 | validated_new | 16.357369 | -0.056167 [-0.058698, -0.053647] | -5.4619 | 63.98 |
| c4 | 128 | validated_new | 16.349494 | -0.056648 [-0.059244, -0.054049] | -5.5074 | 64.53 |
| c4 | 256 | validated_new | 16.576315 | -0.042871 [-0.045025, -0.040732] | -4.1965 | 48.84 |

## mistral7b

| Corpus | N | Status | PPL | ΔNLL [pointwise 95% CI] | Relative PPL (%) | N8 gain retained (%) |
|---|---:|---|---:|---|---:|---:|
| wiki | 8 | validated_reuse | 5.498714 | -0.004463 [-0.005410, -0.003533] | -0.4453 | 100.00 |
| wiki | 16 | validated_reuse | 5.501871 | -0.003889 [-0.004879, -0.002975] | -0.3881 | 87.14 |
| wiki | 32 | blocked_retry_limit | missing | missing | missing | missing |
| wiki | 64 | blocked_retry_limit | missing | missing | missing | missing |
| wiki | 128 | blocked_retry_limit | missing | missing | missing | missing |
| wiki | 256 | blocked_retry_limit | missing | missing | missing | missing |
| c4 | 8 | validated_reuse | 8.045567 | -0.002711 [-0.003450, -0.001955] | -0.2708 | 100.00 |
| c4 | 16 | validated_reuse | 8.047506 | -0.002470 [-0.003659, -0.001561] | -0.2467 | 91.11 |
| c4 | 32 | blocked_retry_limit | missing | missing | missing | missing |
| c4 | 64 | blocked_retry_limit | missing | missing | missing | missing |
| c4 | 128 | blocked_retry_limit | missing | missing | missing | missing |
| c4 | 256 | blocked_retry_limit | missing | missing | missing | missing |

