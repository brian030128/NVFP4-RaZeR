# Completed smaller-model results — full study still pending

Calibration: OpenWebMath and CodeParrot only; evaluation: WikiText-2 and C4 only. One shared causal score pass per model; no calibration-seed replication. All ten settings and both count rules are retained. This file does not represent completion of the three-model study.

Qwen3-4B baseline: Wiki 19.414897; C4 21.928243.

### Qwen3-4B

| Calibration | Sequences | Adaptive E0M3 blocks | Adaptive Wiki PPL | Adaptive C4 PPL | Adaptive average | Fixed E0M3 blocks | Fixed Wiki PPL | Fixed C4 PPL | Fixed average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| math16 | 16 | 0 | 19.414897 | 21.928243 | 20.671570 | 256 | 18.507500 | 21.410291 | 19.958896 |
| math32 | 32 | 1 | 19.397059 | 21.909817 | 20.653438 | 256 | 18.129010 | 21.190370 | 19.659690 |
| math64 | 64 | 1 | 19.397059 | 21.909817 | 20.653438 | 256 | 18.020614 | 21.119963 | 19.570288 |
| code16 | 16 | 0 | 19.414897 | 21.928243 | 20.671570 | 256 | 17.368412 | 21.024071 | 19.196241 |
| code32 | 32 | 0 | 19.414897 | 21.928243 | 20.671570 | 256 | 17.249606 | 20.939862 | 19.094734 |
| code64 | 64 | 1 | 19.397059 | 21.909817 | 20.653438 | 256 | 17.115535 | 20.854547 | 18.985041 |
| math_code16 | 16 | 0 | 19.414897 | 21.928243 | 20.671570 | 256 | 17.919890 | 21.140365 | 19.530127 |
| math_code32 | 32 | 0 | 19.414897 | 21.928243 | 20.671570 | 256 | 17.453015 | 20.996466 | 19.224740 |
| math_code64 | 64 | 1 | 19.397059 | 21.909817 | 20.653438 | 256 | 17.436687 | 20.991402 | 19.214044 |
| math_code128 | 128 | 1 | 19.397059 | 21.909817 | 20.653438 | 256 | 17.426627 | 20.940578 | 19.183602 |

Llama-3.1-8B baseline: Wiki 9.092934; C4 11.600109.

### Llama-3.1-8B

| Calibration | Sequences | Adaptive E0M3 blocks | Adaptive Wiki PPL | Adaptive C4 PPL | Adaptive average | Fixed E0M3 blocks | Fixed Wiki PPL | Fixed C4 PPL | Fixed average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| math16 | 16 | 1 | 9.084794 | 11.615502 | 10.350148 | 256 | 9.055346 | 11.555850 | 10.305598 |
| math32 | 32 | 1 | 9.084794 | 11.615502 | 10.350148 | 256 | 9.042277 | 11.525782 | 10.284030 |
| math64 | 64 | 1 | 9.084794 | 11.615502 | 10.350148 | 256 | 9.030872 | 11.526124 | 10.278498 |
| code16 | 16 | 1 | 9.084794 | 11.615502 | 10.350148 | 256 | 9.052092 | 11.540401 | 10.296247 |
| code32 | 32 | 2 | 9.088794 | 11.587583 | 10.338189 | 256 | 9.038428 | 11.538297 | 10.288363 |
| code64 | 64 | 2 | 9.088794 | 11.587583 | 10.338189 | 256 | 9.043249 | 11.553241 | 10.298245 |
| math_code16 | 16 | 1 | 9.084794 | 11.615502 | 10.350148 | 256 | 9.049960 | 11.545661 | 10.297810 |
| math_code32 | 32 | 1 | 9.084794 | 11.615502 | 10.350148 | 256 | 9.049170 | 11.544846 | 10.297008 |
| math_code64 | 64 | 1 | 9.084794 | 11.615502 | 10.350148 | 256 | 9.028759 | 11.539653 | 10.284206 |
| math_code128 | 128 | 2 | 9.088794 | 11.587583 | 10.338189 | 256 | 9.028599 | 11.513102 | 10.270851 |

The source/sample-count cells share data and often selected blocks. Gain counts are descriptive, not independent replications. Two-SE intervals do not account for WikiText article dependence or multiple comparisons. The adaptive objective is an estimated-curvature surrogate, not a true finite-switch loss bound.
