## Math/code-only calibration: adaptive E0M3 count

Calibration uses **OpenWebMath and CodeParrot only**. Neither C4 nor WikiText supplies calibration examples, gradients, or count-selection feedback. All measurements below evaluate the frozen maps only on **WikiText-2 test and held-out C4**. No seed replication or best-setting election is performed.

**Result: this adaptive rule is not a competitive replacement for fixed-256.** It selected 0–8 blocks and achieved lower PPL than fixed-256 in only 1/60 paired model/setting/dataset comparisons. Fixed-256 improved point PPL over FourOverSix in 57/60 cells; 48 had baseline-relative ΔNLL + 2SE < 0 and 0 had ΔNLL − 2SE > 0. These correlated-cell counts are descriptive, not independent replications. The post hoc curvature audit documents conservatism in the interaction penalty; it does not establish that a less conservative selector would generalize. The study supports a negative finding for this particular adaptive surrogate, not an impossibility claim about adaptive selection.

[Frozen protocol and selection equations](../PROTOCOL.md).

One shared 128-sequence causal scoring pass per model supplies all ten source/sample-count settings. The adaptive method minimizes a directional-benefit plus estimated-curvature penalty over all negative-score prefixes, including zero switches. It has **no count cap**. The fixed-256 comparison uses the same fresh causal CE/KL derivatives and math/code subsets; it is not the older C4-containing pooled map.

Counts are **8x64 E0M3 type blocks**, each containing 512 weights and 32 distinct 16-element scale blocks. Each average is the unweighted arithmetic mean of the two displayed dataset PPLs. PPL is computed from per-token loss over identical scored windows within each model.

| Model | Baseline Wiki PPL | Baseline C4 PPL | Baseline average | Weight-MSE E0M3 blocks | Weight-MSE Wiki PPL | Weight-MSE C4 PPL | Weight-MSE average |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 19.414897 | 21.928243 | 20.671570 | 3,407,142 | 20.108009 | 22.082958 | 21.095483 |
| Llama-3.1-8B | 9.092934 | 11.600109 | 10.346521 | 6,652,753 | 9.088662 | 11.635797 | 10.362229 |
| Qwen3.8-27B | 9.040908 | 12.644977 | 10.842943 | 22,696,327 | 9.026329 | 12.671271 | 10.848800 |

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

### Qwen3.8-27B

| Calibration | Sequences | Adaptive E0M3 blocks | Adaptive Wiki PPL | Adaptive C4 PPL | Adaptive average | Fixed E0M3 blocks | Fixed Wiki PPL | Fixed C4 PPL | Fixed average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| math16 | 16 | 2 | 9.041333 | 12.660161 | 10.850747 | 256 | 9.026444 | 12.630066 | 10.828255 |
| math32 | 32 | 2 | 9.040871 | 12.646751 | 10.843811 | 256 | 9.032305 | 12.649907 | 10.841106 |
| math64 | 64 | 4 | 9.039517 | 12.650800 | 10.845158 | 256 | 9.012561 | 12.632059 | 10.822310 |
| code16 | 16 | 5 | 9.043073 | 12.653983 | 10.848528 | 256 | 9.020848 | 12.650864 | 10.835856 |
| code32 | 32 | 8 | 9.047084 | 12.649668 | 10.848376 | 256 | 9.008592 | 12.645944 | 10.827268 |
| code64 | 64 | 5 | 9.030592 | 12.640839 | 10.835715 | 256 | 9.017231 | 12.639964 | 10.828597 |
| math_code16 | 16 | 2 | 9.048039 | 12.650519 | 10.849279 | 256 | 9.032979 | 12.644239 | 10.838609 |
| math_code32 | 32 | 2 | 9.040062 | 12.647666 | 10.843864 | 256 | 9.006973 | 12.643467 | 10.825220 |
| math_code64 | 64 | 6 | 9.038289 | 12.650116 | 10.844203 | 256 | 9.013104 | 12.643789 | 10.828447 |
| math_code128 | 128 | 1 | 9.042556 | 12.650176 | 10.846366 | 256 | 9.005556 | 12.635394 | 10.820475 |

### Adaptive objective, counts, and fractions

| Model | Setting | Eligible blocks | Selected blocks | Selected % of all type blocks | Predicted linear term | Curvature penalty | Predicted total |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | math16 | 36,648 | 0 | 0.000000% | +0 | 0 | +0 |
| Qwen3-4B | math32 | 36,225 | 1 | 0.000014% | -0.000985113 | 0.000827383 | -0.00015773 |
| Qwen3-4B | math64 | 41,284 | 1 | 0.000014% | -0.00111765 | 0.00082902 | -0.000288635 |
| Qwen3-4B | code16 | 43,376 | 0 | 0.000000% | +0 | 0 | +0 |
| Qwen3-4B | code32 | 50,903 | 0 | 0.000000% | +0 | 0 | +0 |
| Qwen3-4B | code64 | 68,978 | 1 | 0.000014% | -0.000932807 | 0.000770897 | -0.000161909 |
| Qwen3-4B | math_code16 | 36,957 | 0 | 0.000000% | +0 | 0 | +0 |
| Qwen3-4B | math_code32 | 41,958 | 0 | 0.000000% | +0 | 0 | +0 |
| Qwen3-4B | math_code64 | 49,479 | 1 | 0.000014% | -0.000849738 | 0.000849152 | -5.8619e-07 |
| Qwen3-4B | math_code128 | 66,856 | 1 | 0.000014% | -0.00113949 | 0.000799959 | -0.00033953 |
| Llama-3.1-8B | math16 | 48,199 | 1 | 0.000007% | -0.000413753 | 0.000192885 | -0.000220868 |
| Llama-3.1-8B | math32 | 55,568 | 1 | 0.000007% | -0.000362086 | 0.000125147 | -0.000236939 |
| Llama-3.1-8B | math64 | 74,195 | 1 | 0.000007% | -0.000519718 | 0.000122356 | -0.000397362 |
| Llama-3.1-8B | code16 | 71,949 | 1 | 0.000007% | -0.000186382 | 8.74487e-05 | -9.89337e-05 |
| Llama-3.1-8B | code32 | 68,425 | 2 | 0.000015% | -0.000698346 | 0.000304242 | -0.000394104 |
| Llama-3.1-8B | code64 | 89,460 | 2 | 0.000015% | -0.0010628 | 0.000372654 | -0.000690144 |
| Llama-3.1-8B | math_code16 | 35,244 | 1 | 0.000007% | -0.000151038 | 9.79063e-05 | -5.31316e-05 |
| Llama-3.1-8B | math_code32 | 53,709 | 1 | 0.000007% | -0.000400171 | 0.000140167 | -0.000260004 |
| Llama-3.1-8B | math_code64 | 70,518 | 1 | 0.000007% | -0.000437733 | 9.56111e-05 | -0.000342121 |
| Llama-3.1-8B | math_code128 | 99,024 | 2 | 0.000015% | -0.00111688 | 0.000453003 | -0.000663872 |
| Qwen3.8-27B | math16 | 103,418 | 2 | 0.000004% | -2.13376e-05 | 6.07957e-06 | -1.5258e-05 |
| Qwen3.8-27B | math32 | 91,460 | 2 | 0.000004% | -1.28804e-05 | 3.32183e-06 | -9.55855e-06 |
| Qwen3.8-27B | math64 | 98,737 | 4 | 0.000008% | -3.12694e-05 | 2.64946e-05 | -4.77488e-06 |
| Qwen3.8-27B | code16 | 197,733 | 5 | 0.000011% | -8.85365e-05 | 1.61869e-05 | -7.23497e-05 |
| Qwen3.8-27B | code32 | 156,484 | 8 | 0.000017% | -0.000122645 | 5.64435e-05 | -6.62014e-05 |
| Qwen3.8-27B | code64 | 166,509 | 5 | 0.000011% | -9.78824e-05 | 3.40146e-05 | -6.38678e-05 |
| Qwen3.8-27B | math_code16 | 142,192 | 2 | 0.000004% | -3.65514e-05 | 1.49175e-05 | -2.1634e-05 |
| Qwen3.8-27B | math_code32 | 140,668 | 2 | 0.000004% | -2.92506e-05 | 7.14279e-06 | -2.21078e-05 |
| Qwen3.8-27B | math_code64 | 130,718 | 6 | 0.000013% | -6.81974e-05 | 3.42672e-05 | -3.39301e-05 |
| Qwen3.8-27B | math_code128 | 149,033 | 1 | 0.000002% | -1.63431e-05 | 3.18578e-06 | -1.31574e-05 |

### Baseline-relative paired NLL differences

| Model | Policy | Wiki ΔNLL ±2SE | C4 ΔNLL ±2SE |
|---|---|---:|---:|
| Qwen3-4B | adaptive_math16 | +0.000000 ±0.000000 | +0.000000 ±0.000000 |
| Qwen3-4B | fixed256_math16 | -0.047865 ±0.003536 | -0.023904 ±0.003136 |
| Qwen3-4B | adaptive_math32 | -0.000919 ±0.000114 | -0.000841 ±0.000159 |
| Qwen3-4B | fixed256_math32 | -0.068527 ±0.003677 | -0.034229 ±0.003132 |
| Qwen3-4B | adaptive_math64 | -0.000919 ±0.000114 | -0.000841 ±0.000159 |
| Qwen3-4B | fixed256_math64 | -0.074524 ±0.003702 | -0.037557 ±0.003078 |
| Qwen3-4B | adaptive_code16 | +0.000000 ±0.000000 | +0.000000 ±0.000000 |
| Qwen3-4B | fixed256_code16 | -0.111387 ±0.004781 | -0.042107 ±0.003620 |
| Qwen3-4B | adaptive_code32 | +0.000000 ±0.000000 | +0.000000 ±0.000000 |
| Qwen3-4B | fixed256_code32 | -0.118251 ±0.004592 | -0.046121 ±0.003537 |
| Qwen3-4B | adaptive_code64 | -0.000919 ±0.000114 | -0.000841 ±0.000159 |
| Qwen3-4B | fixed256_code64 | -0.126054 ±0.004705 | -0.050203 ±0.003823 |
| Qwen3-4B | adaptive_math_code16 | +0.000000 ±0.000000 | +0.000000 ±0.000000 |
| Qwen3-4B | fixed256_math_code16 | -0.080129 ±0.004014 | -0.036591 ±0.003320 |
| Qwen3-4B | adaptive_math_code32 | +0.000000 ±0.000000 | +0.000000 ±0.000000 |
| Qwen3-4B | fixed256_math_code32 | -0.106528 ±0.004344 | -0.043421 ±0.003420 |
| Qwen3-4B | adaptive_math_code64 | -0.000919 ±0.000114 | -0.000841 ±0.000159 |
| Qwen3-4B | fixed256_math_code64 | -0.107464 ±0.004297 | -0.043663 ±0.003145 |
| Qwen3-4B | adaptive_math_code128 | -0.000919 ±0.000114 | -0.000841 ±0.000159 |
| Qwen3-4B | fixed256_math_code128 | -0.108041 ±0.004074 | -0.046087 ±0.003575 |
| Qwen3-4B | weight_mse | +0.035078 ±0.004814 | +0.007031 ±0.004017 |
| Llama-3.1-8B | adaptive_math16 | -0.000896 ±0.001605 | +0.001326 ±0.002787 |
| Llama-3.1-8B | fixed256_math16 | -0.004142 ±0.001675 | -0.003823 ±0.002576 |
| Llama-3.1-8B | adaptive_math32 | -0.000896 ±0.001605 | +0.001326 ±0.002787 |
| Llama-3.1-8B | fixed256_math32 | -0.005587 ±0.001723 | -0.006428 ±0.002192 |
| Llama-3.1-8B | adaptive_math64 | -0.000896 ±0.001605 | +0.001326 ±0.002787 |
| Llama-3.1-8B | fixed256_math64 | -0.006849 ±0.001782 | -0.006398 ±0.002505 |
| Llama-3.1-8B | adaptive_code16 | -0.000896 ±0.001605 | +0.001326 ±0.002787 |
| Llama-3.1-8B | fixed256_code16 | -0.004502 ±0.001774 | -0.005160 ±0.001985 |
| Llama-3.1-8B | adaptive_code32 | -0.000455 ±0.001657 | -0.001080 ±0.002199 |
| Llama-3.1-8B | fixed256_code32 | -0.006012 ±0.001692 | -0.005343 ±0.002509 |
| Llama-3.1-8B | adaptive_code64 | -0.000455 ±0.001657 | -0.001080 ±0.002199 |
| Llama-3.1-8B | fixed256_code64 | -0.005479 ±0.001804 | -0.004048 ±0.002420 |
| Llama-3.1-8B | adaptive_math_code16 | -0.000896 ±0.001605 | +0.001326 ±0.002787 |
| Llama-3.1-8B | fixed256_math_code16 | -0.004737 ±0.001661 | -0.004705 ±0.002396 |
| Llama-3.1-8B | adaptive_math_code32 | -0.000896 ±0.001605 | +0.001326 ±0.002787 |
| Llama-3.1-8B | fixed256_math_code32 | -0.004825 ±0.001717 | -0.004775 ±0.002336 |
| Llama-3.1-8B | adaptive_math_code64 | -0.000896 ±0.001605 | +0.001326 ±0.002787 |
| Llama-3.1-8B | fixed256_math_code64 | -0.007083 ±0.001723 | -0.005225 ±0.002574 |
| Llama-3.1-8B | adaptive_math_code128 | -0.000455 ±0.001657 | -0.001080 ±0.002199 |
| Llama-3.1-8B | fixed256_math_code128 | -0.007100 ±0.001722 | -0.007529 ±0.002139 |
| Llama-3.1-8B | weight_mse | -0.000470 ±0.002130 | +0.003072 ±0.003394 |
| Qwen3.8-27B | adaptive_math16 | +0.000047 ±0.001380 | +0.001200 ±0.001555 |
| Qwen3.8-27B | fixed256_math16 | -0.001601 ±0.001468 | -0.001180 ±0.001475 |
| Qwen3.8-27B | adaptive_math32 | -0.000004 ±0.001416 | +0.000140 ±0.001461 |
| Qwen3.8-27B | fixed256_math32 | -0.000952 ±0.001429 | +0.000390 ±0.001395 |
| Qwen3.8-27B | adaptive_math64 | -0.000154 ±0.001294 | +0.000460 ±0.001363 |
| Qwen3.8-27B | fixed256_math64 | -0.003140 ±0.001435 | -0.001022 ±0.001582 |
| Qwen3.8-27B | adaptive_code16 | +0.000239 ±0.001428 | +0.000712 ±0.001493 |
| Qwen3.8-27B | fixed256_code16 | -0.002221 ±0.001452 | +0.000465 ±0.001664 |
| Qwen3.8-27B | adaptive_code32 | +0.000683 ±0.001461 | +0.000371 ±0.001464 |
| Qwen3.8-27B | fixed256_code32 | -0.003581 ±0.001472 | +0.000076 ±0.001526 |
| Qwen3.8-27B | adaptive_code64 | -0.001142 ±0.001235 | -0.000327 ±0.001353 |
| Qwen3.8-27B | fixed256_code64 | -0.002622 ±0.001376 | -0.000397 ±0.001473 |
| Qwen3.8-27B | adaptive_math_code16 | +0.000788 ±0.001403 | +0.000438 ±0.001524 |
| Qwen3.8-27B | fixed256_math_code16 | -0.000877 ±0.001431 | -0.000058 ±0.001503 |
| Qwen3.8-27B | adaptive_math_code32 | -0.000094 ±0.001396 | +0.000213 ±0.001539 |
| Qwen3.8-27B | fixed256_math_code32 | -0.003761 ±0.001377 | -0.000119 ±0.001504 |
| Qwen3.8-27B | adaptive_math_code64 | -0.000290 ±0.001368 | +0.000406 ±0.001576 |
| Qwen3.8-27B | fixed256_math_code64 | -0.003080 ±0.001448 | -0.000094 ±0.001501 |
| Qwen3.8-27B | adaptive_math_code128 | +0.000182 ±0.001468 | +0.000411 ±0.001521 |
| Qwen3.8-27B | fixed256_math_code128 | -0.003918 ±0.001466 | -0.000758 ±0.001462 |
| Qwen3.8-27B | weight_mse | -0.001614 ±0.001800 | +0.002077 ±0.001799 |

### Limits

The curvature inequality is exact for the estimated PSD matrix, but its sampled GGN diagonal and straight-through derivatives do not bound the true finite-switch network loss. This is an adaptive surrogate method, not a universal guarantee or a claim of methodological novelty. Two-SE intervals describe evaluation-window variation; they do not adjust for multiple comparisons, WikiText article dependence, or unmeasured calibration-draw variability. Both target families were previously inspected. Source exclusion is not a near-duplicate or pretraining-overlap audit.

The cancelled three-source/five-dataset experiment is historical and is not combined with these results. Simulated nonhead-text-linear W4A4; no generation-accuracy or native-throughput claim.

[Count and transfer-sensitivity figure](sensitivity.pdf). Colors identify math, code, and balanced math+code calibration; solid/dashed lines identify adaptive/fixed-256. Error bars are descriptive evaluation-window two-SE intervals, not calibration-seed intervals.

[Post hoc curvature audit](CURVATURE_AUDIT.md) compares the conservative interaction penalty with the same sampled GGN quadratic form on the already-frozen maps. It changes no maps and does not measure true finite-switch losses.
