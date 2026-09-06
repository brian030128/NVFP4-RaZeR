# Gains beyond FourOverSix

Matched native Qwen3.8-27B, W4A4, 8x64 weight type tiles. The new maps
preserve FourOverSix E2M1 scaling and add E0M3 alpha=1 switches.
All test results include the proposed map even if validation rejects it.

| Seed | Validation | Tiles | Dataset | FourOverSix PPL | Proposed PPL | Delta PPL | Delta NLL ± 2 SE |
|---|---|---:|---|---:|---:|---:|---:|
| 20260912 | accepted | 226 | wikitext | 7.289580 | 6.973420 | -0.316160 | -0.044340 ± 0.007626 |
| 20260912 | accepted | 226 | c4 | 9.117679 | 9.118835 | +0.001156 | +0.000127 ± 0.001908 |
| 20260913 | accepted | 327 | wikitext | 7.289580 | 6.935030 | -0.354550 | -0.049861 ± 0.008349 |
| 20260913 | accepted | 327 | c4 | 9.117679 | 9.111269 | -0.006410 | -0.000703 ± 0.001716 |

All target configurations on the same held-out data:

| Configuration | WikiText PPL | C4 PPL |
|---|---:|---:|
| plain_nvfp4 | 7.555223 | 9.141328 |
| four_over_six | 7.289580 | 9.117679 |
| old_sparse_seed09 | 7.133067 | 9.137327 |
| old_sparse_seed10 | 7.156911 | 9.135967 |
| weight_mse_a1 | 7.154471 | 9.139035 |

Paired comparisons against the reference bank (negative favors the new map):

| New seed | Dataset | Reference | Delta NLL ± 2 SE |
|---|---|---|---:|
| 20260912 | wikitext | plain_nvfp4 | -0.080133 ± 0.014342 |
| 20260912 | wikitext | four_over_six | -0.044340 ± 0.007626 |
| 20260912 | wikitext | old_sparse_seed09 | -0.022636 ± 0.004175 |
| 20260912 | wikitext | old_sparse_seed10 | -0.025973 ± 0.005602 |
| 20260912 | wikitext | weight_mse_a1 | -0.025632 ± 0.004922 |
| 20260912 | c4 | plain_nvfp4 | -0.002464 ± 0.002036 |
| 20260912 | c4 | four_over_six | +0.000127 ± 0.001908 |
| 20260912 | c4 | old_sparse_seed09 | -0.002026 ± 0.001854 |
| 20260912 | c4 | old_sparse_seed10 | -0.001877 ± 0.001974 |
| 20260912 | c4 | weight_mse_a1 | -0.002213 ± 0.002491 |
| 20260913 | wikitext | plain_nvfp4 | -0.085654 ± 0.014971 |
| 20260913 | wikitext | four_over_six | -0.049861 ± 0.008349 |
| 20260913 | wikitext | old_sparse_seed09 | -0.028156 ± 0.004712 |
| 20260913 | wikitext | old_sparse_seed10 | -0.031493 ± 0.006050 |
| 20260913 | wikitext | weight_mse_a1 | -0.031152 ± 0.005342 |
| 20260913 | c4 | plain_nvfp4 | -0.003294 ± 0.002249 |
| 20260913 | c4 | four_over_six | -0.000703 ± 0.001716 |
| 20260913 | c4 | old_sparse_seed09 | -0.002856 ± 0.001996 |
| 20260913 | c4 | old_sparse_seed10 | -0.002707 ± 0.001968 |
| 20260913 | c4 | weight_mse_a1 | -0.003043 ± 0.002525 |

Descriptive retention of each positive reference gain, measured in NLL
relative to plain NVFP4. Ratios are unstable for very small reference gains;
these held-out ratios are not universal certificates.

| New seed | Dataset | Reference | Fraction of reference gain retained |
|---|---|---|---:|
| 20260912 | wikitext | four_over_six | 2.239 |
| 20260912 | wikitext | old_sparse_seed09 | 1.394 |
| 20260912 | wikitext | old_sparse_seed10 | 1.480 |
| 20260912 | wikitext | weight_mse_a1 | 1.470 |
| 20260912 | c4 | four_over_six | 0.951 |
| 20260912 | c4 | old_sparse_seed09 | 5.628 |
| 20260912 | c4 | old_sparse_seed10 | 4.199 |
| 20260912 | c4 | weight_mse_a1 | 9.820 |
| 20260913 | wikitext | four_over_six | 2.393 |
| 20260913 | wikitext | old_sparse_seed09 | 1.490 |
| 20260913 | wikitext | old_sparse_seed10 | 1.581 |
| 20260913 | wikitext | weight_mse_a1 | 1.572 |
| 20260913 | c4 | four_over_six | 1.272 |
| 20260913 | c4 | old_sparse_seed09 | 7.524 |
| 20260913 | c4 | old_sparse_seed10 | 5.614 |
| 20260913 | c4 | weight_mse_a1 | 13.129 |

Four-model transfer panel, with the same frozen procedure:

| Model | Validation | Dataset | FourOverSix PPL | Proposed PPL | Delta NLL ± 2 SE |
|---|---|---|---:|---:|---:|
| qwen3-4b | accepted | wikitext | 14.212163 | 13.172389 | -0.075975 ± 0.004800 |
| qwen3-4b | accepted | c4 | 14.532457 | 14.082534 | -0.031449 ± 0.004550 |
| llama-3.1-8b-local | accepted | wikitext | 6.881153 | 6.781946 | -0.014522 ± 0.002127 |
| llama-3.1-8b-local | accepted | c4 | 8.413324 | 8.407306 | -0.000715 ± 0.003345 |
| qwen3-8b | accepted | wikitext | 10.030983 | 9.328109 | -0.072646 ± 0.002977 |
| qwen3-8b | accepted | c4 | 11.548653 | 11.116696 | -0.038121 ± 0.003393 |
| llama-3.1-8b-ins-local | accepted | wikitext | 7.817656 | 7.565416 | -0.032797 ± 0.002438 |
| llama-3.1-8b-ins-local | accepted | c4 | 9.626479 | 9.547941 | -0.008192 ± 0.003616 |

Prespecified eight-window probe ablations:

| Baseline | Map origin | Intervention | Tiles | Delta NLL ± 2 SE |
|---|---|---|---:|---:|
| nvfp4 | old_nvfp4_map | full | 69 | -0.098736 ± 0.062173 |
| nvfp4 | old_nvfp4_map | channels_3968_4031_only | 52 | -0.090119 ± 0.057591 |
| nvfp4 | old_nvfp4_map | without_channels_3968_4031 | 17 | -0.059739 ± 0.036257 |
| nvfp4 | old_nvfp4_map | random_rows_same_columns | 69 | -0.050688 ± 0.040762 |
| nvfp4 | new_four_over_six_map | full | 226 | -0.111013 ± 0.071407 |
| nvfp4 | new_four_over_six_map | channels_3968_4031_only | 150 | -0.098365 ± 0.063919 |
| nvfp4 | new_four_over_six_map | without_channels_3968_4031 | 76 | -0.062815 ± 0.040366 |
| nvfp4 | new_four_over_six_map | random_rows_same_columns | 226 | -0.048428 ± 0.026171 |
| four_over_six | old_nvfp4_map | full | 69 | -0.038498 ± 0.021777 |
| four_over_six | old_nvfp4_map | channels_3968_4031_only | 52 | -0.034721 ± 0.030181 |
| four_over_six | old_nvfp4_map | without_channels_3968_4031 | 17 | -0.015379 ± 0.011189 |
| four_over_six | old_nvfp4_map | random_rows_same_columns | 69 | -0.020376 ± 0.020482 |
| four_over_six | new_four_over_six_map | full | 226 | -0.057067 ± 0.034223 |
| four_over_six | new_four_over_six_map | channels_3968_4031_only | 150 | -0.052736 ± 0.034834 |
| four_over_six | new_four_over_six_map | without_channels_3968_4031 | 76 | -0.031055 ± 0.019253 |
| four_over_six | new_four_over_six_map | random_rows_same_columns | 226 | -0.012398 ± 0.015104 |

Paired contrasts on the same probe windows (negative favors the full map):

| Baseline | Map origin | Full minus intervention | Delta NLL ± 2 SE |
|---|---|---|---:|
| nvfp4 | old_nvfp4_map | channels_3968_4031_only | -0.008617 ± 0.010839 |
| nvfp4 | old_nvfp4_map | random_rows_same_columns | -0.048049 ± 0.030775 |
| nvfp4 | new_four_over_six_map | channels_3968_4031_only | -0.012648 ± 0.008866 |
| nvfp4 | new_four_over_six_map | random_rows_same_columns | -0.062585 ± 0.047773 |
| four_over_six | old_nvfp4_map | channels_3968_4031_only | -0.003777 ± 0.011748 |
| four_over_six | old_nvfp4_map | random_rows_same_columns | -0.018122 ± 0.003489 |
| four_over_six | new_four_over_six_map | channels_3968_4031_only | -0.004331 ± 0.006911 |
| four_over_six | new_four_over_six_map | random_rows_same_columns | -0.044668 ± 0.031778 |

Among 368 projections with 5120 input channels, 344
have their largest measured input energy in channels 3968–4031.
The median energy fraction in this 64-channel region is 0.412035.
These are measurements on eight NVFP4-reference probe windows, not a universal channel index rule.

Isolated-tile reconstruction geometry (common NVFP4-reference probe inputs):

| Error baseline | Tiles examined | Clear adverse-input witnesses | Lower calibration output error |
|---|---:|---:|---:|
| nvfp4 | 245 | 245 | 84 |
| four_over_six | 245 | 245 | 71 |

These local witnesses refute all-input reconstruction dominance for the
identified tiles. They do not imply that those directions are common in
deployment, or that local reconstruction error alone determines task loss.

Read [GUARANTEE.md](GUARANTEE.md) for the precise gain-retention property,
conditional finite-sample certificate, unconditional fixed-map obstruction,
and online-mixture regret guarantee. The latter requires multiple predictive
configurations and is not a static MixFP4 implementation.
All numerical 2-SE intervals here are descriptive, not universal guarantees.
