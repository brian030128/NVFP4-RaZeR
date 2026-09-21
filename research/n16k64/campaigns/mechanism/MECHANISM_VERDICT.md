# Mechanism verdict

This is a fixed-gate verdict for aggregate ranking, CE/KL veto risk control, and attention/MLP interaction. A failed gate is a negative result, not a reason to redefine an endpoint.

## Verdict summary

| Question | Classification | Frozen development gate passed? |
|---|---|---:|
| ranking not calibration | supported | yes |
| ce kl veto | not_supported_under_strict_gate | no |
| attention mlp interaction | not_supported_under_strict_gate | no |

## Direct answer

The strict development gate supports the proposed reconciliation at group level: the score carries useful aggregate ranking information even though the frozen finite-effect evidence does not calibrate individual tiles reliably.
The CE/KL conjunction does not pass the frozen veto-protection gate across both one-objective-pass classes; any favorable individual endpoint remains partial evidence only.
The frozen interaction gate does not establish a stable attention/MLP non-additivity pattern across all development endpoints.
Veto subclass gates: ce_vetoed_kl_approved=pass, kl_vetoed_ce_approved=fail.

The Mistral results below are one-shot analysis validation only; they do not convert a development finding into a fully independent model-family generalization claim.

## Absolute PPL context

Absolute PPL is descriptive context; the inferential endpoint is paired delta NLL (delta-log-PPL).

| Role | Model | Corpus | FourOverSix PPL | Full-map PPL | Full − baseline delta-log-PPL | Relative PPL |
|---|---|---|---:|---:|---:|---:|
| development | llama8b | c4 | 9.8253676 | 9.7761620 | -0.0050206 | -0.5008% |
| development | llama8b | wiki | 6.8754007 | 6.8432373 | -0.0046890 | -0.4678% |
| one-shot validation | mistral7b | c4 | 8.0674100 | 8.0475064 | -0.0024702 | -0.2467% |
| one-shot validation | mistral7b | wiki | 5.5233091 | 5.5018708 | -0.0038890 | -0.3881% |
| development | qwen4b | c4 | 17.3024041 | 16.0485180 | -0.0752290 | -7.2469% |
| development | qwen4b | wiki | 14.2062126 | 12.1967733 | -0.1525079 | -14.1448% |

## Ranking-not-calibration primary contrasts

| Role | Model | Corpus | Contrast | delta-log-PPL | 95% CI | Relative PPL | Holm p |
|---|---|---|---|---:|---:|---:|---:|
| development | llama8b | c4 | group-only strongest − weakest | -0.0029867 | [-0.0042089, -0.0017804] | -0.2982% | 0.00239976 |
| development | llama8b | c4 | marginal strongest − weakest | -0.0021337 | [-0.0035608, -0.0007025] | -0.2131% | 0.0343966 |
| development | llama8b | wiki | group-only strongest − weakest | -0.0035261 | [-0.0050914, -0.0020860] | -0.3520% | 0.00239976 |
| development | llama8b | wiki | marginal strongest − weakest | -0.0013788 | [-0.0028088, +0.0000089] | -0.1378% | 0.305969 |
| one-shot validation | mistral7b | c4 | group-only strongest − weakest | -0.0011237 | [-0.0018247, -0.0003864] | -0.1123% | 0.00259974 |
| one-shot validation | mistral7b | c4 | marginal strongest − weakest | -0.0012540 | [-0.0019122, -0.0005874] | -0.1253% | 0.00059994 |
| one-shot validation | mistral7b | wiki | group-only strongest − weakest | -0.0026750 | [-0.0035339, -0.0018021] | -0.2671% | 0.00039996 |
| one-shot validation | mistral7b | wiki | marginal strongest − weakest | -0.0021324 | [-0.0028437, -0.0013364] | -0.2130% | 0.00039996 |
| development | qwen4b | c4 | group-only strongest − weakest | -0.0436000 | [-0.0456879, -0.0415595] | -4.2663% | 0.00239976 |
| development | qwen4b | c4 | marginal strongest − weakest | -0.0354679 | [-0.0373929, -0.0335791] | -3.4846% | 0.00239976 |
| development | qwen4b | wiki | group-only strongest − weakest | -0.0897141 | [-0.0962187, -0.0838193] | -8.5807% | 0.00239976 |
| development | qwen4b | wiki | marginal strongest − weakest | -0.0731647 | [-0.0779847, -0.0683989] | -7.0552% | 0.00239976 |

## CE/KL-veto primary contrasts

Positive add-back effects are worse NLL. The matched-random contrast controls composition and approving-margin bins.

| Role | Model | Corpus | Veto class | Contrast | delta-log-PPL | 95% CI | Relative PPL | Holm p |
|---|---|---|---|---|---:|---:|---:|---:|
| development | llama8b | c4 | ce_vetoed_kl_approved | actual add-back − conjunction | -0.0000671 | [-0.0011743, +0.0010206] | -0.0067% | 1 |
| development | llama8b | c4 | ce_vetoed_kl_approved | actual add-back − matched random | -0.0002251 | [-0.0014551, +0.0010071] | -0.0225% | 1 |
| development | llama8b | c4 | kl_vetoed_ce_approved | actual add-back − conjunction | -0.0000424 | [-0.0012264, +0.0010812] | -0.0042% | 1 |
| development | llama8b | c4 | kl_vetoed_ce_approved | actual add-back − matched random | -0.0009792 | [-0.0024080, +0.0003018] | -0.0979% | 1 |
| development | llama8b | wiki | ce_vetoed_kl_approved | actual add-back − conjunction | +0.0002357 | [-0.0012027, +0.0016255] | +0.0236% | 1 |
| development | llama8b | wiki | ce_vetoed_kl_approved | actual add-back − matched random | +0.0008926 | [-0.0002957, +0.0020850] | +0.0893% | 1 |
| development | llama8b | wiki | kl_vetoed_ce_approved | actual add-back − conjunction | -0.0002245 | [-0.0015969, +0.0011553] | -0.0224% | 1 |
| development | llama8b | wiki | kl_vetoed_ce_approved | actual add-back − matched random | -0.0014681 | [-0.0030015, +0.0000325] | -0.1467% | 1 |
| one-shot validation | mistral7b | c4 | kl_vetoed_ce_approved | actual add-back − conjunction | -0.0002352 | [-0.0010075, +0.0006295] | -0.0235% | 1 |
| one-shot validation | mistral7b | c4 | kl_vetoed_ce_approved | actual add-back − matched random | +0.0000314 | [-0.0006037, +0.0006938] | +0.0031% | 1 |
| one-shot validation | mistral7b | wiki | kl_vetoed_ce_approved | actual add-back − conjunction | +0.0001794 | [-0.0006143, +0.0009099] | +0.0179% | 1 |
| one-shot validation | mistral7b | wiki | kl_vetoed_ce_approved | actual add-back − matched random | +0.0001796 | [-0.0005878, +0.0009120] | +0.0180% | 1 |
| development | qwen4b | c4 | ce_vetoed_kl_approved | actual add-back − conjunction | +0.0098459 | [+0.0084026, +0.0112757] | +0.9895% | 0.00559944 |
| development | qwen4b | c4 | ce_vetoed_kl_approved | actual add-back − matched random | +0.0069786 | [+0.0055508, +0.0084033] | +0.7003% | 0.00559944 |
| development | qwen4b | c4 | kl_vetoed_ce_approved | actual add-back − conjunction | -0.0552273 | [-0.0581086, -0.0523978] | -5.3730% | 0.00559944 |
| development | qwen4b | c4 | kl_vetoed_ce_approved | actual add-back − matched random | -0.0423068 | [-0.0447291, -0.0398886] | -4.1424% | 0.00559944 |
| development | qwen4b | wiki | ce_vetoed_kl_approved | actual add-back − conjunction | +0.0053684 | [+0.0024469, +0.0082112] | +0.5383% | 0.0350965 |
| development | qwen4b | wiki | ce_vetoed_kl_approved | actual add-back − matched random | +0.0033029 | [+0.0008384, +0.0056685] | +0.3308% | 0.311569 |
| development | qwen4b | wiki | kl_vetoed_ce_approved | actual add-back − conjunction | -0.0910686 | [-0.0956854, -0.0865907] | -8.7045% | 0.00559944 |
| development | qwen4b | wiki | kl_vetoed_ce_approved | actual add-back − matched random | -0.0668897 | [-0.0704600, -0.0636476] | -6.4702% | 0.00559944 |

### Veto tail differences: actual add-back minus matched random

| Role | Model | Corpus | Veto class | Endpoint | Difference | 95% CI | Holm p |
|---|---|---|---|---|---:|---:|---:|
| development | llama8b | c4 | ce_vetoed_kl_approved | q90_actual_minus_random | +0.0013290 | [-0.0007694, +0.0030186] | 1 |
| development | llama8b | c4 | ce_vetoed_kl_approved | q95_actual_minus_random | -0.0003276 | [-0.0051230, +0.0019974] | 1 |
| development | llama8b | c4 | ce_vetoed_kl_approved | q99_actual_minus_random | -0.0102338 | [-0.0189073, -0.0011046] | 0.734927 |
| development | llama8b | c4 | ce_vetoed_kl_approved | regression_probability_actual_minus_random | +0.0259740 | [-0.0433494, +0.0995078] | 1 |
| development | llama8b | c4 | ce_vetoed_kl_approved | worst_actual_minus_random | -0.0169773 | [-0.0226094, -0.0067183] | 0.0163984 |
| development | llama8b | c4 | kl_vetoed_ce_approved | q90_actual_minus_random | -0.0023168 | [-0.0041794, -0.0000685] | 1 |
| development | llama8b | c4 | kl_vetoed_ce_approved | q95_actual_minus_random | -0.0022212 | [-0.0087603, +0.0005127] | 1 |
| development | llama8b | c4 | kl_vetoed_ce_approved | q99_actual_minus_random | -0.0074835 | [-0.0234850, +0.0103244] | 1 |
| development | llama8b | c4 | kl_vetoed_ce_approved | regression_probability_actual_minus_random | +0.0303030 | [-0.0478879, +0.1079563] | 1 |
| development | llama8b | c4 | kl_vetoed_ce_approved | worst_actual_minus_random | -0.0141705 | [-0.0312377, +0.0103856] | 1 |
| development | llama8b | wiki | ce_vetoed_kl_approved | q90_actual_minus_random | +0.0012432 | [-0.0025530, +0.0056712] | 1 |
| development | llama8b | wiki | ce_vetoed_kl_approved | q95_actual_minus_random | +0.0026326 | [-0.0032827, +0.0068117] | 1 |
| development | llama8b | wiki | ce_vetoed_kl_approved | q99_actual_minus_random | -0.0007724 | [-0.0051026, +0.0047270] | 1 |
| development | llama8b | wiki | ce_vetoed_kl_approved | regression_probability_actual_minus_random | +0.1320755 | [-0.0191226, +0.2827642] | 1 |
| development | llama8b | wiki | ce_vetoed_kl_approved | worst_actual_minus_random | -0.0017276 | [-0.0054338, +0.0053921] | 1 |
| development | llama8b | wiki | kl_vetoed_ce_approved | q90_actual_minus_random | -0.0015272 | [-0.0076053, +0.0048848] | 1 |
| development | llama8b | wiki | kl_vetoed_ce_approved | q95_actual_minus_random | -0.0027403 | [-0.0111250, +0.0064566] | 1 |
| development | llama8b | wiki | kl_vetoed_ce_approved | q99_actual_minus_random | -0.0007102 | [-0.0096156, +0.0071893] | 1 |
| development | llama8b | wiki | kl_vetoed_ce_approved | regression_probability_actual_minus_random | -0.2452830 | [-0.4157887, -0.0572981] | 0.311569 |
| development | llama8b | wiki | kl_vetoed_ce_approved | worst_actual_minus_random | -0.0003044 | [-0.0119677, +0.0086459] | 1 |
| one-shot validation | mistral7b | c4 | kl_vetoed_ce_approved | q90_actual_minus_random | +0.0000207 | [-0.0011819, +0.0010202] | 1 |
| one-shot validation | mistral7b | c4 | kl_vetoed_ce_approved | q95_actual_minus_random | -0.0001686 | [-0.0017813, +0.0009719] | 1 |
| one-shot validation | mistral7b | c4 | kl_vetoed_ce_approved | q99_actual_minus_random | -0.0005181 | [-0.0047151, +0.0179204] | 1 |
| one-shot validation | mistral7b | c4 | kl_vetoed_ce_approved | regression_probability_actual_minus_random | -0.0085106 | [-0.0768957, +0.0592745] | 1 |
| one-shot validation | mistral7b | c4 | kl_vetoed_ce_approved | worst_actual_minus_random | +0.0290371 | [+0.0086796, +0.0390932] | 0.00139986 |
| one-shot validation | mistral7b | wiki | kl_vetoed_ce_approved | q90_actual_minus_random | -0.0004759 | [-0.0020700, +0.0019368] | 1 |
| one-shot validation | mistral7b | wiki | kl_vetoed_ce_approved | q95_actual_minus_random | +0.0001594 | [-0.0028195, +0.0042818] | 1 |
| one-shot validation | mistral7b | wiki | kl_vetoed_ce_approved | q99_actual_minus_random | +0.0035149 | [-0.0011061, +0.0062873] | 1 |
| one-shot validation | mistral7b | wiki | kl_vetoed_ce_approved | regression_probability_actual_minus_random | -0.0350877 | [-0.1750596, +0.1056421] | 1 |
| one-shot validation | mistral7b | wiki | kl_vetoed_ce_approved | worst_actual_minus_random | +0.0041033 | [-0.0019093, +0.0062641] | 1 |
| development | qwen4b | c4 | ce_vetoed_kl_approved | q90_actual_minus_random | +0.0093490 | [+0.0065828, +0.0117466] | 0.00559944 |
| development | qwen4b | c4 | ce_vetoed_kl_approved | q95_actual_minus_random | +0.0107686 | [+0.0074316, +0.0137254] | 0.00559944 |
| development | qwen4b | c4 | ce_vetoed_kl_approved | q99_actual_minus_random | +0.0146792 | [+0.0015752, +0.0205482] | 0.356364 |
| development | qwen4b | c4 | ce_vetoed_kl_approved | regression_probability_actual_minus_random | +0.2121212 | [+0.1468385, +0.2768169] | 0.00559944 |
| development | qwen4b | c4 | ce_vetoed_kl_approved | worst_actual_minus_random | +0.0038046 | [-0.0045134, +0.0157098] | 1 |
| development | qwen4b | c4 | kl_vetoed_ce_approved | q90_actual_minus_random | -0.0286045 | [-0.0373218, -0.0207518] | 0.00559944 |
| development | qwen4b | c4 | kl_vetoed_ce_approved | q95_actual_minus_random | -0.0181818 | [-0.0258084, -0.0117772] | 0.0239976 |
| development | qwen4b | c4 | kl_vetoed_ce_approved | q99_actual_minus_random | +0.0021771 | [-0.0147753, +0.0192060] | 1 |
| development | qwen4b | c4 | kl_vetoed_ce_approved | regression_probability_actual_minus_random | -0.0995671 | [-0.1428268, -0.0605758] | 0.00559944 |
| development | qwen4b | c4 | kl_vetoed_ce_approved | worst_actual_minus_random | +0.0126213 | [-0.0011298, +0.0216584] | 1 |
| development | qwen4b | wiki | ce_vetoed_kl_approved | q90_actual_minus_random | +0.0037103 | [-0.0056442, +0.0110498] | 1 |
| development | qwen4b | wiki | ce_vetoed_kl_approved | q95_actual_minus_random | +0.0046932 | [-0.0092499, +0.0128621] | 1 |
| development | qwen4b | wiki | ce_vetoed_kl_approved | q99_actual_minus_random | -0.0031374 | [-0.0127425, +0.0093728] | 1 |
| development | qwen4b | wiki | ce_vetoed_kl_approved | regression_probability_actual_minus_random | +0.1132075 | [-0.0373019, +0.2645849] | 1 |
| development | qwen4b | wiki | ce_vetoed_kl_approved | worst_actual_minus_random | -0.0028961 | [-0.0111055, +0.0106864] | 1 |
| development | qwen4b | wiki | kl_vetoed_ce_approved | q90_actual_minus_random | -0.0654633 | [-0.0709700, -0.0571790] | 0.00559944 |
| development | qwen4b | wiki | kl_vetoed_ce_approved | q95_actual_minus_random | -0.0623844 | [-0.0723931, -0.0511076] | 0.00559944 |
| development | qwen4b | wiki | kl_vetoed_ce_approved | q99_actual_minus_random | -0.0575610 | [-0.0692590, -0.0491889] | 0.00559944 |
| development | qwen4b | wiki | kl_vetoed_ce_approved | regression_probability_actual_minus_random | -0.0188679 | [-0.0565679, +0.0000358] | 1 |
| development | qwen4b | wiki | kl_vetoed_ce_approved | worst_actual_minus_random | -0.0583931 | [-0.0725405, -0.0497412] | 0.00559944 |

## Attention/MLP interaction primary contrasts

| Role | Model | Corpus | Contrast | delta-log-PPL | 95% CI | Relative PPL | Holm p |
|---|---|---|---|---:|---:|---:|---:|
| development | llama8b | c4 | actual interaction residual | +0.0010942 | [-0.0003563, +0.0026978] | +0.1095% | 0.345865 |
| development | llama8b | c4 | actual − matched-random residual | -0.0007534 | [-0.0027745, +0.0011462] | -0.0753% | 0.441456 |
| development | llama8b | wiki | actual interaction residual | +0.0012755 | [-0.0002687, +0.0029435] | +0.1276% | 0.345865 |
| development | llama8b | wiki | actual − matched-random residual | +0.0029851 | [+0.0006922, +0.0052710] | +0.2990% | 0.0467953 |
| one-shot validation | mistral7b | c4 | actual interaction residual | +0.0004303 | [-0.0005231, +0.0014016] | +0.0430% | 0.768123 |
| one-shot validation | mistral7b | c4 | actual − matched-random residual | -0.0001904 | [-0.0014825, +0.0011138] | -0.0190% | 0.777022 |
| one-shot validation | mistral7b | wiki | actual interaction residual | +0.0010071 | [-0.0001226, +0.0020936] | +0.1008% | 0.29917 |
| one-shot validation | mistral7b | wiki | actual − matched-random residual | +0.0009275 | [-0.0006178, +0.0024359] | +0.0928% | 0.70253 |
| development | qwen4b | c4 | actual interaction residual | +0.0083891 | [+0.0064337, +0.0104210] | +0.8424% | 0.00079992 |
| development | qwen4b | c4 | actual − matched-random residual | +0.0093939 | [+0.0069167, +0.0118330] | +0.9438% | 0.00079992 |
| development | qwen4b | wiki | actual interaction residual | +0.0219833 | [+0.0176199, +0.0263537] | +2.2227% | 0.00079992 |
| development | qwen4b | wiki | actual − matched-random residual | +0.0230524 | [+0.0180714, +0.0277010] | +2.3320% | 0.00079992 |

## Existing individual-tile counterevidence

These frozen attempt7 finite-effect results are the counterpoint to any aggregate result; they are not overwritten by this campaign.

| Model | n | Sign precision | Sign recall | Pearson | Spearman | Resolvable @1.96SE | Resolvable @3SE | Full-map actual/predicted CE ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama8b | 160 | 0.3010 | 0.7045 | 0.1067 | 0.0445 | 0.1625 | 0.0250 | 0.7390 |
| mistral7b | 160 | 0.5816 | 0.5876 | -0.0379 | -0.1275 | 0.0500 | 0.0063 | 0.7907 |
| qwen4b | 160 | 0.3011 | 0.7778 | 0.1550 | 0.2393 | 0.0625 | 0.0187 | 0.6641 |

## Interpretation boundary

The admissible interpretation is group-level and model-level: a selector may carry useful aggregate ranking or risk-control information even when its per-tile first-order estimate is noisy, biased, or interaction-confounded. No result here establishes reliable individual-tile causal signs or magnitudes.

All evaluation is fake-quantized/dequantized BF16 quality evaluation. Native FP4/E0M3 Tensor Core execution, latency, speedup, area, and power claims remain prohibited and untested.
