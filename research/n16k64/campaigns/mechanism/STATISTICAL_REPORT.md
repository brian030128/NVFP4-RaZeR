# Statistical report

- Frozen protocol SHA-256: `9e7c3d1dcb6199ab9bbedf9e0947f6198304f26466f2fd2c58e5533fb747926c`
- Bootstrap: 10,000 deterministic paired natural-cluster replicates.
- P-values use finite Monte Carlo plus-one correction; Holm adjustment is applied within each frozen family.

## A_development

| Model | Corpus | Endpoint | delta-log-PPL | 95% CI | p | Holm p |
|---|---|---|---:|---:|---:|---:|
| llama8b | wiki | group_only:strongest-weakest | -0.0035261 | [-0.0050914, -0.0020860] | 9.999e-05 | 0.00239976 |
| llama8b | wiki | group_only:strongest-random | -0.0025908 | [-0.0042612, -0.0011411] | 0.00259974 | 0.0233977 |
| llama8b | wiki | group_only:weakest-random | +0.0009353 | [-0.0004617, +0.0021574] | 0.156084 | 0.624338 |
| llama8b | wiki | full_context_marginal:strongest-weakest | -0.0013788 | [-0.0028088, +0.0000089] | 0.0556944 | 0.305969 |
| llama8b | wiki | full_context_marginal:strongest-random | +0.0000694 | [-0.0013215, +0.0014649] | 0.923608 | 1 |
| llama8b | wiki | full_context_marginal:weakest-random | +0.0014482 | [+0.0000121, +0.0029187] | 0.0509949 | 0.305969 |
| llama8b | c4 | group_only:strongest-weakest | -0.0029867 | [-0.0042089, -0.0017804] | 9.999e-05 | 0.00239976 |
| llama8b | c4 | group_only:strongest-random | -0.0024949 | [-0.0038415, -0.0012478] | 0.00039996 | 0.0039996 |
| llama8b | c4 | group_only:weakest-random | +0.0004918 | [-0.0011448, +0.0019484] | 0.537146 | 1 |
| llama8b | c4 | full_context_marginal:strongest-weakest | -0.0021337 | [-0.0035608, -0.0007025] | 0.00429957 | 0.0343966 |
| llama8b | c4 | full_context_marginal:strongest-random | -0.0013688 | [-0.0025529, -0.0001643] | 0.0243976 | 0.170783 |
| llama8b | c4 | full_context_marginal:weakest-random | +0.0007649 | [-0.0006555, +0.0023221] | 0.310169 | 0.930507 |
| qwen4b | wiki | group_only:strongest-weakest | -0.0897141 | [-0.0962187, -0.0838193] | 9.999e-05 | 0.00239976 |
| qwen4b | wiki | group_only:strongest-random | -0.0758824 | [-0.0808902, -0.0713806] | 9.999e-05 | 0.00239976 |
| qwen4b | wiki | group_only:weakest-random | +0.0138317 | [+0.0106238, +0.0171392] | 9.999e-05 | 0.00239976 |
| qwen4b | wiki | full_context_marginal:strongest-weakest | -0.0731647 | [-0.0779847, -0.0683989] | 9.999e-05 | 0.00239976 |
| qwen4b | wiki | full_context_marginal:strongest-random | -0.0672295 | [-0.0713598, -0.0631949] | 9.999e-05 | 0.00239976 |
| qwen4b | wiki | full_context_marginal:weakest-random | +0.0059352 | [+0.0034118, +0.0083792] | 9.999e-05 | 0.00239976 |
| qwen4b | c4 | group_only:strongest-weakest | -0.0436000 | [-0.0456879, -0.0415595] | 9.999e-05 | 0.00239976 |
| qwen4b | c4 | group_only:strongest-random | -0.0387351 | [-0.0407861, -0.0366497] | 9.999e-05 | 0.00239976 |
| qwen4b | c4 | group_only:weakest-random | +0.0048649 | [+0.0033476, +0.0063983] | 9.999e-05 | 0.00239976 |
| qwen4b | c4 | full_context_marginal:strongest-weakest | -0.0354679 | [-0.0373929, -0.0335791] | 9.999e-05 | 0.00239976 |
| qwen4b | c4 | full_context_marginal:strongest-random | -0.0311223 | [-0.0329447, -0.0293412] | 9.999e-05 | 0.00239976 |
| qwen4b | c4 | full_context_marginal:weakest-random | +0.0043456 | [+0.0031416, +0.0055181] | 9.999e-05 | 0.00239976 |

## B_development

| Model | Corpus | Endpoint | delta-log-PPL | 95% CI | p | Holm p |
|---|---|---|---:|---:|---:|---:|
| llama8b | wiki | kl_vetoed_ce_approved:mean_actual_minus_full | -0.0002245 | [-0.0015969, +0.0011553] | 0.756624 | 1 |
| llama8b | wiki | kl_vetoed_ce_approved:mean_actual_minus_random | -0.0014681 | [-0.0030015, +0.0000325] | 0.0591941 | 1 |
| llama8b | wiki | kl_vetoed_ce_approved:q90_actual_minus_random | -0.0015272 | [-0.0076053, +0.0048848] | 0.575142 | 1 |
| llama8b | wiki | kl_vetoed_ce_approved:q95_actual_minus_random | -0.0027403 | [-0.0111250, +0.0064566] | 0.486451 | 1 |
| llama8b | wiki | kl_vetoed_ce_approved:q99_actual_minus_random | -0.0007102 | [-0.0096156, +0.0071893] | 0.752425 | 1 |
| llama8b | wiki | kl_vetoed_ce_approved:worst_actual_minus_random | -0.0003044 | [-0.0119677, +0.0086459] | 1 | 1 |
| llama8b | wiki | kl_vetoed_ce_approved:regression_probability_actual_minus_random | -0.2452830 | [-0.4157887, -0.0572981] | 0.00819918 | 0.311569 |
| llama8b | wiki | ce_vetoed_kl_approved:mean_actual_minus_full | +0.0002357 | [-0.0012027, +0.0016255] | 0.742626 | 1 |
| llama8b | wiki | ce_vetoed_kl_approved:mean_actual_minus_random | +0.0008926 | [-0.0002957, +0.0020850] | 0.141086 | 1 |
| llama8b | wiki | ce_vetoed_kl_approved:q90_actual_minus_random | +0.0012432 | [-0.0025530, +0.0056712] | 0.582342 | 1 |
| llama8b | wiki | ce_vetoed_kl_approved:q95_actual_minus_random | +0.0026326 | [-0.0032827, +0.0068117] | 0.258074 | 1 |
| llama8b | wiki | ce_vetoed_kl_approved:q99_actual_minus_random | -0.0007724 | [-0.0051026, +0.0047270] | 0.777922 | 1 |
| llama8b | wiki | ce_vetoed_kl_approved:worst_actual_minus_random | -0.0017276 | [-0.0054338, +0.0053921] | 0.516448 | 1 |
| llama8b | wiki | ce_vetoed_kl_approved:regression_probability_actual_minus_random | +0.1320755 | [-0.0191226, +0.2827642] | 0.0854915 | 1 |
| llama8b | c4 | kl_vetoed_ce_approved:mean_actual_minus_full | -0.0000424 | [-0.0012264, +0.0010812] | 0.938206 | 1 |
| llama8b | c4 | kl_vetoed_ce_approved:mean_actual_minus_random | -0.0009792 | [-0.0024080, +0.0003018] | 0.151585 | 1 |
| llama8b | c4 | kl_vetoed_ce_approved:q90_actual_minus_random | -0.0023168 | [-0.0041794, -0.0000685] | 0.030497 | 1 |
| llama8b | c4 | kl_vetoed_ce_approved:q95_actual_minus_random | -0.0022212 | [-0.0087603, +0.0005127] | 0.166183 | 1 |
| llama8b | c4 | kl_vetoed_ce_approved:q99_actual_minus_random | -0.0074835 | [-0.0234850, +0.0103244] | 0.223278 | 1 |
| llama8b | c4 | kl_vetoed_ce_approved:worst_actual_minus_random | -0.0141705 | [-0.0312377, +0.0103856] | 0.469853 | 1 |
| llama8b | c4 | kl_vetoed_ce_approved:regression_probability_actual_minus_random | +0.0303030 | [-0.0478879, +0.1079563] | 0.433657 | 1 |
| llama8b | c4 | ce_vetoed_kl_approved:mean_actual_minus_full | -0.0000671 | [-0.0011743, +0.0010206] | 0.905009 | 1 |
| llama8b | c4 | ce_vetoed_kl_approved:mean_actual_minus_random | -0.0002251 | [-0.0014551, +0.0010071] | 0.712329 | 1 |
| llama8b | c4 | ce_vetoed_kl_approved:q90_actual_minus_random | +0.0013290 | [-0.0007694, +0.0030186] | 0.185281 | 1 |
| llama8b | c4 | ce_vetoed_kl_approved:q95_actual_minus_random | -0.0003276 | [-0.0051230, +0.0019974] | 0.748925 | 1 |
| llama8b | c4 | ce_vetoed_kl_approved:q99_actual_minus_random | -0.0102338 | [-0.0189073, -0.0011046] | 0.0209979 | 0.734927 |
| llama8b | c4 | ce_vetoed_kl_approved:worst_actual_minus_random | -0.0169773 | [-0.0226094, -0.0067183] | 0.00039996 | 0.0163984 |
| llama8b | c4 | ce_vetoed_kl_approved:regression_probability_actual_minus_random | +0.0259740 | [-0.0433494, +0.0995078] | 0.489351 | 1 |
| qwen4b | wiki | kl_vetoed_ce_approved:mean_actual_minus_full | -0.0910686 | [-0.0956854, -0.0865907] | 9.999e-05 | 0.00559944 |
| qwen4b | wiki | kl_vetoed_ce_approved:mean_actual_minus_random | -0.0668897 | [-0.0704600, -0.0636476] | 9.999e-05 | 0.00559944 |
| qwen4b | wiki | kl_vetoed_ce_approved:q90_actual_minus_random | -0.0654633 | [-0.0709700, -0.0571790] | 9.999e-05 | 0.00559944 |
| qwen4b | wiki | kl_vetoed_ce_approved:q95_actual_minus_random | -0.0623844 | [-0.0723931, -0.0511076] | 9.999e-05 | 0.00559944 |
| qwen4b | wiki | kl_vetoed_ce_approved:q99_actual_minus_random | -0.0575610 | [-0.0692590, -0.0491889] | 9.999e-05 | 0.00559944 |
| qwen4b | wiki | kl_vetoed_ce_approved:worst_actual_minus_random | -0.0583931 | [-0.0725405, -0.0497412] | 9.999e-05 | 0.00559944 |
| qwen4b | wiki | kl_vetoed_ce_approved:regression_probability_actual_minus_random | -0.0188679 | [-0.0565679, +0.0000358] | 0.441356 | 1 |
| qwen4b | wiki | ce_vetoed_kl_approved:mean_actual_minus_full | +0.0053684 | [+0.0024469, +0.0082112] | 0.00089991 | 0.0350965 |
| qwen4b | wiki | ce_vetoed_kl_approved:mean_actual_minus_random | +0.0033029 | [+0.0008384, +0.0056685] | 0.00829917 | 0.311569 |
| qwen4b | wiki | ce_vetoed_kl_approved:q90_actual_minus_random | +0.0037103 | [-0.0056442, +0.0110498] | 0.365763 | 1 |
| qwen4b | wiki | ce_vetoed_kl_approved:q95_actual_minus_random | +0.0046932 | [-0.0092499, +0.0128621] | 0.440056 | 1 |
| qwen4b | wiki | ce_vetoed_kl_approved:q99_actual_minus_random | -0.0031374 | [-0.0127425, +0.0093728] | 0.635636 | 1 |
| qwen4b | wiki | ce_vetoed_kl_approved:worst_actual_minus_random | -0.0028961 | [-0.0111055, +0.0106864] | 0.389761 | 1 |
| qwen4b | wiki | ce_vetoed_kl_approved:regression_probability_actual_minus_random | +0.1132075 | [-0.0373019, +0.2645849] | 0.129387 | 1 |
| qwen4b | c4 | kl_vetoed_ce_approved:mean_actual_minus_full | -0.0552273 | [-0.0581086, -0.0523978] | 9.999e-05 | 0.00559944 |
| qwen4b | c4 | kl_vetoed_ce_approved:mean_actual_minus_random | -0.0423068 | [-0.0447291, -0.0398886] | 9.999e-05 | 0.00559944 |
| qwen4b | c4 | kl_vetoed_ce_approved:q90_actual_minus_random | -0.0286045 | [-0.0373218, -0.0207518] | 9.999e-05 | 0.00559944 |
| qwen4b | c4 | kl_vetoed_ce_approved:q95_actual_minus_random | -0.0181818 | [-0.0258084, -0.0117772] | 0.00059994 | 0.0239976 |
| qwen4b | c4 | kl_vetoed_ce_approved:q99_actual_minus_random | +0.0021771 | [-0.0147753, +0.0192060] | 0.822418 | 1 |
| qwen4b | c4 | kl_vetoed_ce_approved:worst_actual_minus_random | +0.0126213 | [-0.0011298, +0.0216584] | 0.0345965 | 1 |
| qwen4b | c4 | kl_vetoed_ce_approved:regression_probability_actual_minus_random | -0.0995671 | [-0.1428268, -0.0605758] | 9.999e-05 | 0.00559944 |
| qwen4b | c4 | ce_vetoed_kl_approved:mean_actual_minus_full | +0.0098459 | [+0.0084026, +0.0112757] | 9.999e-05 | 0.00559944 |
| qwen4b | c4 | ce_vetoed_kl_approved:mean_actual_minus_random | +0.0069786 | [+0.0055508, +0.0084033] | 9.999e-05 | 0.00559944 |
| qwen4b | c4 | ce_vetoed_kl_approved:q90_actual_minus_random | +0.0093490 | [+0.0065828, +0.0117466] | 9.999e-05 | 0.00559944 |
| qwen4b | c4 | ce_vetoed_kl_approved:q95_actual_minus_random | +0.0107686 | [+0.0074316, +0.0137254] | 9.999e-05 | 0.00559944 |
| qwen4b | c4 | ce_vetoed_kl_approved:q99_actual_minus_random | +0.0146792 | [+0.0015752, +0.0205482] | 0.00989901 | 0.356364 |
| qwen4b | c4 | ce_vetoed_kl_approved:worst_actual_minus_random | +0.0038046 | [-0.0045134, +0.0157098] | 0.59824 | 1 |
| qwen4b | c4 | ce_vetoed_kl_approved:regression_probability_actual_minus_random | +0.2121212 | [+0.1468385, +0.2768169] | 9.999e-05 | 0.00559944 |

## C_development

| Model | Corpus | Endpoint | delta-log-PPL | 95% CI | p | Holm p |
|---|---|---|---:|---:|---:|---:|
| llama8b | wiki | actual_residual | +0.0012755 | [-0.0002687, +0.0029435] | 0.115288 | 0.345865 |
| llama8b | wiki | actual_minus_random_residual | +0.0029851 | [+0.0006922, +0.0052710] | 0.0116988 | 0.0467953 |
| llama8b | c4 | actual_residual | +0.0010942 | [-0.0003563, +0.0026978] | 0.161784 | 0.345865 |
| llama8b | c4 | actual_minus_random_residual | -0.0007534 | [-0.0027745, +0.0011462] | 0.441456 | 0.441456 |
| qwen4b | wiki | actual_residual | +0.0219833 | [+0.0176199, +0.0263537] | 9.999e-05 | 0.00079992 |
| qwen4b | wiki | actual_minus_random_residual | +0.0230524 | [+0.0180714, +0.0277010] | 9.999e-05 | 0.00079992 |
| qwen4b | c4 | actual_residual | +0.0083891 | [+0.0064337, +0.0104210] | 9.999e-05 | 0.00079992 |
| qwen4b | c4 | actual_minus_random_residual | +0.0093939 | [+0.0069167, +0.0118330] | 9.999e-05 | 0.00079992 |

## A_validation

| Model | Corpus | Endpoint | delta-log-PPL | 95% CI | p | Holm p |
|---|---|---|---:|---:|---:|---:|
| mistral7b | wiki | group_only:strongest-weakest | -0.0026750 | [-0.0035339, -0.0018021] | 9.999e-05 | 0.00039996 |
| mistral7b | wiki | full_context_marginal:strongest-weakest | -0.0021324 | [-0.0028437, -0.0013364] | 9.999e-05 | 0.00039996 |
| mistral7b | c4 | group_only:strongest-weakest | -0.0011237 | [-0.0018247, -0.0003864] | 0.00259974 | 0.00259974 |
| mistral7b | c4 | full_context_marginal:strongest-weakest | -0.0012540 | [-0.0019122, -0.0005874] | 0.00029997 | 0.00059994 |

## B_validation

| Model | Corpus | Endpoint | delta-log-PPL | 95% CI | p | Holm p |
|---|---|---|---:|---:|---:|---:|
| mistral7b | wiki | kl_vetoed_ce_approved:mean_actual_minus_full | +0.0001794 | [-0.0006143, +0.0009099] | 0.645335 | 1 |
| mistral7b | wiki | kl_vetoed_ce_approved:mean_actual_minus_random | +0.0001796 | [-0.0005878, +0.0009120] | 0.637036 | 1 |
| mistral7b | wiki | kl_vetoed_ce_approved:q90_actual_minus_random | -0.0004759 | [-0.0020700, +0.0019368] | 0.532747 | 1 |
| mistral7b | wiki | kl_vetoed_ce_approved:q95_actual_minus_random | +0.0001594 | [-0.0028195, +0.0042818] | 0.913309 | 1 |
| mistral7b | wiki | kl_vetoed_ce_approved:q99_actual_minus_random | +0.0035149 | [-0.0011061, +0.0062873] | 0.0885911 | 1 |
| mistral7b | wiki | kl_vetoed_ce_approved:worst_actual_minus_random | +0.0041033 | [-0.0019093, +0.0062641] | 0.0979902 | 1 |
| mistral7b | wiki | kl_vetoed_ce_approved:regression_probability_actual_minus_random | -0.0350877 | [-0.1750596, +0.1056421] | 0.617438 | 1 |
| mistral7b | c4 | kl_vetoed_ce_approved:mean_actual_minus_full | -0.0002352 | [-0.0010075, +0.0006295] | 0.571443 | 1 |
| mistral7b | c4 | kl_vetoed_ce_approved:mean_actual_minus_random | +0.0000314 | [-0.0006037, +0.0006938] | 0.925407 | 1 |
| mistral7b | c4 | kl_vetoed_ce_approved:q90_actual_minus_random | +0.0000207 | [-0.0011819, +0.0010202] | 0.969803 | 1 |
| mistral7b | c4 | kl_vetoed_ce_approved:q95_actual_minus_random | -0.0001686 | [-0.0017813, +0.0009719] | 0.828617 | 1 |
| mistral7b | c4 | kl_vetoed_ce_approved:q99_actual_minus_random | -0.0005181 | [-0.0047151, +0.0179204] | 0.845115 | 1 |
| mistral7b | c4 | kl_vetoed_ce_approved:worst_actual_minus_random | +0.0290371 | [+0.0086796, +0.0390932] | 9.999e-05 | 0.00139986 |
| mistral7b | c4 | kl_vetoed_ce_approved:regression_probability_actual_minus_random | -0.0085106 | [-0.0768957, +0.0592745] | 0.805719 | 1 |

## C_validation

| Model | Corpus | Endpoint | delta-log-PPL | 95% CI | p | Holm p |
|---|---|---|---:|---:|---:|---:|
| mistral7b | wiki | actual_residual | +0.0010071 | [-0.0001226, +0.0020936] | 0.0747925 | 0.29917 |
| mistral7b | wiki | actual_minus_random_residual | +0.0009275 | [-0.0006178, +0.0024359] | 0.234177 | 0.70253 |
| mistral7b | c4 | actual_residual | +0.0004303 | [-0.0005231, +0.0014016] | 0.384062 | 0.768123 |
| mistral7b | c4 | actual_minus_random_residual | -0.0001904 | [-0.0014825, +0.0011138] | 0.777022 | 0.777022 |

