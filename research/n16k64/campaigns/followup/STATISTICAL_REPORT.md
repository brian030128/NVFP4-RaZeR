# Statistical report

- Frozen protocol SHA-256: `22a78ee87078543da7d40ca39fabc56078f250438f0ec3d4a540adeec8c132a9`
- Inference unit: paired natural document/article cluster.
- Bootstrap: 10,000 deterministic replicates; finite Monte Carlo plus-one p-values.
- Multiplicity: Holm correction within the frozen A-development, A-validation, B-development, B-validation, and C-Mistral families.
- delta-log-PPL is delta NLL; relative PPL is `exp(delta_NLL)-1`.

## Dose-response statistics

| Role | Model | Corpus | Metric | Estimate | 95% CI | p | Holm p |
|---|---|---|---|---:|---:|---:|---:|
| development | llama8b | c4 | ce_pearson | +0.6914400 | [+0.3843951, +0.9113497] | 9.999e-05 | 0.00319968 |
| development | llama8b | c4 | ce_spearman | +0.4285714 | [+0.1225333, +0.6939619] | 0.00609939 | 0.0548945 |
| development | llama8b | c4 | ce_kendall | +0.2857143 | [+0.0171786, +0.5886071] | 0.0462954 | 0.231477 |
| development | llama8b | c4 | ce_slope | +0.4066400 | [+0.1609492, +0.6880372] | 0.00439956 | 0.0443956 |
| development | llama8b | c4 | margin_pearson | -0.6899928 | [-0.9104201, -0.3806018] | 0.00019998 | 0.00319968 |
| development | llama8b | c4 | margin_spearman | -0.4523810 | [-0.7153333, -0.1677143] | 0.00379962 | 0.0443956 |
| development | llama8b | c4 | margin_kendall | -0.3571429 | [-0.5854786, -0.0854786] | 0.00979902 | 0.0783922 |
| development | llama8b | c4 | margin_slope | -89.3622178 | [-150.4080868, -35.7956830] | 0.00369963 | 0.0443956 |
| development | llama8b | wiki | ce_pearson | +0.7201205 | [+0.2867188, +1.0263390] | 0.00139986 | 0.019598 |
| development | llama8b | wiki | ce_spearman | +0.3333333 | [-0.1523381, +0.8000429] | 0.222478 | 0.585241 |
| development | llama8b | wiki | ce_kendall | +0.2142857 | [-0.1561714, +0.6295429] | 0.321168 | 0.585241 |
| development | llama8b | wiki | ce_slope | +0.2284510 | [+0.0143266, +0.4420016] | 0.0373963 | 0.224378 |
| development | llama8b | wiki | margin_pearson | -0.7386962 | [-1.0411270, -0.3055930] | 0.00139986 | 0.019598 |
| development | llama8b | wiki | margin_spearman | -0.4285714 | [-0.8970452, +0.1029548] | 0.111589 | 0.446355 |
| development | llama8b | wiki | margin_kendall | -0.2857143 | [-0.6484286, +0.1372857] | 0.19508 | 0.585241 |
| development | llama8b | wiki | margin_slope | -52.3086891 | [-99.1751452, -5.3046442] | 0.0311969 | 0.218378 |
| development | qwen4b | c4 | ce_pearson | +0.9994776 | [+0.9966916, +1.0010893] | 9.999e-05 | 0.00319968 |
| development | qwen4b | c4 | ce_spearman | +1.0000000 | [+0.8678952, +1.0583714] | 9.999e-05 | 0.00319968 |
| development | qwen4b | c4 | ce_kendall | +1.0000000 | [+0.7644143, +1.1215571] | 9.999e-05 | 0.00319968 |
| development | qwen4b | c4 | ce_slope | +0.5308053 | [+0.4967616, +0.5651910] | 9.999e-05 | 0.00319968 |
| development | qwen4b | c4 | margin_pearson | -0.9918138 | [-0.9969345, -0.9852242] | 9.999e-05 | 0.00319968 |
| development | qwen4b | c4 | margin_spearman | -1.0000000 | [-1.0583714, -0.8678952] | 9.999e-05 | 0.00319968 |
| development | qwen4b | c4 | margin_kendall | -1.0000000 | [-1.1215571, -0.7644143] | 9.999e-05 | 0.00319968 |
| development | qwen4b | c4 | margin_slope | -1535.9889443 | [-1635.4184074, -1438.1739531] | 9.999e-05 | 0.00319968 |
| development | qwen4b | wiki | ce_pearson | +0.9938986 | [+0.9875531, +0.9981810] | 9.999e-05 | 0.00319968 |
| development | qwen4b | wiki | ce_spearman | +0.9285714 | [+0.8251095, +1.0155857] | 9.999e-05 | 0.00319968 |
| development | qwen4b | wiki | ce_kendall | +0.7857143 | [+0.6184500, +0.9755929] | 9.999e-05 | 0.00319968 |
| development | qwen4b | wiki | ce_slope | +0.9937837 | [+0.9342370, +1.0564721] | 9.999e-05 | 0.00319968 |
| development | qwen4b | wiki | margin_pearson | -0.9967901 | [-1.0001908, -0.9919713] | 9.999e-05 | 0.00319968 |
| development | qwen4b | wiki | margin_spearman | -0.9285714 | [-1.0155857, -0.8251095] | 9.999e-05 | 0.00319968 |
| development | qwen4b | wiki | margin_kendall | -0.7857143 | [-0.9755929, -0.6184500] | 9.999e-05 | 0.00319968 |
| development | qwen4b | wiki | margin_slope | -2899.3782324 | [-3084.5026929, -2725.0887504] | 9.999e-05 | 0.00319968 |
| validation | mistral7b | c4 | ce_pearson | +0.8796925 | [+0.6910977, +0.9999398] | 9.999e-05 | 0.00159984 |
| validation | mistral7b | c4 | ce_spearman | +0.5714286 | [+0.2182952, +0.8849619] | 0.00079992 | 0.00639936 |
| validation | mistral7b | c4 | ce_kendall | +0.4285714 | [+0.1296357, +0.7010643] | 0.00429957 | 0.0257974 |
| validation | mistral7b | c4 | ce_slope | +0.2462372 | [+0.1757416, +0.3202918] | 9.999e-05 | 0.00159984 |
| validation | mistral7b | c4 | margin_pearson | -0.8816464 | [-1.0015919, -0.6894879] | 9.999e-05 | 0.00159984 |
| validation | mistral7b | c4 | margin_spearman | -0.5714286 | [-0.8849619, -0.2182952] | 0.00079992 | 0.00639936 |
| validation | mistral7b | c4 | margin_kendall | -0.4285714 | [-0.7010643, -0.1296357] | 0.00429957 | 0.0257974 |
| validation | mistral7b | c4 | margin_slope | -212.8474393 | [-276.8256322, -152.1795740] | 9.999e-05 | 0.00159984 |
| validation | mistral7b | wiki | ce_pearson | +0.8736380 | [+0.7013541, +0.9919071] | 9.999e-05 | 0.00159984 |
| validation | mistral7b | wiki | ce_spearman | +0.2619048 | [-0.1101595, +0.6517452] | 0.20478 | 0.409559 |
| validation | mistral7b | wiki | ce_kendall | +0.2857143 | [-0.0624571, +0.5804000] | 0.0921908 | 0.368763 |
| validation | mistral7b | wiki | ce_slope | +0.2335302 | [+0.1473387, +0.3177720] | 9.999e-05 | 0.00159984 |
| validation | mistral7b | wiki | margin_pearson | -0.8793662 | [-0.9953822, -0.7090019] | 9.999e-05 | 0.00159984 |
| validation | mistral7b | wiki | margin_spearman | -0.2619048 | [-0.6517452, +0.1101595] | 0.20478 | 0.409559 |
| validation | mistral7b | wiki | margin_kendall | -0.2857143 | [-0.5804000, +0.0624571] | 0.0921908 | 0.368763 |
| validation | mistral7b | wiki | margin_slope | -202.8151353 | [-275.5978410, -128.7197275] | 9.999e-05 | 0.00159984 |

## Standardized pooled dose-response sensitivity

| Analysis | Slope | 95% CI | p | Models |
|---|---:|---:|---:|---|
| ce_all_models | +0.8597112 | [+0.7657290, +0.9354113] | 9.999e-05 | llama8b, mistral7b, qwen4b |
| ce_leave_qwen4b_out | +0.7912227 | [+0.6507037, +0.9047970] | 9.999e-05 | llama8b, mistral7b |
| margin_all_models | -0.8630509 | [-0.9382667, -0.7689776] | 9.999e-05 | llama8b, mistral7b, qwen4b |
| margin_leave_qwen4b_out | -0.7974254 | [-0.9102675, -0.6557864] | 9.999e-05 | llama8b, mistral7b |

## Matched-budget objective contrasts

| Role | Model | Corpus | Contrast | delta-log-PPL | 95% CI | Relative PPL | p | Holm p |
|---|---|---|---|---:|---:|---:|---:|---:|
| development | llama8b | c4 | ce_matched_k_minus_conjunction | +0.0006366 | [-0.0006142, +0.0018892] | +0.0637% | 0.312769 | 0.938306 |
| development | llama8b | c4 | kl_matched_k_minus_conjunction | -0.0005934 | [-0.0017981, +0.0005648] | -0.0593% | 0.318268 | 0.938306 |
| development | llama8b | c4 | ce_matched_k_minus_kl_matched_k | +0.0012300 | [-0.0000305, +0.0025550] | +0.1231% | 0.0657934 | 0.328967 |
| development | llama8b | c4 | ce_matched_k_minus_random | -0.0019911 | [-0.0030303, -0.0009316] | -0.1989% | 0.00029997 | 0.00239976 |
| development | llama8b | c4 | kl_matched_k_minus_random | -0.0032211 | [-0.0044686, -0.0019467] | -0.3216% | 9.999e-05 | 0.00239976 |
| development | llama8b | c4 | conjunction_minus_random | -0.0026277 | [-0.0039798, -0.0013018] | -0.2624% | 9.999e-05 | 0.00239976 |
| development | llama8b | wiki | ce_matched_k_minus_conjunction | -0.0011798 | [-0.0028959, +0.0004208] | -0.1179% | 0.159284 | 0.637136 |
| development | llama8b | wiki | kl_matched_k_minus_conjunction | -0.0018316 | [-0.0031955, -0.0003081] | -0.1830% | 0.0157984 | 0.0947905 |
| development | llama8b | wiki | ce_matched_k_minus_kl_matched_k | +0.0006517 | [-0.0013926, +0.0025011] | +0.0652% | 0.516148 | 0.938306 |
| development | llama8b | wiki | ce_matched_k_minus_random | -0.0038122 | [-0.0053803, -0.0021583] | -0.3805% | 9.999e-05 | 0.00239976 |
| development | llama8b | wiki | kl_matched_k_minus_random | -0.0044640 | [-0.0060919, -0.0028603] | -0.4454% | 9.999e-05 | 0.00239976 |
| development | llama8b | wiki | conjunction_minus_random | -0.0026324 | [-0.0041644, -0.0009960] | -0.2629% | 0.00109989 | 0.00769923 |
| development | qwen4b | c4 | ce_matched_k_minus_conjunction | -0.0567908 | [-0.0599530, -0.0535952] | -5.5208% | 9.999e-05 | 0.00239976 |
| development | qwen4b | c4 | kl_matched_k_minus_conjunction | +0.0222720 | [+0.0208450, +0.0237107] | +2.2522% | 9.999e-05 | 0.00239976 |
| development | qwen4b | c4 | ce_matched_k_minus_kl_matched_k | -0.0790628 | [-0.0829335, -0.0750288] | -7.6018% | 9.999e-05 | 0.00239976 |
| development | qwen4b | c4 | ce_matched_k_minus_random | -0.0989849 | [-0.1031959, -0.0944108] | -9.4244% | 9.999e-05 | 0.00239976 |
| development | qwen4b | c4 | kl_matched_k_minus_random | -0.0199220 | [-0.0218039, -0.0181250] | -1.9725% | 9.999e-05 | 0.00239976 |
| development | qwen4b | c4 | conjunction_minus_random | -0.0421941 | [-0.0442213, -0.0401889] | -4.1316% | 9.999e-05 | 0.00239976 |
| development | qwen4b | wiki | ce_matched_k_minus_conjunction | -0.0937961 | [-0.0982391, -0.0895410] | -8.9532% | 9.999e-05 | 0.00239976 |
| development | qwen4b | wiki | kl_matched_k_minus_conjunction | +0.0237745 | [+0.0208340, +0.0265906] | +2.4059% | 9.999e-05 | 0.00239976 |
| development | qwen4b | wiki | ce_matched_k_minus_kl_matched_k | -0.1175706 | [-0.1229016, -0.1127308] | -11.0922% | 9.999e-05 | 0.00239976 |
| development | qwen4b | wiki | ce_matched_k_minus_random | -0.1790643 | [-0.1871947, -0.1715061] | -16.3948% | 9.999e-05 | 0.00239976 |
| development | qwen4b | wiki | kl_matched_k_minus_random | -0.0614937 | [-0.0663637, -0.0571520] | -5.9641% | 9.999e-05 | 0.00239976 |
| development | qwen4b | wiki | conjunction_minus_random | -0.0852682 | [-0.0904298, -0.0807167] | -8.1734% | 9.999e-05 | 0.00239976 |
| validation | mistral7b | c4 | ce_matched_k_minus_conjunction | +0.0004716 | [-0.0004077, +0.0015466] | +0.0472% | 0.347665 | 0.69533 |
| validation | mistral7b | c4 | kl_matched_k_minus_conjunction | -0.0007244 | [-0.0013656, -0.0000914] | -0.0724% | 0.0265973 | 0.079792 |
| validation | mistral7b | c4 | ce_matched_k_minus_kl_matched_k | +0.0011960 | [+0.0004146, +0.0021505] | +0.1197% | 0.00819918 | 0.0409959 |
| validation | mistral7b | c4 | ce_matched_k_minus_random | -0.0012326 | [-0.0018935, -0.0005932] | -0.1232% | 0.00019998 | 0.0019998 |
| validation | mistral7b | c4 | kl_matched_k_minus_random | -0.0024286 | [-0.0035093, -0.0015723] | -0.2426% | 9.999e-05 | 0.00119988 |
| validation | mistral7b | c4 | conjunction_minus_random | -0.0017042 | [-0.0029903, -0.0006868] | -0.1703% | 0.00679932 | 0.0407959 |
| validation | mistral7b | wiki | ce_matched_k_minus_conjunction | +0.0000587 | [-0.0008978, +0.0009799] | +0.0059% | 0.90071 | 0.90071 |
| validation | mistral7b | wiki | kl_matched_k_minus_conjunction | -0.0010150 | [-0.0018121, -0.0002006] | -0.1014% | 0.0125987 | 0.050395 |
| validation | mistral7b | wiki | ce_matched_k_minus_kl_matched_k | +0.0010737 | [+0.0002905, +0.0018138] | +0.1074% | 0.00569943 | 0.039896 |
| validation | mistral7b | wiki | ce_matched_k_minus_random | -0.0018343 | [-0.0029282, -0.0007626] | -0.1833% | 0.00079992 | 0.00639936 |
| validation | mistral7b | wiki | kl_matched_k_minus_random | -0.0029080 | [-0.0040350, -0.0018216] | -0.2904% | 9.999e-05 | 0.00119988 |
| validation | mistral7b | wiki | conjunction_minus_random | -0.0018931 | [-0.0028563, -0.0009531] | -0.1891% | 0.00029997 | 0.00269973 |

## Mistral veto-completion family

| Corpus | Endpoint | Estimate | 95% CI | p | Holm p |
|---|---|---:|---:|---:|---:|
| c4 | mean_actual_minus_full | -0.0004313 | [-0.0012084, +0.0004446] | 0.309769 | 1 |
| c4 | mean_actual_minus_random | -0.0003659 | [-0.0010035, +0.0002672] | 0.259874 | 1 |
| c4 | q90_actual_minus_random | -0.0004173 | [-0.0018234, +0.0011308] | 0.592941 | 1 |
| c4 | q95_actual_minus_random | -0.0000219 | [-0.0021161, +0.0014968] | 0.986301 | 1 |
| c4 | q99_actual_minus_random | -0.0002644 | [-0.0050599, +0.0095950] | 0.868713 | 1 |
| c4 | regression_probability_actual_minus_random | +0.0425532 | [-0.0295217, +0.1194145] | 0.265973 | 1 |
| c4 | worst_actual_minus_random | +0.0145944 | [+0.0035799, +0.0199334] | 9.999e-05 | 0.00139986 |
| wiki | mean_actual_minus_full | -0.0008319 | [-0.0015014, -0.0001256] | 0.0193981 | 0.232777 |
| wiki | mean_actual_minus_random | -0.0012870 | [-0.0020351, -0.0004864] | 0.00169983 | 0.0220978 |
| wiki | q90_actual_minus_random | -0.0006253 | [-0.0035303, +0.0014990] | 0.627137 | 1 |
| wiki | q95_actual_minus_random | -0.0027127 | [-0.0056282, +0.0000148] | 0.0557944 | 0.613739 |
| wiki | q99_actual_minus_random | -0.0024074 | [-0.0058725, -0.0002287] | 0.090491 | 0.830917 |
| wiki | regression_probability_actual_minus_random | -0.1052632 | [-0.2465053, +0.0517404] | 0.174283 | 1 |
| wiki | worst_actual_minus_random | -0.0031091 | [-0.0065964, -0.0008029] | 0.0830917 | 0.830917 |

## Interpretation boundary

Correlations across composition-matched tile groups are aggregate ranking evidence. They do not establish reliable individual-tile signs, magnitudes, calibration, or causality. Natural-threshold objective maps have unequal selected counts and are descriptive rule-level comparisons; only matched-K contrasts isolate objective information.

