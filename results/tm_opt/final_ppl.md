### Llama-3.1-8B

| backend | map | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | ΔWiki vs NVFP4 | ΔC4 vs NVFP4 |
|---|---|---:|---:|---|---|---|---|
| native | FourOverSix | 6.8757 | 9.8254 | — | — | -0.00875 ± 0.00209 (better) | -0.01038 ± 0.00203 (better) |
| native | NVFP4 | 6.9361 | 9.9279 | +0.00875 ± 0.00209 (worse) | +0.01038 ± 0.00203 (worse) | — | — |
| native | tmopt-8x64 | 6.7822 | 9.6754 | -0.01369 ± 0.00192 (better) | -0.01539 ± 0.00290 (better) | -0.02244 ± 0.00225 (better) | -0.02577 ± 0.00422 (better) |
| native | tmopt-16x64 | 6.7907 | 9.6813 | -0.01243 ± 0.00194 (better) | -0.01477 ± 0.00282 (better) | -0.02118 ± 0.00233 (better) | -0.02515 ± 0.00412 (better) |
| native | tmopt-256x64 | 6.7957 | 9.7254 | -0.01170 ± 0.00184 (better) | -0.01023 ± 0.00235 (better) | -0.02046 ± 0.00231 (better) | -0.02061 ± 0.00349 (better) |
| native | tc-8x64 | 6.7827 | 9.6795 | -0.01362 ± 0.00177 (better) | -0.01496 ± 0.00284 (better) | -0.02237 ± 0.00226 (better) | -0.02534 ± 0.00411 (better) |
| native | tc-16x64 | 6.7865 | 9.6863 | -0.01305 ± 0.00178 (better) | -0.01426 ± 0.00280 (better) | -0.02180 ± 0.00229 (better) | -0.02464 ± 0.00407 (better) |
| native | tc-256x64 | 6.8012 | 9.7228 | -0.01090 ± 0.00176 (better) | -0.01050 ± 0.00263 (better) | -0.01965 ± 0.00221 (better) | -0.02088 ± 0.00381 (better) |
| native | mropt-8x64 | 6.8134 | 9.7644 | -0.00910 ± 0.00175 (better) | -0.00623 ± 0.00177 (better) | -0.01786 ± 0.00231 (better) | -0.01661 ± 0.00216 (better) |
| native | mropt-16x64 | 6.8259 | 9.7412 | -0.00727 ± 0.00163 (better) | -0.00861 ± 0.00219 (better) | -0.01602 ± 0.00209 (better) | -0.01899 ± 0.00347 (better) |
| native | mropt-256x64 | 6.8369 | 9.7594 | -0.00566 ± 0.00171 (better) | -0.00673 ± 0.00169 (better) | -0.01442 ± 0.00209 (better) | -0.01712 ± 0.00257 (better) |
| fake | FourOverSix | 6.8872 | 9.8294 | — | — | -0.00817 ± 0.00208 (better) | -0.01020 ± 0.00207 (better) |
| fake | NVFP4 | 6.9437 | 9.9302 | +0.00817 ± 0.00208 (worse) | +0.01020 ± 0.00207 (worse) | — | — |
| fake | tmopt-8x64 | 6.7792 | 9.6734 | -0.01581 ± 0.00199 (better) | -0.01599 ± 0.00359 (better) | -0.02398 ± 0.00237 (better) | -0.02619 ± 0.00440 (better) |
| fake | tmopt-16x64 | 6.7796 | 9.6775 | -0.01574 ± 0.00200 (better) | -0.01558 ± 0.00361 (better) | -0.02391 ± 0.00239 (better) | -0.02578 ± 0.00435 (better) |
| fake | tmopt-256x64 | 6.8016 | 9.7164 | -0.01250 ± 0.00190 (better) | -0.01157 ± 0.00311 (better) | -0.02067 ± 0.00241 (better) | -0.02176 ± 0.00387 (better) |
| fake | tc-8x64 | 6.7838 | 9.6816 | -0.01513 ± 0.00222 (better) | -0.01515 ± 0.00331 (better) | -0.02329 ± 0.00271 (better) | -0.02535 ± 0.00420 (better) |
| fake | tc-16x64 | 6.7879 | 9.6851 | -0.01451 ± 0.00195 (better) | -0.01479 ± 0.00330 (better) | -0.02268 ± 0.00242 (better) | -0.02499 ± 0.00410 (better) |
| fake | tc-256x64 | 6.8035 | 9.7386 | -0.01222 ± 0.00181 (better) | -0.00928 ± 0.00228 (better) | -0.02039 ± 0.00232 (better) | -0.01948 ± 0.00323 (better) |
| fake | mropt-8x64 | 6.8178 | 9.7606 | -0.01013 ± 0.00176 (better) | -0.00703 ± 0.00160 (better) | -0.01829 ± 0.00226 (better) | -0.01722 ± 0.00261 (better) |
| fake | mropt-256x64 | 6.8315 | 9.7737 | -0.00811 ± 0.00172 (better) | -0.00569 ± 0.00148 (better) | -0.01628 ± 0.00230 (better) | -0.01588 ± 0.00243 (better) |
| fake | BF16 | 6.2403 | 8.9579 | — | — | — | — |

| backend | TM-OPT+TC minus TM-OPT | ΔWiki | ΔC4 |
|---|---|---|---|
| native | 8x64 | +0.00007 ± 0.00143 (n.s.) | +0.00043 ± 0.00103 (n.s.) |
| native | 16x64 | -0.00062 ± 0.00144 (n.s.) | +0.00051 ± 0.00103 (n.s.) |
| native | 256x64 | +0.00081 ± 0.00152 (n.s.) | -0.00027 ± 0.00124 (n.s.) |
| fake | 8x64 | +0.00068 ± 0.00139 (n.s.) | +0.00084 ± 0.00099 (n.s.) |
| fake | 16x64 | +0.00123 ± 0.00130 (n.s.) | +0.00079 ± 0.00109 (n.s.) |
| fake | 256x64 | +0.00028 ± 0.00157 (n.s.) | +0.00229 ± 0.00178 (worse) |

Checks (FourOverSix window NLLs repeat across the merged processes): {"native items_llama8b_eval_native": true, "native tc_llama8b_eval_native": true, "native eval_native": true, "fake items_llama8b_eval_fake": true, "fake tc_llama8b_eval_fake": true, "fake eval_fake": true}

### Mistral-7B-v0.3

| backend | map | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | ΔWiki vs NVFP4 | ΔC4 vs NVFP4 |
|---|---|---:|---:|---|---|---|---|
| native | FourOverSix | 5.5225 | 8.0660 | — | — | -0.00591 ± 0.00113 (better) | -0.00367 ± 0.00092 (better) |
| native | NVFP4 | 5.5552 | 8.0957 | +0.00591 ± 0.00113 (worse) | +0.00367 ± 0.00092 (worse) | — | — |
| native | tmopt-8x64 | 5.4860 | 8.0274 | -0.00662 ± 0.00101 (better) | -0.00480 ± 0.00081 (better) | -0.01254 ± 0.00122 (better) | -0.00847 ± 0.00094 (better) |
| native | tmopt-16x64 | 5.4915 | 8.0285 | -0.00562 ± 0.00094 (better) | -0.00466 ± 0.00091 (better) | -0.01154 ± 0.00112 (better) | -0.00833 ± 0.00118 (better) |
| native | tmopt-256x64 | 5.4957 | 8.0317 | -0.00486 ± 0.00104 (better) | -0.00426 ± 0.00080 (better) | -0.01078 ± 0.00114 (better) | -0.00793 ± 0.00105 (better) |
| native | tc-8x64 | 5.4862 | 8.0261 | -0.00659 ± 0.00108 (better) | -0.00495 ± 0.00079 (better) | -0.01250 ± 0.00114 (better) | -0.00862 ± 0.00094 (better) |
| native | tc-16x64 | 5.4951 | 8.0264 | -0.00497 ± 0.00225 (better) | -0.00492 ± 0.00122 (better) | -0.01088 ± 0.00232 (better) | -0.00860 ± 0.00109 (better) |
| native | tc-256x64 | 5.4899 | 8.0326 | -0.00591 ± 0.00110 (better) | -0.00415 ± 0.00082 (better) | -0.01182 ± 0.00121 (better) | -0.00782 ± 0.00097 (better) |
| native | mropt-8x64 | 5.4837 | 8.0334 | -0.00704 ± 0.00097 (better) | -0.00405 ± 0.00075 (better) | -0.01295 ± 0.00121 (better) | -0.00772 ± 0.00092 (better) |
| native | mropt-16x64 | 5.4992 | 8.0338 | -0.00423 ± 0.00098 (better) | -0.00400 ± 0.00081 (better) | -0.01014 ± 0.00117 (better) | -0.00767 ± 0.00107 (better) |
| native | mropt-256x64 | 5.4950 | 8.0354 | -0.00498 ± 0.00098 (better) | -0.00380 ± 0.00074 (better) | -0.01089 ± 0.00112 (better) | -0.00747 ± 0.00102 (better) |
| fake | FourOverSix | 5.5260 | 8.0665 | — | — | -0.00488 ± 0.00118 (better) | -0.00363 ± 0.00094 (better) |
| fake | NVFP4 | 5.5531 | 8.0958 | +0.00488 ± 0.00118 (worse) | +0.00363 ± 0.00094 (worse) | — | — |
| fake | tmopt-8x64 | 5.4847 | 8.0290 | -0.00751 ± 0.00101 (better) | -0.00466 ± 0.00123 (better) | -0.01239 ± 0.00129 (better) | -0.00829 ± 0.00103 (better) |
| fake | tmopt-16x64 | 5.4883 | 8.0262 | -0.00685 ± 0.00096 (better) | -0.00501 ± 0.00120 (better) | -0.01173 ± 0.00126 (better) | -0.00864 ± 0.00101 (better) |
| fake | tmopt-256x64 | 5.4938 | 8.0330 | -0.00585 ± 0.00100 (better) | -0.00417 ± 0.00127 (better) | -0.01073 ± 0.00128 (better) | -0.00780 ± 0.00108 (better) |
| fake | tc-8x64 | 5.4910 | 8.0208 | -0.00636 ± 0.00105 (better) | -0.00569 ± 0.00085 (better) | -0.01124 ± 0.00127 (better) | -0.00932 ± 0.00119 (better) |
| fake | tc-16x64 | 5.4900 | 8.0282 | -0.00655 ± 0.00103 (better) | -0.00476 ± 0.00144 (better) | -0.01143 ± 0.00132 (better) | -0.00839 ± 0.00116 (better) |
| fake | tc-256x64 | 5.4880 | 8.0321 | -0.00691 ± 0.00095 (better) | -0.00427 ± 0.00088 (better) | -0.01179 ± 0.00116 (better) | -0.00790 ± 0.00091 (better) |
| fake | mropt-8x64 | 5.4865 | 8.0309 | -0.00717 ± 0.00100 (better) | -0.00442 ± 0.00116 (better) | -0.01205 ± 0.00132 (better) | -0.00805 ± 0.00100 (better) |
| fake | mropt-256x64 | 5.4937 | 8.0365 | -0.00586 ± 0.00095 (better) | -0.00372 ± 0.00125 (better) | -0.01074 ± 0.00128 (better) | -0.00735 ± 0.00105 (better) |
| fake | BF16 | 5.3182 | 7.8306 | — | — | — | — |

| backend | TM-OPT+TC minus TM-OPT | ΔWiki | ΔC4 |
|---|---|---|---|
| native | 8x64 | +0.00003 ± 0.00085 (n.s.) | -0.00015 ± 0.00066 (n.s.) |
| native | 16x64 | +0.00066 ± 0.00229 (n.s.) | -0.00026 ± 0.00142 (n.s.) |
| native | 256x64 | -0.00104 ± 0.00096 (better) | +0.00010 ± 0.00071 (n.s.) |
| fake | 8x64 | +0.00115 ± 0.00084 (worse) | -0.00103 ± 0.00139 (n.s.) |
| fake | 16x64 | +0.00031 ± 0.00094 (n.s.) | +0.00025 ± 0.00074 (n.s.) |
| fake | 256x64 | -0.00106 ± 0.00093 (better) | -0.00011 ± 0.00085 (n.s.) |

Checks (FourOverSix window NLLs repeat across the merged processes): {"native items_mistral7b_eval_native": true, "native tc_mistral7b_eval_native": true, "native eval_native": true, "fake items_mistral7b_eval_fake": true, "fake tc_mistral7b_eval_fake": true, "fake eval_fake": true}

### Phi-4

| backend | map | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | ΔWiki vs NVFP4 | ΔC4 vs NVFP4 |
|---|---|---:|---:|---|---|---|---|
| native | FourOverSix | 6.6649 | 10.5456 | — | — | -0.00594 ± 0.00182 (better) | -0.00388 ± 0.00104 (better) |
| native | NVFP4 | 6.7046 | 10.5866 | +0.00594 ± 0.00182 (worse) | +0.00388 ± 0.00104 (worse) | — | — |
| native | tmopt-8x64 | 6.6089 | 10.4999 | -0.00843 ± 0.00142 (better) | -0.00434 ± 0.00080 (better) | -0.01437 ± 0.00179 (better) | -0.00822 ± 0.00104 (better) |
| native | tmopt-16x64 | 6.6108 | 10.5024 | -0.00815 ± 0.00144 (better) | -0.00410 ± 0.00087 (better) | -0.01409 ± 0.00194 (better) | -0.00798 ± 0.00110 (better) |
| native | tmopt-256x64 | 6.6282 | 10.5131 | -0.00551 ± 0.00142 (better) | -0.00309 ± 0.00084 (better) | -0.01145 ± 0.00178 (better) | -0.00697 ± 0.00101 (better) |
| native | tc-8x64 | 6.6128 | 10.4954 | -0.00784 ± 0.00155 (better) | -0.00477 ± 0.00086 (better) | -0.01378 ± 0.00196 (better) | -0.00865 ± 0.00111 (better) |
| native | tc-16x64 | 6.6161 | 10.5034 | -0.00735 ± 0.00152 (better) | -0.00402 ± 0.00083 (better) | -0.01329 ± 0.00196 (better) | -0.00790 ± 0.00107 (better) |
| native | tc-256x64 | 6.6308 | 10.5137 | -0.00512 ± 0.00141 (better) | -0.00303 ± 0.00083 (better) | -0.01106 ± 0.00182 (better) | -0.00691 ± 0.00107 (better) |
| native | mropt-8x64 | 6.6373 | 10.5216 | -0.00415 ± 0.00131 (better) | -0.00228 ± 0.00080 (better) | -0.01009 ± 0.00176 (better) | -0.00616 ± 0.00105 (better) |
| native | mropt-16x64 | 6.6288 | 10.5113 | -0.00542 ± 0.00137 (better) | -0.00326 ± 0.00087 (better) | -0.01136 ± 0.00193 (better) | -0.00714 ± 0.00106 (better) |
| native | mropt-256x64 | 6.6519 | 10.5237 | -0.00195 ± 0.00147 (better) | -0.00208 ± 0.00083 (better) | -0.00789 ± 0.00195 (better) | -0.00596 ± 0.00104 (better) |
| fake | FourOverSix | 6.6667 | 10.5437 | — | — | -0.00543 ± 0.00181 (better) | -0.00421 ± 0.00106 (better) |
| fake | NVFP4 | 6.7029 | 10.5882 | +0.00543 ± 0.00181 (worse) | +0.00421 ± 0.00106 (worse) | — | — |
| fake | tmopt-8x64 | 6.6123 | 10.5023 | -0.00819 ± 0.00130 (better) | -0.00393 ± 0.00088 (better) | -0.01361 ± 0.00204 (better) | -0.00815 ± 0.00113 (better) |
| fake | tmopt-16x64 | 6.6108 | 10.5043 | -0.00842 ± 0.00147 (better) | -0.00374 ± 0.00082 (better) | -0.01384 ± 0.00212 (better) | -0.00795 ± 0.00111 (better) |
| fake | tmopt-256x64 | 6.6266 | 10.5109 | -0.00602 ± 0.00137 (better) | -0.00311 ± 0.00087 (better) | -0.01145 ± 0.00182 (better) | -0.00732 ± 0.00112 (better) |
| fake | tc-8x64 | 6.6146 | 10.4990 | -0.00784 ± 0.00138 (better) | -0.00425 ± 0.00085 (better) | -0.01327 ± 0.00208 (better) | -0.00846 ± 0.00113 (better) |
| fake | tc-16x64 | 6.6105 | 10.5032 | -0.00846 ± 0.00143 (better) | -0.00384 ± 0.00087 (better) | -0.01389 ± 0.00205 (better) | -0.00805 ± 0.00116 (better) |
| fake | tc-256x64 | 6.6304 | 10.5161 | -0.00545 ± 0.00126 (better) | -0.00261 ± 0.00091 (better) | -0.01088 ± 0.00206 (better) | -0.00682 ± 0.00115 (better) |
| fake | mropt-8x64 | 6.6369 | 10.5149 | -0.00448 ± 0.00129 (better) | -0.00273 ± 0.00078 (better) | -0.00991 ± 0.00188 (better) | -0.00694 ± 0.00108 (better) |
| fake | mropt-256x64 | 6.6458 | 10.5293 | -0.00314 ± 0.00129 (better) | -0.00136 ± 0.00083 (better) | -0.00856 ± 0.00194 (better) | -0.00557 ± 0.00111 (better) |
| fake | BF16 | 6.4615 | 10.3098 | — | — | — | — |

| backend | TM-OPT+TC minus TM-OPT | ΔWiki | ΔC4 |
|---|---|---|---|
| native | 8x64 | +0.00059 ± 0.00120 (n.s.) | -0.00043 ± 0.00071 (n.s.) |
| native | 16x64 | +0.00080 ± 0.00116 (n.s.) | +0.00009 ± 0.00073 (n.s.) |
| native | 256x64 | +0.00039 ± 0.00124 (n.s.) | +0.00006 ± 0.00076 (n.s.) |
| fake | 8x64 | +0.00034 ± 0.00107 (n.s.) | -0.00031 ± 0.00067 (n.s.) |
| fake | 16x64 | -0.00004 ± 0.00106 (n.s.) | -0.00010 ± 0.00073 (n.s.) |
| fake | 256x64 | +0.00057 ± 0.00131 (n.s.) | +0.00050 ± 0.00071 (n.s.) |

Checks (FourOverSix window NLLs repeat across the merged processes): {"native items_phi4_eval_native": true, "native tc_phi4_eval_native": true, "native eval_native": true, "fake items_phi4_eval_fake": true, "fake tc_phi4_eval_fake": true, "fake eval_fake": true}

### Qwen3.8-27B

| backend | map | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | ΔWiki vs NVFP4 | ΔC4 vs NVFP4 |
|---|---|---:|---:|---|---|---|---|
| native | FourOverSix | 7.3092 | 10.1840 | — | — | -0.03596 ± 0.00684 (better) | -0.00359 ± 0.00114 (better) |
| native | NVFP4 | 7.5768 | 10.2206 | +0.03596 ± 0.00684 (worse) | +0.00359 ± 0.00114 (worse) | — | — |
| native | tmopt-8x64 | 7.0739 | 10.1257 | -0.03272 ± 0.00505 (better) | -0.00574 ± 0.00097 (better) | -0.06868 ± 0.00899 (better) | -0.00932 ± 0.00143 (better) |
| native | tmopt-16x64 | 7.1462 | 10.1342 | -0.02255 ± 0.00477 (better) | -0.00490 ± 0.00088 (better) | -0.05851 ± 0.00850 (better) | -0.00849 ± 0.00128 (better) |
| native | tmopt-256x64 | 7.1624 | 10.1488 | -0.02028 ± 0.00376 (better) | -0.00346 ± 0.00093 (better) | -0.05624 ± 0.00831 (better) | -0.00705 ± 0.00136 (better) |
| native | tc-8x64 | 7.1062 | 10.1261 | -0.02816 ± 0.00476 (better) | -0.00570 ± 0.00095 (better) | -0.06412 ± 0.00859 (better) | -0.00928 ± 0.00131 (better) |
| native | tc-16x64 | 7.1144 | 10.1335 | -0.02701 ± 0.00498 (better) | -0.00497 ± 0.00091 (better) | -0.06297 ± 0.00934 (better) | -0.00856 ± 0.00130 (better) |
| native | tc-256x64 | 7.1951 | 10.1482 | -0.01573 ± 0.00366 (better) | -0.00352 ± 0.00076 (better) | -0.05169 ± 0.00772 (better) | -0.00711 ± 0.00111 (better) |
| fake | FourOverSix | 7.3007 | 10.1858 | — | — | -0.03591 ± 0.00743 (better) | -0.00397 ± 0.00111 (better) |
| fake | NVFP4 | 7.5676 | 10.2263 | +0.03591 ± 0.00743 (worse) | +0.00397 ± 0.00111 (worse) | — | — |
| fake | tmopt-8x64 | 7.0696 | 10.1283 | -0.03216 ± 0.00463 (better) | -0.00566 ± 0.00099 (better) | -0.06807 ± 0.00930 (better) | -0.00963 ± 0.00142 (better) |
| fake | tmopt-16x64 | 7.1335 | 10.1326 | -0.02317 ± 0.00409 (better) | -0.00524 ± 0.00096 (better) | -0.05907 ± 0.00870 (better) | -0.00920 ± 0.00123 (better) |
| fake | tmopt-256x64 | 7.1623 | 10.1447 | -0.01914 ± 0.00414 (better) | -0.00405 ± 0.00085 (better) | -0.05505 ± 0.00844 (better) | -0.00801 ± 0.00131 (better) |
| fake | tc-8x64 | 7.0895 | 10.1241 | -0.02935 ± 0.00465 (better) | -0.00608 ± 0.00107 (better) | -0.06526 ± 0.00911 (better) | -0.01005 ± 0.00151 (better) |
| fake | tc-16x64 | 7.1007 | 10.1246 | -0.02777 ± 0.00440 (better) | -0.00603 ± 0.00092 (better) | -0.06368 ± 0.00907 (better) | -0.01000 ± 0.00137 (better) |
| fake | tc-256x64 | 7.1983 | 10.1486 | -0.01412 ± 0.00353 (better) | -0.00366 ± 0.00094 (better) | -0.05003 ± 0.00802 (better) | -0.00763 ± 0.00126 (better) |
| fake | BF16 | 7.0509 | 9.8935 | — | — | — | — |

| backend | TM-OPT+TC minus TM-OPT | ΔWiki | ΔC4 |
|---|---|---|---|
| native | 8x64 | +0.00456 ± 0.00244 (worse) | +0.00004 ± 0.00081 (n.s.) |
| native | 16x64 | -0.00446 ± 0.00264 (better) | -0.00007 ± 0.00079 (n.s.) |
| native | 256x64 | +0.00455 ± 0.00290 (worse) | -0.00006 ± 0.00088 (n.s.) |
| fake | 8x64 | +0.00281 ± 0.00259 (worse) | -0.00042 ± 0.00069 (n.s.) |
| fake | 16x64 | -0.00460 ± 0.00287 (better) | -0.00079 ± 0.00077 (better) |
| fake | 256x64 | +0.00502 ± 0.00293 (worse) | +0.00039 ± 0.00074 (n.s.) |

Checks (FourOverSix window NLLs repeat across the merged processes): {"single_process": true}

