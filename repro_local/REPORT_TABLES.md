# N16K64 reproduction on one RTX PRO 6000 (fake quant + native mixfp4 kernel)

## PPL: original / fake quant (this machine) / native kernel (this machine)

| Model | Corpus | NVFP4 | 4Over6 | N8K64(k=3) | N16K64(k=3) |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B | Wiki | 6.9317 / 6.9296 / 6.9382 | 6.8754 / 6.8807 / 6.8826 | 6.8414 / 6.8441 / 6.8479 | 6.8432 / 6.8416 / 6.8468 |
| Llama-3.1-8B | C4 | 9.9379 / 9.9302 / 9.9386 | 9.8254 / 9.8141 / 9.8209 | 9.7763 / 9.7774 / 9.7802 | 9.7762 / 9.7771 / 9.7831 |
| Qwen3.8-27B | Wiki | 7.5769 / 7.5446 / 7.5420 | 7.2839 / 7.3016 / 7.2911 | 7.2254 / 7.2569 / 7.2515 | 7.2456 / 7.2587 / 7.2581 |
| Qwen3.8-27B | C4 | 10.2237 / 10.2222 / 10.2225 | 10.1894 / 10.1882 / 10.1899 | 10.1491 / 10.1500 / 10.1473 | 10.1564 / 10.1559 / 10.1572 |
| Qwen3-4B | Wiki | 13.9557 / 13.9584 / 13.9313 | 14.2062 / 14.2183 / 14.2106 | 11.8841 / 11.7066 / 11.6925 | 12.1968 / 12.1095 / 12.0888 |
| Qwen3-4B | C4 | 17.2621 / 17.2493 / 17.2596 | 17.3024 / 17.3175 / 17.3062 | 15.8483 / 15.7092 / 15.6958 | 16.0485 / 15.9714 / 15.9623 |
| Mistral-7B-v0.3 | Wiki | 5.5487 / 5.5508 / 5.5511 | 5.5233 / 5.5210 / 5.5213 | 5.4987 / 5.5029 / 5.5049 | 5.5019 / 5.5029 / 5.5035 |
| Mistral-7B-v0.3 | C4 | 8.0938 / 8.0949 / 8.0934 | 8.0674 / 8.0681 / 8.0669 | 8.0456 / 8.0449 / 8.0508 | 8.0475 / 8.0521 / 8.0505 |
| Phi-4 | Wiki | 6.7013 / 6.6998 / 6.7013 | 6.6641 / 6.6627 / 6.6625 | 6.6152 / 6.6324 / 6.6347 | 6.6288 / 6.6366 / 6.6396 |
| Phi-4 | C4 | 10.5838 / 10.5834 / 10.5807 | 10.5485 / 10.5473 / 10.5404 | 10.5000 / 10.5132 / 10.5161 | 10.5073 / 10.5205 / 10.5176 |

Each cell: original / fake quant / native kernel. Tolerance: |rel| ≤ 0.5%.

| Model | Corpus | Policy | fake rel | native rel |
|---|---|---|---:|---:|
| Llama-3.1-8B | Wiki | NVFP4 | -0.031% | +0.093% |
| Llama-3.1-8B | Wiki | 4Over6 | +0.077% | +0.104% |
| Llama-3.1-8B | Wiki | N8K64(k=3) | +0.039% | +0.095% |
| Llama-3.1-8B | Wiki | N16K64(k=3) | -0.024% | +0.052% |
| Llama-3.1-8B | C4 | NVFP4 | -0.077% | +0.008% |
| Llama-3.1-8B | C4 | 4Over6 | -0.115% | -0.046% |
| Llama-3.1-8B | C4 | N8K64(k=3) | +0.012% | +0.041% |
| Llama-3.1-8B | C4 | N16K64(k=3) | +0.009% | +0.071% |
| Qwen3.8-27B | Wiki | NVFP4 | -0.426% | -0.460% |
| Qwen3.8-27B | Wiki | 4Over6 | +0.243% | +0.100% |
| Qwen3.8-27B | Wiki | N8K64(k=3) | +0.436% | +0.362% |
| Qwen3.8-27B | Wiki | N16K64(k=3) | +0.181% | +0.173% |
| Qwen3.8-27B | C4 | NVFP4 | -0.015% | -0.011% |
| Qwen3.8-27B | C4 | 4Over6 | -0.012% | +0.005% |
| Qwen3.8-27B | C4 | N8K64(k=3) | +0.008% | -0.018% |
| Qwen3.8-27B | C4 | N16K64(k=3) | -0.005% | +0.008% |
| Qwen3-4B | Wiki | NVFP4 | +0.020% | -0.175% |
| Qwen3-4B | Wiki | 4Over6 | +0.085% | +0.031% |
| Qwen3-4B | Wiki | N8K64(k=3) | -1.494% ✗ | -1.612% ✗ |
| Qwen3-4B | Wiki | N16K64(k=3) | -0.716% ✗ | -0.885% ✗ |
| Qwen3-4B | C4 | NVFP4 | -0.074% | -0.015% |
| Qwen3-4B | C4 | 4Over6 | +0.087% | +0.022% |
| Qwen3-4B | C4 | N8K64(k=3) | -0.878% ✗ | -0.962% ✗ |
| Qwen3-4B | C4 | N16K64(k=3) | -0.481% | -0.537% ✗ |
| Mistral-7B-v0.3 | Wiki | NVFP4 | +0.037% | +0.042% |
| Mistral-7B-v0.3 | Wiki | 4Over6 | -0.042% | -0.037% |
| Mistral-7B-v0.3 | Wiki | N8K64(k=3) | +0.076% | +0.112% |
| Mistral-7B-v0.3 | Wiki | N16K64(k=3) | +0.019% | +0.030% |
| Mistral-7B-v0.3 | C4 | NVFP4 | +0.013% | -0.006% |
| Mistral-7B-v0.3 | C4 | 4Over6 | +0.009% | -0.007% |
| Mistral-7B-v0.3 | C4 | N8K64(k=3) | -0.008% | +0.065% |
| Mistral-7B-v0.3 | C4 | N16K64(k=3) | +0.058% | +0.037% |
| Phi-4 | Wiki | NVFP4 | -0.022% | +0.000% |
| Phi-4 | Wiki | 4Over6 | -0.022% | -0.024% |
| Phi-4 | Wiki | N8K64(k=3) | +0.260% | +0.295% |
| Phi-4 | Wiki | N16K64(k=3) | +0.118% | +0.163% |
| Phi-4 | C4 | NVFP4 | -0.004% | -0.029% |
| Phi-4 | C4 | 4Over6 | -0.011% | -0.077% |
| Phi-4 | C4 | N8K64(k=3) | +0.127% | +0.153% |
| Phi-4 | C4 | N16K64(k=3) | +0.126% | +0.098% |

## Tile counts (k=3, CE+KL conjunction)

| Model | Map | original | this machine | rel |
|---|---|---:|---:|---:|
| Llama-3.1-8B | N8K64(k=3) | 3,130 | 3,365 | +7.5% ✗ |
| Llama-3.1-8B | N16K64(k=3) | 1,781 | 1,866 | +4.8% |
| Qwen3.8-27B | N8K64(k=3) | 3,946 | 3,450 | -12.6% ✗ |
| Qwen3.8-27B | N16K64(k=3) | 2,168 | 1,957 | -9.7% ✗ |
| Qwen3-4B | N8K64(k=3) | 7,349 | 8,149 | +10.9% ✗ |
| Qwen3-4B | N16K64(k=3) | 4,077 | 4,399 | +7.9% ✗ |
| Mistral-7B-v0.3 | N8K64(k=3) | 7,501 | 6,703 | -10.6% ✗ |
| Mistral-7B-v0.3 | N16K64(k=3) | 4,179 | 3,824 | -8.5% ✗ |
| Phi-4 | N8K64(k=3) | 4,201 | 2,742 | -34.7% ✗ |
| Phi-4 | N16K64(k=3) | 2,184 | 1,452 | -33.5% ✗ |

## Paired effects, ΔlogPPL vs FourOverSix (tierC: same sign, |diff| ≤ max(0.25|orig|, 0.002))

| Model | Corpus | Contrast | Source | local ± 2SE | original | tierC |
|---|---|---|---|---:|---:|---|
| Llama-3.1-8B | Wiki | n8_k3−four_over_six | fake | -0.00534 ± 0.00164 | -0.00496 | PASS |
| Llama-3.1-8B | Wiki | n8_k3−four_over_six | native | -0.00518 ± 0.00162 | -0.00496 | PASS |
| Llama-3.1-8B | Wiki | n16_k3−four_over_six | fake | -0.00570 ± 0.00182 | -0.00469 | PASS |
| Llama-3.1-8B | Wiki | n16_k3−four_over_six | native | -0.00521 ± 0.00152 | -0.00469 | PASS |
| Llama-3.1-8B | Wiki | nvfp4−four_over_six | fake | +0.00708 ± 0.00222 | 0.00816 | PASS |
| Llama-3.1-8B | Wiki | nvfp4−four_over_six | native | +0.00805 ± 0.00202 | 0.00816 | PASS |
| Llama-3.1-8B | C4 | n8_k3−four_over_six | fake | -0.00374 ± 0.00151 | -0.00501 | PASS |
| Llama-3.1-8B | C4 | n8_k3−four_over_six | native | -0.00449 ± 0.00150 | -0.00501 | PASS |
| Llama-3.1-8B | C4 | n16_k3−four_over_six | fake | -0.00378 ± 0.00125 | -0.00502 | PASS |
| Llama-3.1-8B | C4 | n16_k3−four_over_six | native | -0.00385 ± 0.00145 | -0.00502 | PASS |
| Llama-3.1-8B | C4 | nvfp4−four_over_six | fake | +0.01176 ± 0.00240 | 0.01138 | PASS |
| Llama-3.1-8B | C4 | nvfp4−four_over_six | native | +0.01192 ± 0.00242 | 0.01138 | PASS |
| Qwen3.8-27B | Wiki | n8_k3−four_over_six | fake | -0.00614 ± 0.00456 | -0.00806 | PASS |
| Qwen3.8-27B | Wiki | n8_k3−four_over_six | native | -0.00532 ± 0.00326 | -0.00806 | FAIL |
| Qwen3.8-27B | Wiki | n16_k3−four_over_six | fake | -0.00589 ± 0.00368 | -0.00526 | PASS |
| Qwen3.8-27B | Wiki | n16_k3−four_over_six | native | -0.00454 ± 0.00363 | -0.00526 | PASS |
| Qwen3.8-27B | Wiki | nvfp4−four_over_six | fake | +0.03274 ± 0.00760 | 0.03944 | PASS |
| Qwen3.8-27B | Wiki | nvfp4−four_over_six | native | +0.03383 ± 0.00641 | 0.03944 | PASS |
| Qwen3.8-27B | C4 | n8_k3−four_over_six | fake | -0.00376 ± 0.00079 | -0.00396 | PASS |
| Qwen3.8-27B | C4 | n8_k3−four_over_six | native | -0.00389 ± 0.00082 | -0.00396 | PASS |
| Qwen3.8-27B | C4 | n16_k3−four_over_six | fake | -0.00317 ± 0.00074 | -0.00324 | PASS |
| Qwen3.8-27B | C4 | n16_k3−four_over_six | native | -0.00321 ± 0.00075 | -0.00324 | PASS |
| Qwen3.8-27B | C4 | nvfp4−four_over_six | fake | +0.00333 ± 0.00119 | 0.00336 | PASS |
| Qwen3.8-27B | C4 | nvfp4−four_over_six | native | +0.00320 ± 0.00103 | 0.00336 | PASS |
| Qwen3-4B | Wiki | n8_k3−four_over_six | fake | -0.19438 ± 0.00759 | -0.17848 | PASS |
| Qwen3-4B | Wiki | n8_k3−four_over_six | native | -0.19259 ± 0.00712 | -0.17848 | PASS |
| Qwen3-4B | Wiki | n16_k3−four_over_six | fake | -0.16054 ± 0.00612 | -0.15251 | PASS |
| Qwen3-4B | Wiki | n16_k3−four_over_six | native | -0.16171 ± 0.00642 | -0.15251 | PASS |
| Qwen3-4B | Wiki | nvfp4−four_over_six | fake | -0.01845 ± 0.00486 | -0.01779 | PASS |
| Qwen3-4B | Wiki | nvfp4−four_over_six | native | -0.01985 ± 0.00483 | -0.01779 | PASS |
| Qwen3-4B | C4 | n8_k3−four_over_six | fake | -0.09747 ± 0.00398 | -0.08778 | PASS |
| Qwen3-4B | C4 | n8_k3−four_over_six | native | -0.09698 ± 0.00399 | -0.08778 | PASS |
| Qwen3-4B | C4 | n16_k3−four_over_six | fake | -0.08092 ± 0.00329 | -0.07523 | PASS |
| Qwen3-4B | C4 | n16_k3−four_over_six | native | -0.08084 ± 0.00337 | -0.07523 | PASS |
| Qwen3-4B | C4 | nvfp4−four_over_six | fake | -0.00395 ± 0.00209 | -0.00233 | PASS |
| Qwen3-4B | C4 | nvfp4−four_over_six | native | -0.00270 ± 0.00213 | -0.00233 | PASS |
| Mistral-7B-v0.3 | Wiki | n8_k3−four_over_six | fake | -0.00328 ± 0.00093 | -0.00446 | PASS |
| Mistral-7B-v0.3 | Wiki | n8_k3−four_over_six | native | -0.00377 ± 0.00087 | -0.00446 | PASS |
| Mistral-7B-v0.3 | Wiki | n16_k3−four_over_six | fake | -0.00328 ± 0.00090 | -0.00389 | PASS |
| Mistral-7B-v0.3 | Wiki | n16_k3−four_over_six | native | -0.00322 ± 0.00091 | -0.00389 | PASS |
| Mistral-7B-v0.3 | Wiki | nvfp4−four_over_six | fake | +0.00538 ± 0.00121 | 0.00459 | PASS |
| Mistral-7B-v0.3 | Wiki | nvfp4−four_over_six | native | +0.00538 ± 0.00120 | 0.00459 | PASS |
| Mistral-7B-v0.3 | C4 | n8_k3−four_over_six | fake | -0.00288 ± 0.00124 | -0.00271 | PASS |
| Mistral-7B-v0.3 | C4 | n8_k3−four_over_six | native | -0.00199 ± 0.00073 | -0.00271 | PASS |
| Mistral-7B-v0.3 | C4 | n16_k3−four_over_six | fake | -0.00198 ± 0.00085 | -0.00247 | PASS |
| Mistral-7B-v0.3 | C4 | n16_k3−four_over_six | native | -0.00203 ± 0.00070 | -0.00247 | PASS |
| Mistral-7B-v0.3 | C4 | nvfp4−four_over_six | fake | +0.00332 ± 0.00107 | 0.00327 | PASS |
| Mistral-7B-v0.3 | C4 | nvfp4−four_over_six | native | +0.00328 ± 0.00083 | 0.00327 | PASS |
| Phi-4 | Wiki | n8_k3−four_over_six | fake | -0.00456 ± 0.00136 | -0.00737 | FAIL |
| Phi-4 | Wiki | n8_k3−four_over_six | native | -0.00464 ± 0.00116 | -0.00737 | FAIL |
| Phi-4 | Wiki | n16_k3−four_over_six | fake | -0.00393 ± 0.00134 | -0.00532 | PASS |
| Phi-4 | Wiki | n16_k3−four_over_six | native | -0.00344 ± 0.00138 | -0.00532 | PASS |
| Phi-4 | Wiki | nvfp4−four_over_six | fake | +0.00556 ± 0.00174 | 0.00556 | PASS |
| Phi-4 | Wiki | nvfp4−four_over_six | native | +0.00581 ± 0.00154 | 0.00556 | PASS |
| Phi-4 | C4 | n8_k3−four_over_six | fake | -0.00323 ± 0.00079 | -0.00461 | PASS |
| Phi-4 | C4 | n8_k3−four_over_six | native | -0.00287 ± 0.00075 | -0.00461 | PASS |
| Phi-4 | C4 | n16_k3−four_over_six | fake | -0.00254 ± 0.00081 | -0.00391 | PASS |
| Phi-4 | C4 | n16_k3−four_over_six | native | -0.00217 ± 0.00075 | -0.00391 | PASS |
| Phi-4 | C4 | nvfp4−four_over_six | fake | +0.00341 ± 0.00097 | 0.00334 | PASS |
| Phi-4 | C4 | nvfp4−four_over_six | native | +0.00382 ± 0.00095 | 0.00334 | PASS |

## Native kernel − fake quant, per-window paired ΔNLL

| Model | Corpus | Policy | mean ± 2SE |
|---|---|---|---:|
| Llama-3.1-8B | Wiki | NVFP4 | +0.00124 ± 0.00164 |
| Llama-3.1-8B | Wiki | 4Over6 | +0.00027 ± 0.00181 |
| Llama-3.1-8B | Wiki | N8K64(k=3) | +0.00056 ± 0.00142 |
| Llama-3.1-8B | Wiki | N16K64(k=3) | +0.00077 ± 0.00165 |
| Llama-3.1-8B | C4 | NVFP4 | +0.00084 ± 0.00143 |
| Llama-3.1-8B | C4 | 4Over6 | +0.00069 ± 0.00133 |
| Llama-3.1-8B | C4 | N8K64(k=3) | +0.00029 ± 0.00172 |
| Llama-3.1-8B | C4 | N16K64(k=3) | +0.00062 ± 0.00122 |
| Qwen3.8-27B | Wiki | NVFP4 | -0.00034 ± 0.00486 |
| Qwen3.8-27B | Wiki | 4Over6 | -0.00143 ± 0.00370 |
| Qwen3.8-27B | Wiki | N8K64(k=3) | -0.00074 ± 0.00392 |
| Qwen3.8-27B | Wiki | N16K64(k=3) | -0.00008 ± 0.00381 |
| Qwen3.8-27B | C4 | NVFP4 | +0.00003 ± 0.00090 |
| Qwen3.8-27B | C4 | 4Over6 | +0.00017 ± 0.00076 |
| Qwen3.8-27B | C4 | N8K64(k=3) | -0.00026 ± 0.00077 |
| Qwen3.8-27B | C4 | N16K64(k=3) | +0.00013 ± 0.00071 |
| Qwen3-4B | Wiki | NVFP4 | -0.00195 ± 0.00347 |
| Qwen3-4B | Wiki | 4Over6 | -0.00054 ± 0.00410 |
| Qwen3-4B | Wiki | N8K64(k=3) | -0.00121 ± 0.00187 |
| Qwen3-4B | Wiki | N16K64(k=3) | -0.00170 ± 0.00203 |
| Qwen3-4B | C4 | NVFP4 | +0.00060 ± 0.00132 |
| Qwen3-4B | C4 | 4Over6 | -0.00065 ± 0.00146 |
| Qwen3-4B | C4 | N8K64(k=3) | -0.00086 ± 0.00111 |
| Qwen3-4B | C4 | N16K64(k=3) | -0.00057 ± 0.00120 |
| Mistral-7B-v0.3 | Wiki | NVFP4 | +0.00005 ± 0.00093 |
| Mistral-7B-v0.3 | Wiki | 4Over6 | +0.00005 ± 0.00077 |
| Mistral-7B-v0.3 | Wiki | N8K64(k=3) | +0.00036 ± 0.00080 |
| Mistral-7B-v0.3 | Wiki | N16K64(k=3) | +0.00011 ± 0.00078 |
| Mistral-7B-v0.3 | C4 | NVFP4 | -0.00019 ± 0.00065 |
| Mistral-7B-v0.3 | C4 | 4Over6 | -0.00015 ± 0.00097 |
| Mistral-7B-v0.3 | C4 | N8K64(k=3) | +0.00073 ± 0.00079 |
| Mistral-7B-v0.3 | C4 | N16K64(k=3) | -0.00020 ± 0.00072 |
| Phi-4 | Wiki | NVFP4 | +0.00022 ± 0.00137 |
| Phi-4 | Wiki | 4Over6 | -0.00002 ± 0.00136 |
| Phi-4 | Wiki | N8K64(k=3) | +0.00035 ± 0.00124 |
| Phi-4 | Wiki | N16K64(k=3) | +0.00046 ± 0.00129 |
| Phi-4 | C4 | NVFP4 | -0.00025 ± 0.00075 |
| Phi-4 | C4 | 4Over6 | -0.00065 ± 0.00076 |
| Phi-4 | C4 | N8K64(k=3) | +0.00027 ± 0.00072 |
| Phi-4 | C4 | N16K64(k=3) | -0.00028 ± 0.00075 |
