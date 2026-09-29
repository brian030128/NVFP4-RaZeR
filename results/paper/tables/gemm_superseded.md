# SUPERSEDED (fixed-order measurement; results/paper/PROTOCOL.md, deviation 1): GEMM tables

### SUPERSEDED (fixed-order measurement; results/paper/PROTOCOL.md, deviation 1): GEMM latency

#### SUPERSEDED (fixed-order measurement; results/paper/PROTOCOL.md, deviation 1): Llama-3.1-8B: GEMM kernel time per forward, µs (every quantized text Linear; CUPTI median of 20, fixed order)

| T | stock wA (NVFP4, FourOverSix) | stock wB | FlipQuant (ours) 16x64 (n16k64_wA) | FlipQuant (ours) 256x64 (n16k64_wA) | FlipQuant (ours) 8x64 (n8k64_wB) | FlipQuant (ours) 16x64 (n16k64_wA) vs stock wA (NVFP4, FourOverSix) | FlipQuant (ours) 256x64 (n16k64_wA) vs stock wA (NVFP4, FourOverSix) | FlipQuant (ours) 8x64 (n8k64_wB) vs stock wB | FlipQuant (ours) 8x64 (n8k64_wB) vs stock wA (NVFP4, FourOverSix) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 2883 | 4442 | 3151 | 3150 | 4896 | +9.3 % | +9.3 % | +10.2 % | +69.8 % |
| 256 | 4165 | 5404 | 4383 | 4397 | 5976 | +5.2 % | +5.6 % | +10.6 % | +43.5 % |
| 512 | 6125 | 6472 | 6442 | 6433 | 7389 | +5.2 % | +5.0 % | +14.2 % | +20.6 % |
| 1024 | 11846 | 12022 | 12380 | 12368 | 13620 | +4.5 % | +4.4 % | +13.3 % | +15.0 % |
| 2048 | 20933 | 20812 | 21950 | 21929 | 23348 | +4.9 % | +4.8 % | +12.2 % | +11.5 % |
| 4096 | 43210 | 43197 | 45322 | 45406 | 48645 | +4.9 % | +5.1 % | +12.6 % | +12.6 % |
| 8192 | 83866 | 83882 | 88387 | 88613 | 94671 | +5.4 % | +5.7 % | +12.9 % | +12.9 % |

FourOverSix and NVFP4 run the same stock GEMM (stock wA); their activation quantizers differ (supplementary):

| T | NVFP4 quantizer, µs / forward | FourOverSix quantizer, µs / forward | FourOverSix vs NVFP4 |
|---|---:|---:|---:|
| 128 | 799 | 716 | -10.4 % |
| 256 | 965 | 890 | -7.8 % |
| 512 | 1653 | 1502 | -9.1 % |
| 1024 | 2472 | 2810 | +13.7 % |
| 2048 | 4281 | 4717 | +10.2 % |
| 4096 | 8821 | 9205 | +4.4 % |
| 8192 | 17128 | 18233 | +6.5 % |

Quantizer launches net of the reuse measured in step 05.

#### SUPERSEDED (fixed-order measurement; results/paper/PROTOCOL.md, deviation 1): Mistral-7B-v0.3: GEMM kernel time per forward, µs (every quantized text Linear; CUPTI median of 20, fixed order)

| T | stock wA (NVFP4, FourOverSix) | stock wB | FlipQuant (ours) 16x64 (n16k64_wA) | FlipQuant (ours) 256x64 (n16k64_wA) | FlipQuant (ours) 8x64 (n8k64_wB) | FlipQuant (ours) 16x64 (n16k64_wA) vs stock wA (NVFP4, FourOverSix) | FlipQuant (ours) 256x64 (n16k64_wA) vs stock wA (NVFP4, FourOverSix) | FlipQuant (ours) 8x64 (n8k64_wB) vs stock wB | FlipQuant (ours) 8x64 (n8k64_wB) vs stock wA (NVFP4, FourOverSix) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 2872 | 4431 | 3154 | 3141 | 4890 | +9.8 % | +9.4 % | +10.4 % | +70.3 % |
| 256 | 4157 | 5394 | 4370 | 4388 | 5979 | +5.1 % | +5.5 % | +10.8 % | +43.8 % |
| 512 | 6131 | 6524 | 6489 | 6495 | 7393 | +5.8 % | +5.9 % | +13.3 % | +20.6 % |
| 1024 | 11823 | 12002 | 12308 | 12324 | 13263 | +4.1 % | +4.2 % | +10.5 % | +12.2 % |
| 2048 | 20721 | 20646 | 21678 | 21701 | 23086 | +4.6 % | +4.7 % | +11.8 % | +11.4 % |
| 4096 | 42751 | 42705 | 44727 | 44745 | 48023 | +4.6 % | +4.7 % | +12.5 % | +12.3 % |
| 8192 | 82818 | 82732 | 87090 | 87236 | 93453 | +5.2 % | +5.3 % | +13.0 % | +12.8 % |

FourOverSix and NVFP4 run the same stock GEMM (stock wA); their activation quantizers differ (supplementary):

| T | NVFP4 quantizer, µs / forward | FourOverSix quantizer, µs / forward | FourOverSix vs NVFP4 |
|---|---:|---:|---:|
| 128 | 795 | 714 | -10.2 % |
| 256 | 962 | 887 | -7.8 % |
| 512 | 1649 | 1509 | -8.5 % |
| 1024 | 2457 | 2809 | +14.4 % |
| 2048 | 4231 | 4669 | +10.4 % |
| 4096 | 8771 | 9183 | +4.7 % |
| 8192 | 16980 | 18041 | +6.2 % |

Quantizer launches net of the reuse measured in step 05.

#### SUPERSEDED (fixed-order measurement; results/paper/PROTOCOL.md, deviation 1): Phi-4: GEMM kernel time per forward, µs (every quantized text Linear; CUPTI median of 20, fixed order)

| T | stock wA (NVFP4, FourOverSix) | stock wB | FlipQuant (ours) 16x64 (n16k64_wA) | FlipQuant (ours) 256x64 (n16k64_wA) | FlipQuant (ours) 8x64 (n8k64_wB) | FlipQuant (ours) 16x64 (n16k64_wA) vs stock wA (NVFP4, FourOverSix) | FlipQuant (ours) 256x64 (n16k64_wA) vs stock wA (NVFP4, FourOverSix) | FlipQuant (ours) 8x64 (n8k64_wB) vs stock wB | FlipQuant (ours) 8x64 (n8k64_wB) vs stock wA (NVFP4, FourOverSix) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 3925 | 5341 | 4213 | 4210 | 5809 | +7.3 % | +7.3 % | +8.7 % | +48.0 % |
| 256 | 5935 | 6580 | 6077 | 6065 | 7286 | +2.4 % | +2.2 % | +10.7 % | +22.8 % |
| 512 | 10661 | 10648 | 11144 | 11152 | 12013 | +4.5 % | +4.6 % | +12.8 % | +12.7 % |
| 1024 | 20921 | 20618 | 21524 | 21580 | 23115 | +2.9 % | +3.2 % | +12.1 % | +10.5 % |
| 2048 | 42027 | 41935 | 43603 | 43676 | 46339 | +3.8 % | +3.9 % | +10.5 % | +10.3 % |
| 4096 | 80028 | 79842 | 83520 | 83716 | 89005 | +4.4 % | +4.6 % | +11.5 % | +11.2 % |
| 8192 | 159911 | 171123 | 177622 | 179553 | 190973 | +11.1 % | +12.3 % | +11.6 % | +19.4 % |

FourOverSix and NVFP4 run the same stock GEMM (stock wA); their activation quantizers differ (supplementary):

| T | NVFP4 quantizer, µs / forward | FourOverSix quantizer, µs / forward | FourOverSix vs NVFP4 |
|---|---:|---:|---:|
| 128 | 1114 | 1076 | -3.4 % |
| 256 | 1957 | 1729 | -11.6 % |
| 512 | 3182 | 2871 | -9.8 % |
| 1024 | 4343 | 5005 | +15.2 % |
| 2048 | 8221 | 8835 | +7.5 % |
| 4096 | 14356 | 15211 | +6.0 % |
| 8192 | 27194 | 28983 | +6.6 % |

Quantizer launches net of the reuse measured in step 05.

#### SUPERSEDED (fixed-order measurement; results/paper/PROTOCOL.md, deviation 1): Qwen3.8-27B: GEMM kernel time per forward, µs (every quantized text Linear; CUPTI median of 20, fixed order)

| T | stock wA (NVFP4, FourOverSix) | stock wB | FlipQuant (ours) 16x64 (n16k64_wA) | FlipQuant (ours) 256x64 (n16k64_wA) | FlipQuant (ours) 8x64 (n8k64_wB) | FlipQuant (ours) 16x64 (n16k64_wA) vs stock wA (NVFP4, FourOverSix) | FlipQuant (ours) 256x64 (n16k64_wA) vs stock wA (NVFP4, FourOverSix) | FlipQuant (ours) 8x64 (n8k64_wB) vs stock wB | FlipQuant (ours) 8x64 (n8k64_wB) vs stock wA (NVFP4, FourOverSix) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 7854 | 11829 | 8752 | 8763 | 12938 | +11.4 % | +11.6 % | +9.4 % | +64.7 % |
| 256 | 12077 | 14607 | 12576 | 12583 | 16152 | +4.1 % | +4.2 % | +10.6 % | +33.7 % |
| 512 | 18677 | 20039 | 19874 | 19923 | 22694 | +6.4 % | +6.7 % | +13.2 % | +21.5 % |
| 1024 | 37402 | 38476 | 39098 | 39005 | 42787 | +4.5 % | +4.3 % | +11.2 % | +14.4 % |
| 2048 | 75692 | 76225 | 79004 | 79096 | 84576 | +4.4 % | +4.5 % | +11.0 % | +11.7 % |
| 4096 | 145138 | 144865 | 151675 | 151682 | 161866 | +4.5 % | +4.5 % | +11.7 % | +11.5 % |
| 8192 | 286908 | 289776 | 303099 | 304240 | 325131 | +5.6 % | +6.0 % | +12.2 % | +13.3 % |

FourOverSix and NVFP4 run the same stock GEMM (stock wA); their activation quantizers differ (supplementary):

| T | NVFP4 quantizer, µs / forward | FourOverSix quantizer, µs / forward | FourOverSix vs NVFP4 |
|---|---:|---:|---:|
| 128 | 1760 | 1642 | -6.7 % |
| 256 | 2825 | 2539 | -10.1 % |
| 512 | 4905 | 4456 | -9.2 % |
| 1024 | 6492 | 7571 | +16.6 % |
| 2048 | 12811 | 13769 | +7.5 % |
| 4096 | 23238 | 24599 | +5.9 % |
| 8192 | 43385 | 46552 | +7.3 % |

Quantizer launches net of the reuse measured in step 05.

### SUPERSEDED: GEMM vs end-to-end consistency (deviation 1)

Per-forward GEMM time difference against the end-to-end CUDA-graph prefill difference, both in ms and as % of the reference prefill; FLAG when they differ by more than 1 % of the reference prefill (the activation quantizer and the rest of the forward are the same on both sides).

| model | comparison | batch x prompt | end to end, ms (%) [per-round range, %] | GEMM per forward, ms (%) | gap, pp | check |
|---|---|---|---:|---:|---:|---:|
| Llama-3.1-8B | FlipQuant (ours) 16x64 vs FourOverSix | 1x128 | +0.116 (+1.5 %) [+1.5, +1.7] | +0.268 (+3.6 %) | -2.0 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 16x64 vs FourOverSix | 1x256 | +0.135 (+1.4 %) [+1.4, +1.5] | +0.218 (+2.3 %) | -0.9 |  |
| Llama-3.1-8B | FlipQuant (ours) 16x64 vs FourOverSix | 1x512 | +0.535 (+3.5 %) [+1.6, +4.3] | +0.317 (+2.1 %) | +1.4 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 16x64 vs FourOverSix | 1x1024 | +0.754 (+2.4 %) [+1.9, +2.4] | +0.534 (+1.7 %) | +0.7 |  |
| Llama-3.1-8B | FlipQuant (ours) 16x64 vs FourOverSix | 1x2048 | +1.115 (+1.8 %) [+1.7, +2.0] | +1.016 (+1.6 %) | +0.2 |  |
| Llama-3.1-8B | FlipQuant (ours) 16x64 vs FourOverSix | 1x4096 | +2.638 (+2.0 %) [+1.7, +2.0] | +2.113 (+1.6 %) | +0.4 |  |
| Llama-3.1-8B | FlipQuant (ours) 16x64 vs FourOverSix | 1x8192 | +4.537 (+1.4 %) [+1.3, +1.6] | +4.521 (+1.4 %) | +0.0 |  |
| Llama-3.1-8B | FlipQuant (ours) 16x64 vs FourOverSix | 4x2048 | +4.622 (+1.7 %) [+1.6, +1.9] | +4.521 (+1.6 %) | +0.0 |  |
| Llama-3.1-8B | FlipQuant (ours) 256x64 vs FourOverSix | 1x128 | +0.118 (+1.6 %) [+1.6, +1.6] | +0.267 (+3.6 %) | -2.0 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 256x64 vs FourOverSix | 1x256 | +0.147 (+1.6 %) [+1.5, +1.6] | +0.232 (+2.5 %) | -0.9 |  |
| Llama-3.1-8B | FlipQuant (ours) 256x64 vs FourOverSix | 1x512 | +0.369 (+2.4 %) [+1.9, +5.1] | +0.308 (+2.0 %) | +0.4 |  |
| Llama-3.1-8B | FlipQuant (ours) 256x64 vs FourOverSix | 1x1024 | +0.852 (+2.7 %) [+2.5, +3.1] | +0.522 (+1.7 %) | +1.0 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 256x64 vs FourOverSix | 1x2048 | +1.306 (+2.1 %) [+1.8, +2.3] | +0.996 (+1.6 %) | +0.5 |  |
| Llama-3.1-8B | FlipQuant (ours) 256x64 vs FourOverSix | 1x4096 | +2.820 (+2.1 %) [+1.9, +2.2] | +2.197 (+1.6 %) | +0.5 |  |
| Llama-3.1-8B | FlipQuant (ours) 256x64 vs FourOverSix | 1x8192 | +5.014 (+1.5 %) [+1.4, +1.7] | +4.747 (+1.5 %) | +0.1 |  |
| Llama-3.1-8B | FlipQuant (ours) 256x64 vs FourOverSix | 4x2048 | +4.954 (+1.8 %) [+1.7, +2.0] | +4.747 (+1.7 %) | +0.1 |  |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x128 | +0.339 (+3.8 %) [+3.8, +3.9] | +0.454 (+5.1 %) | -1.3 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x256 | +0.429 (+4.1 %) [+4.0, +4.1] | +0.572 (+5.4 %) | -1.4 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x512 | +1.418 (+9.1 %) [+5.9, +9.8] | +0.916 (+5.9 %) | +3.2 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x1024 | +1.804 (+5.8 %) [+5.6, +6.3] | +1.598 (+5.1 %) | +0.7 |  |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x2048 | +3.214 (+5.1 %) [+4.9, +5.6] | +2.536 (+4.1 %) | +1.1 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x4096 | +6.598 (+4.9 %) [+4.6, +5.4] | +5.448 (+4.0 %) | +0.9 |  |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x8192 | +12.471 (+3.8 %) [+3.6, +4.2] | +10.789 (+3.3 %) | +0.5 |  |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 4x2048 | +12.870 (+4.6 %) [+4.4, +5.0] | +10.789 (+3.9 %) | +0.7 |  |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix | 1x128 | +1.752 (+23.5 %) [+23.4, +23.5] | +2.013 (+27.0 %) | -3.5 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix | 1x256 | +1.541 (+16.4 %) [+16.3, +16.4] | +1.811 (+19.3 %) | -2.9 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix | 1x512 | +1.824 (+12.0 %) [+7.7, +12.0] | +1.263 (+8.3 %) | +3.7 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix | 1x1024 | +1.458 (+4.6 %) [+4.0, +4.8] | +1.774 (+5.6 %) | -1.0 |  |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix | 1x2048 | +3.479 (+5.6 %) [+5.0, +5.9] | +2.415 (+3.9 %) | +1.7 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix | 1x4096 | +6.985 (+5.2 %) [+4.5, +5.2] | +5.435 (+4.0 %) | +1.1 | FLAG |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix | 1x8192 | +13.676 (+4.2 %) [+3.9, +4.5] | +10.805 (+3.3 %) | +0.9 |  |
| Llama-3.1-8B | FlipQuant (ours) 8x64 vs FourOverSix | 4x2048 | +13.872 (+5.0 %) [+4.7, +5.2] | +10.805 (+3.9 %) | +1.1 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 16x64 vs FourOverSix | 1x128 | +0.114 (+1.6 %) [+1.6, +1.7] | +0.282 (+4.0 %) | -2.4 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 16x64 vs FourOverSix | 1x256 | +0.115 (+1.3 %) [+1.3, +1.3] | +0.213 (+2.4 %) | -1.1 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 16x64 vs FourOverSix | 1x512 | +0.553 (+4.1 %) [+1.6, +4.7] | +0.358 (+2.7 %) | +1.5 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 16x64 vs FourOverSix | 1x1024 | +0.796 (+2.8 %) [+2.4, +3.0] | +0.485 (+1.7 %) | +1.1 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 16x64 vs FourOverSix | 1x2048 | +1.126 (+2.0 %) [+1.8, +2.1] | +0.957 (+1.7 %) | +0.3 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 16x64 vs FourOverSix | 1x4096 | +2.083 (+1.7 %) [+1.5, +1.9] | +1.976 (+1.6 %) | +0.1 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 16x64 vs FourOverSix | 1x8192 | +4.566 (+1.5 %) [+1.1, +1.6] | +4.272 (+1.4 %) | +0.1 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 16x64 vs FourOverSix | 4x2048 | +4.623 (+1.8 %) [+1.5, +1.9] | +4.272 (+1.7 %) | +0.1 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 256x64 vs FourOverSix | 1x128 | +0.119 (+1.7 %) [+1.7, +1.7] | +0.269 (+3.8 %) | -2.1 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 256x64 vs FourOverSix | 1x256 | +0.128 (+1.5 %) [+1.4, +1.5] | +0.230 (+2.6 %) | -1.2 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 256x64 vs FourOverSix | 1x512 | +0.561 (+4.2 %) [+3.5, +5.0] | +0.364 (+2.7 %) | +1.5 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 256x64 vs FourOverSix | 1x1024 | +0.971 (+3.4 %) [+2.4, +4.0] | +0.500 (+1.8 %) | +1.7 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 256x64 vs FourOverSix | 1x2048 | +1.253 (+2.2 %) [+2.2, +2.7] | +0.980 (+1.7 %) | +0.5 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 256x64 vs FourOverSix | 1x4096 | +2.436 (+2.0 %) [+1.8, +2.4] | +1.994 (+1.6 %) | +0.4 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 256x64 vs FourOverSix | 1x8192 | +5.045 (+1.7 %) [+1.5, +2.0] | +4.418 (+1.5 %) | +0.2 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 256x64 vs FourOverSix | 4x2048 | +5.120 (+2.0 %) [+1.9, +2.3] | +4.418 (+1.7 %) | +0.3 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x128 | +0.338 (+4.0 %) [+4.0, +4.1] | +0.459 (+5.5 %) | -1.4 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x256 | +0.437 (+4.4 %) [+4.4, +4.5] | +0.584 (+5.9 %) | -1.5 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x512 | +1.102 (+8.0 %) [+7.7, +8.7] | +0.869 (+6.3 %) | +1.7 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x1024 | +2.113 (+7.6 %) [+6.5, +8.1] | +1.261 (+4.5 %) | +3.1 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x2048 | +3.219 (+5.7 %) [+5.2, +6.0] | +2.440 (+4.3 %) | +1.4 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x4096 | +6.825 (+5.5 %) [+5.2, +5.9] | +5.318 (+4.3 %) | +1.2 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x8192 | +12.934 (+4.3 %) [+4.1, +4.6] | +10.720 (+3.5 %) | +0.7 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 4x2048 | +13.253 (+5.2 %) [+4.8, +5.5] | +10.720 (+4.2 %) | +1.0 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix | 1x128 | +1.761 (+25.2 %) [+25.2, +25.2] | +2.018 (+28.9 %) | -3.7 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix | 1x256 | +1.557 (+17.7 %) [+17.6, +17.7] | +1.821 (+20.7 %) | -3.0 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix | 1x512 | +1.441 (+10.8 %) [+10.1, +12.6] | +1.262 (+9.5 %) | +1.3 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix | 1x1024 | +1.558 (+5.5 %) [+5.1, +6.6] | +1.439 (+5.1 %) | +0.4 |  |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix | 1x2048 | +3.411 (+6.1 %) [+5.7, +6.3] | +2.365 (+4.2 %) | +1.9 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix | 1x4096 | +6.945 (+5.6 %) [+5.4, +5.8] | +5.272 (+4.3 %) | +1.4 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix | 1x8192 | +14.147 (+4.7 %) [+4.6, +4.8] | +10.635 (+3.5 %) | +1.2 | FLAG |
| Mistral-7B-v0.3 | FlipQuant (ours) 8x64 vs FourOverSix | 4x2048 | +14.376 (+5.7 %) [+5.5, +5.7] | +10.635 (+4.2 %) | +1.5 | FLAG |
| Phi-4 | FlipQuant (ours) 16x64 vs FourOverSix | 1x128 | +0.095 (+0.8 %) [+0.8, +0.9] | +0.288 (+2.5 %) | -1.7 | FLAG |
| Phi-4 | FlipQuant (ours) 16x64 vs FourOverSix | 1x256 | +0.407 (+2.7 %) [+0.7, +3.1] | +0.142 (+0.9 %) | +1.8 | FLAG |
| Phi-4 | FlipQuant (ours) 16x64 vs FourOverSix | 1x512 | +0.654 (+2.4 %) [+2.2, +2.7] | +0.483 (+1.8 %) | +0.6 |  |
| Phi-4 | FlipQuant (ours) 16x64 vs FourOverSix | 1x1024 | +1.150 (+2.1 %) [+1.8, +2.1] | +0.603 (+1.1 %) | +1.0 |  |
| Phi-4 | FlipQuant (ours) 16x64 vs FourOverSix | 1x2048 | +2.123 (+1.9 %) [+1.7, +2.0] | +1.576 (+1.4 %) | +0.5 |  |
| Phi-4 | FlipQuant (ours) 16x64 vs FourOverSix | 1x4096 | +4.050 (+1.7 %) [+1.6, +1.9] | +3.492 (+1.5 %) | +0.2 |  |
| Phi-4 | FlipQuant (ours) 16x64 vs FourOverSix | 1x8192 | +8.708 (+1.6 %) [+1.4, +1.7] | +17.712 (+3.2 %) | -1.6 | FLAG |
| Phi-4 | FlipQuant (ours) 16x64 vs FourOverSix | 4x2048 | +8.375 (+1.7 %) [+1.6, +1.8] | +17.712 (+3.6 %) | -1.9 | FLAG |
| Phi-4 | FlipQuant (ours) 256x64 vs FourOverSix | 1x128 | +0.096 (+0.8 %) [+0.8, +0.9] | +0.285 (+2.5 %) | -1.7 | FLAG |
| Phi-4 | FlipQuant (ours) 256x64 vs FourOverSix | 1x256 | +0.446 (+3.0 %) [+2.7, +3.9] | +0.130 (+0.9 %) | +2.1 | FLAG |
| Phi-4 | FlipQuant (ours) 256x64 vs FourOverSix | 1x512 | +0.661 (+2.4 %) [+2.2, +2.8] | +0.492 (+1.8 %) | +0.6 |  |
| Phi-4 | FlipQuant (ours) 256x64 vs FourOverSix | 1x1024 | +1.160 (+2.1 %) [+1.8, +2.5] | +0.660 (+1.2 %) | +0.9 |  |
| Phi-4 | FlipQuant (ours) 256x64 vs FourOverSix | 1x2048 | +2.246 (+2.0 %) [+1.8, +2.2] | +1.649 (+1.5 %) | +0.5 |  |
| Phi-4 | FlipQuant (ours) 256x64 vs FourOverSix | 1x4096 | +4.444 (+1.9 %) [+1.6, +2.1] | +3.689 (+1.6 %) | +0.3 |  |
| Phi-4 | FlipQuant (ours) 256x64 vs FourOverSix | 1x8192 | +9.338 (+1.7 %) [+1.5, +1.9] | +19.642 (+3.5 %) | -1.8 | FLAG |
| Phi-4 | FlipQuant (ours) 256x64 vs FourOverSix | 4x2048 | +9.344 (+1.9 %) [+1.7, +2.1] | +19.642 (+4.0 %) | -2.1 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x128 | +0.355 (+2.9 %) [+2.8, +2.9] | +0.467 (+3.8 %) | -0.9 |  |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x256 | +0.633 (+4.2 %) [+3.4, +5.3] | +0.706 (+4.6 %) | -0.5 |  |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x512 | +1.710 (+6.1 %) [+5.1, +6.5] | +1.365 (+4.9 %) | +1.2 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x1024 | +3.067 (+5.5 %) [+5.1, +5.7] | +2.497 (+4.5 %) | +1.0 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x2048 | +6.176 (+5.5 %) [+5.4, +5.8] | +4.404 (+3.9 %) | +1.6 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x4096 | +12.322 (+5.2 %) [+4.9, +5.3] | +9.163 (+3.8 %) | +1.3 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x8192 | +24.039 (+4.3 %) [+4.1, +4.4] | +19.850 (+3.5 %) | +0.7 |  |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 4x2048 | +24.228 (+4.9 %) [+4.8, +5.1] | +19.850 (+4.0 %) | +0.9 |  |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix | 1x128 | +1.318 (+11.6 %) [+11.6, +11.6] | +1.883 (+16.6 %) | -5.0 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix | 1x256 | +0.883 (+5.9 %) [+5.4, +6.3] | +1.352 (+9.0 %) | -3.1 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix | 1x512 | +1.964 (+7.1 %) [+5.6, +7.3] | +1.352 (+4.9 %) | +2.2 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix | 1x1024 | +3.321 (+6.0 %) [+5.5, +6.3] | +2.194 (+4.0 %) | +2.0 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix | 1x2048 | +6.701 (+6.0 %) [+5.5, +6.4] | +4.312 (+3.9 %) | +2.1 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix | 1x4096 | +13.521 (+5.7 %) [+5.2, +5.8] | +8.977 (+3.8 %) | +1.9 | FLAG |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix | 1x8192 | +27.502 (+4.9 %) [+4.5, +5.1] | +31.063 (+5.5 %) | -0.6 |  |
| Phi-4 | FlipQuant (ours) 8x64 vs FourOverSix | 4x2048 | +27.749 (+5.7 %) [+5.2, +5.8] | +31.063 (+6.4 %) | -0.7 |  |
| Qwen3.8-27B | FlipQuant (ours) 16x64 vs FourOverSix | 1x128 | +0.093 (+0.2 %) [-0.0, +0.2] | +0.899 (+1.6 %) | -1.4 | FLAG |
| Qwen3.8-27B | FlipQuant (ours) 16x64 vs FourOverSix | 1x256 | +0.405 (+0.6 %) [+0.4, +0.7] | +0.499 (+0.7 %) | -0.1 |  |
| Qwen3.8-27B | FlipQuant (ours) 16x64 vs FourOverSix | 1x512 | +1.225 (+1.1 %) [+0.9, +1.2] | +1.197 (+1.1 %) | +0.0 |  |
| Qwen3.8-27B | FlipQuant (ours) 16x64 vs FourOverSix | 1x1024 | +2.361 (+1.2 %) [+0.8, +1.4] | +1.696 (+0.9 %) | +0.3 |  |
| Qwen3.8-27B | FlipQuant (ours) 16x64 vs FourOverSix | 1x2048 | +4.218 (+1.1 %) [+1.1, +1.4] | +3.312 (+0.9 %) | +0.2 |  |
| Qwen3.8-27B | FlipQuant (ours) 16x64 vs FourOverSix | 1x4096 | +7.745 (+1.0 %) [+0.8, +1.1] | +6.536 (+0.8 %) | +0.1 |  |
| Qwen3.8-27B | FlipQuant (ours) 16x64 vs FourOverSix | 1x8192 | +15.083 (+0.8 %) [+0.7, +1.0] | +16.191 (+0.9 %) | -0.1 |  |
| Qwen3.8-27B | FlipQuant (ours) 16x64 vs FourOverSix | 4x2048 | +16.546 (+1.0 %) [+0.8, +1.1] | +16.191 (+1.0 %) | +0.0 |  |
| Qwen3.8-27B | FlipQuant (ours) 256x64 vs FourOverSix | 1x128 | +0.121 (+0.2 %) [+0.2, +0.5] | +0.910 (+1.6 %) | -1.4 | FLAG |
| Qwen3.8-27B | FlipQuant (ours) 256x64 vs FourOverSix | 1x256 | +0.408 (+0.6 %) [+0.4, +0.6] | +0.506 (+0.7 %) | -0.1 |  |
| Qwen3.8-27B | FlipQuant (ours) 256x64 vs FourOverSix | 1x512 | +1.113 (+1.0 %) [+1.0, +1.3] | +1.246 (+1.1 %) | -0.1 |  |
| Qwen3.8-27B | FlipQuant (ours) 256x64 vs FourOverSix | 1x1024 | +2.375 (+1.2 %) [+0.9, +1.3] | +1.603 (+0.8 %) | +0.4 |  |
| Qwen3.8-27B | FlipQuant (ours) 256x64 vs FourOverSix | 1x2048 | +4.097 (+1.1 %) [+0.9, +1.2] | +3.404 (+0.9 %) | +0.2 |  |
| Qwen3.8-27B | FlipQuant (ours) 256x64 vs FourOverSix | 1x4096 | +8.369 (+1.0 %) [+0.8, +1.1] | +6.544 (+0.8 %) | +0.2 |  |
| Qwen3.8-27B | FlipQuant (ours) 256x64 vs FourOverSix | 1x8192 | +16.112 (+0.9 %) [+0.7, +1.0] | +17.332 (+1.0 %) | -0.1 |  |
| Qwen3.8-27B | FlipQuant (ours) 256x64 vs FourOverSix | 4x2048 | +16.255 (+1.0 %) [+0.9, +1.0] | +17.332 (+1.0 %) | -0.1 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x128 | +0.531 (+0.9 %) [+0.6, +1.0] | +1.108 (+1.8 %) | -0.9 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x256 | +1.146 (+1.5 %) [+1.5, +1.6] | +1.545 (+2.1 %) | -0.5 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x512 | +2.774 (+2.5 %) [+2.3, +2.7] | +2.655 (+2.4 %) | +0.1 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x1024 | +5.500 (+2.8 %) [+2.5, +2.9] | +4.311 (+2.2 %) | +0.6 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x2048 | +11.283 (+3.0 %) [+2.7, +3.0] | +8.351 (+2.2 %) | +0.8 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x4096 | +22.082 (+2.7 %) [+2.4, +2.8] | +17.001 (+2.1 %) | +0.6 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 1x8192 | +42.001 (+2.3 %) [+2.2, +2.5] | +35.355 (+1.9 %) | +0.4 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix (wB) | 4x2048 | +44.064 (+2.6 %) [+2.4, +2.7] | +35.355 (+2.1 %) | +0.5 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix | 1x128 | +3.565 (+6.2 %) [+6.1, +6.2] | +5.084 (+8.8 %) | -2.6 | FLAG |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix | 1x256 | +3.232 (+4.4 %) [+4.2, +4.5] | +4.074 (+5.6 %) | -1.2 | FLAG |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix | 1x512 | +3.288 (+3.0 %) [+2.7, +3.1] | +4.017 (+3.7 %) | -0.7 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix | 1x1024 | +6.484 (+3.3 %) [+2.8, +3.7] | +5.385 (+2.8 %) | +0.6 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix | 1x2048 | +12.607 (+3.3 %) [+3.0, +3.4] | +8.884 (+2.3 %) | +1.0 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix | 1x4096 | +23.356 (+2.9 %) [+2.6, +3.0] | +16.728 (+2.1 %) | +0.8 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix | 1x8192 | +45.931 (+2.5 %) [+2.3, +2.7] | +38.222 (+2.1 %) | +0.4 |  |
| Qwen3.8-27B | FlipQuant (ours) 8x64 vs FourOverSix | 4x2048 | +47.068 (+2.8 %) [+2.6, +2.9] | +38.222 (+2.3 %) | +0.5 |  |

### SUPERSEDED (fixed-order measurement; results/paper/PROTOCOL.md, deviation 1): GEMM kernel time per shape, µs (wN = CTA tile width picked by the tile table)

| model | projection | out x in | modules | T | stock wA (NVFP4, FourOverSix) | stock wB | FlipQuant (ours) 16x64 (n16k64_wA) | FlipQuant (ours) 256x64 (n16k64_wA) | FlipQuant (ours) 8x64 (n8k64_wB) | NVFP4 quantizer | FourOverSix quantizer |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | q_proj | 4096x4096 | 32 | 128 | 9.2 (w64) | 15.2 | 10.6 (w32) | 10.7 (w32) | 16.7 | 4.3 | 3.8 |
| Llama-3.1-8B | q_proj | 4096x4096 | 32 | 256 | 11.3 (w64) | 15.4 | 11.7 (w64) | 11.6 (w64) | 16.9 | 4.8 | 4.5 |
| Llama-3.1-8B | q_proj | 4096x4096 | 32 | 512 | 16.2 (w128) | 16.0 | 16.8 (w128) | 16.7 (w128) | 17.7 | 8.0 | 7.4 |
| Llama-3.1-8B | q_proj | 4096x4096 | 32 | 1024 | 30.9 (w128) | 30.6 | 32.0 (w128) | 31.9 (w128) | 35.0 | 12.0 | 13.8 |
| Llama-3.1-8B | q_proj | 4096x4096 | 32 | 2048 | 47.0 (w128) | 45.9 | 50.4 (w128) | 49.2 (w128) | 54.3 | 22.5 | 25.1 |
| Llama-3.1-8B | q_proj | 4096x4096 | 32 | 4096 | 101.2 (w128) | 100.0 | 106.3 (w128) | 105.9 (w128) | 113.2 | 35.3 | 39.7 |
| Llama-3.1-8B | q_proj | 4096x4096 | 32 | 8192 | 198.8 (w128) | 194.0 | 210.6 (w128) | 210.9 (w128) | 223.9 | 82.1 | 87.4 |
| Llama-3.1-8B | k_proj | 1024x4096 | 32 | 128 | 6.7 (w16) | 15.1 | 7.7 (w16) | 7.6 (w16) | 16.5 | 4.3 | 3.8 |
| Llama-3.1-8B | k_proj | 1024x4096 | 32 | 256 | 7.0 (w32) | 15.1 | 8.5 (w32) | 8.5 (w32) | 16.8 | 4.8 | 4.5 |
| Llama-3.1-8B | k_proj | 1024x4096 | 32 | 512 | 9.1 (w64) | 15.1 | 9.8 (w32) | 9.8 (w32) | 16.8 | 8.0 | 7.4 |
| Llama-3.1-8B | k_proj | 1024x4096 | 32 | 1024 | 10.9 (w64) | 15.3 | 11.5 (w64) | 11.4 (w64) | 16.8 | 12.0 | 13.8 |
| Llama-3.1-8B | k_proj | 1024x4096 | 32 | 2048 | 16.1 (w128) | 16.1 | 16.8 (w128) | 16.6 (w128) | 17.5 | 19.6 | 21.8 |
| Llama-3.1-8B | k_proj | 1024x4096 | 32 | 4096 | 30.7 (w128) | 30.9 | 33.2 (w128) | 34.9 (w128) | 37.2 | 35.4 | 39.8 |
| Llama-3.1-8B | k_proj | 1024x4096 | 32 | 8192 | 47.6 (w128) | 47.2 | 50.5 (w128) | 50.5 (w128) | 55.5 | 68.6 | 78.5 |
| Llama-3.1-8B | v_proj | 1024x4096 | 32 | 128 | 6.8 (w16) | 15.1 | 7.7 (w16) | 7.7 (w16) | 16.5 | 4.3 | 3.8 |
| Llama-3.1-8B | v_proj | 1024x4096 | 32 | 256 | 7.0 (w32) | 15.1 | 8.5 (w32) | 8.6 (w32) | 16.7 | 4.8 | 4.5 |
| Llama-3.1-8B | v_proj | 1024x4096 | 32 | 512 | 9.1 (w64) | 15.1 | 9.9 (w32) | 9.8 (w32) | 16.7 | 8.0 | 7.4 |
| Llama-3.1-8B | v_proj | 1024x4096 | 32 | 1024 | 10.9 (w64) | 15.3 | 11.5 (w64) | 11.4 (w64) | 16.8 | 12.0 | 13.7 |
| Llama-3.1-8B | v_proj | 1024x4096 | 32 | 2048 | 16.0 (w128) | 16.1 | 16.8 (w128) | 16.8 (w128) | 17.5 | 19.6 | 21.7 |
| Llama-3.1-8B | v_proj | 1024x4096 | 32 | 4096 | 30.8 (w128) | 30.7 | 31.9 (w128) | 32.4 (w128) | 37.4 | 35.2 | 40.0 |
| Llama-3.1-8B | v_proj | 1024x4096 | 32 | 8192 | 47.8 (w128) | 47.7 | 50.4 (w128) | 51.0 (w128) | 55.5 | 68.8 | 78.6 |
| Llama-3.1-8B | o_proj | 4096x4096 | 32 | 128 | 9.3 (w64) | 15.2 | 10.5 (w32) | 10.5 (w32) | 16.7 | 4.3 | 3.8 |
| Llama-3.1-8B | o_proj | 4096x4096 | 32 | 256 | 11.3 (w64) | 15.4 | 11.6 (w64) | 11.5 (w64) | 17.0 | 4.8 | 4.5 |
| Llama-3.1-8B | o_proj | 4096x4096 | 32 | 512 | 16.1 (w128) | 16.0 | 16.8 (w128) | 16.8 (w128) | 17.6 | 8.0 | 7.4 |
| Llama-3.1-8B | o_proj | 4096x4096 | 32 | 1024 | 30.8 (w128) | 30.6 | 31.7 (w128) | 31.7 (w128) | 37.0 | 12.6 | 13.9 |
| Llama-3.1-8B | o_proj | 4096x4096 | 32 | 2048 | 48.1 (w128) | 47.4 | 49.7 (w128) | 50.0 (w128) | 52.6 | 22.0 | 24.8 |
| Llama-3.1-8B | o_proj | 4096x4096 | 32 | 4096 | 102.0 (w128) | 102.1 | 106.2 (w128) | 106.8 (w128) | 114.0 | 35.4 | 39.7 |
| Llama-3.1-8B | o_proj | 4096x4096 | 32 | 8192 | 201.3 (w128) | 198.6 | 210.4 (w128) | 210.9 (w128) | 223.7 | 82.2 | 87.6 |
| Llama-3.1-8B | gate_proj | 14336x4096 | 32 | 128 | 16.3 (w128) | 16.2 | 17.0 (w128) | 16.9 (w128) | 17.8 | 4.3 | 3.8 |
| Llama-3.1-8B | gate_proj | 14336x4096 | 32 | 256 | 30.9 (w128) | 30.8 | 32.1 (w128) | 32.2 (w128) | 33.8 | 4.8 | 4.6 |
| Llama-3.1-8B | gate_proj | 14336x4096 | 32 | 512 | 46.0 (w128) | 45.9 | 49.2 (w128) | 48.3 (w128) | 53.3 | 9.1 | 8.4 |
| Llama-3.1-8B | gate_proj | 14336x4096 | 32 | 1024 | 87.4 (w128) | 87.4 | 92.0 (w128) | 91.6 (w128) | 97.6 | 13.0 | 14.8 |
| Llama-3.1-8B | gate_proj | 14336x4096 | 32 | 2048 | 179.8 (w128) | 178.8 | 187.8 (w128) | 187.9 (w128) | 199.7 | 21.5 | 23.3 |
| Llama-3.1-8B | gate_proj | 14336x4096 | 32 | 4096 | 368.8 (w128) | 367.4 | 385.4 (w128) | 385.7 (w128) | 412.1 | 42.9 | 45.6 |
| Llama-3.1-8B | gate_proj | 14336x4096 | 32 | 8192 | 721.4 (w128) | 722.2 | 758.4 (w128) | 759.7 (w128) | 811.5 | 82.2 | 87.2 |
| Llama-3.1-8B | up_proj | 14336x4096 | 32 | 128 | 16.4 (w128) | 16.3 | 17.0 (w128) | 17.1 (w128) | 17.9 | 4.3 | 3.9 |
| Llama-3.1-8B | up_proj | 14336x4096 | 32 | 256 | 31.1 (w128) | 30.9 | 32.4 (w128) | 32.6 (w128) | 34.3 | 4.8 | 4.6 |
| Llama-3.1-8B | up_proj | 14336x4096 | 32 | 512 | 47.1 (w128) | 46.5 | 49.3 (w128) | 49.7 (w128) | 54.9 | 9.2 | 8.5 |
| Llama-3.1-8B | up_proj | 14336x4096 | 32 | 1024 | 87.5 (w128) | 87.4 | 91.9 (w128) | 92.0 (w128) | 97.7 | 12.9 | 14.8 |
| Llama-3.1-8B | up_proj | 14336x4096 | 32 | 2048 | 179.9 (w128) | 178.3 | 188.2 (w128) | 188.1 (w128) | 199.8 | 21.5 | 23.6 |
| Llama-3.1-8B | up_proj | 14336x4096 | 32 | 4096 | 368.4 (w128) | 367.3 | 387.2 (w128) | 386.9 (w128) | 412.9 | 43.0 | 46.0 |
| Llama-3.1-8B | up_proj | 14336x4096 | 32 | 8192 | 721.6 (w128) | 722.7 | 762.4 (w128) | 763.2 (w128) | 814.2 | 82.5 | 87.4 |
| Llama-3.1-8B | down_proj | 4096x14336 | 32 | 128 | 25.5 (w64) | 45.8 | 27.8 (w32) | 27.8 (w32) | 50.8 | 12.0 | 10.9 |
| Llama-3.1-8B | down_proj | 4096x14336 | 32 | 256 | 31.6 (w64) | 46.1 | 32.0 (w64) | 32.4 (w64) | 51.3 | 15.8 | 14.2 |
| Llama-3.1-8B | down_proj | 4096x14336 | 32 | 512 | 47.9 (w128) | 47.5 | 49.5 (w128) | 49.9 (w128) | 53.8 | 26.6 | 23.7 |
| Llama-3.1-8B | down_proj | 4096x14336 | 32 | 1024 | 111.8 (w64) | 109.0 | 116.4 (w64) | 116.5 (w64) | 124.7 | 39.7 | 45.4 |
| Llama-3.1-8B | down_proj | 4096x14336 | 32 | 2048 | 167.3 (w128) | 167.9 | 176.3 (w128) | 176.7 (w128) | 188.1 | 67.8 | 74.2 |
| Llama-3.1-8B | down_proj | 4096x14336 | 32 | 4096 | 348.5 (w128) | 351.5 | 366.2 (w128) | 366.4 (w128) | 393.4 | 162.0 | 162.6 |
| Llama-3.1-8B | down_proj | 4096x14336 | 32 | 8192 | 682.3 (w128) | 688.9 | 719.5 (w128) | 722.9 (w128) | 774.1 | 288.7 | 307.5 |
| Mistral-7B-v0.3 | q_proj | 4096x4096 | 32 | 128 | 9.2 (w64) | 15.2 | 10.6 (w32) | 10.6 (w32) | 16.7 | 4.3 | 3.8 |
| Mistral-7B-v0.3 | q_proj | 4096x4096 | 32 | 256 | 11.3 (w64) | 15.4 | 11.7 (w64) | 11.6 (w64) | 17.1 | 4.8 | 4.5 |
| Mistral-7B-v0.3 | q_proj | 4096x4096 | 32 | 512 | 16.1 (w128) | 16.0 | 16.7 (w128) | 16.7 (w128) | 17.8 | 8.0 | 7.4 |
| Mistral-7B-v0.3 | q_proj | 4096x4096 | 32 | 1024 | 30.6 (w128) | 30.3 | 31.7 (w128) | 31.7 (w128) | 34.4 | 12.0 | 13.9 |
| Mistral-7B-v0.3 | q_proj | 4096x4096 | 32 | 2048 | 46.9 (w128) | 46.4 | 49.1 (w128) | 48.9 (w128) | 52.1 | 21.3 | 23.4 |
| Mistral-7B-v0.3 | q_proj | 4096x4096 | 32 | 4096 | 99.0 (w128) | 97.0 | 105.0 (w128) | 103.3 (w128) | 111.5 | 35.0 | 39.6 |
| Mistral-7B-v0.3 | q_proj | 4096x4096 | 32 | 8192 | 194.8 (w128) | 186.8 | 208.0 (w128) | 204.6 (w128) | 221.7 | 81.2 | 86.5 |
| Mistral-7B-v0.3 | k_proj | 1024x4096 | 32 | 128 | 6.6 (w16) | 15.1 | 7.9 (w16) | 7.7 (w16) | 16.8 | 4.3 | 3.8 |
| Mistral-7B-v0.3 | k_proj | 1024x4096 | 32 | 256 | 7.0 (w32) | 15.1 | 8.8 (w32) | 8.6 (w32) | 17.1 | 4.8 | 4.5 |
| Mistral-7B-v0.3 | k_proj | 1024x4096 | 32 | 512 | 9.1 (w64) | 15.2 | 9.8 (w32) | 9.8 (w32) | 17.0 | 8.0 | 7.4 |
| Mistral-7B-v0.3 | k_proj | 1024x4096 | 32 | 1024 | 10.9 (w64) | 15.3 | 11.6 (w64) | 11.4 (w64) | 17.0 | 12.0 | 13.7 |
| Mistral-7B-v0.3 | k_proj | 1024x4096 | 32 | 2048 | 16.1 (w128) | 16.1 | 16.9 (w128) | 16.7 (w128) | 17.6 | 19.6 | 21.7 |
| Mistral-7B-v0.3 | k_proj | 1024x4096 | 32 | 4096 | 30.7 (w128) | 30.5 | 31.9 (w128) | 32.0 (w128) | 37.6 | 35.1 | 39.8 |
| Mistral-7B-v0.3 | k_proj | 1024x4096 | 32 | 8192 | 46.8 (w128) | 46.0 | 49.5 (w128) | 50.2 (w128) | 57.8 | 68.2 | 78.4 |
| Mistral-7B-v0.3 | v_proj | 1024x4096 | 32 | 128 | 6.7 (w16) | 15.0 | 7.7 (w16) | 7.7 (w16) | 16.6 | 4.3 | 3.8 |
| Mistral-7B-v0.3 | v_proj | 1024x4096 | 32 | 256 | 7.0 (w32) | 15.1 | 8.6 (w32) | 8.6 (w32) | 16.8 | 4.8 | 4.4 |
| Mistral-7B-v0.3 | v_proj | 1024x4096 | 32 | 512 | 9.1 (w64) | 15.2 | 9.8 (w32) | 9.8 (w32) | 16.9 | 8.0 | 7.4 |
| Mistral-7B-v0.3 | v_proj | 1024x4096 | 32 | 1024 | 10.9 (w64) | 15.3 | 11.4 (w64) | 11.4 (w64) | 16.9 | 12.0 | 13.8 |
| Mistral-7B-v0.3 | v_proj | 1024x4096 | 32 | 2048 | 16.1 (w128) | 16.2 | 16.7 (w128) | 17.1 (w128) | 17.7 | 19.6 | 21.8 |
| Mistral-7B-v0.3 | v_proj | 1024x4096 | 32 | 4096 | 30.8 (w128) | 30.7 | 32.3 (w128) | 32.1 (w128) | 37.5 | 34.9 | 39.9 |
| Mistral-7B-v0.3 | v_proj | 1024x4096 | 32 | 8192 | 47.1 (w128) | 46.9 | 49.6 (w128) | 49.9 (w128) | 55.5 | 68.4 | 78.5 |
| Mistral-7B-v0.3 | o_proj | 4096x4096 | 32 | 128 | 9.3 (w64) | 15.2 | 10.5 (w32) | 10.6 (w32) | 16.8 | 4.3 | 3.8 |
| Mistral-7B-v0.3 | o_proj | 4096x4096 | 32 | 256 | 11.3 (w64) | 15.4 | 11.6 (w64) | 11.8 (w64) | 17.1 | 4.8 | 4.5 |
| Mistral-7B-v0.3 | o_proj | 4096x4096 | 32 | 512 | 16.3 (w128) | 16.2 | 16.8 (w128) | 17.0 (w128) | 17.7 | 8.0 | 7.4 |
| Mistral-7B-v0.3 | o_proj | 4096x4096 | 32 | 1024 | 31.1 (w128) | 30.7 | 31.9 (w128) | 32.2 (w128) | 34.2 | 12.6 | 13.8 |
| Mistral-7B-v0.3 | o_proj | 4096x4096 | 32 | 2048 | 47.2 (w128) | 46.4 | 49.7 (w128) | 49.3 (w128) | 53.0 | 22.3 | 25.3 |
| Mistral-7B-v0.3 | o_proj | 4096x4096 | 32 | 4096 | 100.7 (w128) | 101.1 | 105.3 (w128) | 105.5 (w128) | 112.0 | 35.2 | 39.6 |
| Mistral-7B-v0.3 | o_proj | 4096x4096 | 32 | 8192 | 198.8 (w128) | 196.2 | 208.7 (w128) | 209.2 (w128) | 221.7 | 81.6 | 86.7 |
| Mistral-7B-v0.3 | gate_proj | 14336x4096 | 32 | 128 | 16.3 (w128) | 16.2 | 17.0 (w128) | 16.9 (w128) | 17.7 | 4.3 | 3.8 |
| Mistral-7B-v0.3 | gate_proj | 14336x4096 | 32 | 256 | 31.0 (w128) | 30.8 | 32.1 (w128) | 32.3 (w128) | 33.8 | 4.8 | 4.5 |
| Mistral-7B-v0.3 | gate_proj | 14336x4096 | 32 | 512 | 46.5 (w128) | 46.4 | 49.0 (w128) | 49.3 (w128) | 54.3 | 9.0 | 8.4 |
| Mistral-7B-v0.3 | gate_proj | 14336x4096 | 32 | 1024 | 86.7 (w128) | 87.4 | 90.7 (w128) | 90.9 (w128) | 95.7 | 13.0 | 14.8 |
| Mistral-7B-v0.3 | gate_proj | 14336x4096 | 32 | 2048 | 178.0 (w128) | 177.5 | 185.5 (w128) | 186.0 (w128) | 197.1 | 21.3 | 23.4 |
| Mistral-7B-v0.3 | gate_proj | 14336x4096 | 32 | 4096 | 364.8 (w128) | 363.5 | 380.8 (w128) | 381.0 (w128) | 404.8 | 42.6 | 45.4 |
| Mistral-7B-v0.3 | gate_proj | 14336x4096 | 32 | 8192 | 712.2 (w128) | 714.6 | 746.3 (w128) | 749.8 (w128) | 799.3 | 81.6 | 86.5 |
| Mistral-7B-v0.3 | up_proj | 14336x4096 | 32 | 128 | 16.4 (w128) | 16.2 | 17.0 (w128) | 17.0 (w128) | 17.8 | 4.3 | 3.8 |
| Mistral-7B-v0.3 | up_proj | 14336x4096 | 32 | 256 | 30.9 (w128) | 30.8 | 32.1 (w128) | 32.3 (w128) | 34.2 | 4.8 | 4.5 |
| Mistral-7B-v0.3 | up_proj | 14336x4096 | 32 | 512 | 46.8 (w128) | 47.4 | 51.1 (w128) | 50.5 (w128) | 54.9 | 9.1 | 8.5 |
| Mistral-7B-v0.3 | up_proj | 14336x4096 | 32 | 1024 | 87.2 (w128) | 87.0 | 91.3 (w128) | 91.2 (w128) | 96.7 | 12.8 | 14.6 |
| Mistral-7B-v0.3 | up_proj | 14336x4096 | 32 | 2048 | 178.0 (w128) | 177.1 | 186.0 (w128) | 186.6 (w128) | 198.1 | 21.3 | 23.3 |
| Mistral-7B-v0.3 | up_proj | 14336x4096 | 32 | 4096 | 364.6 (w128) | 363.3 | 381.0 (w128) | 382.5 (w128) | 407.6 | 42.6 | 45.5 |
| Mistral-7B-v0.3 | up_proj | 14336x4096 | 32 | 8192 | 713.5 (w128) | 714.0 | 749.0 (w128) | 751.5 (w128) | 801.8 | 81.9 | 86.7 |
| Mistral-7B-v0.3 | down_proj | 4096x14336 | 32 | 128 | 25.3 (w64) | 45.6 | 27.8 (w32) | 27.7 (w32) | 50.4 | 11.9 | 10.8 |
| Mistral-7B-v0.3 | down_proj | 4096x14336 | 32 | 256 | 31.5 (w64) | 45.9 | 31.6 (w64) | 31.8 (w64) | 50.8 | 15.8 | 14.1 |
| Mistral-7B-v0.3 | down_proj | 4096x14336 | 32 | 512 | 47.7 (w128) | 47.6 | 49.5 (w128) | 49.8 (w128) | 52.6 | 26.5 | 23.9 |
| Mistral-7B-v0.3 | down_proj | 4096x14336 | 32 | 1024 | 112.1 (w64) | 109.0 | 115.9 (w64) | 116.4 (w64) | 119.6 | 39.2 | 45.3 |
| Mistral-7B-v0.3 | down_proj | 4096x14336 | 32 | 2048 | 165.2 (w128) | 165.4 | 173.6 (w128) | 173.7 (w128) | 185.9 | 67.3 | 73.9 |
| Mistral-7B-v0.3 | down_proj | 4096x14336 | 32 | 4096 | 345.5 (w128) | 348.4 | 361.4 (w128) | 361.9 (w128) | 389.8 | 161.3 | 162.3 |
| Mistral-7B-v0.3 | down_proj | 4096x14336 | 32 | 8192 | 674.9 (w128) | 680.9 | 710.5 (w128) | 710.9 (w128) | 762.6 | 286.4 | 304.1 |
| Phi-4 | o_proj | 5120x5120 | 40 | 128 | 11.1 (w64) | 18.3 | 12.9 (w64) | 13.0 (w64) | 20.0 | 4.9 | 4.6 |
| Phi-4 | o_proj | 5120x5120 | 40 | 256 | 14.8 (w64) | 18.7 | 15.3 (w64) | 15.3 (w64) | 20.4 | 7.7 | 6.8 |
| Phi-4 | o_proj | 5120x5120 | 40 | 512 | 20.4 (w128) | 20.3 | 20.8 (w128) | 20.9 (w128) | 21.7 | 11.1 | 10.0 |
| Phi-4 | o_proj | 5120x5120 | 40 | 1024 | 37.9 (w128) | 37.7 | 39.0 (w128) | 39.2 (w128) | 41.8 | 16.4 | 20.0 |
| Phi-4 | o_proj | 5120x5120 | 40 | 2048 | 85.9 (w128) | 86.1 | 89.3 (w128) | 89.5 (w128) | 94.0 | 26.2 | 30.1 |
| Phi-4 | o_proj | 5120x5120 | 40 | 4096 | 146.5 (w128) | 146.4 | 153.9 (w128) | 154.6 (w128) | 164.7 | 49.2 | 55.3 |
| Phi-4 | o_proj | 5120x5120 | 40 | 8192 | 301.0 (w128) | 300.0 | 315.6 (w128) | 315.9 (w128) | 337.1 | 105.5 | 116.0 |
| Phi-4 | qkv_proj | 7680x5120 | 40 | 128 | 13.2 (w64) | 18.6 | 14.0 (w64) | 13.9 (w64) | 20.3 | 4.9 | 4.6 |
| Phi-4 | qkv_proj | 7680x5120 | 40 | 256 | 19.3 (w128) | 19.2 | 19.9 (w128) | 20.0 (w128) | 21.2 | 7.8 | 6.8 |
| Phi-4 | qkv_proj | 7680x5120 | 40 | 512 | 38.1 (w128) | 38.1 | 39.3 (w128) | 39.6 (w128) | 43.1 | 11.1 | 10.0 |
| Phi-4 | qkv_proj | 7680x5120 | 40 | 1024 | 64.6 (w128) | 58.0 | 62.1 (w128) | 62.4 (w128) | 68.5 | 18.3 | 20.2 |
| Phi-4 | qkv_proj | 7680x5120 | 40 | 2048 | 123.0 (w128) | 123.1 | 127.1 (w128) | 127.1 (w128) | 135.8 | 26.2 | 30.1 |
| Phi-4 | qkv_proj | 7680x5120 | 40 | 4096 | 237.8 (w128) | 235.5 | 247.8 (w128) | 248.1 (w128) | 263.3 | 49.7 | 55.6 |
| Phi-4 | qkv_proj | 7680x5120 | 40 | 8192 | 457.8 (w128) | 457.5 | 479.2 (w128) | 479.8 (w128) | 512.2 | 106.3 | 116.3 |
| Phi-4 | gate_up_proj | 35840x5120 | 40 | 128 | 40.4 (w128) | 40.4 | 41.2 (w128) | 41.2 (w128) | 43.4 | 5.1 | 4.8 |
| Phi-4 | gate_up_proj | 35840x5120 | 40 | 256 | 70.9 (w128) | 69.6 | 72.5 (w128) | 72.1 (w128) | 77.9 | 12.2 | 10.7 |
| Phi-4 | gate_up_proj | 35840x5120 | 40 | 512 | 144.5 (w128) | 144.4 | 150.7 (w128) | 150.4 (w128) | 162.1 | 18.4 | 16.9 |
| Phi-4 | gate_up_proj | 35840x5120 | 40 | 1024 | 286.1 (w128) | 285.3 | 296.8 (w128) | 297.4 (w128) | 315.6 | 24.0 | 26.8 |
| Phi-4 | gate_up_proj | 35840x5120 | 40 | 2048 | 557.6 (w128) | 555.5 | 579.3 (w128) | 580.9 (w128) | 618.1 | 36.9 | 38.8 |
| Phi-4 | gate_up_proj | 35840x5120 | 40 | 4096 | 1100.7 (w128) | 1096.7 | 1147.6 (w128) | 1150.1 (w128) | 1224.0 | 56.4 | 61.3 |
| Phi-4 | gate_up_proj | 35840x5120 | 40 | 8192 | 2186.3 (w128) | 2458.9 | 2544.4 (w128) | 2585.8 (w128) | 2744.7 | 115.1 | 115.6 |
| Phi-4 | down_proj | 5120x17920 | 40 | 128 | 33.5 (w64) | 56.3 | 37.1 (w64) | 37.2 (w64) | 61.5 | 13.0 | 12.9 |
| Phi-4 | down_proj | 5120x17920 | 40 | 256 | 43.3 (w64) | 57.0 | 44.2 (w64) | 44.2 (w64) | 62.7 | 21.2 | 18.9 |
| Phi-4 | down_proj | 5120x17920 | 40 | 512 | 63.5 (w128) | 63.4 | 67.8 (w128) | 67.9 (w128) | 73.4 | 38.9 | 34.8 |
| Phi-4 | down_proj | 5120x17920 | 40 | 1024 | 134.4 (w128) | 134.5 | 140.2 (w128) | 140.6 (w128) | 152.0 | 49.9 | 58.1 |
| Phi-4 | down_proj | 5120x17920 | 40 | 2048 | 284.2 (w128) | 283.7 | 294.4 (w128) | 294.3 (w128) | 310.6 | 116.2 | 121.9 |
| Phi-4 | down_proj | 5120x17920 | 40 | 4096 | 515.7 (w128) | 517.4 | 538.8 (w128) | 540.1 (w128) | 573.1 | 203.7 | 208.2 |
| Phi-4 | down_proj | 5120x17920 | 40 | 8192 | 1052.7 (w128) | 1061.7 | 1101.3 (w128) | 1107.3 (w128) | 1180.4 | 352.9 | 376.7 |
| Qwen3.8-27B | out_proj | 5120x6144 | 48 | 128 | 12.6 (w64) | 21.4 | 14.9 (w64) | 14.8 (w64) | 23.4 | 5.0 | 4.7 |
| Qwen3.8-27B | out_proj | 5120x6144 | 48 | 256 | 17.4 (w64) | 21.7 | 18.1 (w64) | 18.1 (w64) | 23.8 | 8.0 | 7.2 |
| Qwen3.8-27B | out_proj | 5120x6144 | 48 | 512 | 23.2 (w128) | 22.9 | 23.8 (w128) | 23.9 (w128) | 25.1 | 11.6 | 10.7 |
| Qwen3.8-27B | out_proj | 5120x6144 | 48 | 1024 | 45.0 (w128) | 45.9 | 47.5 (w128) | 47.1 (w128) | 49.8 | 19.1 | 21.1 |
| Qwen3.8-27B | out_proj | 5120x6144 | 48 | 2048 | 99.1 (w128) | 98.9 | 102.6 (w128) | 102.6 (w128) | 109.2 | 28.7 | 31.8 |
| Qwen3.8-27B | out_proj | 5120x6144 | 48 | 4096 | 179.1 (w128) | 178.0 | 186.4 (w128) | 186.5 (w128) | 197.6 | 54.1 | 60.1 |
| Qwen3.8-27B | out_proj | 5120x6144 | 48 | 8192 | 361.3 (w128) | 361.2 | 377.9 (w128) | 379.0 (w128) | 404.0 | 123.8 | 131.2 |
| Qwen3.8-27B | in_proj_qkv | 10240x5120 | 48 | 128 | 15.6 (w64) | 18.9 | 16.1 (w64) | 16.1 (w64) | 20.7 | 4.9 | 4.6 |
| Qwen3.8-27B | in_proj_qkv | 10240x5120 | 48 | 256 | 20.3 (w128) | 20.0 | 20.8 (w128) | 20.8 (w128) | 21.9 | 7.7 | 6.8 |
| Qwen3.8-27B | in_proj_qkv | 10240x5120 | 48 | 512 | 38.1 (w128) | 37.9 | 38.9 (w128) | 39.1 (w128) | 41.6 | 11.9 | 10.7 |
| Qwen3.8-27B | in_proj_qkv | 10240x5120 | 48 | 1024 | 88.0 (w128) | 88.0 | 90.9 (w128) | 91.0 (w128) | 94.3 | 16.3 | 19.8 |
| Qwen3.8-27B | in_proj_qkv | 10240x5120 | 48 | 2048 | 147.1 (w128) | 147.6 | 154.0 (w128) | 154.4 (w128) | 164.5 | 29.4 | 33.3 |
| Qwen3.8-27B | in_proj_qkv | 10240x5120 | 48 | 4096 | 312.1 (w128) | 310.0 | 325.9 (w128) | 325.6 (w128) | 346.5 | 56.5 | 61.4 |
| Qwen3.8-27B | in_proj_qkv | 10240x5120 | 48 | 8192 | 619.9 (w128) | 619.7 | 648.7 (w128) | 649.7 (w128) | 692.3 | 105.1 | 113.9 |
| Qwen3.8-27B | in_proj_z | 6144x5120 | 48 | 128 | 11.7 (w64) | 18.4 | 13.3 (w64) | 13.3 (w64) | 20.2 | 4.9 | 4.6 |
| Qwen3.8-27B | in_proj_z | 6144x5120 | 48 | 256 | 19.1 (w128) | 18.9 | 19.6 (w128) | 19.6 (w128) | 20.8 | 7.7 | 6.8 |
| Qwen3.8-27B | in_proj_z | 6144x5120 | 48 | 512 | 34.2 (w64) | 37.1 | 36.4 (w64) | 36.5 (w64) | 42.9 | 11.5 | 10.4 |
| Qwen3.8-27B | in_proj_z | 6144x5120 | 48 | 1024 | 57.6 (w128) | 59.6 | 64.4 (w128) | 61.6 (w128) | 70.6 | 18.3 | 22.0 |
| Qwen3.8-27B | in_proj_z | 6144x5120 | 48 | 2048 | 106.4 (w128) | 106.3 | 110.1 (w128) | 110.1 (w128) | 116.1 | 25.8 | 29.8 |
| Qwen3.8-27B | in_proj_z | 6144x5120 | 48 | 4096 | 189.7 (w128) | 188.6 | 196.9 (w128) | 197.0 (w128) | 209.6 | 46.0 | 52.2 |
| Qwen3.8-27B | in_proj_z | 6144x5120 | 48 | 8192 | 366.1 (w128) | 365.8 | 382.3 (w128) | 382.7 (w128) | 408.0 | 105.8 | 116.2 |
| Qwen3.8-27B | in_proj_b | 48x5120 | 48 | 128 | 7.1 (w16) | 18.2 | 8.3 (w16) | 8.3 (w16) | 19.5 | 4.9 | 4.6 |
| Qwen3.8-27B | in_proj_b | 48x5120 | 48 | 256 | 7.2 (w16) | 18.2 | 8.3 (w16) | 8.3 (w16) | 19.4 | 7.7 | 6.8 |
| Qwen3.8-27B | in_proj_b | 48x5120 | 48 | 512 | 7.7 (w32) | 18.1 | 9.4 (w32) | 9.4 (w32) | 19.5 | 11.1 | 9.9 |
| Qwen3.8-27B | in_proj_b | 48x5120 | 48 | 1024 | 9.9 (w32) | 18.1 | 10.1 (w32) | 10.1 (w32) | 19.6 | 16.3 | 19.7 |
| Qwen3.8-27B | in_proj_b | 48x5120 | 48 | 2048 | 10.6 (w64) | 18.4 | 12.3 (w64) | 12.3 (w64) | 19.8 | 25.9 | 29.6 |
| Qwen3.8-27B | in_proj_b | 48x5120 | 48 | 4096 | 17.4 (w64) | 18.6 | 17.7 (w64) | 17.7 (w64) | 20.0 | 45.2 | 51.4 |
| Qwen3.8-27B | in_proj_b | 48x5120 | 48 | 8192 | 20.2 (w128) | 20.2 | 20.4 (w128) | 20.4 (w128) | 20.7 | 85.0 | 99.6 |
| Qwen3.8-27B | in_proj_a | 48x5120 | 48 | 128 | 7.1 (w16) | 18.1 | 8.2 (w16) | 8.3 (w16) | 19.6 | 4.9 | 4.6 |
| Qwen3.8-27B | in_proj_a | 48x5120 | 48 | 256 | 7.1 (w16) | 18.1 | 8.3 (w16) | 8.3 (w16) | 19.5 | 7.6 | 6.7 |
| Qwen3.8-27B | in_proj_a | 48x5120 | 48 | 512 | 7.7 (w32) | 18.1 | 9.4 (w32) | 9.4 (w32) | 19.6 | 11.1 | 9.9 |
| Qwen3.8-27B | in_proj_a | 48x5120 | 48 | 1024 | 10.0 (w32) | 18.1 | 10.1 (w32) | 10.1 (w32) | 19.6 | 16.5 | 19.7 |
| Qwen3.8-27B | in_proj_a | 48x5120 | 48 | 2048 | 10.6 (w64) | 18.3 | 12.3 (w64) | 12.2 (w64) | 19.8 | 25.8 | 29.7 |
| Qwen3.8-27B | in_proj_a | 48x5120 | 48 | 4096 | 17.4 (w64) | 18.6 | 17.7 (w64) | 17.7 (w64) | 20.0 | 45.2 | 51.7 |
| Qwen3.8-27B | in_proj_a | 48x5120 | 48 | 8192 | 20.3 (w128) | 20.2 | 20.4 (w128) | 20.4 (w128) | 20.7 | 84.8 | 99.6 |
| Qwen3.8-27B | gate_proj | 17408x5120 | 64 | 128 | 20.0 (w128) | 19.8 | 20.7 (w128) | 20.5 (w128) | 21.6 | 4.9 | 4.6 |
| Qwen3.8-27B | gate_proj | 17408x5120 | 64 | 256 | 38.2 (w128) | 38.0 | 39.3 (w128) | 39.2 (w128) | 42.2 | 8.0 | 7.2 |
| Qwen3.8-27B | gate_proj | 17408x5120 | 64 | 512 | 61.5 (w128) | 61.6 | 64.3 (w128) | 64.5 (w128) | 69.3 | 14.4 | 12.8 |
| Qwen3.8-27B | gate_proj | 17408x5120 | 64 | 1024 | 125.3 (w128) | 125.1 | 130.5 (w128) | 130.3 (w128) | 138.8 | 16.8 | 20.4 |
| Qwen3.8-27B | gate_proj | 17408x5120 | 64 | 2048 | 272.5 (w128) | 271.0 | 284.0 (w128) | 284.1 (w128) | 301.8 | 32.9 | 33.8 |
| Qwen3.8-27B | gate_proj | 17408x5120 | 64 | 4096 | 535.2 (w128) | 534.8 | 560.0 (w128) | 558.2 (w128) | 596.7 | 55.6 | 60.2 |
| Qwen3.8-27B | gate_proj | 17408x5120 | 64 | 8192 | 1055.9 (w128) | 1064.5 | 1120.5 (w128) | 1111.1 (w128) | 1192.0 | 105.9 | 114.5 |
| Qwen3.8-27B | up_proj | 17408x5120 | 64 | 128 | 19.9 (w128) | 19.9 | 20.7 (w128) | 20.9 (w128) | 21.6 | 4.9 | 4.6 |
| Qwen3.8-27B | up_proj | 17408x5120 | 64 | 256 | 38.3 (w128) | 38.1 | 39.3 (w128) | 39.3 (w128) | 43.4 | 8.0 | 7.3 |
| Qwen3.8-27B | up_proj | 17408x5120 | 64 | 512 | 61.2 (w128) | 61.5 | 64.5 (w128) | 64.8 (w128) | 69.3 | 14.5 | 12.7 |
| Qwen3.8-27B | up_proj | 17408x5120 | 64 | 1024 | 125.0 (w128) | 125.1 | 130.2 (w128) | 130.8 (w128) | 139.0 | 16.8 | 20.4 |
| Qwen3.8-27B | up_proj | 17408x5120 | 64 | 2048 | 272.4 (w128) | 271.3 | 284.9 (w128) | 285.2 (w128) | 302.6 | 32.9 | 33.6 |
| Qwen3.8-27B | up_proj | 17408x5120 | 64 | 4096 | 535.3 (w128) | 533.5 | 560.1 (w128) | 561.1 (w128) | 598.3 | 55.6 | 60.1 |
| Qwen3.8-27B | up_proj | 17408x5120 | 64 | 8192 | 1056.6 (w128) | 1063.0 | 1111.8 (w128) | 1131.6 (w128) | 1181.8 | 105.9 | 114.7 |
| Qwen3.8-27B | down_proj | 5120x17408 | 64 | 128 | 30.4 (w64) | 54.8 | 36.6 (w64) | 36.7 (w64) | 60.4 | 12.8 | 11.8 |
| Qwen3.8-27B | down_proj | 5120x17408 | 64 | 256 | 41.8 (w64) | 55.5 | 42.8 (w64) | 43.0 (w64) | 61.1 | 20.2 | 18.3 |
| Qwen3.8-27B | down_proj | 5120x17408 | 64 | 512 | 60.6 (w128) | 60.4 | 65.9 (w128) | 65.9 (w128) | 70.8 | 38.5 | 35.2 |
| Qwen3.8-27B | down_proj | 5120x17408 | 64 | 1024 | 131.3 (w128) | 131.0 | 136.2 (w128) | 136.7 (w128) | 148.0 | 49.2 | 57.1 |
| Qwen3.8-27B | down_proj | 5120x17408 | 64 | 2048 | 274.9 (w128) | 274.5 | 286.6 (w128) | 287.2 (w128) | 303.7 | 110.0 | 117.0 |
| Qwen3.8-27B | down_proj | 5120x17408 | 64 | 4096 | 502.2 (w128) | 502.4 | 525.6 (w128) | 526.6 (w128) | 561.1 | 197.1 | 202.8 |
| Qwen3.8-27B | down_proj | 5120x17408 | 64 | 8192 | 1023.4 (w128) | 1054.1 | 1095.6 (w128) | 1100.8 (w128) | 1202.9 | 342.9 | 367.5 |
| Qwen3.8-27B | q_proj | 12288x5120 | 16 | 128 | 19.3 (w128) | 19.0 | 19.8 (w128) | 19.9 (w128) | 20.9 | 4.9 | 4.6 |
| Qwen3.8-27B | q_proj | 12288x5120 | 16 | 256 | 34.3 (w64) | 37.2 | 37.1 (w64) | 36.8 (w64) | 42.7 | 8.1 | 7.1 |
| Qwen3.8-27B | q_proj | 12288x5120 | 16 | 512 | 56.8 (w128) | 56.4 | 62.4 (w128) | 62.2 (w128) | 69.4 | 12.7 | 11.7 |
| Qwen3.8-27B | q_proj | 12288x5120 | 16 | 1024 | 108.6 (w128) | 108.7 | 112.5 (w128) | 112.2 (w128) | 119.3 | 16.5 | 19.5 |
| Qwen3.8-27B | q_proj | 12288x5120 | 16 | 2048 | 192.0 (w128) | 190.8 | 199.5 (w128) | 199.7 (w128) | 212.7 | 26.5 | 30.4 |
| Qwen3.8-27B | q_proj | 12288x5120 | 16 | 4096 | 379.4 (w128) | 377.0 | 396.1 (w128) | 396.1 (w128) | 422.0 | 55.6 | 60.6 |
| Qwen3.8-27B | q_proj | 12288x5120 | 16 | 8192 | 742.3 (w128) | 743.3 | 778.5 (w128) | 779.4 (w128) | 830.9 | 105.6 | 114.9 |
| Qwen3.8-27B | k_proj | 1024x5120 | 16 | 128 | 7.7 (w16) | 18.1 | 8.8 (w16) | 8.9 (w16) | 19.8 | 4.8 | 4.6 |
| Qwen3.8-27B | k_proj | 1024x5120 | 16 | 256 | 8.1 (w32) | 18.1 | 9.9 (w32) | 10.0 (w32) | 20.0 | 7.7 | 6.8 |
| Qwen3.8-27B | k_proj | 1024x5120 | 16 | 512 | 10.7 (w64) | 18.2 | 11.6 (w32) | 11.6 (w32) | 20.0 | 11.1 | 10.0 |
| Qwen3.8-27B | k_proj | 1024x5120 | 16 | 1024 | 13.0 (w64) | 18.4 | 13.5 (w64) | 13.4 (w64) | 20.1 | 16.0 | 19.4 |
| Qwen3.8-27B | k_proj | 1024x5120 | 16 | 2048 | 19.3 (w128) | 19.4 | 19.8 (w128) | 19.8 (w128) | 21.0 | 25.9 | 29.7 |
| Qwen3.8-27B | k_proj | 1024x5120 | 16 | 4096 | 37.3 (w128) | 37.3 | 40.2 (w128) | 40.6 (w128) | 45.8 | 45.7 | 51.9 |
| Qwen3.8-27B | k_proj | 1024x5120 | 16 | 8192 | 59.4 (w128) | 59.3 | 62.0 (w128) | 62.3 (w128) | 72.2 | 86.2 | 101.0 |
| Qwen3.8-27B | v_proj | 1024x5120 | 16 | 128 | 7.7 (w16) | 18.1 | 8.9 (w16) | 9.0 (w16) | 20.0 | 4.9 | 4.6 |
| Qwen3.8-27B | v_proj | 1024x5120 | 16 | 256 | 8.1 (w32) | 18.2 | 10.0 (w32) | 10.1 (w32) | 20.2 | 7.7 | 6.8 |
| Qwen3.8-27B | v_proj | 1024x5120 | 16 | 512 | 10.7 (w64) | 18.2 | 11.8 (w32) | 11.8 (w32) | 20.2 | 11.1 | 10.0 |
| Qwen3.8-27B | v_proj | 1024x5120 | 16 | 1024 | 12.9 (w64) | 18.5 | 13.6 (w64) | 13.7 (w64) | 20.2 | 16.0 | 19.3 |
| Qwen3.8-27B | v_proj | 1024x5120 | 16 | 2048 | 19.4 (w128) | 19.4 | 20.1 (w128) | 20.2 (w128) | 21.2 | 25.9 | 29.8 |
| Qwen3.8-27B | v_proj | 1024x5120 | 16 | 4096 | 37.1 (w128) | 37.3 | 39.9 (w128) | 38.7 (w128) | 44.0 | 45.6 | 51.8 |
| Qwen3.8-27B | v_proj | 1024x5120 | 16 | 8192 | 59.3 (w128) | 59.1 | 62.6 (w128) | 62.5 (w128) | 67.0 | 86.5 | 100.8 |
| Qwen3.8-27B | o_proj | 5120x6144 | 16 | 128 | 12.7 (w64) | 21.3 | 15.0 (w64) | 15.0 (w64) | 23.5 | 4.9 | 4.7 |
| Qwen3.8-27B | o_proj | 5120x6144 | 16 | 256 | 17.6 (w64) | 21.8 | 18.2 (w64) | 18.3 (w64) | 23.9 | 8.0 | 7.2 |
| Qwen3.8-27B | o_proj | 5120x6144 | 16 | 512 | 23.1 (w128) | 23.0 | 23.9 (w128) | 24.0 (w128) | 25.3 | 11.6 | 10.7 |
| Qwen3.8-27B | o_proj | 5120x6144 | 16 | 1024 | 45.3 (w128) | 45.6 | 47.2 (w128) | 47.8 (w128) | 49.9 | 18.9 | 21.0 |
| Qwen3.8-27B | o_proj | 5120x6144 | 16 | 2048 | 99.2 (w128) | 98.9 | 102.6 (w128) | 103.2 (w128) | 110.6 | 28.5 | 31.7 |
| Qwen3.8-27B | o_proj | 5120x6144 | 16 | 4096 | 179.5 (w128) | 178.8 | 187.1 (w128) | 187.3 (w128) | 199.1 | 54.1 | 60.3 |
| Qwen3.8-27B | o_proj | 5120x6144 | 16 | 8192 | 363.6 (w128) | 362.2 | 380.1 (w128) | 380.4 (w128) | 406.6 | 124.4 | 131.7 |
