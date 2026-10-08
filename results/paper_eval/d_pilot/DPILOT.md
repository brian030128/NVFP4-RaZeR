# D pilot: generation lengths, decode steps, memory and the full-run estimate

Amendments 4, 7 and 8; not a result. flipquant `evaluation.accuracy`, natively in n16k64-fast, `--recommended-decoding` (thinking on), seed 0, every task at a 32768-token cap. Lengths: BF16 and FlipQuant 16x64 on 16 problems of GSM8K, MATH-500 and IFEval, and 2 problems x 8 samples of AIME 2024 and 2025, batch 16 (the Qwen3.8-27B BF16 AIME job ran out of memory at batch 16 and its batch-8 rerun was cancelled: amendments 7 and 8). Batch scaling: all six policies on GSM8K at 512 tokens, batch 16 / 32 / 64. The other FP4 policies' lengths are taken to be FlipQuant 16x64's.

## Nemotron-Nano-9B-v2

Generated tokens per sample at the 32768 cap (BF16 / FlipQuant 16x64):

| task | samples | mean | median | p90 | max | ≥ 4k | ≥ 8k | ≥ 16k | ≥ 32k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 16 / 16 | 1028 / 952 | 924 / 927 | 1876 / 1615 | 2360 / 1765 | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % |
| math500 | 16 / 16 | 2619 / 2724 | 2047 / 1920 | 7428 / 6668 | 8014 / 9500 | 12 % / 12 % | 0 % / 6 % | 0 % / 0 % | 0 % / 0 % |
| ifeval | 16 / 16 | 833 / 903 | 622 / 764 | 2356 / 2477 | 2444 / 2624 | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % |
| aime24 | 16 / 16 | 6175 / 8093 | 5144 / 4921 | 8869 / 19664 | 23855 / 20058 | 50 % / 50 % | 19 % / 44 % | 6 % / 19 % | 0 % / 0 % |
| aime25 | 16 / 16 | 6842 / 6355 | 5399 / 5316 | 13032 / 13277 | 17861 / 16206 | 50 % / 50 % | 44 % / 31 % | 6 % / 0 % | 0 % / 0 % |

AIME per problem (8 samples each; BF16 / FlipQuant 16x64): generated tokens min–max, samples truncated at 32k, samples correct:

| problem | tokens | truncated | correct |
|---|---:|---:|---:|
| aime24/0 | 2074–3293 / 2169–3277 | 0/8 / 0/8 | 8/8 / 8/8 |
| aime24/1 | 6994–23855 / 6565–20058 | 0/8 / 0/8 | 7/8 / 5/8 |
| aime25/0 | 1601–2906 / 1572–2918 | 0/8 / 0/8 | 8/8 / 8/8 |
| aime25/1 | 7892–17861 / 7713–16206 | 0/8 / 0/8 | 7/8 / 7/8 |

Static batches: a batch runs until its longest sample ends, so at batch 16 the share of decode slots that produce tokens (mean length / expected longest in the batch) is gsm8k 47 % / 56 %; math500 36 % / 34 %; ifeval 36 % / 36 %; aime24 36 % / 55 %; aime25 41 % / 44 % (BF16 / FlipQuant 16x64).

Decode step (ms) = a + c x context; c = 2.05 (BF16) / 0.93 (FP4) µs per context token at batch 16 (scaled with the batch). a per policy and batch, and the step at 8k / 32k context:

| policy | a, b16 | b32 | b64 | step at 8k, b16 | 32k, b16 | 32k, b64 |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 23.6 | 24.4 | 32.8 | 40.3 | 90.6 | 300.8 |
| nvfp4 | 28.0 | 29.5 | 29.7 | 35.6 | 58.5 | 151.4 |
| fo6 | 29.9 | 29.4 | 30.2 | 37.5 | 60.3 | 151.9 |
| fq-8x64 | 30.2 | 29.2 | 30.7 | 37.8 | 60.6 | 152.4 |
| fq-16x64 | 29.4 | 29.9 | 29.0 | 37.0 | 59.9 | 150.8 |
| fq-256x64 | 27.9 | 29.0 | 29.8 | 35.5 | 58.4 | 151.5 |

Fit (predicted / measured seconds) of the long batches: bf16 math500 M=8014: 255 / 249, bf16 aime25 M=17861: 747 / 748, fq-16x64 math500 M=9500: 322 / 298, fq-16x64 ifeval M=2624: 80 / 79, fq-16x64 aime25 M=16206: 599 / 607; the 512-token batches within 5 %.

Estimated hours per policy and task for the full run (BF16 / mean of the five FP4 policies; AIME per year, 30 problems x 8 samples); ✗ = does not fit in memory at that cap:

| task | cap | batch 8 | batch 16 | batch 32 | batch 64 |
|---|---:|---:|---:|---:|---:|
| gsm8k | 4k | 2.2 / 2.2 | 1.3 / 1.2 | 0.8 / 0.6 | 0.6 / 0.3 |
| gsm8k | 8k | 2.2 / 2.2 | 1.3 / 1.2 | 0.8 / 0.6 | 0.6 / 0.3 |
| gsm8k | 16k | 2.2 / 2.2 | 1.3 / 1.2 | 0.8 / 0.6 | 0.6 / 0.3 |
| gsm8k | 32k | 2.2 / 2.2 | 1.3 / 1.2 | 0.8 / 0.6 | ✗ / 0.3 |
| math500 | 4k | 1.7 / 2.0 | 1.0 / 1.1 | 0.6 / 0.6 | 0.5 / 0.3 |
| math500 | 8k | 2.9 / 3.3 | 2.0 / 2.1 | 1.4 / 1.3 | 1.2 / 0.8 |
| math500 | 16k | 2.9 / 3.6 | 2.0 / 2.4 | 1.4 / 1.5 | 1.2 / 1.0 |
| math500 | 32k | 2.9 / 3.6 | 2.0 / 2.4 | 1.4 / 1.5 | ✗ / 1.0 |
| ifeval | 4k | 1.0 / 1.2 | 0.6 / 0.7 | 0.3 / 0.4 | 0.3 / 0.2 |
| ifeval | 8k | 1.0 / 1.2 | 0.6 / 0.7 | 0.3 / 0.4 | 0.3 / 0.2 |
| ifeval | 16k | 1.0 / 1.2 | 0.6 / 0.7 | 0.3 / 0.4 | 0.3 / 0.2 |
| ifeval | 32k | 1.0 / 1.2 | 0.6 / 0.7 | 0.3 / 0.4 | ✗ / 0.2 |
| aime24 | 4k | 0.8 / 0.9 | 0.4 / 0.5 | 0.3 / 0.3 | 0.2 / 0.2 |
| aime24 | 8k | 1.3 / 1.4 | 0.9 / 0.9 | 0.7 / 0.6 | 0.6 / 0.4 |
| aime24 | 16k | 2.5 / 2.6 | 2.2 / 2.0 | 1.9 / 1.5 | 1.8 / 1.1 |
| aime24 | 32k | 3.3 / 2.9 | 3.2 / 2.3 | 3.2 / 1.8 | ✗ / 1.4 |
| aime25 | 4k | 0.8 / 0.9 | 0.4 / 0.5 | 0.3 / 0.3 | 0.2 / 0.2 |
| aime25 | 8k | 1.3 / 1.5 | 0.9 / 0.9 | 0.7 / 0.6 | 0.6 / 0.4 |
| aime25 | 16k | 2.5 / 2.6 | 2.1 / 2.0 | 1.9 / 1.5 | 1.8 / 1.1 |
| aime25 | 32k | 3.3 / 2.9 | 3.1 / 2.3 | 3.2 / 1.8 | ✗ / 1.4 |

Full AIME (2024 + 2025, 60 problems x 8 samples) per policy, hours, at the 16k and 32k caps; ✗ = does not fit in memory (expandable segments):

| policy | 16k, b8 | b16 | b32 | b64 | 32k, b8 | b16 | b32 | b64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 5.0 | 4.3 | 3.9 | 3.6 | 6.7 | 6.4 | 6.4 | ✗ |
| nvfp4 | 5.1 | 3.8 | 3.0 | 2.2 | 5.6 | 4.4 | 3.7 | 2.9 |
| fo6 | 5.4 | 4.0 | 3.0 | 2.2 | 6.0 | 4.6 | 3.7 | 2.9 |
| fq-8x64 | 5.4 | 4.1 | 3.0 | 2.2 | 6.0 | 4.7 | 3.7 | 2.9 |
| fq-16x64 | 5.3 | 4.0 | 3.0 | 2.1 | 5.9 | 4.6 | 3.7 | 2.8 |
| fq-256x64 | 5.1 | 3.8 | 3.0 | 2.2 | 5.6 | 4.4 | 3.7 | 2.9 |

Predicted peak memory (GiB) at cap + 512 prompt tokens, BF16 / FP4 (fits: ≤ 82 default allocator, ≤ 92 with expandable segments):

| batch | 4k | 8k | 16k | 32k |
|---:|---:|---:|---:|---:|
| 8 | 20 / 10 | 21 / 11 | 24 / 13 | 28 / 18 |
| 16 | 22 / 11 | 24 / 14 | 28 / 18 | 37 / 27 |
| 32 | 24 / 14 | 29 / 18 | 38 / 27 | 56 / 45 |
| 64 | 29 / 19 | 38 / 28 | 56 / 46 | 92 / 82 |

## Qwen3.8-27B

Generated tokens per sample at the 32768 cap (BF16 / FlipQuant 16x64):

| task | samples | mean | median | p90 | max | ≥ 4k | ≥ 8k | ≥ 16k | ≥ 32k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 16 / 16 | 743 / 442 | 299 / 303 | 1646 / 785 | 5855 / 1768 | 6 % / 0 % | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % |
| math500 | 16 / 16 | 1796 / 1884 | 596 / 599 | 9763 / 10318 | 10502 / 12091 | 12 % / 12 % | 12 % / 12 % | 0 % / 0 % | 0 % / 0 % |
| ifeval | 16 / 16 | 1497 / 2254 | 922 / 1084 | 4684 / 6067 | 6272 / 10795 | 12 % / 12 % | 0 % / 6 % | 0 % / 0 % | 0 % / 0 % |
| aime24 | — / 16 | — / 17017 | — / 16710 | — / 32768 | — / 32768 | — / 50 % | — / 50 % | — / 50 % | — / 44 % |
| aime25 | — / 16 | — / 16366 | — / 16064 | — / 32768 | — / 32768 | — / 50 % | — / 50 % | — / 50 % | — / 19 % |

AIME per problem (8 samples each; BF16 / FlipQuant 16x64): generated tokens min–max, samples truncated at 32k, samples correct:

| problem | tokens | truncated | correct |
|---|---:|---:|---:|
| aime24/0 | — / 1104–2126 | — / 0/8 | — / 8/8 |
| aime24/1 | — / 31295–32768 | — / 7/8 | — / 2/8 |
| aime25/0 | — / 730–2185 | — / 0/8 | — / 8/8 |
| aime25/1 | — / 29942–32768 | — / 3/8 | — / 7/8 |

Static batches: a batch runs until its longest sample ends, so at batch 16 the share of decode slots that produce tokens (mean length / expected longest in the batch) is gsm8k 18 % / 32 %; math500 19 % / 18 %; ifeval 28 % / 26 %; aime24 — / 68 %; aime25 — / 65 % (BF16 / FlipQuant 16x64).

Decode step (ms) = a + c x context; c = 9.28 (BF16) / 8.41 (FP4) µs per context token at batch 16 (scaled with the batch). a per policy and batch, and the step at 8k / 32k context:

| policy | a, b16 | b32 | b64 | step at 8k, b16 | 32k, b16 | 32k, b64 |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 67.6 | 86.6 | 111.2 | 143.6 | 371.6 | 1327.3 |
| nvfp4 | 88.1 | 86.6 | 87.9 | 157.0 | 363.7 | 1190.3 |
| fo6 | 87.1 | 86.0 | 88.4 | 156.0 | 362.7 | 1190.8 |
| fq-8x64 | 84.6 | 89.3 | 84.5 | 153.5 | 360.2 | 1187.0 |
| fq-16x64 | 86.9 | 85.3 | 84.6 | 155.8 | 362.5 | 1187.0 |
| fq-256x64 | 87.0 | 87.6 | 87.4 | 155.9 | 362.7 | 1189.9 |

Fit (predicted / measured seconds) of the long batches: bf16 math500 M=10502: 1221 / 1236, bf16 ifeval M=6272: 606 / 564, fq-16x64 math500 M=12091: 1665 / 1383, fq-16x64 ifeval M=10795: 1428 / 1150, fq-16x64 aime25 M=32768: 7363 / 7432; the 512-token batches within 2 %.

Estimated hours per policy and task for the full run (BF16 / mean of the five FP4 policies; AIME per year, 30 problems x 8 samples); ✗ = does not fit in memory at that cap:

| task | cap | batch 8 | batch 16 | batch 32 | batch 64 |
|---|---:|---:|---:|---:|---:|
| gsm8k | 4k | 7.9 / 4.7 | 6.0 / 3.0 | 5.3 / 1.9 | 4.4 / 1.2 |
| gsm8k | 8k | 10.9 / 4.7 | 9.0 / 3.0 | 8.5 / 1.9 | ✗ / 1.2 |
| gsm8k | 16k | 10.9 / 4.7 | 9.0 / 3.0 | ✗ / 1.9 | ✗ / ✗ |
| gsm8k | 32k | 10.9 / 4.7 | ✗ / 3.0 | ✗ / ✗ | ✗ / ✗ |
| math500 | 4k | 4.1 / 5.1 | 2.9 / 3.4 | 2.2 / 2.2 | 1.7 / 1.4 |
| math500 | 8k | 8.4 / 10.4 | 6.8 / 7.7 | 5.8 / 5.6 | ✗ / 4.1 |
| math500 | 16k | 10.9 / 15.2 | 9.3 / 12.2 | ✗ / 9.7 | ✗ / ✗ |
| math500 | 32k | 10.9 / 15.2 | ✗ / 12.2 | ✗ / ✗ | ✗ / ✗ |
| ifeval | 4k | 4.8 / 6.9 | 3.1 / 4.0 | 2.4 / 2.3 | 1.9 / 1.6 |
| ifeval | 8k | 6.5 / 11.4 | 4.8 / 8.0 | 4.1 / 5.7 | ✗ / 4.5 |
| ifeval | 16k | 6.5 / 14.0 | 4.8 / 10.6 | ✗ / 8.4 | ✗ / ✗ |
| ifeval | 32k | 6.5 / 14.0 | ✗ / 10.6 | ✗ / ✗ | ✗ / ✗ |
| aime24 | 4k | 1.9 / 2.4 | 1.3 / 1.5 | 1.1 / 1.1 | 0.8 / 0.7 |
| aime24 | 8k | 3.5 / 4.3 | 2.9 / 3.3 | 2.7 / 2.6 | ✗ / 2.0 |
| aime24 | 16k | 7.7 / 8.9 | 7.5 / 8.1 | ✗ / 7.5 | ✗ / ✗ |
| aime24 | 32k | 19.8 / 21.5 | ✗ / 23.0 | ✗ / ✗ | ✗ / ✗ |
| aime25 | 4k | 2.0 / 2.5 | 1.3 / 1.5 | 1.1 / 1.1 | 0.8 / 0.7 |
| aime25 | 8k | 3.6 / 4.4 | 2.9 / 3.3 | 2.7 / 2.6 | ✗ / 2.0 |
| aime25 | 16k | 8.0 / 9.2 | 7.5 / 8.2 | ✗ / 7.5 | ✗ / ✗ |
| aime25 | 32k | 20.5 / 22.4 | ✗ / 23.1 | ✗ / ✗ | ✗ / ✗ |

Full AIME (2024 + 2025, 60 problems x 8 samples) per policy, hours, at the 16k and 32k caps; ✗ = does not fit in memory (expandable segments):

| policy | 16k, b8 | b16 | b32 | b64 | 32k, b8 | b16 | b32 | b64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 † | 15.6 | 15.0 | ✗ | ✗ | 40.3 | ✗ | ✗ | ✗ |
| nvfp4 | 18.3 | 16.4 | 15.0 | ✗ | 44.3 | 46.4 | ✗ | ✗ |
| fo6 | 18.2 | 16.3 | 15.0 | ✗ | 44.0 | 46.2 | ✗ | ✗ |
| fq-8x64 | 17.8 | 16.0 | 15.2 | ✗ | 43.3 | 45.7 | ✗ | ✗ |
| fq-16x64 | 18.1 | 16.3 | 14.9 | ✗ | 43.9 | 46.1 | ✗ | ✗ |
| fq-256x64 | 18.1 | 16.3 | 15.1 | ✗ | 44.0 | 46.2 | ✗ | ✗ |

† Estimate: BF16's AIME lengths are FlipQuant 16x64's (the BF16 length job was cancelled, amendment 8), with BF16's step model (batch 8, not measured: a and c / 2).

Predicted peak memory (GiB) at cap + 512 prompt tokens, BF16 / FP4 (fits: ≤ 82 default allocator, ≤ 92 with expandable segments):

| batch | 4k | 8k | 16k | 32k |
|---:|---:|---:|---:|---:|
| 8 | 57 / 24 | 59 / 27 | 65 / 32 | 76 / 43 |
| 16 | 60 / 27 | 65 / 33 | 76 / 44 | 98 / 66 |
| 32 | 66 / 33 | 77 / 44 | 99 / 66 | 143 / 110 |
| 64 | 78 / 46 | 100 / 68 | 144 / 112 | 232 / 200 |

## Full D run under the candidate options

(i) caps per model and task: the smallest of 4k / 8k / 16k / 32k above every pilot sample (Nemotron-Nano-9B-v2: gsm8k 4k, math500 16k, ifeval 4k, aime24 32k, aime25 32k; Qwen3.8-27B: gsm8k 8k, math500 16k, ifeval 16k, aime24 32k, aime25 32k). With 16 samples per task (32 for AIME) a < 1 % truncation rate cannot be verified; Qwen3.8-27B's AIME exceeds every measured cap (it truncates at 32k: see above), so (i) keeps 32k there. Every task runs at the requested batch or, when that does not fit at its cap, at the largest batch that does (marked *). Hours, all policies (AIME hours in brackets):

| option | batch | Nemotron-Nano-9B-v2 | Qwen3.8-27B | total |
|---|---:|---:|---:|---:|
| (i) caps above every pilot sample, AIME avg@8, 6 policies | 16* | 54 (29) | 423 (271) | 477 |
| (i) caps above every pilot sample, AIME avg@8, 6 policies | 32* | 40 (25) | 393 (271) | 434 |
| (i) caps above every pilot sample, AIME avg@8, 6 policies | 64* | 31 (21) | 390 (271) | 420 |
| (ii) = (i) with AIME avg@4 | 16* | 42 (17) | 332 (180) | 375 |
| (ii) = (i) with AIME avg@4 | 32* | 29 (13) | 303 (180) | 331 |
| (ii) = (i) with AIME avg@4 | 64* | 20 (11) | 299 (180) | 320 |
| (iii) = (i) with AIME at a 16k cap | 16 | 49 (24) | 248 (96) | 297 |
| (iii) = (i) with AIME at a 16k cap | 32* | 34 (19) | 213 (90) | 247 |
| (iii) = (i) with AIME at a 16k cap | 64* | 24 (14) | 209 (90) | 233 |
| (iv) = (i) with AIME for BF16, FO6, FQ 16x64 only | 16* | 41 (16) | 285 (133) | 325 |
| (iv) = (i) with AIME for BF16, FO6, FQ 16x64 only | 32* | 29 (14) | 255 (133) | 284 |
| (iv) = (i) with AIME for BF16, FO6, FQ 16x64 only | 64* | 22 (12) | 252 (133) | 273 |
| (ii) + (iv) | 16* | 34 (9) | 242 (90) | 276 |
| (ii) + (iv) | 32* | 23 (7) | 213 (90) | 235 |
| (ii) + (iv) | 64* | 16 (6) | 209 (90) | 225 |
| (iii) + (iv) | 16 | 37 (12) | 200 (48) | 237 |
| (iii) + (iv) | 32* | 25 (10) | 167 (45) | 193 |
| (iii) + (iv) | 64* | 18 (8) | 164 (45) | 182 |

Limits of the estimate: 16 problems per task (2 per AIME year) set the length distributions, and sampling makes a rerun's lengths differ; the step model is fitted on 512-token batches and a few long ones; the FP4 policies other than FlipQuant 16x64 are assumed to generate FlipQuant 16x64's lengths; batch 8 is not measured.

Pilot jobs without a summary: qwen3.8-27b bf16 aime.

