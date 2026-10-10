# tab:ptq, GSM8K with lm-eval's gsm8k_llama, thinking off (part N)

lm-eval 0.4.11 `gsm8k_llama` as shipped: 8-shot CoT (first_n), apply_chat_template + fewshot_as_multiturn, greedy, max_gen_toks 1024, filters strict_match / flexible_extract; the full test set (1,319). Thinking off: Qwen3.8-27B `enable_thinking=False` (HFLM chat_template_args), Nemotron-Nano-9B-v2 system instruction `/no_think`. lm-eval's HFLM on the natively quantized model (flipquant `evaluation.downstream`, native sm120, build_V, `--kernel-set auto`, n16k64-fast); part H's artifacts and activations. Accuracy ± 2 SE (binomial).

| model | method | format | batch | strict-match (%) | flexible-extract (%) | cannot extract (strict / flexible) | used the 1024 budget | mean / max tokens |
|---|---|---|---:|---:|---:|---|---:|---|
| Nemotron-Nano-9B-v2 | RTN | NVFP4 | 64 | 87.34 ± 1.83 | 88.86 ± 1.73 | 35 / 1 | 0 | 112 / 608 |
| Nemotron-Nano-9B-v2 | RTN | FourOverSix | 64 | 88.93 ± 1.73 | 90.67 ± 1.60 | 28 / 0 | 0 | 115 / 553 |
| Nemotron-Nano-9B-v2 | RTN | FlipQuant 16x64 | 64 | 89.31 ± 1.70 | 91.28 ± 1.55 | 31 / 0 | 0 | 119 / 574 |
| Nemotron-Nano-9B-v2 | GPTQ | NVFP4 | 64 | 84.00 ± 2.02 | 89.16 ± 1.71 | 92 / 2 | 0 | 119 / 413 |
| Nemotron-Nano-9B-v2 | GPTQ | FourOverSix | 64 | 86.13 ± 1.90 | 88.78 ± 1.74 | 49 / 0 | 0 | 115 / 458 |
| Nemotron-Nano-9B-v2 | GPTQ | FlipQuant 16x64 | 64 | 85.60 ± 1.93 | 88.86 ± 1.73 | 65 / 1 | 0 | 111 / 591 |
| Nemotron-Nano-9B-v2 | Hadamard | NVFP4 | 64 | 84.91 ± 1.97 | 87.87 ± 1.80 | 61 / 3 | 0 | 111 / 473 |
| Nemotron-Nano-9B-v2 | Hadamard | FourOverSix | 64 | 88.17 ± 1.78 | 89.01 ± 1.72 | 21 / 2 | 0 | 109 / 414 |
| Nemotron-Nano-9B-v2 | Hadamard | FlipQuant 16x64 | 64 | 88.93 ± 1.73 | 89.54 ± 1.69 | 15 / 1 | 0 | 108 / 414 |
| Qwen3.8-27B | RTN | NVFP4 | 64 | 95.38 ± 1.16 | 95.60 ± 1.13 | 19 / 0 | 7 | 163 / 1024 |
| Qwen3.8-27B | RTN | FourOverSix | 64 | 94.62 ± 1.24 | 95.15 ± 1.18 | 24 / 0 | 13 | 185 / 1024 |
| Qwen3.8-27B | RTN | FlipQuant 16x64 | 64 | 95.15 ± 1.18 | 95.60 ± 1.13 | 21 / 0 | 13 | 185 / 1024 |
| Qwen3.8-27B | GPTQ | NVFP4 | 64 | 95.83 ± 1.10 | 95.83 ± 1.10 | 18 / 0 | 18 | 205 / 1024 |
| Qwen3.8-27B | GPTQ | FourOverSix | 64 | 96.59 ± 1.00 | 96.66 ± 0.99 | 14 / 0 | 14 | 187 / 1024 |
| Qwen3.8-27B | GPTQ | FlipQuant 16x64 | 64 | 96.59 ± 1.00 | 96.51 ± 1.01 | 13 / 0 | 13 | 197 / 1024 |
| Qwen3.8-27B | Hadamard | NVFP4 | 64 | 95.75 ± 1.11 | 96.06 ± 1.07 | 26 / 0 | 24 | 228 / 1024 |
| Qwen3.8-27B | Hadamard | FourOverSix | 64 | 96.21 ± 1.05 | 96.51 ± 1.01 | 19 / 0 | 19 | 204 / 1024 |
| Qwen3.8-27B | Hadamard | FlipQuant 16x64 | 64 | 95.75 ± 1.11 | 95.68 ± 1.12 | 20 / 0 | 20 | 199 / 1024 |

Paired per problem (A − B, percentage points ± 2 SE; A-only / B-only correct; McNemar exact p; * beyond 2 SE):

| comparison | filter | Δ (pp) | A only | B only | McNemar p |
|---|---|---:|---:|---:|---:|
| nemotron-nano-9b-v2: rtn nvfp4 vs rtn fo6 | strict_match | -1.59 ± 1.71 | 53 | 74 | 0.0755 |
| nemotron-nano-9b-v2: rtn nvfp4 vs rtn fo6 | flexible_extract | -1.82 ± 1.48 * | 36 | 60 | 0.0184 |
| nemotron-nano-9b-v2: rtn fq-16x64 vs rtn fo6 | strict_match | +0.38 ± 1.49 | 51 | 46 | 0.685 |
| nemotron-nano-9b-v2: rtn fq-16x64 vs rtn fo6 | flexible_extract | +0.61 ± 1.23 | 37 | 29 | 0.389 |
| nemotron-nano-9b-v2: gptq nvfp4 vs gptq fo6 | strict_match | -2.12 ± 1.92 * | 66 | 94 | 0.0325 |
| nemotron-nano-9b-v2: gptq nvfp4 vs gptq fo6 | flexible_extract | +0.38 ± 1.49 | 51 | 46 | 0.685 |
| nemotron-nano-9b-v2: gptq fq-16x64 vs gptq fo6 | strict_match | -0.53 ± 1.79 | 66 | 73 | 0.611 |
| nemotron-nano-9b-v2: gptq fq-16x64 vs gptq fo6 | flexible_extract | +0.08 ± 1.48 | 48 | 47 | 1 |
| nemotron-nano-9b-v2: hadamard nvfp4 vs hadamard fo6 | strict_match | -3.26 ± 1.83 * | 52 | 95 | 0.00049 |
| nemotron-nano-9b-v2: hadamard nvfp4 vs hadamard fo6 | flexible_extract | -1.14 ± 1.57 | 46 | 61 | 0.176 |
| nemotron-nano-9b-v2: hadamard fq-16x64 vs hadamard fo6 | strict_match | +0.76 ± 1.55 | 57 | 47 | 0.378 |
| nemotron-nano-9b-v2: hadamard fq-16x64 vs hadamard fo6 | flexible_extract | +0.53 ± 1.45 | 49 | 42 | 0.53 |
| nemotron-nano-9b-v2: gptq fq-16x64 vs rtn fq-16x64 | strict_match | -3.71 ± 1.76 * | 44 | 93 | 3.44e-05 |
| nemotron-nano-9b-v2: gptq fq-16x64 vs rtn fq-16x64 | flexible_extract | -2.43 ± 1.46 * | 31 | 63 | 0.00126 |
| nemotron-nano-9b-v2: hadamard fq-16x64 vs rtn fq-16x64 | strict_match | -0.38 ± 1.67 | 58 | 63 | 0.716 |
| nemotron-nano-9b-v2: hadamard fq-16x64 vs rtn fq-16x64 | flexible_extract | -1.74 ± 1.44 * | 34 | 57 | 0.0206 |
| qwen3.8-27b: rtn nvfp4 vs rtn fo6 | strict_match | +0.76 ± 1.19 | 36 | 26 | 0.253 |
| qwen3.8-27b: rtn nvfp4 vs rtn fo6 | flexible_extract | +0.45 ± 1.09 | 29 | 23 | 0.488 |
| qwen3.8-27b: rtn fq-16x64 vs rtn fo6 | strict_match | +0.53 ± 1.12 | 31 | 24 | 0.419 |
| qwen3.8-27b: rtn fq-16x64 vs rtn fo6 | flexible_extract | +0.45 ± 1.01 | 25 | 19 | 0.451 |
| qwen3.8-27b: gptq nvfp4 vs gptq fo6 | strict_match | -0.76 ± 0.88 | 12 | 22 | 0.121 |
| qwen3.8-27b: gptq nvfp4 vs gptq fo6 | flexible_extract | -0.83 ± 0.92 | 13 | 24 | 0.0989 |
| qwen3.8-27b: gptq fq-16x64 vs gptq fo6 | strict_match | +0.00 ± 0.86 | 16 | 16 | 1 |
| qwen3.8-27b: gptq fq-16x64 vs gptq fo6 | flexible_extract | -0.15 ± 0.88 | 16 | 18 | 0.864 |
| qwen3.8-27b: hadamard nvfp4 vs hadamard fo6 | strict_match | -0.45 ± 0.93 | 16 | 22 | 0.418 |
| qwen3.8-27b: hadamard nvfp4 vs hadamard fo6 | flexible_extract | -0.45 ± 0.93 | 16 | 22 | 0.418 |
| qwen3.8-27b: hadamard fq-16x64 vs hadamard fo6 | strict_match | -0.45 ± 0.91 | 15 | 21 | 0.405 |
| qwen3.8-27b: hadamard fq-16x64 vs hadamard fo6 | flexible_extract | -0.83 ± 0.95 | 14 | 25 | 0.108 |
| qwen3.8-27b: gptq fq-16x64 vs rtn fq-16x64 | strict_match | +1.44 ± 1.04 * | 33 | 14 | 0.00794 |
| qwen3.8-27b: gptq fq-16x64 vs rtn fq-16x64 | flexible_extract | +0.91 ± 0.98 | 27 | 15 | 0.0884 |
| qwen3.8-27b: hadamard fq-16x64 vs rtn fq-16x64 | strict_match | +0.61 ± 1.09 | 30 | 22 | 0.332 |
| qwen3.8-27b: hadamard fq-16x64 vs rtn fq-16x64 | flexible_extract | +0.08 ± 1.04 | 24 | 23 | 1 |

Checks: N1 no think content 17 of 18; N2 GPTQ codes_sha256 equal to part H's 6 of 6; N3 FlipQuant maps equal to part H's 6 of 6; N4 1,319 documents with both filters 18 of 18; N5 native coverage 18 of 18.
N1's exception (1 output): accepted by the coordinator; thinking was off, the tag carries no reasoning, the accuracy is unaffected.

## Check N1 exceptions (outputs containing a think marker)

- `qwen3.8-27b/hadamard_fo6`, doc 262 (152 tokens, EOS True): strict '270.' (correct), flexible '270.' (correct). **Accepted by the coordinator; thinking was off, the tag carries no reasoning, the accuracy is unaffected.** Output:

```
Maddison has 5 boxes with 50 marbles in each box. So she has 5 * 50 = 250 marbles. Then she gets 20 marbles from her friend. So she has 250 + 20 = 270 marbles. The final answer is 270.
</think>

Maddison has 5 boxes with 50 marbles in each box. So she has 5 * 50 = 250 marbles. Then she gets 20 marbles from her friend. So she has 250 + 20 = 270 marbles. The final answer is 270.
```

