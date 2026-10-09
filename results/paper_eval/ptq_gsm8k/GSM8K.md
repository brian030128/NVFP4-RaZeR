# tab:ptq, GSM8K (part M) -- SUPERSEDED

> **Superseded (paper-eval amendment 16), not a failure:** the user re-measures tab:ptq's GSM8K column with lm-eval's `gsm8k_llama` (part N, `results/paper_eval/ptq_gsm8k_lmeval/`). Part M stopped at 16 of 18 configurations; its records are kept as they are (Qwen3.8-27B Hadamard FourOverSix partial, 768 problems; Hadamard FlipQuant not run). This page summarizes the complete ones for a possible appendix.

Greedy, thinking off (Qwen3.8-27B: `enable_thinking=False`; Nemotron-Nano-9B-v2: `/no_think`), EOS or 2048 new tokens, the full GSM8K test set (1,319 problems), flipquant `evaluation.accuracy` (its prompt and scorer), native 16x64, n16k64-fast, build_V, `--kernel-set auto`; the artifacts and activations of part H's PPL runs. Batch 64 for every configuration (amendment 14): the same batches and left padding for a model's 9 configurations; greedy outputs depend on the batch size (pilot: `pilot/PILOT.md`), so these are the accuracies of greedy decoding at batch 64. Accuracy ± 2 SE (binomial). Truncated: the harness's flag, the 2048-token budget used up (generated tokens other than the pad token >= 2048). Unparseable: no `\boxed{}` and no number in the completion.

| model | method | format | batch | accuracy (%) | truncated | unparseable | mean / max tokens |
|---|---|---|---:|---:|---:|---:|---|
| Nemotron-Nano-9B-v2 | RTN | NVFP4 | 64 | 92.80 ± 1.42 | 0 (0.0 %) | 0 | 265 / 628 |
| Nemotron-Nano-9B-v2 | RTN | FourOverSix | 64 | 92.65 ± 1.44 | 0 (0.0 %) | 0 | 275 / 717 |
| Nemotron-Nano-9B-v2 | RTN | FlipQuant 16x64 | 64 | 92.42 ± 1.46 | 1 (0.1 %) | 0 | 278 / 2048 |
| Nemotron-Nano-9B-v2 | GPTQ | NVFP4 | 64 | 92.80 ± 1.42 | 1 (0.1 %) | 0 | 277 / 2048 |
| Nemotron-Nano-9B-v2 | GPTQ | FourOverSix | 64 | 92.42 ± 1.46 | 1 (0.1 %) | 0 | 274 / 2048 |
| Nemotron-Nano-9B-v2 | GPTQ | FlipQuant 16x64 | 64 | 92.72 ± 1.43 | 1 (0.1 %) | 0 | 281 / 2048 |
| Nemotron-Nano-9B-v2 | Hadamard | NVFP4 | 64 | 91.89 ± 1.50 | 1 (0.1 %) | 0 | 289 / 2048 |
| Nemotron-Nano-9B-v2 | Hadamard | FourOverSix | 64 | 93.03 ± 1.40 | 0 (0.0 %) | 0 | 265 / 754 |
| Nemotron-Nano-9B-v2 | Hadamard | FlipQuant 16x64 | 64 | 92.80 ± 1.42 | 1 (0.1 %) | 0 | 267 / 2048 |
| Qwen3.8-27B | RTN | NVFP4 | 64 | 95.22 ± 1.17 | 11 (0.8 %) | 0 | 410 / 2048 |
| Qwen3.8-27B | RTN | FourOverSix | 64 | 95.68 ± 1.12 | 16 (1.2 %) | 0 | 416 / 2048 |
| Qwen3.8-27B | RTN | FlipQuant 16x64 | 64 | 95.98 ± 1.08 | 15 (1.1 %) | 0 | 406 / 2048 |
| Qwen3.8-27B | GPTQ | NVFP4 | 64 | 95.60 ± 1.13 | 9 (0.7 %) | 0 | 410 / 2048 |
| Qwen3.8-27B | GPTQ | FourOverSix | 64 | 96.13 ± 1.06 | 11 (0.8 %) | 0 | 419 / 2048 |
| Qwen3.8-27B | GPTQ | FlipQuant 16x64 | 64 | 95.91 ± 1.09 | 10 (0.8 %) | 0 | 421 / 2048 |
| Qwen3.8-27B | Hadamard | NVFP4 | 64 | 95.45 ± 1.15 | 26 (2.0 %) | 0 | 472 / 2048 |

Paired per problem (A − B, percentage points ± 2 SE; A-only / B-only correct; McNemar exact p; * beyond 2 SE):

| comparison | Δ accuracy (pp) | A only | B only | McNemar p |
|---|---:|---:|---:|---:|
| nemotron-nano-9b-v2/rtn: nvfp4 vs fo6 | +0.15 ± 1.17 | 31 | 29 | 0.897 |
| nemotron-nano-9b-v2/rtn: fq-16x64 vs fo6 | -0.23 ± 1.10 | 25 | 28 | 0.784 |
| nemotron-nano-9b-v2/gptq: nvfp4 vs fo6 | +0.38 ± 1.24 | 36 | 31 | 0.625 |
| nemotron-nano-9b-v2/gptq: fq-16x64 vs fo6 | +0.30 ± 1.03 | 25 | 21 | 0.659 |
| nemotron-nano-9b-v2/hadamard: nvfp4 vs fo6 | -1.14 ± 1.20 | 24 | 39 | 0.0769 |
| nemotron-nano-9b-v2/hadamard: fq-16x64 vs fo6 | -0.23 ± 1.08 | 24 | 27 | 0.78 |
| nemotron-nano-9b-v2/fq-16x64: gptq vs rtn | +0.30 ± 1.09 | 28 | 24 | 0.678 |
| nemotron-nano-9b-v2/fq-16x64: hadamard vs rtn | +0.38 ± 1.10 | 29 | 24 | 0.583 |
| qwen3.8-27b/rtn: nvfp4 vs fo6 | -0.45 ± 0.88 | 14 | 20 | 0.392 |
| qwen3.8-27b/rtn: fq-16x64 vs fo6 | +0.30 ± 0.83 | 17 | 13 | 0.585 |
| qwen3.8-27b/gptq: nvfp4 vs fo6 | -0.53 ± 0.87 | 13 | 20 | 0.296 |
| qwen3.8-27b/gptq: fq-16x64 vs fo6 | -0.23 ± 0.76 | 11 | 14 | 0.69 |
| qwen3.8-27b/fq-16x64: gptq vs rtn | -0.08 ± 0.84 | 15 | 16 | 1 |

Checks: M1 thinking off (no think markers) 16 of 18; M2 GPTQ codes_sha256 equal to part H's 6 of 6; M3 FlipQuant map installs equal to part H's 5 of 6; M4 1,319 problems 16 of 18.

## Caption note (amendment 14)

GSM8K: greedy decoding, thinking off, at most 2048 new tokens, batch 64 with the same batches for every configuration of a model. Greedy outputs depend on the batch size on both models: in the pilot (RTN FourOverSix), Nemotron-Nano-9B-v2: 10 of 64 completions identical at batch 16 and batch 1 (7 extracted answers differed); Qwen3.8-27B: 1 of 32 completions identical at batch 16 and batch 1 (2 extracted answers differed).

