# Part E: cost estimate (nothing run)

lm-eval 0.4.11 log-likelihood suite (MMLU 5-shot, MMLU-Pro 5-shot by log-likelihood, ARC-C/E, HellaSwag, PIQA, WinoGrande, BoolQ, LAMBADA), 6 models x 6 policies (BF16, NVFP4, FourOverSix, FlipQuant 8x64 / 16x64 / 256x64), from the FlipQuant paper's timed lm-eval runs; assumptions in `experiments/paper_eval/estimate_e.py`.

| model | hours (low / mid / high) | without MMLU-Pro (mid) |
|---|---:|---:|
| qwen3-1.7b | 1.3 / 1.5 / 1.9 | 0.8 |
| qwen3-8b | 3.0 / 3.6 / 4.5 | 1.7 |
| mistral-7b | 3.0 / 3.6 / 4.5 | 1.7 |
| nemotron-nano-9b-v2 | 3.5 / 4.1 / 5.1 | 2.0 |
| phi4-14b | 4.4 / 5.2 / 6.5 | 2.5 |
| qwen3.8-27b | 7.5 / 11.5 / 21.7 | 6.9 |
| total | 22.8 / 29.5 / 44.1 | 15.7 |

Measured basis (paper step 04, seconds per task, mean over its FP4 policies; BF16 in brackets):

| paper model | mmlu | arc_challenge | arc_easy | hellaswag | piqa |
|---|---:|---:|---:|---:|---:|
| llama8b | 595 (917) | 21 (25) | 31 (37) | 146 (262) | 19 (22) |
| mistral7b | 589 (961) | 20 (25) | 33 (39) | 138 (263) | 19 (22) |
| phi4 | 861 (1485) | 23 (33) | 35 (51) | 215 (436) | 21 (29) |
| qwen27b | 2397 (3563) | 221 (220) | 411 (409) | 2088 (2073) | 191 (188) |
