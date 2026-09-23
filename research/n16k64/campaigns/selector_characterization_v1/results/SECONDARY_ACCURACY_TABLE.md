# Secondary accuracy: fixed seed0, full eight tasks

Equal-task macro points; MMLU is question-weighted within its task. No confidence interval, non-inferiority, population-risk or five-draw accuracy claim. Only admitted complete runs enter these tables.

| Model | Policy | Valid tasks | Macro accuracy (%) | Change vs baseline (pp) | Negative task changes |
|---|---|---:|---:|---:|---:|
| llama8b | baseline | 8/8 | 68.8243 | +0.0000 | 0/8 |
| llama8b | joint | 8/8 | 68.7150 | -0.1093 | 3/8 |
| llama8b | ce_matched | 0/8 | missing | missing | missing |
| llama8b | kl_matched | 0/8 | missing | missing | missing |
| qwen4b | baseline | 8/8 | 64.2625 | +0.0000 | 0/8 |
| qwen4b | joint | 8/8 | 64.9119 | +0.6494 | 1/8 |
| qwen4b | ce_matched | 8/8 | 65.4314 | +1.1689 | 0/8 |
| qwen4b | kl_matched | 8/8 | 64.5073 | +0.2448 | 3/8 |
| mistral7b | baseline | 8/8 | 68.7956 | +0.0000 | 0/8 |
| mistral7b | joint | 8/8 | 69.0335 | +0.2379 | 2/8 |
| mistral7b | ce_matched | 8/8 | 68.7988 | +0.0032 | 3/8 |
| mistral7b | kl_matched | 8/8 | 69.1640 | +0.3684 | 2/8 |
