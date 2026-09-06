# Additional-model extension

The frozen 64-window fit/backtracking rule was extended without changing its
thresholds. Array 329090 completed two accessible models with seed 20260909:

| Model | E0M3 8x64 tiles | Accepted fit budget | WikiText delta PPL | C4 delta PPL |
|---|---:|---:|---:|---:|
| Qwen3-8B | 275 | 0.1 | -0.717138 | -0.421764 |
| Llama3.1-8B-Instruct | 2654 | 0.1 | -0.209375 | -0.034045 |

Both passed the independent validation check. The Llama C4 change remains
uncertain: paired delta NLL -0.003521 with SE 0.002107.

Tasks 0 and 1 could not download Llama3.2-3B and Llama2-7B because the configured
credentials did not provide gated repository access (HTTP 401). These are
access failures, not quantization results. Pending tasks 4–15 were cancelled
to prioritize the user's proposed final target, Qwen/Qwen3.8-27B.
`slurm/task_sensitivity_extension.sbatch` preserves the extension manifest.

The two completed models are additional calibration repeats of models in the
initial six-model panel, not two newly evaluated architectures. The 27B target
adds a substantially different hybrid architecture if its experiment completes.
