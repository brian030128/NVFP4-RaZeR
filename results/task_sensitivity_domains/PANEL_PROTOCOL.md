# Model transfer panel

Declared before the target-domain measurements. Apply the same four rules to
Qwen3-4B and Llama-3.1-8B, using native Transformers 5.16.1 model code and
quantizing every non-head Linear weight and input. This is a new matched
baseline, not a comparison of absolute PPL against the old copied model code.
Use fresh WikiText seed 20260918 and C4 seed 20260920. Same 64-window single
domain and 32+32 mixed budgets, 16-window single-domain validation and 8+8
mixed/consensus validation, thresholds, candidate formulas, and 8x64 tiles.
Compute the single-domain moments by exactly pooling two 32-window chunks.
No old scores or maps are reused. No isolated-tile diagnostic is repeated.

Freeze all maps before testing. Use the target's fresh WikiText split, C4
shard and sampling seed, and GSM8K/MBPP row selections, tokenized for each model.
C4 qualifies documents by token length, so different tokenizers can select
different qualifying documents. WikiText window boundaries also vary with
tokenization. Pair all policies on identical examples WITHIN each model;
do not claim the C4 document lists are identical across tokenizers.
Record model and dataset revisions, tokens, selected tiles, gate outcomes,
paired candidate NLLs and exported fallback behavior. One fresh seed per panel
model is a transfer check, not a seed-robustness claim. Runs use at most two
H100 GPUs in total, after the target and stress jobs.
