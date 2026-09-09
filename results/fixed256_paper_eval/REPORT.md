# Fixed-256 results under the released evaluation protocol

[All separate WikiText/C4 results, means, block counts, and signed differences](job_335428/REPORT.md) ·
[CSV](job_335428/fixed256_cells.csv) · [Machine-readable paired results](job_335428/summary.json).

All 33 cases / 66 PPL cells passed validation. Only the ten existing
math/code-only fixed-256 maps and one matched FourOverSix baseline per
model were evaluated. No adaptive selection or recalibration was performed.
Fixed-256 improves point PPL in 54/60 comparisons: all Qwen3-4B and
Llama-3.1-8B cells, and 14/20 Qwen3.8-27B cells. These are correlated
comparisons, not independent replications.

The previous evaluation differed from the released evaluator in context
length (512 instead of 2048), C4 sampling, per-token versus tensor-wide
activation scales, and WikiText cache behavior. The corrected evaluation
matches the released 2048-token dataset protocol, float32 aggregation,
SDPA backend, fresh WikiText cache, and disabled C4 cache. Every targeted
linear input is quantized, including Qwen o_proj; the historical release's
omission is not used in this full-W4A4 comparison. Tensor-wide activation
scales retain the paper-style simulation's dependence on future tokens.

The Llama baseline exactly reproduces the archived released-code result
at 6.875524520874023 / 9.82373332977295. Qwen3-4B baseline 14.269062 /
17.326633 matches the corrected repository wrapper and includes o_proj
activation quantization. Published-table residuals remain documented in
the separate released-code reproduction; these results use fresh matched
baselines, not substituted paper values.

The common math_code128 map improves both datasets for Qwen3-4B and Llama.
For Qwen3.8-27B, it worsens WikiText by +0.005361 PPL and improves C4 by
-0.020648 PPL. All maps select exactly 256 E0M3 blocks; no winning setting
was chosen using these evaluation results.

[Protocol](PROTOCOL.md) · [Execution and cache diagnostic](RUN.md).
