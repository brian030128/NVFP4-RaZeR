# Audit of the discrepancy from RaZeR Table 3

No calibration or E0M3 selection was performed. All models use pinned original weights, BF16 execution, eager attention, and unquantized KV tensors. The released evaluator also loads BF16 despite its FP16 label. FourOverSix quantizes the same nonhead linear weights and inputs as the math/code study.

Wiki uses identical complete 2048-token spans, split into four windows for the 512 condition. C4 paper-protocol crops use shard 00000 and the released seed-0 sampling procedure; their 512 condition splits the same parent crops into four windows. More window boundaries also change which first tokens are excluded from loss. The historical C4 study uses different documents from shard 00001.

| Model | Context | Method | Wiki PPL | C4 paper-sample PPL | C4 historical-sample PPL |
|---|---:|---|---:|---:|---:|
| Qwen3-4B | 512 | bf16 | 18.414227 | 20.252757 | 20.887771 |
| Qwen3-4B | 512 | four_over_six_row | 19.414897 | 21.336690 | 21.928243 |
| Qwen3-4B | 512 | four_over_six_tensor | 19.464810 | 21.333247 | 21.967226 |
| Qwen3-4B | 2048 | bf16 | 13.663008 | 16.645096 | — |
| Qwen3-4B | 2048 | four_over_six_row | 14.201139 | 17.312386 | — |
| Qwen3-4B | 2048 | four_over_six_tensor | 14.213994 | 17.299487 | — |
| Qwen3-4B | 2048 | nvfp4_tensor | 13.981052 | 17.281944 | — |
| Llama-3.1-8B | 512 | bf16 | 8.180760 | 10.729659 | 10.547444 |
| Llama-3.1-8B | 512 | four_over_six_row | 9.092934 | 11.865411 | 11.600109 |
| Llama-3.1-8B | 512 | four_over_six_tensor | 9.085640 | 11.866302 | 11.601548 |
| Llama-3.1-8B | 2048 | bf16 | 6.240317 | 8.957994 | — |
| Llama-3.1-8B | 2048 | four_over_six_row | 6.875328 | 9.818250 | — |
| Llama-3.1-8B | 2048 | four_over_six_tensor | 6.879243 | 9.823488 | — |
| Llama-3.1-8B | 2048 | nvfp4_tensor | 6.939384 | 9.929869 | — |

## Direct comparison with the published values

The audit column uses 2048 tokens and tensor-wide activation factors. Paper values are rounded to two decimals. Δ is audit minus paper; the comparison does not assert identical checkpoint revisions or attention backends to the original experiment.

| Model | Method | Paper Wiki | Audit Wiki | ΔWiki | Paper C4 | Audit C4 | ΔC4 |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | bf16 | 13.66 | 13.663008 | +0.003008 | 16.65 | 16.645096 | -0.004904 |
| Qwen3-4B | four_over_six_tensor | 13.88 | 14.213994 | +0.333994 | 17.21 | 17.299487 | +0.089487 |
| Qwen3-4B | nvfp4_tensor | 13.88 | 13.981052 | +0.101052 | 17.21 | 17.281944 | +0.071944 |
| Llama-3.1-8B | bf16 | 6.24 | 6.240317 | +0.000317 | 8.96 | 8.957994 | -0.002006 |
| Llama-3.1-8B | four_over_six_tensor | 6.88 | 6.879243 | -0.000757 | 9.83 | 9.823488 | -0.006512 |
| Llama-3.1-8B | nvfp4_tensor | 6.95 | 6.939384 | -0.010616 | 9.94 | 9.929869 | -0.010131 |

## Historical Qwen attention-output activation bug

Repository commit [abab3c6](qwen_o_proj_fix.patch) (2026-06-17) changed o_proj(attn_output) to o_proj(attn_output_quant). The earlier Qwen wrapper computed but discarded quantized attention-output activations. The following diagnostic reproduces that behavior by leaving only those projection inputs unquantized; their weights remain quantized. It uses exactly the same checkpoints, input hashes and 2048-token windows as the corrected audit.

| Method | Paper Wiki | Historical-behavior Wiki | Corrected Wiki | Paper C4 | Historical-behavior C4 | Corrected C4 |
|---|---:|---:|---:|---:|---:|---:|
| four_over_six_tensor | 13.88 | 14.137439 | 14.213994 | 17.21 | 17.286469 | 17.299487 |
| nvfp4_tensor | 13.88 | 13.948580 | 13.981052 | 17.21 | 17.222591 | 17.281944 |

This comparison measures the effect of the historical behavior. Any remaining gap to the paper is still unresolved; the code history does not independently prove which exact revision generated its table. The corrected W4A4 baseline remains the appropriate comparison for a method that quantizes all these inputs.

## Sequential protocol differences for FourOverSix

These are changes along a fixed comparison path, not independently additive causal effects. The residual includes paper rounding and remaining implementation/checkpoint differences.

| Model | Dataset | Historical 512 PPL | Aligned-data 512 PPL | Per-token 2048 PPL | Tensor-wide 2048 PPL | Paper PPL |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B | wiki | 19.414897 | 19.414897 | 14.201139 | 14.213994 | 13.880000 |
| Qwen3-4B | c4 | 21.928243 | 21.336690 | 17.312386 | 17.299487 | 17.210000 |
| Llama-3.1-8B | wiki | 9.092934 | 9.092934 | 6.875328 | 6.879243 | 6.880000 |
| Llama-3.1-8B | c4 | 11.600109 | 11.865411 | 9.818250 | 9.823488 | 9.830000 |

The native hook implementation and released quantized model wrappers produced exactly equal logits on the first 2048-token Wiki window for BF16 and tensor-wide FourOverSix at matched eager attention. Every historical C4 baseline loss and every shared historical Wiki baseline loss replayed within 1e-6. These checks constrain, but do not exhaust, possible bugs.

Tensor-wide activation factors may depend on later tokens within the evaluated window; matching the released evaluator is a reproduction condition, not a causal deployment guarantee. The historical sparse-map gains remain specific to their 512-token protocol; this audit does not evaluate those maps at 2048 or validate a paper-comparable improvement.

[Protocol](PROTOCOL.md). [RaZeR Table 3](https://arxiv.org/html/2501.04052v2#S4.T3).
