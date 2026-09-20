### Full-model native Llama-3.1-8B latency on GB200 (diagnostic)

This measures the complete model using the sibling SM100 GEMM with a tested
per-256×64 format map. All 224 transformer projections use native packed FP4;
embeddings, normalization, attention, residuals, and the vocabulary head retain
their normal model implementation. Every packed weight decodes to its frozen
fake-quantized BF16 reference bitwise. All 672 projection checks across three
policies passed (<0.005 relative error against a decoded FP64 dot reference),
activation codes/scales matched exactly, and native repeated forwards were
bitwise reproducible. The warmed follow-up reuses these audits after hash/AST checks
and repeats every source-weight and packed-weight audit.

**This is a diagnostic latency measurement, not validated native model quality.**
The free-running full-model reference-equivalence gate failed for mixed policies.
Small GEMM rounding differences amplify through subsequent FP4 quantization;
the first divergent layer has relative error 4.586e-6, while arranged final
logits differ by about 10.8% from the FP32 epilogue-order reference. The published
PPL figures remain the prior BF16 fake-quantized quality measurements. Native
PPL is unmeasured, and previous full-output gate failures remain failures.

Batch size is one. Each request performs prefill and 32 actual KV-cached decode
steps using identical fixed tokens across policies. The head computes last-token
logits, as in generation. Results are medians of 5 paired runs. This first pilot warmed only one decode position; cold initialization affected its baseline samples, so it must not establish a speedup. Loading, offline packing, tokenizer work, and sampling are excluded.
Activation amax, FourOverSix activation quantization, attention, normalization,
cache updates, and permutations are included. These are measurements of this
matched eager Transformers/native backend, not an optimized serving engine.

Columns are fused into the packed activation store. This first full-model
integration restores permuted output rows in a separate kernel; it does not
claim the fully consumer-fused microbenchmark implementation.

| Prompt tokens | Policy | Prefill ms | Decode ms/token | Request ms (prefill + 32 tokens) | Paired request overhead, mean ±2SE |
|---:|---|---:|---:|---:|---:|
| 128 | FourOverSix NVFP4 | 30.231 | 28.718 | 949.492 | — |
| 128 | Raw MixFP4 256×64 (187 tiles) | 33.940 | 30.210 | 1003.791 | -4.12% ± 20.73% |
| 128 | Arranged MixFP4 256×64 (147 tiles) | 30.746 | 28.189 | 932.781 | -8.72% ± 20.57% |
| 2048 | FourOverSix NVFP4 | 56.462 | 30.189 | 1022.582 | — |
| 2048 | Raw MixFP4 256×64 (187 tiles) | 56.556 | 28.648 | 973.279 | -13.83% ± 19.71% |
| 2048 | Arranged MixFP4 256×64 (147 tiles) | 56.689 | 30.211 | 1023.433 | -10.10% ± 22.23% |

The ±2SE values describe paired run variation across these paired repetitions;
small negative overhead does not establish a speedup. CUDA elapsed time is shown;
synchronized host-wall measurements and each repetition are retained in the report.

For warmed job 406828, the 128-token paired request overhead is informative only
for this eager backend. At 2,048 tokens, decode time varies roughly 27–53 ms/token
across all policies. **The 2,048-token overhead is inconclusive**; retain every
sample and do not interpret its point estimate as an algorithmic penalty.
The earlier job 406751 is retained as a cold-initialization pilot, not the primary
request comparison. No post-hoc samples are removed.

The following **GEMM-only** measurements use the actual frozen final-MLP weights
and heterogeneous format masks. Activation quantization and output restoration
are outside this timing. Each sample replays a graph containing 100 GEMM calls;
values are medians of five paired repetitions. These short graph measurements
also vary between jobs: for example, gate M=1 changes from +0.32% in pilot 406751
to -12.91% in 406828. Negative point estimates do not establish a speedup.
This measures the complete arranged-versus-baseline GEMM difference, not an
isolated format-branch cost. The uniform-format 8192-cubed control above has a
different scope.

| Projection | M | NVFP4 µs | Arranged MixFP4 µs | GEMM-only overhead |
|---|---:|---:|---:|---:|
| down_proj | 1 | 20.067 | 22.239 | +10.83% |
| down_proj | 128 | 23.614 | 23.741 | +0.54% |
| down_proj | 2048 | 39.173 | 39.612 | +1.12% |
| gate_proj | 1 | 13.695 | 13.739 | +0.32% |
| gate_proj | 128 | 10.713 | 10.844 | +1.22% |
| gate_proj | 2048 | 42.352 | 43.761 | +3.33% |
| up_proj | 1 | 12.351 | 13.201 | +6.88% |
| up_proj | 128 | 10.604 | 11.426 | +7.75% |
| up_proj | 2048 | 42.769 | 43.507 | +1.72% |

[Complete measurements](results/task_reorder/full_model_20260920/llama_406751/report.json), [summary](results/task_reorder/full_model_20260920/llama_406751/summary.json), and [frozen protocol](results/task_reorder/full_model_20260920/plan.json).
