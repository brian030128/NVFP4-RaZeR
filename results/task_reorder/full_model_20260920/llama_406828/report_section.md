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
logits, as in generation. Results are medians of 6 paired runs. Two complete requests per policy/context warmed all 32 decode positions; all six policy orders were measured and Python GC was disabled. Loading, offline packing, tokenizer work, and sampling are excluded.
Activation amax, FourOverSix activation quantization, attention, normalization,
cache updates, and permutations are included. These are measurements of this
matched eager Transformers/native backend, not an optimized serving engine.

Columns are fused into the packed activation store. This first full-model
integration restores permuted output rows in a separate kernel; it does not
claim the fully consumer-fused microbenchmark implementation.

| Prompt tokens | Policy | Prefill ms | Decode ms/token | Request ms (prefill + 32 tokens) | Paired request overhead, mean ±2SE |
|---:|---|---:|---:|---:|---:|
| 128 | FourOverSix NVFP4 | 29.597 | 27.530 | 911.380 | — |
| 128 | Raw MixFP4 256×64 (187 tiles) | 30.000 | 27.664 | 914.977 | +0.36% ± 0.82% |
| 128 | Arranged MixFP4 256×64 (147 tiles) | 29.620 | 27.670 | 915.747 | +0.53% ± 0.53% |
| 2048 | FourOverSix NVFP4 | 56.742 | 34.230 | 1152.110 | — |
| 2048 | Raw MixFP4 256×64 (187 tiles) | 56.479 | 38.462 | 1287.276 | +2.75% ± 11.22% |
| 2048 | Arranged MixFP4 256×64 (147 tiles) | 57.774 | 47.938 | 1594.820 | +14.81% ± 20.64% |

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
| down_proj | 1 | 19.518 | 21.655 | +10.95% |
| down_proj | 128 | 18.902 | 19.202 | +1.58% |
| down_proj | 2048 | 39.530 | 39.692 | +0.41% |
| gate_proj | 1 | 13.431 | 11.696 | -12.91% |
| gate_proj | 128 | 10.581 | 10.676 | +0.90% |
| gate_proj | 2048 | 44.473 | 43.787 | -1.54% |
| up_proj | 1 | 14.811 | 15.758 | +6.40% |
| up_proj | 128 | 12.465 | 12.842 | +3.02% |
| up_proj | 2048 | 42.574 | 43.328 | +1.77% |

[Complete measurements](results/task_reorder/full_model_20260920/llama_406828/report.json), [summary](results/task_reorder/full_model_20260920/llama_406828/summary.json), and [frozen protocol](results/task_reorder/full_model_20260920/plan.json).
