### Latency overhead: GEMM only, projection pipeline, and full model

**Hardware:** the available native measurements were made on **NVIDIA GB200
(SM100)**, not on a separately tested B200 system. They are relevant Blackwell
measurements, but should retain their actual hardware label.

| Timing scope | What is timed | Measured overhead | Status |
|---|---|---|---|
| Matrix multiplication only | Same GEMM executable, E2M1 activations, uniform E2M1 versus E0M3 weights | Approximately zero; −0.012% ordinary / −0.093% graph | Measured format-path comparison; differences within observed run-to-run spread |
| Per-projection pipeline | FourOverSix quantization → GEMM → SiLU/multiply or residual-add; permutations fused into producer/consumer | +1.4% to +5.1% | Measured permutation overhead; common tensor-amax excluded |
| Full-model prefill/decode, Llama | Complete native model including attention, all layers, KV cache, activation amax/quantization and permutations | 128-token prompt +32 decode: arranged +0.53% ±0.53% (2SE); 2048-token overhead inconclusive | Diagnostic eager-backend timing; native accuracy unverified; Qwen unmeasured |
| Actual heterogeneous GEMM only, Llama | Frozen final-MLP weights/maps; quantization and row restoration excluded | Shape-dependent; all nine shapes below, including run variation | Measured separately from uniform-format control |

#### Matrix multiplication only

The saved sibling-kernel experiment compares the same mixed GEMM executable and
its default launch configuration, holding activations at E2M1 while switching
weights from E2M1 to E0M3. Thus the E2M1/E2M1 row is the matched NVFP4 arithmetic
path, not a separately tuned cuBLAS denominator. Shape is **8192×8192×8192**,
GEMM tile 256×256×256, format granularity compatible with 256×64 weight blocks,
BF16 output, FP32 accumulation, and PDL enabled.

| Timing mode | NVFP4 arithmetic µs | E0M3-weight arithmetic µs | Added latency µs | Overhead |
|---|---:|---:|---:|---:|
| CUDA Graph | 136.455 | 136.328 | -0.127 | -0.093% |
| Ordinary launch | 136.691 | 136.674 | -0.017 | -0.012% |

These latency values are converted from the median of three recorded throughput
samples, using `latency_us = 2 M N K / (TFLOP/s × 10^6)`. Tiny negative overheads
are consistent with measurement variation, not a claimed speedup. This directly
supports negligible **uniform format-selection** cost. It does not benchmark a
heterogeneous per-tile format map or the full accepted quantized model. The user's
previous verification that raw 256×64 MixFP4 matches NVFP4 is retained separately;
no unrecorded per-map latency is invented here.

[Saved GEMM-only samples](results/task_reorder/transfer_20260920/latency_scope_20260920/sm100_graph.json),
[ordinary-launch samples](results/task_reorder/transfer_20260920/latency_scope_20260920/sm100_launch.json),
and [derived latency table](results/task_reorder/transfer_20260920/latency_scope_20260920/gemm_only_summary.json)
come from historical job 400605. These completed controls were reused without
spending additional GPU hours.

#### Quantization → GEMM → consumer pipeline

The table below gives the measured local pipeline comparison. Each denominator
contains the same FourOverSix producer, GEMM and consumer, with no permutation;
the numerator fuses the required permutations into producer/consumer indexing.
This is the **measured projection-pipeline overhead**, not full-model end-to-end
latency and not an isolated heterogeneous-format GEMM comparison.

| Projection (N×K) | Tokens M | Baseline pipeline µs | Arranged pipeline µs | Added latency µs | Overhead |
|---|---:|---:|---:|---:|---:|
| up (17408×5120) | 1 | 20.779 | 21.156 | +0.377 | +1.81% |
| up (17408×5120) | 128 | 28.487 | 29.628 | +1.141 | +4.01% |
| up (17408×5120) | 512 | 69.934 | 72.993 | +3.059 | +4.37% |
| up (17408×5120) | 2,048 | 240.623 | 253.004 | +12.381 | +5.15% |
| down (5120×17408) | 1 | 28.967 | 29.713 | +0.746 | +2.58% |
| down (5120×17408) | 128 | 48.334 | 49.243 | +0.909 | +1.88% |
| down (5120×17408) | 512 | 117.144 | 118.899 | +1.755 | +1.50% |
| down (5120×17408) | 2,048 | 421.932 | 427.891 | +5.959 | +1.41% |

Job 405810 uses exact compacted Qwen maps, synthetic weights, fixed E2M1 GEMM
format and a shared global activation scale reapplied through GEMM's alpha.
Allocation, offline weight preparation, common tensor-amax, and graph setup are
excluded. The absolute quantizer implementation cost affects the denominator.
The up and down tests are independent; their overhead percentages must not be
summed or reported as a full MLP/model overhead. Matrix-only and pipeline results
are different experiments and cannot be added to manufacture a full-model number.

The native Llama measurement below now covers every model layer and runtime
activation quantization. Its diagnostic qualification is essential: native
full-output equivalence failed and native PPL is unmeasured. The historical
projection pipeline above used its recorded earlier producer implementation;
subsequent real-input quantizer corrections are documented in the native
[implementation notes](results/task_reorder/full_model_20260920/IMPLEMENTATION.md).
Qwen full-model native integration and latency remain outstanding.

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
