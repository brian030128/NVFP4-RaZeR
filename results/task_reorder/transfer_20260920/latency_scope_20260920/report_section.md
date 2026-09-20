### Latency overhead: GEMM only, projection pipeline, and full model

**Hardware:** the available native measurements were made on **NVIDIA GB200
(SM100)**, not on a separately tested B200 system. They are relevant Blackwell
measurements, but should retain their actual hardware label.

| Timing scope | What is timed | Measured overhead | Status |
|---|---|---|---|
| Matrix multiplication only | Same GEMM executable, E2M1 activations, uniform E2M1 versus E0M3 weights | Approximately zero; −0.012% ordinary / −0.093% graph | Measured format-path comparison; differences within observed run-to-run spread |
| Per-projection pipeline | FourOverSix quantization → GEMM → SiLU/multiply or residual-add; permutations fused into producer/consumer | +1.4% to +5.1% | Measured permutation overhead; common tensor-amax excluded |
| Full-model end-to-end prefill/decode | Complete native model, including attention, all layers, normalization, KV-cache work and runtime scheduling | **Not measured** | No full-model percentage, TTFT or decode latency can be claimed from these tests |

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

**Full-model end-to-end measurement remains outstanding.** It requires a native
model path using the frozen heterogeneous 256×64 format maps and the permutation
contracts, followed by matched prefill and cached-decode tests against NVFP4.
The fake-quantized PyTorch perplexity harness is a quality reference and cannot
supply that native latency measurement.
