### Continued Llama research: independent gate passed, both PPLs improved

The initial Llama transfer failure above remains a failure. The renewed study
found that most of its degradation came from `down_proj`, and that summing
single-tile improvements was unreliable: tile effects interacted in the actual
quantized model. A gate/up-only candidate and an eight-down-tile refinement both
improved mean fresh CE but failed their frozen confidence gates. Neither was
sent to perplexity evaluation.

The successful refinement keeps the learned row/column arrangements and searches
**subsets of the already joint-CE/KL-k3-elected tiles**. It changes actual format
masks, not the original codebooks, quantization scales, or legal 256×64 geometry.
It uses no rotation or correction GEMM.

1. Cache the final MLP input and residual from the raw quantized model. Replay
   the final MLP, final norm, and vocabulary head for each trial. Verify cached
   losses bit-for-bit against full-model losses before searching.
2. Use all **192 previously observed documents as development data**, including
   the rejected confirmation sets. These documents are explicitly no longer
   independent validation. Preserve each prior failure.
3. Evaluate each proposed tile flip with the actual joint quantized-model CE,
   retaining interactions between gate, up, and down. Search at most four flips
   among the 50 originally elected tiles. Minimize the worst of pooled CE+2SE,
   each-domain CE+1SE, and each observed 64-document set's CE mean versus raw.
4. The four changes remove one gate tile and one up tile and add two down tiles.
   The final MLP has **3 gate + 6 up + 10 down = 19 E0M3 tiles**. The unchanged
   raw background contributes 128, giving **147 total**. The 192-document
   development CE change is −0.001490 versus raw.
5. Freeze this new map before drawing **64 new documents**, excluding all prior
   calibration, confirmation, and published C4 documents. Require pooled CE+2SE
   below zero versus both raw and matched identity, and nonpositive math/code
   CE means versus raw. Only after that gate passes, evaluate PPL on the exact
   published WikiText/C4 token windows.

The independent confirmation (job 405692) passed:

| Comparison | Mean ΔCE | SE | Mean + 2SE |
|---|---:|---:|---:|
| Candidate − raw 256×64 | -0.001675449 | 0.000447776 | -0.000779897 |
| Candidate − matched identity | -0.001712369 | 0.000417071 | -0.000878227 |

Math/code mean ΔCE versus raw are −0.002173/−0.001178. KL also improves,
but remains diagnostic under this prospective CE-primary protocol. No failed
candidate was promoted or retested unchanged on another fresh draw.

| Llama-3.1-8B policy | E0M3 tiles | WikiText PPL | Δ vs FourOverSix | C4 PPL | Δ vs FourOverSix |
|---|---:|---:|---:|---:|---:|
| FourOverSix | 0 | 6.875525 | +0.000000 | 9.823733 | +0.000000 |
| Published MixFP4 8×64 | 3,345 | 6.849275 | -0.026249 | 9.773040 | -0.050694 |
| Supplied raw MixFP4 256×64 | 187 | 6.866879 | -0.008646 | 9.801361 | -0.022372 |
| **Refined both-axis 256×64** | 147 | 6.864886 | -0.010638 | 9.796946 | -0.026788 |

The new result improves on supplied raw 256×64 by **-0.001993 WikiText**
and **-0.004415 C4**, recovering **40.5%/52.8%** of the published 8×64
gain over FourOverSix. The original controls were reused; the raw deltas use
the user-supplied rounded values. Job 405707 verified identical published token
windows and source weights. This is a modest, independently confirmed quality
gain; it does not meet the historical 90% target.

[Fresh confirmation](results/task_reorder/transfer_20260920/renewed_llama/joint192_confirm/report.json),
[PPL report](results/task_reorder/transfer_20260920/renewed_llama/joint192_ppl/report.json),
and [joint search](results/task_reorder/transfer_20260920/renewed_llama/joint192/report.json)
preserve the evidence, including the preceding rejected candidates.

### GB200: remove standalone permutations through producer/consumer fusion

The successful native approach preserves the existing SM100 GEMM mainloop and
TMA epilogue. A direct scatter epilogue was bitwise correct but much slower
(job 405618), so it is not the recommended implementation.

**Columns:** the FourOverSix activation quantizer stores each original 16-value
group's packed 64-bit code word and E4M3 scale byte directly at its permuted
destination. The scale byte uses the actual CUTLASS SM100 SFA layout. Whole-group
permutation preserves tensor-wide absolute maximum and every within-group
calculation. The final pipeline reapplies the common global activation scale
through GEMM's alpha; the common maximum computation is outside timing.

**Rows:** keep GEMM's fast output order and make the next consumer read the
inverse permutation. For up, SiLU/multiply reads the matching gate and up
channels and emits down's column order. For down, residual-add reads the inverse
row map. The vector version processes two BF16 values per thread, using paired
loads when consecutive and scalar loads where needed. These operations replace
standalone restoration passes rather than adding more kernels. In a connected
MLP, apply down's column permutation exactly once: either the SiLU consumer
emits that order or the down quantizer applies it. These independent projection
prototypes are not an end-to-end MLP implementation.

The native quantizer's first independent Python comparison found a signed-zero
mismatch. After preserving negative zero, **all 22,528 tested BF16 values match
`quant_nvfp4_4over6` bit-for-bit**. Separate checks compare fused/unfused packed
codes and scale bytes. Every measured pipeline shape passes bitwise output
comparison and a deliberately wrong no-permutation negative control.

The final table (job 405810) times **FourOverSix producer + GEMM + consumer**,
using the exact compacted **Qwen** maps. All three variants use the same
quantizer, GEMM, and BF16 arithmetic. Values are microseconds, medians of five
alternating baseline/fused repetitions with 100 CUDA-graph iterations each.

| Projection | Tokens | No permutation µs | Separate passes µs | Fully fused µs | Added cost µs | Overhead |
|---|---:|---:|---:|---:|---:|---:|
| up | 1 | 20.78 | 26.35 | 21.16 | +0.38 | +1.8% |
| up | 128 | 28.49 | 43.49 | 29.63 | +1.14 | +4.0% |
| up | 512 | 69.93 | 111.54 | 72.99 | +3.06 | +4.4% |
| up | 2,048 | 240.62 | 417.37 | 253.00 | +12.38 | +5.1% |
| down | 1 | 28.97 | 32.62 | 29.71 | +0.75 | +2.6% |
| down | 128 | 48.33 | 54.76 | 49.24 | +0.91 | +1.9% |
| down | 512 | 117.14 | 132.00 | 118.90 | +1.76 | +1.5% |
| down | 2,048 | 421.93 | 465.68 | 427.89 | +5.96 | +1.4% |

This is a per-projection prototype, **not a full-model speedup claim**. It uses
synthetic weights and the mixed kernel's fixed E2M1 path to isolate permutation
cost; it does not execute the complete 212-tile mixed-format model or benchmark
Llama's native latency. Its FourOverSix producer is not claimed to be an optimal
quantizer, so its absolute cost affects percentage overhead. Common tensor-amax,
setup, weight preparation, and graph construction are excluded. Do not compare
these percentages directly with the earlier GEMM-only baseline.

The intermediate GEMM+consumer benchmark, which still includes input gathering,
reduces up's 2,048-token pipeline from 319.50 to 159.79 µs (142.21 µs baseline).
Producer fusion then removes that remaining gather. The isolated producer
experiment adds only about 0.02–0.03 µs at one token; small negative differences
at other shapes are timing variation, not a claimed quantization speedup.

[Final pipeline measurements](results/task_reorder/transfer_20260920/full_pipeline_405810/summary.json),
[independent quantizer check](results/task_reorder/transfer_20260920/quant_audit_405697/python_reference.json),
and [implementation notes](native/PERMUTATION_FUSION.md) document scope and code.
The renewed [job ledger](results/task_reorder/transfer_20260920/renewed_job_ledger.json)
includes failed builds and rejected candidates. All work used `gov113008`, with
Slurm workers for heavy compute and attached completion monitors; the measured
peak concurrency remained below the four-GPU limit.
