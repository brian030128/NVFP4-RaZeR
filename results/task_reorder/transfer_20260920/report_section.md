## Arranging weights for 256×64 MixFP4 tiles

The Blackwell follow-up keeps the two weight codebooks and the joint CE/KL
`k=3` rule, but makes each format decision cover **256 rows × 64 columns**.
A large tile often mixes weights that benefit from E0M3 with weights that are
hurt by it. Arranging groups compatible weights into the same legal tile.
The accepted Qwen result uses permutations only; it uses no rotation.

### How the arranging algorithm works

1. **Score small pieces using the task loss.** Starting from FourOverSix,
   compute the weight-gradient inner product with the E0M3-minus-E2M1 weight
   change. Store separate CE and teacher-KL scores for each **1-row × 16-column**
   atom on each of 128 original OpenWebMath/CodeParrot calibration documents.
   This is a first-order loss prediction, not an MSE ranking. Keeping scores
   per document preserves covariance when many atoms are summed into a tile.
2. **Fit a legal arrangement.** Use 64 source-balanced documents to group rows
   into sets of 256 and intact 16-column scale groups into sets of four.
   Alternate row and column assignments, using capacity-constrained linearized
   assignments and exact pair swaps. A smooth objective transitions toward the
   hard joint confidence-bound objective. The fixed search uses four starts,
   six rounds, 4,096 sampled swaps, two swap passes, seed 0, and 32 MiB scratch.
   Columns never split a 16-element quantization scale group.
3. **Elect formats independently.** Freeze the arrangement, then use the other
   64 source-balanced documents to recompute each tile's summed CE and KL
   scores. Choose E0M3 only when **both `mean + 3 SE < 0`**. Otherwise retain
   FourOverSix E2M1. Election data do not choose the permutation.
4. **Keep the scope small.** The accepted experiment arranges only the final
   MLP's gate, up, and down projections (Qwen layer 63). All other matrices keep
   their raw 256×64 map. The elected final-MLP counts are 0, 6, and 14 tiles,
   respectively; 192 tiles elsewhere give **212 E0M3 tiles total**.
5. **Compact without changing the selected weights.** Reassign active row
   groups and column bands to positions that retain as many original indices
   as possible; fill unused positions with identities. This step uses no loss
   data. It preserves the original-coordinate format map and was verified to
   preserve the actual quantized weights **bitwise**. The zero-tile gate
   projection becomes identity.

For an output-by-input weight matrix, write the arranged matrix as
`W′ = P W Qᵀ`. Its input is `X′ = X Qᵀ`, so
`X′ W′ᵀ = X Wᵀ Pᵀ`; restoring output order cancels `P`.
Weights can be arranged offline. Runtime work is the matching input permutation
and output restoration, potentially fused into activation production and the
GEMM epilogue. Whole 16-element codes/scale groups must move together, preserving
the tensor-wide activation quantizer decision. Independent gate/up permutations
also require consistent ordering before their elementwise MLP product.

### Measured Qwen3.8-27B results

All PPL rows below use the same released 2,048-token protocol as the main report.
Lower is better. Deltas are relative to FourOverSix.

| Policy | E0M3 tiles | WikiText-2 | ΔWiki | C4 | ΔC4 |
|---|---:|---:|---:|---:|---:|
| FourOverSix | 0 | 7.287076 | — | 10.188365 | — |
| Published MixFP4 8×64, k=3 | 3,785 | 7.214750 | −0.072327 | 10.149866 | −0.038499 |
| Raw 256×64, supplied reference | 198 | 7.275704 | −0.011373 | 10.177685 | −0.010680 |
| Raw 256×64, local reconstruction | 195 | 7.266300 | −0.020776 | 10.176030 | −0.012335 |
| Earlier row-only arrangement | 207 | 7.259481 | −0.027595 | 10.169099 | −0.019266 |
| **Accepted both-axis arrangement** | **212** | **7.255834** | **−0.031243** | **10.167336** | **−0.021029** |
| Independently confirmed compacted row candidate | 201 | 7.262465 | −0.024611 | 10.172032 | −0.016333 |

The accepted both-axis model beats the earlier row-only result on both datasets.
It recovers **43.2% of the 8×64 WikiText improvement and 54.6% of the C4
improvement** over FourOverSix. It does **not** reach the former 90% target;
the user accepted this endpoint. The supplied raw-198 and reconstructed raw-195
maps differ and are retained as separate controls.

The both-axis model has measured held-out PPL gains, but did not pass the later
strict fresh joint CE/KL confirmation procedure. The independently confirmed
row candidate is a separate model. Later extensions and rotations did not produce
a new validated PPL leader; in particular, the 218-tile Fisher extension failed
its frozen fresh CE-primary gate and was not evaluated for PPL. These failures
are not relabeled as successes.

### Exact compaction and deployment cost

| Final-MLP projection | Rows moved before → after | Columns moved before → after |
|---|---:|---:|
| Gate | 17,408 → 0 | 4,768 → 0 |
| Up | 17,407 → 1,921 | 5,072 → 288 |
| Down | 5,119 → 3,080 | 17,376 → 1,104 |
| **Total** | **39,934 → 5,001** | **27,216 → 1,392** |

Compaction reduces moved rows by 87.5% and moved columns by 94.9%, with identical
quantized weights and therefore reused parent PPL. These percentages measure
index movement, **not latency savings**. The user has verified that the raw
256×64 MixFP4 GEMM matches NVFP4 throughput; the added permutation cost is a
separate measurement. Llama transfer and native GB200 overhead use the frozen
[follow-up plan](results/task_reorder/transfer_20260920/plan.json).
The measured native costs are reported below; fused overhead remains unmeasured.

### Llama-3.1-8B transfer: fresh confirmation failed

We transferred the accepted search settings without tuning: last MLP (layer 31),
1×16 atoms, both axes, four starts, six rounds, and independent `k=3` election.
The regenerated calibration reproduced all historical adaptive/fixed maps exactly
and recovered the expected counts of 187 raw 256×64 and 3,345 fine 8×64 tiles.
The new arrangement elected **4 gate, 7 up, and 39 down tiles**, giving **178 total**
with the 128 unchanged tiles elsewhere. Compaction preserved all three actual
quantized matrices bitwise.

Before any PPL promotion, one frozen candidate was checked on 64 new, balanced
math/code documents (512 tokens each), excluding both models' calibration data,
six earlier Qwen fresh-data manifests, and published C4 evaluation documents.
The prospective gate required pooled CE `mean + 2 SE < 0` against both raw256
and matched identity, plus nonpositive math/code mean CE changes versus raw256.
KL was diagnostic. Matched identity used the same 64-document tile-election
split, yielding 161 total tiles, and therefore controls for the split as well
as the unchanged background.

| Fresh comparison | Documents | ΔCE | 2 SE | ΔCE + 2 SE | ΔKL |
|---|---:|---:|---:|---:|---:|
| Arranged − raw256, pooled | 64 | +0.005206 | 0.001771 | +0.006978 | +0.004720 |
| Arranged − matched identity, pooled | 64 | +0.005148 | 0.001795 | +0.006942 | +0.004581 |
| Arranged − raw256, math | 32 | +0.004309 | 0.002867 | +0.007176 | +0.003467 |
| Arranged − raw256, code | 32 | +0.006104 | 0.002079 | +0.008183 | +0.005972 |

**This transfer failed:** loss worsened on both sources, and also against matched
identity. Thus the failure cannot be explained solely by changing from 128 to
64 election documents. Successful Qwen arranging did not establish a model-general
method. No Llama arranging PPL gain is claimed; the failed candidate was **not
promoted to full WikiText/C4 PPL evaluation**, preserving the frozen gate and GPU budget.

| Llama policy | E0M3 tiles | WikiText-2 PPL | ΔWiki vs FourOverSix | C4 PPL | ΔC4 vs FourOverSix |
|---|---:|---:|---:|---:|---:|
| FourOverSix, published | 0 | 6.875525 | — | 9.823733 | — |
| MixFP4 8×64, k=3, published | 3,345 | 6.849275 | −0.026249 | 9.773040 | −0.050694 |
| Raw MixFP4 256×64, supplied reference | 187 | 6.866879 | −0.008645 | 9.801361 | −0.022372 |
| Both-axis 256×64 transfer | 178 | Not run: fresh gate failed | — | Not run: fresh gate failed | — |

The first three PPL rows reuse existing results. Matching regenerated tile counts
alone is not a bitwise audit of the supplied raw-256 mask, whose full artifact was
not provided. The new measurement here is the controlled fresh-loss comparison,
not another raw-PPL run. Sources: [confirmation report](results/task_reorder/transfer_20260920/llama_confirmation/report.json),
[frozen plan](results/task_reorder/transfer_20260920/llama_confirmation/plan.json), and
[fresh document manifest](results/task_reorder/transfer_20260920/llama_confirmation/fresh_manifest.json).

The PPL quality path restores quantized weights to original coordinates. It does
not measure rounding changes from native GEMM accumulation in permuted K order.

### Native GB200 permutation overhead

Jobs 405441 and 405458 ran on **NVIDIA GB200, SM100, driver 580.105.08**,
using the existing `../mixfp4/src/mixed_nvfp4_gemm_sm100.cu` kernel and the exact
compacted Qwen up/down permutations. Gate needs no permutation. We measured:

- A full out-of-place gather of packed FP4 activation codes and their existing
  16-element scales, followed by GEMM and full BF16 output restoration.
- The same input gather and GEMM, with two-pass **in-place restoration of only
  moved output channels**: first read all moved values into scratch, then write
  their destinations. Separating reads and writes prevents permutation-cycle
  races. Fixed output channels remain where GEMM wrote them.

The table reports graph-replay medians of five repetitions, 100 iterations each,
with baseline/pipeline order alternating. Setup, allocation, graph construction,
weight preparation, and activation quantization are excluded. Timings are per
projection, in microseconds; these are not full-model latencies.

| Projection | Tokens | GEMM only µs | Full input/output permutation + GEMM µs | Full overhead | Sparse-output pipeline µs | Sparse overhead |
|---|---:|---:|---:|---:|---:|---:|
| down | 1 | 21.15 | 27.57 | +30.3% | 29.47 | +39.5% |
| down | 16 | 20.81 | 26.49 | +27.3% | 28.28 | +36.0% |
| down | 128 | 21.11 | 28.19 | +33.5% | 30.03 | +42.2% |
| down | 512 | 22.45 | 36.65 | +63.3% | 38.56 | +73.2% |
| down | 2048 | 63.16 | 111.27 | +76.2% | 115.47 | +82.8% |
| up | 1 | 14.23 | 19.62 | +37.9% | 21.36 | +50.2% |
| up | 16 | 11.83 | 18.40 | +55.6% | 19.63 | +65.8% |
| up | 128 | 12.18 | 23.20 | +90.4% | 20.41 | +67.4% |
| up | 512 | 15.30 | 46.48 | +203.7% | 28.80 | +88.4% |
| up | 2048 | 57.33 | 174.27 | +204.0% | 92.17 | +60.9% |

Each overhead uses its own job's matched GEMM baseline; the displayed GEMM column
is from the full-permutation job. Sparse restoration helps the up projection at
larger token counts, reducing its 2,048-token pipeline from 174.27 to 92.17 µs.
It does not help the down projection: 3,080 of 5,120 output channels still move,
and two launches outweigh the saved traffic. At small token counts, launch cost
also makes sparse restoration slower. Use full restoration for down, and choose
up restoration according to workload; those initial jobs did not implement fusion; the follow-up below does.

**The added permutations are not free.** Compaction preserves quality but does
not by itself remove memory passes or launches. The measured per-projection
cost cannot be extrapolated to a whole-model slowdown: only the last MLP is
arranged, and no native end-to-end model benchmark was run.

Correctness checks used nonconstant FP4 codes/scales and verified their gather
byte-for-byte; both output restoration variants passed bitwise checks. A separate
[eight-second correctness job](results/task_reorder/transfer_20260920/gb200_check_405476/checks.txt)
used distinct output bit patterns at all 20 shape/variant combinations. Its
unchanged-output negative control failed for exactly every moved channel. This
strengthens the initial synthetic-GEMM check, whose output columns can coincide. The
existing GEMM also passed its small host-reference check. This overhead harness
uses the mixed kernel's fixed E2M1 format path, isolating permutation cost; it
**does not validate execution of the complete 212-tile mixed-format model**.
It adds no activation quantization or rotation. Those initial jobs make no fused-kernel latency claim.
Ordinary-launch timings, individual pass timings, all repetitions, and provenance
are retained in the [full-gather results](results/task_reorder/transfer_20260920/gb200_405441/SUMMARY.md)
and [sparse-output results](results/task_reorder/transfer_20260920/gb200_405458/SUMMARY.md).
The two native jobs used one GPU each for about 2 minutes per job.

Implementation: `quantize/task_reorder.py`, `run_task_reorder.py`,
`quantize/compact_rows.py`, and `quantize/compact_both.py`.
See [study status](REORDERING_STUDY_STATUS.md), the
[full comparison](results/task_reorder/cluster_20260919/FULL_COMPARISON.md), and
[measurement snapshots](results/task_reorder/cluster_20260919/published_evidence/INDEX.json)
for provenance and the unsuccessful follow-ups.

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
