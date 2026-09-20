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
up restoration according to workload; a fused epilogue remains unimplemented.

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
It adds no activation quantization or rotation. No fused-kernel latency is claimed.
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
