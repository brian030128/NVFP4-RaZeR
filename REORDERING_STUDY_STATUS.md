# MixFP4 256×64 reordering: accepted study result

The user accepted the measured both-axis result on 2026-09-20. Further GPU experiments are stopped. The earlier 90% recovery target was not reached: the accepted result recovers **43.2% of the 8×64 WikiText gain and 54.6% of its C4 gain** over FourOverSix. This is an experimental endpoint, not a proof that reordering cannot do better.

## Measured Qwen3.8-27B comparison

Lower perplexity is better. All deltas below use FourOverSix as the reference.

| Policy | E0M3 tiles | WikiText PPL | ΔWiki | C4 PPL | ΔC4 |
|---|---:|---:|---:|---:|---:|
| FourOverSix | 0 | 7.287076 | — | 10.188365 | — |
| Published MixFP4 8×64 | 3,785 | 7.214750 | −0.072327 | 10.149866 | −0.038499 |
| User-provided raw 256×64 | 198 | 7.275704 | −0.011373 | 10.177685 | −0.010680 |
| Locally reproduced raw 256×64 | 195 | 7.266300 | −0.020776 | 10.176030 | −0.012335 |
| Earlier row-only reordering | 207 | 7.259481 | −0.027595 | 10.169099 | −0.019266 |
| **Accepted both-axis reordering** | **212** | **7.255834** | **−0.031243** | **10.167336** | **−0.021029** |
| Separately confirmed compacted rows | 201 | 7.262465 | −0.024611 | 10.172032 | −0.016333 |
| Earlier 90% recovery target | — | ≤7.221982 | — | ≤10.153716 | — |

The supplied and locally reproduced raw maps differ and are reported separately. Both-axis reordering beats row-only in the measured comparison. The separately confirmed row result has stronger independent-validation evidence but does not beat the historical best PPL.

The [full comparison](results/task_reorder/cluster_20260919/FULL_COMPARISON.md) includes all prior policies and Llama references. [Measurement snapshots](results/task_reorder/cluster_20260919/published_evidence/README.md) preserve the supporting reports and source hashes.

## How the gain is obtained

A 256×64 tile ties together 32 former 8×64 tiles. Rows that benefit from E0M3 can therefore be grouped with rows that lose from it, causing the whole coarse tile to fail selection.

The search groups channels according to their **task-loss effects**. For each calibration document and each 1-row × 16-column atom, it computes the directional effect of replacing FourOverSix E2M1 weights with E0M3 weights, for both next-token cross-entropy (CE) and KL to the BF16 teacher. Summing these document-level scores inside a proposed tile retains covariance between its atoms.

The search alternates assignments of rows into groups of 256 and intact 16-column scale groups into groups of four. Capacity-constrained assignments and pair swaps improve a smooth approximation before the final hard objective. E0M3 election requires both CE and KL directional means plus three standard errors to be negative. Layout fitting and election use separate, source-balanced halves of the original 128 math/code calibration documents: 64 for fitting and 64 for election. Actual quantized-model PPL is evaluated afterward; more elected tiles or a better surrogate is not itself a quality result.

The winning experiment changes only the final MLP in layer 63: **six E0M3 tiles in `up_proj`, fourteen in `down_proj`, and none in `gate_proj`**. The remaining model retains 192 raw-256 E0M3 tiles, giving 212 total. The accepted result uses no rotation, extra correction GEMM, changed codebook, or larger scale groups.

For a row permutation P and column permutation Q, the algebra is:

```text
W' = P W Qᵀ
X' = X Qᵀ
X' W'ᵀ = X Wᵀ Pᵀ
```

Thus the matching activation permutation cancels the column permutation, and the output is restored by the inverse row permutation. In deployment, weights are arranged offline; activation reordering can be integrated into quantization/stores, and output restoration can be fused into the GEMM epilogue. Preserve quantized codes, scales, and the original tensor-wide activation-factor decision when permuting them. The gain comes from changing which weights share a format decision while retaining this equivalence for the underlying unquantized linear operation.

## Reducing permutation work without changing quality

After selecting the type map, exact compaction reassigns active groups to preserve as many original positions as possible. It keeps every original-coordinate format choice and the actual dequantized weights bitwise identical.

| Projection | Rows moved before | Rows moved after | Columns moved before | Columns moved after |
|---|---:|---:|---:|---:|
| Gate | 17,408 | 0 | 4,768 | 0 |
| Up | 17,407 | 1,921 | 5,072 | 288 |
| Down | 5,119 | 3,080 | 17,376 | 1,104 |
| **Total** | **39,934** | **5,001** | **27,216** | **1,392** |

That is 87.5% fewer moved rows and 94.9% fewer moved columns. These are movement reductions, not latency reductions. The user verified equal GEMM speed for raw 256×64 MixFP4 and NVFP4; native B200 permutation and full-MLP overhead remain unmeasured.

## What later experiments established

| Experiment | Outcome |
|---|---|
| Wider row packing and raw-context individual-row scores | New candidates failed their applicable development or fresh gates; no new best PPL |
| H16 rotations in E0-active column bands | No supported improvement over the accepted parent |
| H16 rotations only on existing E0 tiles | All three projection scopes worsened CE with matched numerical controls |
| Twenty individual E0 tile rotations | No eligible subset; no further evaluation |
| Eight-tile Fisher-aware row extension | Failed the incremental development gate |
| Six-tile subset selected using cross-matrix curvature | Passed development, then failed fresh confirmation |

The final six-tile extension would have produced 218 total tiles. On 64 new confirmation documents, its CE difference versus raw256 was −0.00164422 with SE 0.00110921, giving a **two-SE upper bound of +0.00057421**. Its math and code means improved, but the preset pooled gate failed. It received **no PPL run** and is not the accepted result.

A known-good 8×64 positive control had failed the earlier strict CE/KL confirmation rule. New branches therefore used a prospectively declared CE-primary rule: pooled CE mean + 2SE < 0 against raw and matched identity, with nonpositive domain CE means; KL remained diagnostic. Earlier failures were not relabeled. Every fresh set excluded original calibration, development, previous confirmation documents, and published C4 evaluation documents.

The accepted both-axis result is supported by measured held-out PPL. It is not claimed to have passed the later independent CE/KL confirmation gate; exact compaction preserves its existing quality and trade-offs rather than adding new validation evidence.

No GPUs are allocated at this endpoint. Future work is limited to at most four concurrent GPUs and must use monitored H200 Slurm jobs for heavy computation.
