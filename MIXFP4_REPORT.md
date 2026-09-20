# MixFP4: implementation, selection, reordering, and results

<!-- CURATED MIXFP4 SUMMARY: keep experiment history in supporting reports. -->

MixFP4 keeps NVFP4's 4-bit storage and 16-element scale groups, but chooses
E2M1 or E0M3 for each weight tile. Activations remain E2M1. Reordering groups
compatible weights into the larger 256×64 tiles used by our B200 kernel.

## 1. MixFP4 implementation on B200 and sm_120

| Path | Weight type granularity | Format selection |
|---|---|---|
| B200 / SM100 (`sm_100a` build) | 256×64 in the implemented kernel | Set the operand-format field of the `tcgen05.mma` descriptor; the same instruction handles E2M1 and E0M3. |
| SM120 (`sm_120a` build) | Hardware minimum 8×64 for operand B | Patch compiled `mma.sync` format bits to create E0M3 variants. Dispatch around a pipeline iteration to avoid issuing predicated-off tensor instructions. |

E2M1 magnitudes are `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`; E0M3 magnitudes are
`{0, 1, 2, 3, 4, 5, 6, 7}`. Both retain sign bits and FP8 block scaling.
E0M3 uses an undocumented hardware interface. The 256×64 geometry is our
chosen SM100 implementation, not a universal hardware minimum. GEMM launch
tiles and weight type tiles are distinct.

Our datacenter measurements use **GB200**, not a separately tested B200 system:

| Measurement | Result and scope |
|---|---|
| Uniform-format GEMM, 8192³ | Approximately zero format-selection overhead; not a heterogeneous model-map test. |
| Actual heterogeneous Llama GEMMs | Shape- and run-dependent; all nine shapes are in the linked timing tables. |
| Fused Qwen projection pipeline | **+1.4–5.1%** versus the same quantizer/GEMM/consumer without permutation; common activation amax excluded. |
| Full Llama, batch 1, prompt 128 + decode 32 | **915.747 ms arranged / 911.380 ms FourOverSix**; paired request overhead **+0.53% ±0.53% (2SE)**. |
| Full Llama, prompt 2048 | Overhead inconclusive: large timing variation across all policies. |
| SM120 GEMM, RTX 5090 | Sibling implementation reports **+1.0–4.9%** for its tested patterns/shapes. |

Full-model timing includes all layers, activation amax/quantization, attention,
and KV cache. It is an eager-backend **diagnostic**, with fused column permutation
and separate row restoration: operator checks pass, but mixed-policy full-output
equivalence fails and native PPL is unmeasured. Qwen native full-model latency
is also unmeasured.

Sources: [SM100 kernel](../mixfp4/src/mixed_nvfp4_gemm_sm100.cu),
[SM120 implementation](../mixfp4/docs/mixed_nvfp4_report.md),
[full timing tables](results/task_reorder/transfer_20260920/latency_scope_20260920/report_section.md),
[native correctness limitations](results/task_reorder/full_model_20260920/IMPLEMENTATION.md).

## 2. How to choose tile type

Start with FourOverSix E2M1 weights, `Q0`, and an E0M3 candidate, `Q1`, with
alpha fixed at 1. For each tile, `D = Q1 − Q0`. On math/code calibration
sequences, score the change at the **quantized model**:

```text
g_CE[i, tile] = ⟨gradient of next-token CE, D_tile⟩
g_KL[i, tile] = ⟨gradient of KL(BF16 teacher || quantized model), D_tile⟩

Choose E0M3 iff both:
    mean(g_CE) + 3 SE(g_CE) < 0
    mean(g_KL) + 3 SE(g_KL) < 0
Otherwise keep E2M1.
```

Negative scores predict lower loss. Scoring uses a straight-through derivative
for activation quantization. This is task-loss selection, not weight-MSE
selection; tile count follows from the rule rather than a fixed budget. The
bounds are a selection heuristic, not a multiple-testing guarantee. Combined
changes need exact finite-loss checks because gradients can miss quantization
effects.

## 3. How to decide reordering

1. **Score small units:** retain per-document CE/KL scores for each 1-row ×
   16-column atom. Sum within each document to preserve covariance for a
   proposed tile.
2. **Fit a legal arrangement:** group rows into sets of 256 and intact
   16-column scale groups into sets of four. Alternate capacity-constrained
   row/column assignments and pair swaps to improve joint score bounds.
   Never split a scale group. The original 128-document calibration uses
   64 documents for fitting and 64 for tile election after freezing the layout.
3. **Verify combined changes:** Llama needed exact final-MLP replay and format
   refinement among previously elected tiles on 192 development documents.
   Freeze the map, then require pooled CE mean+2SE < 0 versus raw and matched
   identity on **64 new documents**, with nonpositive math/code CE means versus
   raw. KL is diagnostic in this later CE-primary gate; original tile election
   still requires both objectives. Prior failed gates remain failures.
4. **Compact for deployment:** retain original indices for inactive rows/column
   groups wherever possible, preserving effective quantized weights bitwise.
   Pack weights offline; fuse runtime permutations into adjacent operations.

For `W′ = P W Qᵀ`, use `X′ = X Qᵀ`; then `X′W′ᵀ = XWᵀPᵀ`.
Restoring output order cancels the row permutation. Gate/up ordering must agree
before their elementwise product. This preserves the unquantized operation;
the new grouping changes which weights receive E0M3.

Accepted layouts change **only the final MLP's gate/up/down projections**;
other layers retain raw 256×64 MixFP4. Cached inputs make exact trials cheap
and the small scope limits runtime overhead. This is not proven globally
optimal. Qwen has 0/6/14 final-MLP E0M3 tiles; refined Llama has 3/6/10.
Neither uses rotation. [Algorithm details](results/task_reorder/transfer_20260920/report_section.md).

## 4. PPL: 8×64, raw 256×64, and 256×64 + MLP reordering

Lower is better. These are **fake-quantized W4A4 quality measurements**, not
native-kernel PPL. Evaluation uses 2,048-token WikiText-2 windows and 256 seed-0
C4 crops, tensor-wide activation factors, and the released aggregation protocol.
E0M3 tile counts differ in area across geometries.

| Model | Policy | E0M3 tiles | WikiText-2 | C4 |
|---|---|---:|---:|---:|
| Llama-3.1-8B | FourOverSix | 0 | 6.875525 | 9.823733 |
| | MixFP4 8×64, k=3 | 3,345 | 6.849275 | 9.773040 |
| | Raw MixFP4 256×64, supplied | 187 | 6.866879 | 9.801361 |
| | **256×64 + final-MLP reordering and refined selection** | **147** | **6.864886** | **9.796946** |
| Qwen3.8-27B | FourOverSix | 0 | 7.287076 | 10.188365 |
| | MixFP4 8×64, k=3 | 3,785 | 7.214750 | 10.149866 |
| | Raw MixFP4 256×64, supplied | 198 | 7.275704 | 10.177685 |
| | Raw MixFP4 256×64, local reconstruction | 195 | 7.266300 | 10.176030 |
| | **256×64 + final-MLP both-axis reordering** | **212** | **7.255834** | **10.167336** |

Llama improves supplied raw PPL by **0.001993 / 0.004415** and passes its fresh
CE gate. Its gain includes format-mask refinement, not permutation alone.
Qwen's accepted endpoint improves both PPLs but **failed the later strict fresh
joint CE/KL gate**; supplied and local raw maps are distinct controls.
Recovery of the 8×64 gain over FourOverSix is **40.5% / 52.8% for Llama** and
**43.2% / 54.6% for Qwen**. The historical 90% target was not achieved.

PPL gains do not establish answer-accuracy gains: Llama's separate non-STEM MMLU
and ARC-Challenge comparisons were inconclusive.
[Quality results and gates](results/task_reorder/transfer_20260920/report_section.md),
[answer-accuracy evaluation](results/task_reorder/llama_accuracy_20260920/REPORT.md).

## 5. Ablation study: KL only, CE only, and more reordered layers

### Tile-selection objective

Hold calibration, 8×64 geometry, and `k=3` fixed; drop one selection objective.
Tile counts are shown because a fixed threshold does **not** give equal budgets.

| Model | Objective | E0M3 tiles | WikiText-2 | C4 |
|---|---|---:|---:|---:|
| Llama-3.1-8B | CE + KL | 3,345 | 6.849275 | 9.773040 |
| | KL only | 32,774 | 7.356566 | 10.394302 |
| | CE only | 22,906 | **6.836686** | **9.768658** |
| Qwen3-4B | CE + KL | 7,912 | 11.862908 | **15.824034** |
| | KL only | 21,528 | 12.007304 | 16.192286 |
| | CE only | 125,611 | **11.804390** | 16.745670 |

KL-only loses to the joint rule on both corpora here. **CE-only wins both Llama
PPLs**, using 6.85× as many tiles, and trades better WikiText for worse C4 on
Qwen3-4B. Requiring both is a conservative rule, not a universal optimum.
Threshold sweeps also contain CE-only wins with fewer tiles; they do not prove
KL is always necessary. No equivalent objective ablation was run on Qwen3.8-27B.
[Full threshold/count and accuracy comparisons](MIXFP4_REPORT_DETAILS.md#the-conjunction-measured).

### Extending reordering to more layers

Matched Qwen study using eight-row groups and the same layers-56–63 background:

| Reordered scope | E0M3 tiles | WikiText-2 | C4 |
|---|---:|---:|---:|
| Final MLP only | 192 | **7.263466** | 10.177821 |
| Last eight MLPs | 201 | 7.263998 | **10.175365** |

Extension improves C4 but slightly worsens WikiText. This is a separate matched
study, not an extension of the accepted 212-tile model. Later wider candidates
failed applicable gates; the 218-tile Fisher extension failed fresh CE
confirmation and was not evaluated for PPL.

Llama has **no exhaustive earlier-layer layout search**. A diagnostic transfers
the final-layer layout to layers 0/15/31: predicted versus actual loss change
agrees in sign on 54.2%/54.2%/91.7% of fresh cases. Freezing downstream
activation-quantization residuals reduces early prediction-error magnitude by
about 10–12× but does not fix sign agreement. This explains a difficulty with
early-layer scoring; it does not establish that optimized earlier layouts
cannot work.
[Matched scope tables](results/task_reorder/cluster_20260919/FULL_COMPARISON.md#qwen-expanded-scope-layers-5663),
[Llama depth diagnosis](results/task_reorder/llama_diagnosis_20260920/REPORT.md).

The [archived detailed report](MIXFP4_REPORT_DETAILS.md) retains the full protocol,
threshold sweeps, diagnostics, timing tables, and experiment history.
