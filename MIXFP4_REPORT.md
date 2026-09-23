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

Lower is better. The table below is **fake-quantized W4A4**; §4a confirms the
Llama rows on the native SM100 kernel. Evaluation uses 2,048-token WikiText-2
windows and 256 seed-0 C4 crops, tensor-wide activation factors, and the released
aggregation protocol. E0M3 tile counts differ in area across geometries.

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

### 4a. The same numbers on the native kernel

Quality above came from the simulator and the timings of §1 from the kernel, so
no single artifact had shown both. Job 416794 ran the complete native SM100 model
over the same 397 published windows, loading the library whose digest matches the
passed kernel gate 406633, with every one of the 224 matrices packed and decoded
bitwise against its simulator weights first.

| Policy | WikiText-2 native / simulated | C4 native / simulated |
|---|---|---|
| FourOverSix | 6.878384 / 6.875525 | 9.826777 / 9.823733 |
| Refined 147-tile | 6.862185 / 6.864886 | 9.793517 / 9.796946 |

**Native and simulated agree.** Paired per window, on token windows whose
`token_sha256` were checked identical between the two runs, the refined map's
native-minus-simulated NLL is **−0.000394 ± 0.001570** on WikiText and
**−0.000350 ± 0.001258** on C4, i.e. within noise at t = −0.50 and −0.56. The
simulator is a faithful proxy for the kernel on perplexity, and the ±0.003
aggregate differences are not evidence of anything.

**The improvement is significant on the kernel.** Paired within the native run,
the refined map beats FourOverSix by **−0.002358 ± 0.001598 NLL (t = −2.95)** on
WikiText and **−0.003391 ± 0.001600 (t = −4.24)** on C4, which is −0.016199 and
−0.033259 in perplexity.

This also settles the full-output gate recorded as failed in the native
implementation notes. That gate compares logits, where 0–4 BF16 differences per
projection are amplified by later FP4 activation quantization into a ~10.8%
relative gap; two correct implementations rounding in different orders diverge
exactly that way. Perplexity is the metric that matters and it agrees. One model,
one seed; Qwen native perplexity is still unmeasured.
[Native run](results/task_reorder/native_ppl_20260921/run_416794/report.json),
[native versus simulated](results/task_reorder/native_ppl_20260921/native_vs_simulated.json),
[native gain](results/task_reorder/native_ppl_20260921/native_gain_paired.json).

## 5. Multi-round KL-only election

One-shot election (§2) scores every tile once, at FourOverSix, and elects all
tiles whose bound is negative. Summing thousands of first-order scores
overstates the combined effect of a step by 5–50×, which is why low thresholds
and fine tiles collapse. Multi-round election works like training with a line
search:

1. **Score.** At the *current* quantized model, compute per-sequence gradients of
   KL(BF16 teacher ‖ quantized) on the 128 calibration sequences, and the
   directional score of every legal flip (E2M1→E0M3, or undo). The activation
   and straight-through conventions are those of §2.
2. **Rank.** Candidates are flips with mean + 2 SE < 0; the 2 SE filter is fixed,
   not tuned.
3. **Backtrack.** Apply the top n, n/2, n/4, … candidates. Accept the first step
   that lowers mean KL on 192 held-out math/code development documents. These
   are the three recorded confirmation sets per model, disjoint from calibration.
4. **Repeat** until no step lowers development KL. WikiText-2 and C4 are
   evaluated once, on the final map; no choice looks at them.

The output is an ordinary per-tile E2M1/E0M3 map, so it has no runtime cost. The
procedure is deterministic: an independent Llama 256×64 re-run reproduced the
map and every evaluation NLL bitwise.

### Results (W4A4 fake quantization, released protocol)

| Model | Policy | E0M3 tiles | WikiText-2 | C4 | paired ΔNLL vs FourOverSix ± 2SE (wiki / c4) |
|---|---|---:|---:|---:|---|
| Llama-3.1-8B | FourOverSix | 0 | 6.875525 | 9.823733 | — |
| | One-shot 256×64, k=3 (§4) | 187 | 6.866879 | 9.801361 | −0.00126±0.00155 / −0.00228±0.00145 |
| | One-shot 8×64, k=3 (§4) | 3,345 | 6.849275 | 9.773040 | — |
| | **Multi-round KL, 256×64** | 8,405 | **6.841998** | **9.774137** | −0.00489±0.00185 / −0.00506±0.00220 |
| | **Multi-round KL, 8×64** | 3,654 | **6.819751** | **9.750492** | −0.00815±0.00173 / −0.00748±0.00240 |
| Qwen3.8-27B | FourOverSix | 0 | 7.287076 | 10.188365 | — |
| | One-shot 256×64, k=3 (local) | 195 | 7.266300 | 10.176030 | — |
| | One-shot 8×64, k=3 (§4) | 3,785 | 7.214750 | 10.149866 | — |
| | **Multi-round KL, 256×64**¹ | 39,099 | **7.246839** | **10.157245** | −0.00554±0.00343 / −0.00306±0.00092 |
| | **Multi-round KL, 8×64** | *running* | — | — | — |

¹ Stopped by request after round 5 of the tail, when rounds accepted 1–17 flips
each (dev KL 0.04625 → 0.04289).

- **Llama:** at 256×64, multi-round KL gains 3.9× (WikiText) and 2.2× (C4) the
  one-shot k=3 map, and edges 8×64 k=3 on WikiText. At 8×64 it reaches
  −0.0558 / −0.0732. That exceeds the non-deployable 1×16 MSE-selected reference
  (−0.0418 / −0.0613), so the MSE per-block choice is not a ceiling for
  task-aware selection.
- **Qwen:** at 256×64, multi-round KL gains about 2× the one-shot k=3 map but
  stays below one-shot 8×64 k=3. On Qwen, the one-shot threshold results in
  `results/mixfp4_potential/MULTIROUND.md` show that much larger elections keep
  improving PPL, which the KL acceptance test does not reach.
- **Generalization (Llama, five unseen domains).** Fresh math and code, PG-19,
  arXiv and GovReport, with maps frozen before any of these documents were read.
  Multi-round KL has the lowest teacher KL of every map on all five domains.
  Its CE ties the CE+KL and CE-only variants.
- **One-shot KL was not the problem objective.** One-shot KL-only at 8×64 (§6) is
  catastrophic; the same objective with re-scoring and backtracking is the best
  Llama policy measured. Earlier objective ablations were one-shot artifacts.

### Calibration time and memory

Measured by the runs themselves: wall time per phase, `torch.cuda` peak memory
and process peak RSS. The Qwen 256×64 CPU figure is Slurm MaxRSS.

| Run | Hardware | Scoring passes | Dev evaluations | Setup | Optimization | Final PPL eval | Peak GPU memory | Peak CPU memory |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Llama 256×64 | 1× H200 | 5 | 42 | 2.0 min | 43.9 min | 4.7 min | 46.1 GiB (48.9 reserved) | 41.2 GiB |
| Llama 8×64 | 1× H200 | 10 | 133 | 2.3 min | 2 h 15 min | 4.8 min | 47.3 GiB (50.6 reserved) | 41.3 GiB |
| Qwen 256×64 (to round 5) | 2× H200 | 6 | 63 | ≤ 37 min³ | 3 h 58 min | 16.8 min² | not logged⁴ | 78.1 GiB |
| Qwen 8×64 | 2× H200 | *running* | | | | | | |

² Separate 1-GPU evaluation job of the saved map (whole job, including model load).
³ Job wall time 4 h 35 min minus the logged optimization rounds. This is an upper bound: it also
  includes the unfinished round 6 that was cancelled.
⁴ Started before resource logging was added; the Qwen 8×64 run logs it, and its footprint (model,
  candidates and teachers) is the same apart from the small per-tile score arrays.

- **Where the time goes.** One scoring pass (128 × 512 tokens, forward plus CE and
  KL backward) takes about 2 min on Llama and 8 min on Qwen. One development
  evaluation (192 × 512 tokens, forward only) takes about 0.6 min on Llama and
  3 min on Qwen.
- **Backtracking dominates, mostly in the tail.** Rounds that accept a handful
  of flips each cost many evaluations. Round 0 alone takes 3.5 min (Llama
  256×64) or 11 min (Qwen 256×64) and delivers most of the development-KL gain.
- **Memory.** GPU memory is the BF16 model plus the FourOverSix and E0M3
  candidate weights plus 512-token activations. CPU memory is dominated by the
  BF16 teacher log-probabilities cached for the calibration and development
  documents.
- **All of this is one-time and offline.** The deployed artifact is a tile map.

## 6. Ablation study: KL only, CE only, and more reordered layers

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

## 7. Tried and failed

Every entry below was measured and rejected. Links point to the retained report
for each. Two caveats on reading them. First, **regime matters**: rows marked
8×64 W4A16 come from the earlier `mix_4_6` rounds and do not automatically carry
to the deployed 256×64 W4A4 geometry, and vice versa. Second, **"failed" is
relative to a stated reference** — several entries improve on FourOverSix while
losing to a better alternative, and the reference is named in each case.

### 7.1 Rearranging weights between tiles

The largest single line of failed work. Permutation preserves the multiset of
per-atom scores and only rearranges them into rectangles, and the objective used
to choose the rearrangement is not additive under that operation.

| What was tried | Result | Report |
|---|---|---|
| Both-axis 256×64 transfer to Llama | Fresh CE **+0.005206** vs raw256, also worse vs matched identity; not promoted to PPL | [llama_confirmation](results/task_reorder/transfer_20260920/llama_confirmation/report.json) |
| Gate/up-only candidate | Improved mean fresh CE, failed its frozen gate | [gate_up_confirm](results/task_reorder/transfer_20260920/renewed_llama/gate_up_confirm/report.json) |
| Eight-down-tile refinement | Improved mean fresh CE, failed its frozen gate | [tile_refine_confirm](results/task_reorder/transfer_20260920/renewed_llama/tile_refine_confirm/report.json) |
| KL-grouped spectral co-clustering, refinement deleted | Development CE **+0.007347 ± 0.000556 (t = +13.2)** vs raw256, **+0.007132** vs matched identity, on 192 documents with 192/192 bitwise suffix audits | [development192](results/task_reorder/step2_20260921/development192_report.json) |
| Rows-only arrangement | Development CE **+0.006919 ± 0.000587 (t = +11.8)** vs raw256 | [rows_only192](results/task_reorder/step2_20260921/rows_only192_report.json) |
| Fisher / logit-Gauss-Newton row grouping | Failed fresh CE confirmation; the 218-tile extension was never evaluated for PPL | [fisher_validate](results/task_reorder/cluster_20260919/published_evidence/fisher_validate/report.json), [fisher_subset_validate_v2_confirm](results/task_reorder/cluster_20260919/published_evidence/fisher_subset_validate_v2_confirm/report.json) |
| Individual-row scoring in full raw256 context | Failed its development gate | [raw_context_fisher](results/task_reorder/cluster_20260919/published_evidence/raw_context_fisher/report.json) |
| Preserving inactive raw maps, last-MLP rows | CE improved on 64 new windows but teacher KL did not (**+0.0000263** mean+1SE); math KL positive | [preserved_row_confirmation](results/task_reorder/cluster_20260919/preserved_row_confirmation_summary.json) |
| Wider matrix scope (24 matrices, layers 56–63) | **17 of 24 matrices elected zero tiles** on held-out data; median election/fit objective ratio **0** | [fine_rows_v2_diagnosis](results/task_reorder/cluster_20260919/fine_rows_v2_diagnosis.json) |
| Extending reordering past the final MLP | Trade-off, not a win: WikiText 7.263466 → 7.263998, C4 10.177821 → 10.175365 | [matched scope tables](results/task_reorder/cluster_20260919/FULL_COMPARISON.md#qwen-expanded-scope-layers-5663) |
| Row permutation at 8×64 (`_perm`, W4A16) | `perm_h1.5` **−0.0086** vs plain `h1.5` **−0.0111** — permutation costs ~+0.0025 | [decide_r2](results/decide_r2/REPORT.md), [decide_r3](results/decide_r3/REPORT.md) |

**The column axis specifically is not worth its cost.** Decomposing the deployed
256×64 arrangement by axis on Llama `down_proj`, against a free-assignment
ceiling of 27904 over an identity floor of 925: rows capture **40.7%** of that
headroom, columns **0.19%**, and both axes together **35.4%** — so adding the
column axis makes the result *worse* while raising the fit objective. The column
permutation is also the half that needs the activation gather fused into the
producer. [Capacity bracket](results/task_reorder/capacity_20260921/down_proj_capacity.json),
[axis and objective summary](results/task_reorder/step1_20260921/summary.json),
[analysis](REORDER_OBJECTIVE_REDESIGN.md).

### 7.2 Rotation

| What was tried | Result | Report |
|---|---|---|
| Unconditional Hadamard, 8×64 W4A16 | `_rot` **+0.0942** WikiText / **+0.1250** C4; `_rotcol` **+0.0946** / **+0.1431** | [decide_r3](results/decide_r3/REPORT.md) |
| Fixed selective H16 column bands | No incremental CE winner across three scopes | [cached_rotation_ablation](results/task_reorder/cluster_20260919/published_evidence/cached_rotation_ablation/report.json), [cached_rotation_panel](results/task_reorder/cluster_20260919/published_evidence/cached_rotation_panel/report.json) |
| Rotating the tiles that elected **E0M3** | Worse than the same arrangement unrotated: `up` **+0.000343**, `down` **+0.000232**, `both` **+0.000600** | [cached_tile_rotation_precise](results/task_reorder/cluster_20260919/published_evidence/cached_tile_rotation_precise/report.json), [paired arms](results/task_reorder/rotation_pairing_20260921/summary.json) |
| Rotating the tiles that stayed **E2M1** | Also worse: **+0.000776**, **+0.000466**, **+0.000197** | same |
| 20 individual E0M3 tile rotations | Zero eligible tiles; branch closed | recorded in [cached_tile_rotation_precise](results/task_reorder/cluster_20260919/published_evidence/cached_tile_rotation_precise/report.json) (`selected: null`, `passed_calibration_check: false`) |

E0M3 is uniform with scale `block_max / 7` and no coarse top codes, so it should
in principle gain more from outlier suppression than log-spaced E2M1 — and it
does cost about half as much on `up` and `down`. The asymmetry reverses on
`both`, and every arm is harmful, so it is moot. Selective rotation also flags
`reference_only_expanded_k`: rotation lives on K and is shared with the
activation across all rows, so rotating only some tiles requires duplicating K.

### 7.3 Selection objectives

| What was tried | Result | Report |
|---|---|---|
| Clipping the block scale (`alpha < 1`) | Harmful on WikiText, e.g. `clipe2_m2_8x64` **+0.0070**; C4 moves the other way, so it is not a clean win anywhere | [decide_r1](results/decide_r1/REPORT.md) |
| MAE and L*p* selection losses | Indistinguishable from MSE: `mae_m2_8x64` **−0.0016** WikiText / **+0.0008** C4, `l1.5_m2_32x128` **+0.0011** / **−0.0020** — the squared-error criterion is not what needs fixing | [decide_r1](results/decide_r1/REPORT.md) |
| Coherent-error objective `corr<r>` | Near no-op: the coherent and incoherent terms are measured equal, ratio **0.998–1.005** for every grid and clip preset | [analyze_coherent_error.py](analyze_coherent_error.py) |
| Calibration-free proxies for `diag(S)` | Preceding RMSNorm `gamma²` correlates **+0.63** on q/k/v but **−0.50** on gate/up; weight column energy has no consistent sign | [quantize/importance.py](quantize/importance.py), [measured importance](results/mix_4_6_sweep/importance_llama-2-7b.pt) |
| Regularizing the arrangement search | Deleting the hinge refinement raises fit-over-null **3.46 → 28.66** and KL grouping recovers the tile count, yet both candidates still fail finite loss (7.1) | [objective ablation](results/task_reorder/objective_ablation_20260921/summary.json), [step 1 summary](results/task_reorder/step1_20260921/summary.json) |

A sign-flip placebo search reaches **93%** of the deployed fit objective on Llama
`gate_proj`, so the fit objective alone is not evidence that a layout is usable.

### 7.4 Scale and format variants

| What was tried | Result |
|---|---|
| E2M1 headroom family (`head`, `headx`, `alpha > 1`) | **Removed from `CLIP_PRESETS`.** Not universally safe: on Qwen3-4B plain NVFP4 scores 13.6584/16.8723 while FourOverSix scores 14.0407/17.0153, i.e. **+0.38 WikiText worse**. `alpha > 1` discards the sparse top codes that absorb a block's outlier. |
| E0M3 headroom (`heade0`, `heade0x`) | **Removed.** It entangles the element-type decision with a second scale search on the E0M3 branch; `test_no_e0m3_headroom` enforces the removal. |
| Type blocks coarser than 1×16, MSE-selected | Every shape coarser than one scale block lost to plain NVFP4 in the original sweep. Superseded by the task-loss selection of §2, which is what makes 256×64 viable. See [mixfp4_sweep](results/mixfp4_sweep/REPORT.md). |

Both preset removals are recorded in the `CLIP_PRESETS` comments in
[quantize/quantizer.py](quantize/quantizer.py).

### 7.5 What this leaves

Electing a tile's format **in place** is the one mechanism with a clean record:
it only ever sums the scores of atoms already in that tile and never assumes
anything about moving them. That is the raw 256×64 map of §4 and the refined
Llama map that passed its fresh gate. Rearranging those atoms, rotating them, or
changing the tile geometry has not paid.

The [archived detailed report](MIXFP4_REPORT_DETAILS.md) retains the full protocol,
threshold sweeps, diagnostics, timing tables, and experiment history.
