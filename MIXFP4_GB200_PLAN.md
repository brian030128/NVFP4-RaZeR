# MixFP4 on GB200: weight reordering and rotation plan

Proposed experiments, 2026-09-18. This plan starts from the **root
[MIXFP4_REPORT.md](MIXFP4_REPORT.md)**, especially its CE+KL selection rule and
channel concentration analysis. It contains no new accuracy or speed measurements.

The recommendation is to **cluster channels with compatible task-loss effects,
then test selective rotations in that layout, and re-elect the weight formats at
256x64**. Optimize the accuracy attainable by the large-tile kernel. More E0M3
tiles, lower weight MSE, and lower latency are separate outcomes.

## 1. Fix the deployment contract

Use the requested `m256n256k64` kernel target. For `Y = X W^T`, with activations
in A and weights in B, this means a **weight type tile of N256 x K64**. M256 is
the token/batch axis; it is not a third dimension of the weight format map.

| Property | Existing report | GB200 target |
|---|---|---|
| Weight type tile | 8x64 | 256x64 |
| Weights sharing one format | 512 | 16,384 |
| Independent 16-element scale blocks inside it | 32 | 1,024 |
| Weight candidates | Canonical FourOverSix E2M1 / E0M3 alpha=1 | Same, rebuilt in the deployed basis |
| Activation element type | Fixed E2M1 | Fixed E2M1 |

A new tile joins **32 old type tiles along N; K stays at 64**. Promoting a whole
tile because it contains one old elected tile changes the other weights too;
neither an OR nor a majority
vote over the old map is the new selection rule. Keep scale groups at 16 and
UE4M3 block scales. A 64-wide type tile still contains four separate scale groups
per row; it must not become a 64-wide scale group.

Here, “weight-only” means **only weights get mixed element types**. The primary
experiment remains W4A4 with the report's fixed activation quantizer and scale
policy. W4A16 is a diagnostic for separating weight effects from activation
quantization effects. An activation permutation or rotation needed to preserve a
GEMM is allowed, but its numerical effects and runtime cost must be counted.

Treat 256x64 as the target kernel contract, rather than claiming it is the
universal minimum for every Blackwell instruction. NVIDIA documents shapes and
types through the `tcgen05` instruction descriptor; the public PTX documentation
does not expose E0M3 by that name. A first-hand SM100 experiment reports mixed
E0M3/E2M1 execution, but our exact kernel still needs verification.
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor),
[SM100 experiment](https://forums.developer.nvidia.com/t/does-blackwell-support-int4-native/326513/14).

Before a full port, verify B-only format switching with known packed values and
scales, alternating B types across K iterations, FP32 accumulation, and partial
tiles. Record whether 256 is an instruction dimension or a composed kernel tile,
and at which level the format bit actually applies. The experiments below retain
the requested 256x64 contract either way.

## 2. Use what the report actually establishes

The k=3 maps select only 0.0245%, 0.1115%, and 0.0080% of old tiles on Llama-3.1-8B,
Qwen3-4B, and Qwen3.8-27B. The benefit is sparse, but its location has structure:

- `down_proj` holds 45.2% and 41.8% of the elected tiles on Llama and Qwen3-4B.
- Roughly half of each model's elected tiles occur in its final depth quarter.
- Qwen3-4B residual channels 0–63 recur in 146 of 169 residual-reading matrices.
  Llama has stronger concentration inside individual matrices, including
  `L31.down_proj` input channels 12672–12735.

The corrected geometry makes **grouping compatible output rows** the first
question: the new constraint is entirely along N. K regrouping can still improve
which four scale groups share a tile, and rotation can change row preferences.
Use the last quarter's MLPs as the first pilot. These observations do not prove
that favorable tiles can be packed into legal 256x64 rectangles: every
permutation must apply to a complete
channel axis, and the many currently unelected weights also contribute.

The existing code supplies useful machinery, but its old objective is insufficient.
`quantize/reorder.py` searches balanced groups using reconstruction-error gains.
The report selects with task gradients. Historical experiments summarized in
`CLAUDE.md` also found that simple row sorting and unconditional Hadamard rotation
could hurt perplexity despite attractive weight-error statistics. Reuse the
search mechanics, not those selection conclusions.

## 3. First measure the loss from coarsening

On Slurm workers, reproduce the 8x64 baseline and construct identity-layout
256x64 scores from **all** fine-tile scores, not just the elected subset. For
each calibration sequence i and coarse tile T:

```text
g_CE[i,T] = sum(j inside T) g_CE[i,j]
g_KL[i,T] = sum(j inside T) g_KL[i,j]
U[T] = max(mean(g_CE[:,T]) + 3 SE(g_CE[:,T]),
           mean(g_KL[:,T]) + 3 SE(g_KL[:,T]))
elect E0M3 iff U[T] < 0
```

**Sum within each sequence before computing SE.** Fine-tile scores are correlated;
their individual means and SEs alone cannot recover the coarse SE. This summation
is exact for the directional score when the candidates and quantized baseline
are unchanged. It is not an exact prediction of the finite switch's loss.

Measure identity layouts at 8x64, 32x64, 64x64, 128x64, and 256x64 to locate where
row coarsening loses the benefit. Keep K fixed at 64. The intermediate shapes
are diagnostics, not deployment claims. Evaluate the complete elected 256x64
map with forward passes to detect
interactions. Save CE/KL margins, cancellation between favorable and unfavorable
contributions, selected weight fraction, and actual validation losses.

Existing per-sequence shards can support this first experiment if available.
`run_kse_paper.py` notes that some original score shards were removed; regenerate
missing scores on workers and check the original 8x64 election before coarsening.

## 4. Reorder within the model's legal channel symmetries

For input permutation P and output permutation S, define
`X' = X P`, `W' = S^T W P`. Then `X' W'^T = Y S`. The model must consistently
propagate or undo S. Permuting packed weights alone changes the function.

Start with whole 16-channel K groups so existing scale groups survive. A target
tile groups 256 output rows and four such K groups. Initially, regroup whole
8-row bands into sets of 32 using the existing per-sequence 8x64 scores. Refine
to individual rows and 16-column groups only after collecting scores at that
resolution. **Moving intact 64-column bands alone cannot improve selection at
K64**: it only relocates existing tiles. Useful K search must regroup 16-column
scale groups or change the basis; do not spend a sweep testing mere K64 moves.

| Location | Legal, deployable transformation | Constraint |
|---|---|---|
| MLP intermediate channels | One shared permutation of `gate_proj`/`up_proj` output rows and `down_proj` input columns | Optimize all three matrices jointly; permute biases too |
| Attention value channels | Permute within V heads and matching `o_proj` columns | Respect head boundaries and GQA replication; KV representation must agree |
| Residual stream | One global permutation propagated through embeddings, norms, residual writes/reads, and output head | Shared across layers; no independent order for every q/k/v/gate/up matrix |
| Arbitrary matrix rows or columns | Explicit gather/scatter or kernel layout support | Diagnostic until its full runtime cost is measured |

Prioritize the MLP transformation: it targets the largest reported contribution
and can be folded offline. It groups `gate/up_proj` rows and `down_proj` columns;
it does **not** independently reorder `down_proj` output rows. Grouping those
rows requires a common residual permutation or an output permutation undone by
the kernel epilogue, whose store cost must be measured. Pilot that epilogue option
if the unconstrained row-grouping diagnostic shows substantial headroom.
Next test legal V/O permutations. Test a global residual permutation separately
because its cross-layer coupling makes it a
larger intervention. Defer arbitrary Q/K reordering across RoPE coordinates and
hybrid attention/state channels until their invariances are implemented explicitly.

Replace the old MSE grouping objective with a task-score objective. For example,
sum the positive margins `max(0, -U[T])` over tiles, using CE/KL normalized by
fixed positive baseline loss scales for ranking moves. Positive normalization
does not change the conjunction's eligibility test. Evaluate proposed changes
jointly over every matrix sharing the permuted axis, using balanced clustering
and pair swaps from `quantize/reorder.py`.

A useful inexpensive search stage uses fit-sequence means; compute exact
sequence-level bounds for shortlisted layouts. Include identity, seeded random
legal permutations, and shuffled-score search controls. Row/column preferences
must be vectors across the opposite axis, not one scalar E0M3 count per row.

Score transport through a permutation is valid only when the quantized baseline
and its scale groups are preserved throughout the affected graph. A permutation
that changes another matrix's K grouping changes its baseline; rebuild its
candidates and recollect gradients. Revalidate every final transformed model.

## 5. Rotate selectively, then regroup the resulting preferences

Test normalized signed Hadamard blocks of width **16 and 64**, plus identity.
Width 64 aligns with the deployment tile; width 16 preserves scale-group
boundaries. Keep width 256 as a later ablation that spans four K64 tiles and
requires coordinated activation transformation; it is not the target type width.
Start with four fixed sign seeds on pilot modules,
using the same candidate budget for both pilot models.

For an orthogonal input transform R, `X' = X R` and `W' = W R` preserve
`X' W'^T = X W^T` before quantization. R must be shared across all output rows of
that GEMM. An independently chosen K rotation for each N256 weight tile would
require separately transformed activation copies or extra work; it is excluded
from the initial fast path.

Distinguish two uses of reordering:

1. **Before rotation:** compare grouping large channels together with spreading
   them across rotation blocks. Spreading can give each block outliers to mix;
   simply sorting by magnitude is not automatically the right preconditioner.
2. **After rotation:** rebuild E2M1/E0M3 candidates and regroup their task
   preferences into 256x64 rectangles. Whole rotation-block moves preserve the
   transform structure; any finer move needs an explicitly composed transform.

For `T = P R`, deployment uses `X T` and `W T`; transform order is part of the
exported artifact. Recompute tensor factors and 16-element scales in that basis.
Keep canonical FourOverSix and E0M3 alpha=1 as the two candidates; do not expand
the scale search while comparing transformations.

Rotation need not favor E0M3. It can reduce outliers, improve E2M1, erase useful
channel structure, or change activation errors. Therefore, each rotation gets
both a **transformed all-E2M1 control** and a **transformed MixFP4 arm**, calibrated
at that transformed all-E2M1 model. Old format maps and gradients do not transfer.

Deploy rotations according to the graph:

- MLP permutations commute with the elementwise gate; a general rotation does
  not. A rotation of the `down_proj` input must run **after** SiLU and the product,
  ideally fused with fixed-E2M1 activation quantization.
- Within-head V rotations may be folded through attention into `o_proj`, subject
  to GQA and the actual attention implementation. Verify full-precision equality.
- A residual rotation must be common across connected residual branches. Fuse
  RMSNorm gains into consumers before relying on RMSNorm's orthogonal invariance,
  and transform embeddings, residual writers, final norm consumers, and head
  consistently. This is a later arm, not a per-matrix shortcut.

These graph transformations follow the computational-invariance approach of
[QuaRot](https://arxiv.org/abs/2404.00456). If fixed transforms leave substantial
headroom, a later experiment can learn orthogonal block transforms using task
loss, as motivated by [SpinQuant](https://arxiv.org/abs/2405.16406). Keep this out
of the first sweep: arbitrary dense online rotations can defeat the speed goal.

The existing `_rotate_chunks` and rotate–quantize–inverse path are useful W4A16
references. They are not evidence of a packed fast path. For W4A4, explicitly
apply the same transform before activation quantization and use it during both
scoring and evaluation; quantization does not commute with rotation.

## 6. Keep search, election, and evaluation separate

Use OpenWebMath and CodeParrot, as in the report, with disjoint sequences for:

1. Transform search and candidate pruning.
2. Fresh CE/KL gradient scoring and k=3 format election after transforms freeze.
3. Forward validation of the complete transformed, mixed model and selection of
   the final policy against identity and transformed E2M1 controls.

Start with 128 sequences per split, balanced across the two sources, and record
source IDs, seeds, and token lengths. Repeat the shortlisted policy with three
calibration seeds. Preserve the report's original 8x64 run as a reproduction
control; compare the new methods under matched new calibration budgets too.

Do not choose rotations or permutations on the same samples used to advertise
their k-SE evidence. k=3 remains a heuristic, not a familywise guarantee after
search. If joint switches fail forward validation, retain the transformed E2M1
control or identity. Any rollback/relinearization variant must be named separately
from the unchanged one-shot k=3 rule and frozen before final evaluation.

WikiText-2, C4, and downstream task outcomes remain outside all selection stages.

## 7. Run a staged experiment matrix

| Stage | Work | Completion criterion |
|---|---|---|
| A: geometry and correctness | Reproduce 8x64; coarsen unchanged scores; check transform/inverse and whole-model FP32/BF16 equivalence | Correct masks, covariance handling, padding, and graph transformations |
| B: inexpensive pilot | Qwen3-4B and Llama-3.1-8B; final-quarter MLPs plus representative early/middle layers | Rank legal reorder, rotation, and combined candidates on independent calibration validation |
| C: full dense models | Apply shortlisted graph-legal transforms across both models; three calibration seeds | Stable full-model gains beyond the corresponding transformed E2M1 control |
| D: transfer | Freeze recipe and search budget, then run Qwen3.8-27B | New format election allowed; no model-specific retuning on test metrics; audit hybrid paths |
| E: native GB200 | Pack transformed weights, scales, maps; run numerical and latency comparisons | Simulator/native agreement and measured total serving benefit |

The minimum comparison set at the full-model stage is:

| Arm | Layout/basis | Weight selection |
|---|---|---|
| Reference | Original | BF16, NVFP4, and FourOverSix |
| Fine control | Original | Report's 8x64 MixFP4 |
| Coarse control | Original | 256x64 MixFP4 |
| Reorder | Legal permutations | Both E2M1-only and 256x64 MixFP4 |
| Rotation | Selected legal rotations | Both E2M1-only and 256x64 MixFP4 |
| Combined | Preorder, rotate, regroup | Both E2M1-only and 256x64 MixFP4 |

Measure the same 2048-token WikiText/C4 protocol and paired NLL differences as the
report. Include its six-task zero-shot panel and paired document comparisons;
retain gsm8k for Llama because its existing panel improvement over FourOverSix
was inconclusive. Comparisons should run on the same GPU model and software
stack. Report repeat-seed variability separately from evaluation-window SEs.

Report three distinct quality differences: transformation benefit with E2M1,
incremental MixFP4 benefit in that basis, and total benefit over the better
untransformed baseline. Also report coarse-versus-fine NLL gain retention where
the fine gain is reliably positive; avoid a ratio when its denominator is near
zero. Selected-tile count is a diagnostic, never the success target.

## 8. Measure the serving cost explicitly

Export transformed packed FP4 weights, UE4M3 scales, tensor factors, one type bit
per logical 256x64 weight tile, shared-axis permutations, rotation sizes/signs,
padding, and model/calibration identifiers. Weights stay in the transformed basis
at runtime. Include actual metadata alignment and indexing overhead, not just
the ideal one-bit storage calculation.

Switch weight format inside the native mainloop at its verified granularity;
keep the activation operand E2M1. Avoid materializing a dequantized BF16 weight
matrix or launching an independent GEMM for every selected tile. Check output
against a reference decoded from the **same packed codes and scales**.

Benchmark representative M values 1, 16, 64, 256, and 2048 and the actual N/K
shapes. Separate prefill from decode: an M256 target can waste work on small M.
Measure activation transforms, activation quantization, GEMM, epilogues, padding,
and full-model tokens/s. Use the same shapes and activation policy for native
E2M1 controls. Include the old fine-tile implementation only if a runnable native
baseline exists; simulated timings are not a hardware baseline.

An initial engineering target is at most 5% total latency overhead relative to
the matching native E2M1 path, while improving quality. This is a proposed gate,
not a measured result. Larger-tile execution may accelerate a fine-tile path;
choosing E0M3 does not itself reduce the operation count versus E2M1. Report
speedup against BF16 and the existing deployment separately, with the complete
transform cost included. No significant B200 speedup is established yet.

## 9. Implementation order and cluster execution

The initial joint row/column solver is now described in
[TASK_REORDER.md](TASK_REORDER.md), including fine-score collection, independent
format election, and the activation-gather/output-epilogue contract.

The first implementation should be a tile-geometry adapter plus a score
coarsening analysis, followed by a deployable MLP permutation pass. Add rotations
only after those establish the amount and source of coarsening loss.

| Existing component | Planned change |
|---|---|
| `run_math_code_calibration.py`, `analyze_task_sensitivity.py` | Parameterize score reductions; expose transformed-model calibration and per-sequence coarse scores |
| `run_kse_paper.py`, `quantize/interacting_format.py::apply_mask` | Remove 8x64 assumptions from a new geometry-aware path; preserve historical defaults and reproduction checks |
| `quantize/reorder.py`, `quantize/blockorder.py` | Reuse balanced search and spread/group initialization; add CE/KL features and graph-shared axis constraints |
| `quantize/quantizer.py` | Reuse candidate grids and Hadamard reference; retain packed-basis metadata instead of only returning inverse-transformed fake weights |
| New transform/export adapter | Apply legal graph transformations, biases, tied parameters, inverse maps, and explicit tail masking |
| New GB200 benchmark | Validate real operand types and compare packed kernels with transform-inclusive latency |

Tests must cover exact permutation identities, rotation orthogonality and graph
equivalence, coarse scores versus direct contractions, correlated-score SEs,
256x64 format uniformity, and packed decode correctness. Explicitly pad and mask
tails; do not silently shrink the type tile using the generic quantizer's
small-dimension convenience behavior.

**All computation, including CPU score aggregation, tests, model loading, packing,
and compilation, runs through Slurm.** Use `--account=gov113008` and `taide`
with H100 or `taide_h200` with H200 for simulation/calibration. Override old
`slurm/reorder*.sbatch` account directives; they still name `MST114554`. Use
job-specific worker `/tmp` caches and bounded CPU thread counts. GB200 kernel
timing requires an actual GB200 allocation; the documented taide queues supply
H100/H200, so their results cannot establish B200 throughput.

This planning change submits no jobs. Keep the existing report's measured tables
intact. Append new GB200 results only after the quality and native execution
checks above, with separate transformation, mixed-format, and throughput claims.
