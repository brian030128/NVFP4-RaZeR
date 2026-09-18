# Joint row/column MixFP4 reordering

`quantize/task_reorder.py` implements a task-aware permutation search for the
**N256 x K64 weight type tile**. It chooses complete row and column permutations
and elects E0M3 tiles together. This is an offline optimizer and a deployment
layout contract; the fused GB200 kernel is not implemented here.

## Objective and search

Input atoms are per-sequence CE and teacher-KL directional scores for changing
canonical FourOverSix E2M1 into E0M3 alpha=1. One atom covers **one output row and
16 K elements**. Columns move as intact 16-element groups, so the candidate
weights and their scales are unchanged. Rows can move individually.

For a candidate layout, sum all atoms inside each 256x64 tile **within each
sequence**, then compute each objective's mean plus k standard errors. The tile
is E0M3 only when both bounds are negative. The search maximizes the sum of
positive margins below zero, after normalizing CE and KL by separate, fixed
fit-derived RMS scales. This normalization affects the search tradeoff but not
the sign of the final conjunction. It is a directional surrogate, not a promise
that the complete switched model has lower loss.

The algorithm has four parts:

1. Identity, marginal, rank-one and angular rank-two spectral, and seeded random
   layouts provide different starting partitions. Refine identity and the best
   other starting layouts rather than committing to a single initializer.
2. Alternate row and column assignments into fixed-capacity groups. Differentiate
   the current tile objective, score every item's possible destination, then use
   regret-ordered balanced assignment. Commit only if the actual objective improves.
3. Refine with pair exchanges. Half the proposals are random and half target
   destinations favored by the objective gradient. Evaluate their actual objective
   changes, including sequence covariance, in bounded batches. Commit only
   improving swaps that touch disjoint groups, so their gains remain valid.
4. Anneal from a smooth CE/KL maximum and smooth positive-part objective to the
   hard k-SE objective. This supplies a gradient when no initial tile is elected.
   Track the best **hard** objective throughout, with identity as the fallback.

The search is a multistart heuristic for balanced co-clustering, not a global
optimum certificate. It guarantees that the returned **fit surrogate** is at
least the identity fit score. That guarantee does not apply to fresh election
scores, finite-step loss, or downstream quality.

The runner freezes the winning permutations, then re-elects formats on disjoint
sequences, stratified by math/code source. The default split uses 64 sequences
for search and 64 for election from the existing 128-sequence calibration.
Independent model-level forward validation remains a subsequent step. The
identity election on the second split is recorded as a diagnostic, not used to
select the layout.

## Collect usable scores

Old 8x64 scores cannot be subdivided to optimize column grouping. The existing
calibration runner now accepts `--reorder-modules REGEX` and additionally streams
fine CE/KL atoms for the matched modules to
`reorder_scores/000/`, `reorder_scores/001/`, etc. Each directory contains a
manifest and one paired score shard per sequence. Historical outputs and
calibration behavior remain available without the option.

For example, submit a Qwen3-4B pilot covering the last MLP:

```bash
sbatch --account=gov113008 --partition=taide --array=0 \
  slurm/math_code_calibration_small.sbatch \
  --reorder-modules 'layers\.35\.mlp\.(gate_proj|up_proj|down_proj)$'
```

Use the existing `--allow-source-drift` option only if the calibration runner
reports an actual quantizer source mismatch and that mismatch has been reviewed.
Fine scores are much larger than historical scores: float32 CE+KL for 128
sequences costs approximately **64 bytes per weight**, or about 1 GiB for a
4096x4096 matrix. Start with a few modules. Shards stream to disk during scoring;
the search subsequently processes one matrix at a time.

The collector uses the existing scorer's causal per-token activation factors.
The report's 2048-token evaluation uses its own tensor-wide factor convention.
The manifest records this distinction; the method does not silently substitute
one evaluation protocol for the other.

## Search and export

Once the selected module's score collection has completed:

```bash
sbatch slurm/task_reorder.sbatch \
  --scores results/math_code_adaptive/calibration_JOB_qwen4b/reorder_scores/000 \
  --out results/task_reorder/qwen4b_last_mlp_000 \
  --tile-rows 256 --tile-cols 64 --starts 4 --rounds 6 --seed 0
```

`--axes rows` and `--axes cols` support ablations. `--scratch-mb` bounds the swap
evaluation temporaries, not the full score tensor. The CPU search uses float64
accumulation to reduce cancellation error and requests 96 GiB in the supplied
worker script. It is not a low-memory whole-model process.

The output directory must be new. It receives `report.json` and `layout.pt`:

| Field | Meaning |
|---|---|
| `row_perm`, `col_perm` | New position → original channel index |
| `inverse_row_perm`, `inverse_col_perm` | Original position → new position |
| `row_atom_perm`, `col_atom_perm` | Same mapping at the score-atom granularity |
| `mask` | Final E0M3 map from independent election sequences |
| `fit_mask` | Search diagnostic only; not the deployed map |
| `election_upper_ce`, `election_upper_kl` | Raw, unnormalized final tile bounds |
| `weight_shape`, `padded_shape` | Real and native tile-padded shapes |
| `provenance`, sequence IDs | Score origin and disjoint search/election membership |
| `trace`, `objective_scales` | Search trajectory and fixed normalization |

Tail groups keep their exact real-channel capacity. Padding therefore remains
at the end of each axis rather than being mixed into the interior and removed
incorrectly during export. Zero padding and masked stores are the kernel's
responsibility.

## Kernel contract

For original weights `W[N,K]`, store:

```text
Wp = W[row_perm][:, col_perm]
Xp = X[:, col_perm]
Yp = Xp @ Wp.T
Y[:, row_perm] = Yp
```

The last line is the epilogue scatter that restores output order. Equivalently,
`Y = Yp[:, inverse_row_perm]`. Permute biases with output rows before scattering;
keep residual additions in a consistent output coordinate system.

The input permutation cancels because **both** operands use it. An arbitrary
per-matrix column permutation still needs a matching activation gather, fused
load/quantization, or a consistent producer transformation. It does not disappear
from runtime just because the matrix identity is exact. Columns within each
16-element group remain ordered, so fixed-E2M1 activation quantization can
preserve its original scale groups too.

This version optimizes each matrix independently, which is valid with per-GEMM
input gathering and output scattering. Do not fold independently selected
permutations into a shared producer without reconciling its consumers. Shared
graph-axis optimization and the native fused implementation are later work.

`reordered_weight_reference(base, alternative, layout, layout['mask'])` returns
the mixed dequantized matrix **in deployed order**, useful for checking packing
and the epilogue. It does not hide the permutation by undoing it on the weights,
and it does not implement a native FP4 GEMM. Every 256x64 tile shares one type;
each row's four 16-element scale groups retain independent scales.

## Verification

```bash
sbatch slurm/task_reorder_tests.sbatch
```

The focused tests cover recovery of planted 256x64 structure requiring both
axes, correlated sequence scores, independent-election rejection, direct
gradient contractions, permutation/GEMM equivalence, tail tiles, deterministic
search, axis restrictions, invalid inputs, and the shard-to-artifact runner.
All tests and score processing run on Slurm workers under `gov113008`.

Verified 2026-09-18: **9/9 tests passed** in Slurm step `348924.0` on the
`gov113008/taide_h200` allocation (CPU only, no GPU requested by the step).
Worker-side compilation of the solver, runner, and modified calibration script
also passed in step `348924.1`. The redundant pending batch job `348927` was
cancelled after these checks succeeded. No real-model reordering quality or
native-kernel speed result has been measured yet.
