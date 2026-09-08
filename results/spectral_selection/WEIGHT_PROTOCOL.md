# Real-weight feasibility protocol v0 — frozen before results

Date: 2026-09-07. Follow-up to synthetic Slurm job 331501. No model-loss results
may select a map, configuration, numerical budget, stopping rule, or checkpoint.

## Panel and scope

Use already available pinned Llama-3.2-1B-Instruct revision
9213176726f574b556790deb65791e0c5aa438b6 and Llama-3.1-8B revision
d04e592bb4f6aa9cfee91e2e20afa771667e1d4b. These are development checkpoints from
one family, not unseen-family transfer evidence. Read q_proj and down_proj
weights from layers 0, floor(n_layers/2), and n_layers-1: twelve full matrices.
Access safetensors directly without tokenizer, dataset, activations or forward
passes. No checkpoint download is needed. Use full-matrix canonical FourOverSix
and E0M3 alpha=1 candidates; never recompute global normalization on crops.

## Numerical approximation, fixed before execution

Keep the spectral objective and 16-step cap. Exact dense SVD per candidate is
infeasible at full size, so evaluate a single fixed approximation, not a sweep:
rank-8 block power iteration, 64 iterations, deterministic seed 1729, float32
with highest matmul precision. Rayleigh-Ritz values are estimates, not certified
upper bounds. Record relative eigen residuals; these alone do not prove the
largest eigenvalue was found.

From the all-E2M1 map, obtain estimated top right singular direction v. Rank
all single-tile toggles by the exact finite change in ||Ev||^2 for that fixed v:
2 <Ev, Delta_b v> + ||Delta_b v||^2. Both toggle directions are permitted.
Evaluate ONLY the best-ranked toggle on the full rank-8/64-iteration objective.
Accept if it reduces the estimate by more than 1e-6 relative; otherwise stop.
No prefix sweep, loss evaluation, per-model parameter tuning or restart.
This is an approximation feasibility experiment, not an exact-descent guarantee.

Freeze each selected map before a separate rank-16/128-iteration audit of the
baseline, weight-MSE map, and selected map. Audit results do not change maps.
Report any disagreement or regression instead of silently reverting the map.
Only sampled-matrix maps are saved; this is not a whole-model export.

## Exact checks and feasibility measurements

For every matrix, take 32x128 crops at the first, middle, and last diagonal
positions, aligning offsets down to full 8x64 tile boundaries. Run existing
exact coordinate descent and the fixed approximation on the identical candidate
triples. Evaluate baseline, MSE, exact-reference and approximate maps with exact
float64 SVD on CPU. Record all gaps; do not select a numerical configuration.
This yields 36 crop cases. Passing small crops cannot establish full-matrix
approximation accuracy, so keep that limitation explicit.

Report full-matrix spectral estimates/residuals, Frobenius error and associated
isotropic output error, estimated stable rank, selected tile counts, stop
reasons, all proposal traces, wall time, peak memory, weight/source hashes and
revisions. Exact candidate and numerical tests run first on allocated compute.
No broad LLM accuracy claim follows from reduced matrix error. Both spectral
benefit and average-error costs must be reported, even for unchanged maps.

## Decision after the panel

Assess whether spectral structure exists, whether the approximation reproduces
the reference on crops, whether full-matrix audits agree with selection, and
whether cost/benefit merits a whole-model run. The synthetic average-error
counterexamples remain evidence against the objective. Do not repair them by
choosing coefficients against text loss. Any objective change is a separately
documented development version. A future transfer experiment needs its own
frozen model/task manifest and fresh evaluation data before execution.
