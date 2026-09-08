# Dataset-free spectral format selection: development protocol v0

Written 2026-09-07 before executing the new synthetic panel. This records a
mechanism experiment, not a completed transfer study or a novelty claim.

## Hypothesis and limits

Controlling the directional concentration of mixed-format weight error may
transfer better than independent weight-MSE election, without text calibration.
For E = Q(W)-W, max over positive semidefinite Sigma with trace(Sigma)<=1 of
tr(E Sigma E^T) equals ||E||_2^2. The top right singular vector attains it.
This bounds layer reconstruction for unit input second moment; reducing the
maximum does NOT imply improvement for each Sigma, downstream NLL, or W4A4
activation errors. Different maps have different worst-case directions.

Earlier corr<r> experiments failed: a prescribed equicorrelated input geometry
gave almost unchanged scores at scale-block granularity. This study instead
measures the full matrix operator norm, including arbitrary error directions.
It must demonstrate useful differences empirically, not reinterpret that old
negative result as support. A large stable rank is evidence against a strongly
concentrated error spectrum and is reported for every policy.

## Fixed candidates and reference solver

Use repository canonical FourOverSix E2M1 as baseline; E0M3 uses alpha=1,
the same tensor normalization convention, and E4M3 scale groups of 16.
Candidates are dequantized to BF16 exactly as in existing fake quantization.
Decisions occupy full 8x64 tiles. No scale search, rotation, weight fitting,
text, activations, labels, loss gate, per-domain configuration, or seed election.
Non-divisible matrices are rejected by this reference implementation.

Start all E2M1. At each iteration evaluate every single-tile toggle using a
float64 dense SVD of the FULL matrix error. Accept the best strict improvement,
breaking ties in row-major order. Allow toggles in both directions. Stop when
no toggle improves the objective by more than 1e-10 times its current value,
or after 16 accepted steps. Export the resulting map, including an unchanged
baseline. Record the termination reason. This is coordinate descent, not a
globally optimal algorithm. Dense SVD per toggle is deliberately restricted
to tiny matrices; it is not a deployable LLM-scale solver.

## First experiment, fixed before execution

Twelve BF16 32x128 matrices: seeds 101/102/103 crossed with Gaussian, heavy
tailed (Gaussian divided by clipped absolute Gaussian), column outliers,
and shared-row-component weights. Exact generators are in the hashed runner.
Eight format decisions permit an exhaustive 256-map spectral optimum.

Report all cases for FourOverSix, all-E0M3, independent weight-MSE, the spectral
solver, a count-matched random map, and the exhaustive spectral oracle.
Never elect a policy using these comparisons. Report operator norm squared,
Frobenius error, stable rank, solver/oracle gap, map, trace, source/weight hashes,
and analytical expected output error for unit-trace isotropic, random rank-4,
and coordinate-subspace input second moments constructed after map selection.
These synthetic input geometries are diagnostic stress tests, not real domains.

Correctness requirements: canonical candidate identity, legal tile maps,
monotone accepted objective values, deterministic replay, scale invariance for
fixed candidate triples, zero/tied candidates, the variational identity, and a
counterexample to per-distribution dominance. All tests and experiments run
inside a Slurm H100 allocation; no model or dataset download is needed.

## Transfer study boundary

This stage can establish solver correctness, an optimization gap, and whether
spectral improvement trades away typical-input error. It cannot establish the
headline generalization hypothesis. Report failures and costs alongside gains.

Before any new LLM task evaluation, specify a scalable solver using only weight
geometry and freeze its settings, quantization scope, model revisions, model
panel, task splits, metrics and statistical procedure in a separate transfer
manifest. Existing Qwen/Llama results are development evidence, not fresh model
generalization evidence. Choose available unseen families before reading their
results; record any availability exclusions before execution.

Construct one map per model without dataset access and hash it before task
evaluation. Apply that same map to every evaluation domain, with activations
fixed at FourOverSix for W4A4 and matched native baselines. Evaluate NLL and
generated-task performance separately. Include MSE and existing calibrated
selectors as named baselines, never as candidates from which to pick a winner.
Report calibration/selection cost and every regression. No NLL backtracking or
validation-based fallback. Any method revision after viewing transfer results
creates a new development version and requires new held-out evidence.

Do not advance to a large transfer run merely because one synthetic case wins.
First establish that the scalable solver approximates this reference objective
at acceptable cost. Pure spectral minimization may fail to improve task quality;
that outcome rejects this direction rather than authorizing per-dataset tuning.
