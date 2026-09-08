# Conditional format cost during error compensation — development v0

Date: 2026-09-07. The spectral and activation-bound approaches are closed in
their tested forms. This experiment changes the mechanism: select a format
while actively compensating future weight coordinates, rather than swapping
fixed candidates into an already quantized network.

## Basis and novelty boundary

For a row's quadratic output error delta^T H delta, fixing a coordinate group
B to change e while remaining coordinates R can compensate yields minimum
e^T [(H^-1)_BB]^-1 e. The optimal remaining update is
(H^-1)_RB [(H^-1)_BB]^-1 e. This is the established OBS/OBQ principle applied
to a whole legal format tile; the algebra is not a novelty claim.

MixFP4 (https://arxiv.org/html/2605.31035, Appendix C) reports fixed formats
before GPTQ and dynamic MSE format selection during its SpinQuant/GPTQ path.
The comparator must therefore include dynamic MSE with identical compensation.
Simply integrating mixed formats with GPTQ is not a new contribution.

## Frozen algorithm and controls

One calibration pass: first 32 non-overlapping 512-token windows of WikiText-2
raw training text, pinned revision b08601e04326c79dfdd32d625aee71d232d685c3.
Collect complete input second moments once from pristine BF16 native model
inputs. Share those same moments across all policies; no loss-based selection.
Use diagonal damping 0.01 times mean diagonal, fixed for every matrix.

Process columns in contiguous groups of 64, with one format per eight rows.
At each step quantize the current compensated weights under canonical
FourOverSix E2M1 or E0M3 alpha=1. Preserve the original tensor global scale;
block scales retain E4M3 rounding/saturation. Candidate weights are BF16, with
all compensation and cost accounting in float64. No column reordering.

Compare four named policies, all using the SAME groupwise compensation:
E2M1 only; static MSE type map from pristine weights; dynamic MSE type choice;
conditional quadratic type choice. Conditional sums the optimized group cost
over each legal eight-row tile. Update all remaining columns by the analytical
compensator. Never use model losses, backtracking, thresholds or a preferred
evaluation dataset to choose among these policies.

This is groupwise compensation, not a claim to reproduce columnwise GPTQ.
Formats and scale groups remain legal, but rounding within each 64-column
candidate is ordinary nearest rounding, not an optimal correlated rounding
solver. Conditional local optimality does not imply global discrete optimality.

## Initial panel and transfer probes

Pinned local Llama-3.2-1B-Instruct revision
9213176726f574b556790deb65791e0c5aa438b6. Complete q/k/v matrices from layers
0, 8, 15 (nine matrices); shared QKV inputs permit one covariance per layer.
Native Transformers 4.57.3, eager attention. Record revisions, token hashes,
weight hashes, compensated outputs/maps, moment hashes and full cost traces.

Freeze all weights before loading held-out inputs: first sixteen 512-token
windows of Wiki validation; first sixteen GSM8K test examples; first sixteen
MBPP test examples, at pinned revisions in the runner. Math/code use reference
text, not generated-answer accuracy. Measure paired per-example layer output
errors on identical pristine model inputs. Evaluation does not update weights.

Tests: canonical fixed-global-scale candidate fidelity, exact optimized group
cost, zero future-coordinate gradient after compensation, telescoping cost
identity for final quantized weights, deterministic outputs, isotropic-MSE
equivalence, and legal tile shape. All compute and downloads use Slurm H100s
with per-job worker-local HuggingFace cache.

## Continue or change direction

Call this close enough for a larger study only if conditional versus dynamic
MSE improves geometric-mean held-out layer reconstruction by at least 0.5% in
each of the three domains, with no matrix/domain regression above 1%. This is a
development screening criterion, not a significance claim or a config selector.
If it fails, report the outcome and move to another mechanism rather than tune
damping, windows, layer subsets, or thresholds against these results.

Even a pass establishes only promising reconstruction transfer. A paper still
requires fresh models, matched stronger quantizers, full-model W4A4 accuracy,
cost measurements, and a contribution beyond existing error compensation.
