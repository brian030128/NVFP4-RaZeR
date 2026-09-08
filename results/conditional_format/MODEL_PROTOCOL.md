# Frozen full-model continuation after mechanism job 331863

The initial mechanism panel passed its predeclared screening criterion:
conditional cost versus dynamic MSE reduced reconstruction error in all 27
matrix/domain cases, by 2.62–2.78% in domain geometric means. Continue with the
same conditional rule. No damping, ordering, scale, tile, or calibration-budget
change is permitted in response to evaluation results.

## Models and shared calibration

Models: pinned Llama-3.2-1B-Instruct (same development model) and facebook/opt-350m
(new architecture for this study). Resolve the OPT revision and record it before
loading data or evaluating losses. OPT is chosen for a cheap architectural
transfer test, not from performance measurements. Neither model size represents
the original 27B deployment target.

Quantize every native non-head Linear weight, including OPT projection layers;
preserve biases, embeddings, norms, attention operators and language-model head.
Require matrix dimensions divisible by 8x64. Collect H on the SAME first 32 Wiki
training windows of 512 tokens as the mechanism panel, in one pristine-model
pass per model. Shared QKV and gate/up input sites share one covariance. Every
quantizer receives exactly the same statistics. No validation or loss gate.

Comparators: canonical FourOverSix without compensation, group-compensated
E2M1, dynamic-MSE mixed formats with compensation, and conditional-cost mixed
formats with compensation. Include BF16 as a reference. All compensated
methods retain fixed original tensor normalization, 64-column steps, 8-row
format choices and 1%-mean-diagonal damping from the mechanism protocol.

Save all quantized weights and their hashes before evaluation data is loaded.
Compensated outputs cannot be replayed by the old pristine fixed-candidate map
loader: saved weight dictionaries and calibration regeneration are authoritative.

## Evaluation, fixed before the run

Use 32 disjoint 512-token windows from WikiText raw TEST (the mechanism panel
used validation); GSM8K and MBPP test examples with indices 16 through 47 (the
mechanism panel used 0 through 15). Keep pinned dataset revisions. No evaluation
example chooses a configuration. These sets are new to this particular method
stage, not claimed disjoint from every historical repository experiment.

Evaluate identical frozen weight maps under A16 and A4. A4 applies canonical
FourOverSix to each non-head linear's input tensor, with existing tensor-scale
semantics. Calibration remains A16 for BOTH; A4 measures transfer to activation
quantization, with no recalibration. Biases retain original precision. The
primary comparison is W4A4 conditional versus dynamic MSE; A16 is diagnostic.

Report example-mean NLL/PPL and paired mean difference ± two standard errors
for every domain and both models, plus contrasts against E2M1 compensation and
FourOverSix. Math/code reference-text losses do not measure reasoning/pass@k.
No claim of independent statistical confirmations for correlated Wiki windows.
No packed-kernel speed claim; include calibration and quantization costs.

## Continue criterion

A promising full-model result requires negative primary ΔNLL point estimates
in all six model/domain cells and descriptive improvement intervals in at least
four, while showing no supported harm relative to compensated E2M1. A failure
triggers mechanism review or another direction; it does not authorize selecting
A16, a domain, model, or policy after the fact as the preferred configuration.
This is a development screening criterion, not a certification procedure.

The method is not paper-ready on a pass: it still needs larger models, more
calibration/evaluation seeds, generated tasks, packed cost accounting and a
thorough comparison to blockwise quantization/conditional-error prior art.
