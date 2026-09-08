# Activation-aware asymmetric compensation: fixed second direction

Declared after full-model array 331873 failed the W4A4 transfer screen.
Conditional selection improved all six A16 comparisons versus dynamic MSE,
but only three of six A4 comparisons. This motivates explicitly accounting for
activation rounding; it does not establish that rounding is the unique cause.

## Objective and prior art

For pristine local input x, deployment-quantized input xq, and source W, solve

    min_V E ||V xq-W x||^2 + lambda ||V-W||_F^2.

With H=E[xq xq^T], C=E[x xq^T], the continuous optimum is
V*=W(C+lambda I)(H+lambda I)^-1. Quantizing around V* under metric H+lambda I
then minimizes the excess quadratic objective. This is ridge regression and
asymmetric calibration; related work includes GPTAQ/GPTQv2
(https://arxiv.org/abs/2504.02692). Neither the continuous solution nor the idea
of compensating activation error is claimed as new. Our test concerns legal
mixed-format group decisions within this objective.

This stage uses quantization of the current pristine layer input, not an
iteratively quantized upstream model trajectory. It does not reproduce all
asymmetric upstream-error features of GPTAQ. That limitation must be reported.

## Frozen implementation

Models unchanged: Llama-3.2-1B-Instruct and OPT-350M. One pass over the same 32
Wiki training windows computes H and C together for every input group. All
policies share that pass. No new calibration corpus, scale search, damping
search, tile change, or loss acceptance. lambda remains 1% mean diagonal of H.

Reuse the 64-column compensation and 8-row type decisions unchanged. Construct
all three compensated policies (E2M1, dynamic MSE, conditional) around V*.
Preserve original W's tensor global scale for every policy, so continuous
centering does not introduce a separate normalization search. Keep FourOverSix
without compensation as a reference.

An explicit ablation, quantized_input_conditional, uses the same H from xq but
centers at W instead of V*. It isolates changing the input geometry from
correcting the activation-induced output mismatch. It is never elected by loss.

All non-head native linear weights are quantized. Biases/other operators stay
fixed. Evaluate W4A4 only as the primary deployment target; A4 uses canonical
FourOverSix with the same per-input-tensor scaling as the first full-model run.
Freeze and save all quantized weights before evaluation.

## Fresh evaluation and decision

Use Wiki test windows 32 through 63 (512 tokens) and GSM8K/MBPP test examples
48 through 79, with unchanged pinned revisions. They do not overlap the first
full-model stage's examples/windows. No previously viewed evaluation loss
chooses a map. Math/code remain reference-text NLL, not generated task success.

Report all policies and paired conditional-versus-dynamic, versus-E2M1,
versus-FourOverSix, and versus-quantized-input-only contrasts. Keep the prior
screen: negative conditional-versus-dynamic mean in all six model/domain cells,
descriptive improvement intervals in at least four, and no supported harm
versus compensated E2M1. Report failure without tuning this formulation.

Tests verify the new continuous optimum's stationarity and exact decomposition
of the asymmetric objective into a constant plus conditional quadratic costs.
All earlier candidate-fidelity/compensation tests must also pass. Formal paper
claims still require more models/seeds/tasks and a substantive novelty case.
