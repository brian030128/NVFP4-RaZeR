# A common calibrated rule for FP4 tile types

The procedure chooses one static format map per model. That same map serves
every evaluation domain. It does not select a configuration by evaluating its
loss on the destination dataset.

## Fixed candidates and one shared score table

For each nonhead linear weight matrix W, form canonical FourOverSix Q0 and
E0M3-alpha1 Q1 once. An8x64 tile j has fixed difference D_j=Q1_j−Q0_j. No
element values, scales, floating-point weights or rotations are optimized.

Use a fixed pool of64 C4,64 OpenWebMath and64 CodeParrot sequences, each512
tokens. Source revisions, document hashes and token hashes are recorded. At
the unchanged FourOverSix W4A4 model, collect each sequence's tile derivative
for next-token cross entropy and for KL from the pristine BF16 teacher:

    g_CE[i,j] = <gradient_Wj CE_i, D_j>
    g_KL[i,j] = <gradient_Wj KL(teacher || student)_i, D_j>.

One student forward and two backward passes per sequence supply the shared
table. Inputs to quantized linear layers use identity STE during scoring.
These are approximate sensitivities of a discontinuous fake-quantized model.

## One election rule, unchanged across models

For n scored sequences, calculate

    u_j = max(mean(g_CE[:,j]) + 2 SE(g_CE[:,j]),
              mean(g_KL[:,j]) + 2 SE(g_KL[:,j])).

Choose the at-most256 most negative u_j. All other tiles remain FourOverSix.
This exactly solves the additive surrogate

    minimize sum_j u_j s_j,
    with s_j in {0,1} and sum_j s_j <=256.

For each objective separately, the sum of its estimated upper directional
scores is bounded above by this sum of per-tile maxima. This algebra explains
the election. It does not prove that a finite joint flip reduces the actual
network loss. Two SE also does not control all the many tile comparisons
simultaneously.256 and the factor2 are common empirical constants, not
parameters derived from a universal theorem; they are never tuned per model
or destination domain in this confirmation.

## What differs from configuration search

The algorithm never installs candidate maps to choose among their measured
losses. There is no configuration grid, loss backtracking, validation acceptance
gate, selected checkpoint, or chosen calibration source for each configuration.
The final map freezes before confirmation-domain inputs are loaded. Fitting
audits and evaluation then measure the already-fixed result.

Controls also reuse subsets of this single score table: C4-only64 and a
matched-token mixed64 pool (22 web,21 math,21 code). Their results distinguish
data diversity from the increase to192 observations. They do not choose the
primary map.

## Scope of a positive result

The three original primary maps replay exactly. Pythia1.4B and OLMo1B test new
model families under the unchanged rule. Literature, science and government
documents test data families not inspected during method development. This
is stronger evidence of a transferable procedure than fitting independent
settings for each reported dataset.

It remains a calibrated rule. It does not determine types from weight
statistics alone, and it cannot promise improvement for every possible input
distribution. Gradient selection, distillation and sparse optimization are
established tools; their use here is not by itself a novelty claim. The
specific empirical contribution and comparisons must justify an academic
paper. Older failed studies are retained as exploratory evidence, not ranked
as if all had used identical evaluation examples.

The implementation simulates quantization in native Transformers. Nonhead
linear weights and inputs are quantized; other operations remain native.
The shared scoring pass and original confirmation use activation tensor
factors spanning the teacher-forced window. The later
[causal audit](../causal_replay/REPORT_332374.md) replays those frozen maps
with per-token factors and checks exact prefix independence. The
[4B/8B extension](../pooled_scale/REPORT_332389.md) uses the same scoring and
causal-replay procedure at larger sizes. It does not recalibrate a different
map for the row convention. The combined causal results are in the
[consolidated report](../transfer_rule/REPORT.md).

Per-token factors change the activation representation; no free modification
to a single-tensor-factor CUDA interface is claimed. These are reference-text
loss measurements, not generation accuracy or native FP4-kernel throughput.
