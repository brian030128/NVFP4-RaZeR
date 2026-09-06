# What the experiments establish

The useful common rule is to estimate the task-loss effect of a tile change
**at the baseline that will actually be deployed**, then validate a small
joint proposal. It is not a model-independent list of good block locations.
Here the baseline preserves FourOverSix's per-scale-block E2M1 choice, and
the proposal adds legal 8x64 E0M3 type changes.

For tile b, the score is the calibration average of

    <gradient_Wb loss(W_baseline), Wb_E0M3 - Wb_baseline>.

Eligibility requires a negative mean plus two standard errors. Eligible tiles
are ranked and selected up to a fixed 0.1 predicted-NLL improvement budget.
Fit-loss backtracking shrinks the budget if measured improvement is less than
25% of the prediction or its two-SE check fails. A separate 16-window
validation check accepts the resulting map once, or exports the baseline.
The 64-window fit, thresholds, and procedure are shared across models.
This avoids exhaustive configuration evaluation; it still requires task data,
backward passes, and a few joint forward checks. It does not predict performance
from weight statistics alone, and its first-order magnitude is approximate.

On the target Qwen/Qwen3.8-27B, the two new calibration seeds select 226 and
327 tiles. Both pass validation and improve held-out WikiText perplexity
relative to FourOverSix: 7.289580 -> 6.973420 / 6.935030. C4 changes relative
to FourOverSix are inconclusive: 9.117679 -> 9.118835 / 9.111269. The first
seed's C4 point estimate retains about 95% of FourOverSix's gain over plain
NVFP4; the second exceeds that gain. These ratios are descriptive, not
certified lower bounds. Both target seeds beat the older sparse alpha=1 maps
and ordinary alpha=1 weight-MSE selection on WikiText in point estimates.
The paired new-versus-MSE WikiText changes are -0.025632 +/- 0.004922 and
-0.031152 +/- 0.005342 NLL (two SE), so this stronger-baseline combination
also separates from ordinary alpha=1 MSE selection in these measurements.

The four-model transfer panel is complete. All four maps pass independent
validation. WikiText FourOverSix -> selected perplexities are Qwen3-4B
14.212163 -> 13.172389; Qwen3-8B 10.030983 -> 9.328109; Llama3.1-8B
6.881153 -> 6.781946; and Llama3.1-8B-Instruct 7.817656 -> 7.565416.
C4 improves beyond the descriptive two-SE interval on both Qwen3 models
and Llama Instruct; base Llama's change is inconclusive. Together with the
target, this is five models with WikiText gains over FourOverSix using the
same procedure, not a guarantee for untested models or workloads.

The reserved eight-window mechanism probe shows that channels 3968–4031
carry many useful selected corrections on this target. Keeping only that
region retains much of the full map's probe improvement, while removing it
still leaves useful corrections. The two separate improvements sum to more
than the full improvement: tile effects interact. Randomizing output-row
groups while preserving each projection/input-column count loses much of
the new map's gain. Input location alone is therefore an incomplete selector.
These interventions are diagnostics on eight windows, not additional held-out
benchmarks or a universal channel-index prescription.
Measured input energy peaks lie in this region for 344 of 368 projections
with 5120 input channels. The region is only 1.25% of those channels but
contains a median 41.2% of input energy on the probe. Under FourOverSix,
the calibrated full new map beats the randomized-row control by
0.044668 +/- 0.031778 NLL (two SE). Thus high-energy columns help explain
where to look, while task sensitivity distinguishes useful output blocks.

The same old map changes the probe NLL by -0.098736 on the NVFP4 background
but -0.038498 on FourOverSix. Recalibrating on FourOverSix gives -0.057067
there. Baseline choice changes the value of a correction; independently good
maps cannot be assumed to combine additively. Direct paired contrasts and
activation measurements are included in the generated REPORT.md.

An unconditional guarantee for a single changed predictor on every possible
input/label distribution is impossible, including the weaker request to
retain a positive fraction of each competing configuration's gain. The
normalization counterexample and exact local tensor criterion are in
GUARANTEE.md. This does not rule out reliable improvements on declared
workloads. A distribution-specific finite-sample retention certificate is
implemented for independent examples and bounded loss. Our current contiguous
windows and raw perplexity do not satisfy those assumptions automatically;
the numerical two-SE checks are not that certificate.
All 245 examined tiles have explicit adverse-input reconstruction witnesses
against both NVFP4 and FourOverSix. Only 84/71 respectively reduce isolated
tile reconstruction error under the measured reference input second moments.
This is additional evidence that local reconstruction error does not by
itself explain the task-loss gains. It excludes cross terms with other tiles
and is not a whole-model loss counterexample.

Slurm jobs 329279, 329286, all four tasks of 329289, and 329294 completed
with exit code zero. Structural, saturation, export, backtracking, certificate,
and local-geometry regression checks passed. The native exported FourOverSix
map reproduced direct installation exactly across all 496 text weight matrices.

The scope remains fake-quantized text linear W4A4 accuracy, with native
recurrent, convolution, normalization, vision, embeddings, and head components.
No kernel throughput or broad reasoning/long-context result is implied.
The reference bank covers NVFP4, FourOverSix, alpha=1 weight-MSE selection,
and earlier sparse maps; reordering/rotation configurations require separate
matched comparisons before extending the retention claim.
