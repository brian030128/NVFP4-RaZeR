# What the final-target experiment establishes

**8x64 MixFP4 works on Qwen/Qwen3.8-27B for held-out WikiText accuracy.**
The same calibration procedure used on smaller models selects 69 and 62 tiles
on two seeds, with no target-specific threshold tuning or reordering. WikiText
PPL falls from 7.555223 to 7.133067 and 7.156911: reductions of 5.59% and 5.27%.
Native BF16 is 7.050639. The maps share 44 tiles, with 87 in their union.
Both maps concentrate many switches on input channels 3968–4031 across several
projections, especially linear-attention input projections. This is a concrete
tensor region to investigate, rather than a model-independent channel-index
rule; its causal role has not been isolated by an additional ablation.

The gain is not unique to the gradient rule on this target. Ordinary weight-MSE
selection reaches 7.154471 WikiText PPL, switching 43,487,115 tiles. Its paired
differences from the two sparse maps are inconclusive. A 69-tile random control,
matched by projection, reaches 7.530938; its change from the baseline is also
inconclusive. Thus the evidence supports identifying a tiny set of consequential
tiles, not an accuracy-superiority claim over MSE on this particular model.

## Predicting performance before test evaluation

The first-order score for a tile is the final calibration-loss gradient dotted
with the E0M3-minus-E2M1 weight change. This scores all legal switches together
using 64 backward passes, rather than evaluating a model for each possible map.
Activation quantization retains its exact forward values; its backward uses an
explicit identity straight-through estimator. This is a surrogate prediction,
not an analytic guarantee.

The frozen rule retains stable negative scores within a 0.1 predicted-NLL
budget. Measured fit loss checks the approximation; backtracking halves the
budget if needed. Both target seeds pass at the first budget. An independent
16-window validation set then accepts or rejects the resulting map once.

Validation forecasts were -9.59% and -4.70% relative WikiText PPL changes;
observed test changes were -5.59% and -5.27%. Both test changes fall within the
descriptive validation mean ± 2 SE intervals transformed to PPL. The first
point estimate is substantially optimistic. This predicts direction and a
rough range, not exact test performance.

## What does not generalize yet

WikiText-calibrated maps yield only -0.0040/-0.0054 C4 PPL changes, both
inconclusive. MSE selection is also inconclusive on C4 (-0.0023 PPL).
The WikiText validation forecasts do not transfer to C4. A final workload
therefore needs representative calibration and independent validation;
the current results cannot establish a domain-independent performance rule.

Across the broader study, the reusable object is the scoring, fit-checking,
and validation procedure. The selected tile locations and trust step depend
on the model and calibration distribution. This is stronger evidence than
choosing a preset by held-out perplexity, but it is not a universal guarantee.

## Verification and limits

Probe 329113, two-seed experiment 329114, regression/summary 329119, and
controls/replay 329124 completed successfully. All heavy work ran in Slurm,
with at most two H100s active. Downloads and temporary framework files used
worker-local storage.

The target exposed six illegal magnitude-8 outputs in the old NVFP4 baseline
when FP8 subnormal scales rounded down. Explicit E2M1 saturation fixes this;
full-matrix and synthetic regressions pass, and all 496 text projections match
the MixFP4 E2M1 branch exactly. Gradient and ordinary forward losses match on
every calibration window. Applying the exported map to pristine weights on a
fresh native model reproduces 16 held-out losses exactly.

Quantization covers text linear weights and inputs. Embeddings, output head,
normalization, recurrent state, depthwise convolution, and vision remain native
precision. This is fake-quantization accuracy evidence, not kernel throughput,
vision performance, long-context decoding, or reasoning-task validation.

See [the numerical report](REPORT.md), [protocol](PROTOCOL.md), and
[seed 20260909 map](seed20260909/type_map.json). Both accepted maps are preserved;
no winner is selected by test results.
