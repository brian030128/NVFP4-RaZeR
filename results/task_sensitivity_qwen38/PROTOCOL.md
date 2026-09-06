# Qwen3.8-27B target experiment

The priority target is `Qwen/Qwen3.8-27B`, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. This is a native Qwen3.5-family
hybrid architecture, loaded with Transformers 5.16.1; the older Qwen3 model
copies in this repository are not used.

## Frozen decision rule

Use the existing task-sensitivity method without tuning on target test results:
64 WikiText training windows of 2048 tokens for gradient scoring and fit-loss
backtracking, followed by 16 independent training windows for a single
validation decision. Repeat with seeds 20260909 and 20260910.

Eligible 8x64 weight tiles have estimated loss change plus twice its sequence
standard error below zero. Rank by estimated benefit. Start with total predicted
NLL reduction at most 0.1, halving at most eight times until measured fit loss
improves with mean + 2 SE < 0 and achieves at least 25% of the predicted benefit.
Validation accepts the resulting map only if mean + 2 SE < 0; otherwise export
the E2M1 baseline. Validation does not choose a step size. Evaluate the proposed
map even when rejected, to expose false predictions.

Full WikiText test and 64 fixed C4 windows are held out from selection. Record
paired NLL changes, perplexity, and validation's forecast before test evaluation.
The first seed also measures the native BF16 reference. Report every completed
seed, including failures; do not choose the better seed after evaluation.

## Quantization boundary

Quantize all 496 text `nn.Linear` weights, excluding the output head, with
16-element E4M3 scale groups and 8x64 E2M1/E0M3 type tiles, alpha = 1.
Their inputs use `nvfp4_4over6` A4. Embeddings, output head, recurrent state,
depthwise convolution, normalization, and vision remain native precision.
No reordering or scale-preset search is used. These are simulated W4A4 accuracy
measurements; they do not measure an FP4 kernel's throughput.

`probe_qwen38.apply_native_type_map` applies exported maps to pristine native
weights. The map's explicit architecture scope prevents the legacy loader from
silently applying it to vision projections.

## Integration checks and baseline correction

Before calibration, require a successful two-H100 probe: native tiny hybrid
forward/backward, correct SiLU gate, exact plain-versus-STE forward agreement,
full-model gradient scoring, and E2M1 candidate equality on all 496 matrices.

The target exposed six elements in one weight matrix where the old NVFP4
implementation emitted magnitude code 8 after an E4M3 subnormal scale rounded
down. E2M1's maximum is 6. The baseline now explicitly saturates to [-6, 6].
Job 329107 reproduced the bug and verified exact equality of corrected NVFP4,
an independently saturated reference, and the MixFP4 E2M1 branch on the full
matrix and a synthetic regression case. Evidence is in
`../task_sensitivity_qwen38_probe/saturation_check.json`.

## Resources

All compute, including tests and summaries, runs through Slurm. At most two
H100s run concurrently. Each target job uses two H100s, 24 CPU cores, and 400 GB
host memory. Model downloads and framework caches use job-specific worker
`/tmp` storage. Only persistent results and code use the shared filesystem.

## Target controls

After the first seed completed, register two diagnostic controls, neither used
to select the target map: ordinary weight-MSE E2M1/E0M3 selection at 8x64, and
random switches matched to seed 20260909's tile count in every projection
(random seed 20260911). Evaluate both on the same full held-out datasets.
Also apply the exported first-seed map to pristine native weights and require
exact replay of the first 16 WikiText NLL values. The controls run only after
both primary seeds and regression checks finish, preserving the two-GPU cap.
