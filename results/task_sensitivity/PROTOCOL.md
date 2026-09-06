# Predicting MixFP4 tile switches from task loss

Started 2026-09-06. All model work, tensor processing, and tests run in Slurm
allocations. At most two concurrent H100 GPUs; one GPU per job.

## Question and fixed scope

Can a single calibration procedure choose useful E0M3 tiles across models,
without selecting a model-specific election rule on evaluation perplexity?
Weights use 8x64 type tiles and 1x16 scale blocks. Both grids have alpha=1.
Activations use the existing nvfp4_4over6 forward quantizer. No reordering,
scale search, or weight fine-tuning is included in the initial experiment.
The model runs prefill at sequence length 2048 with use_cache=False throughout.
All comparisons use newly measured paired baselines; cache-enabled historical
absolute perplexities are not substituted. A cache-toggle diagnostic records
its first-window effect in later runs, since discrete activation rounding can
amplify changes in floating-point execution paths.

Existing evidence: MIXFP4_REPORT.md sections 1-5, particularly the failure of
clean-input full-covariance reconstruction to rank election rules. This does
not establish impossibility of task-aware calibration.

## Initial method (fixed before observing outcomes)

Reference: all-E2M1 NVFP4 weights in the actual W4A4 model.
For each tile, d = dequantized E0M3 candidate - dequantized E2M1 candidate.
For each calibration sequence, accumulate s = <gradient of NLL, d> per tile.
The backward uses identity straight-through derivatives through activation
quantization; the forward uses the exact original quantized values. Parameters
are frozen; hooks compute the directional derivatives from linear inputs and
output gradients. No parameter optimizer or full Hessian is used.

Two predeclared policies:

* gradient_sign: switch when mean(s) < 0.
* gradient_confident: switch when mean(s) + 2 standard_errors(s) < 0.

The latter is a heuristic across-sequence stability margin, not a simultaneous
confidence guarantee over millions of tiles. Quantized finite switches and
cross-tile interactions are not guaranteed by an STE derivative.

Controls: NVFP4, hess_h1.5, hess_impg16_h10. All share identical source weights,
scale convention, activation quantization, and evaluation sequences. Existing
rules use clean-input importance from four fit sequences.

## Data and validation

Default sequence length 2048. Split tokenized WikiText training text into
contiguous nonoverlapping windows; partition its window list into thirds
before sampling. Fit: 16 windows from the first third. Validation: 16 windows
from the second. Probe: 8 windows from the third. Record SHA256 fingerprints.
Validation chooses among the fixed policies and the baseline; the held-out
test is never used for selection. Report both new policies even if rejected.

First models: Llama-3.1-8B and Qwen3-4B, chosen for the existing sign reversal.
Before extrapolating to other models, inspect bounded interventions: switch
the 256 or 4096 strongest predicted beneficial or harmful tiles, measuring
actual NLL on fit and probe sequences. This tests sign, extrapolation in switch
count, and generalization separately from whole-model policy evaluation.

Final evaluation: full WikiText test and 64 C4 validation windows, paired
identically between configurations. Preserve per-window NLL, paired mean
differences and standard errors, and perplexity. This is a C4 subset, not the
historical 256-window C4 result. Windows are not assumed independent tokens;
reported standard errors remain descriptive if sequences share documents.

## Adaptive follow-ups

Decide follow-ups from intervention/validation outcomes, and record each
decision before evaluating its final test:

* If small interventions have wrong signs even on fit: inspect gradient
  correctness and activation-rounding discontinuities; test W4A16 as a
  mechanistic control, then directly measure discrete block interventions.
* If fit signs are right but probe signs wrong: increase independent
  calibration, examine stability, and reject unsupported extrapolation.
* If small groups work but full policies fail: measure curvature and
  interactions; use conservative groups and sequential recalibration.
* If validation succeeds: freeze the procedure and test additional models
  and calibration seeds, including a model outside the initial pair.

Success is a frozen, useful calibration procedure with measured cost and
held-out behavior, not a per-model oracle. Failure means the tested approach
does not meet that criterion at the measured calibration budget. No finite
study can establish that every possible predictor is impossible.

## Reproduction

Submit at most two of these one-GPU jobs concurrently:

```bash
sbatch slurm/task_sensitivity.sbatch qwen3-4b --skip-final
sbatch slurm/task_sensitivity.sbatch llama-3.1-8b-local --skip-final
```

Each job runs structural tests before loading the model. Initial diagnostic
runs skip final test evaluation so adaptive methodological decisions do not
consume it. Outputs are written after each intervention and policy.

## Follow-up frozen after the initial probe (before final test evaluation)

The Qwen3-4B 256/4096-tile interventions have the predicted signs on separate
probe sequences, while switching all negative or confidently negative tiles
harms both fit and validation loss. Llama's first 256-tile interventions also
have the predicted signs. This motivates bounding the *total* first-order
extrapolation rather than trying more per-model election thresholds.

One new policy, gradient_trust_0p1, is fixed for the follow-up:

1. Eligible tiles have mean(score) + 2 SE(score) < 0.
2. Sort eligible tiles by mean(score), most negative first.
3. Select the largest prefix whose cumulative predicted NLL reduction is at
   most 0.1 nats/token. If the first tile alone exceeds the budget, select none.

The 0.1 budget is a development choice, not a theoretical guarantee and not a
parameter to tune separately per model. The initial pair are development
models; additional models test transfer of this frozen procedure. It predicts
useful directions and limits intervention size, not exact final perplexity.

Add a random control with the identical selected tile count in each projection
matrix; only the tile identities differ. Its seed is fit seed + 1. Existing
hess controls remain. Compare on full WikiText test and 64 C4 windows, without
choosing the policy using those results. Reuse initial scores for the first
pair; subsequent models run the same 16-sequence calibration themselves.

Replication predeclared before the first sparse-policy test results: repeat
Qwen3-4B and Llama-3.1-8B with fit seed 20260907 (initial 20260906), changing
calibration/validation windows but leaving the rule and final test data fixed.
For these repeats, final evaluation includes baseline, gradient_trust_0p1,
and hess_impg16_h10. The random and permissive controls remain in fit/validation
but their full-test evaluation is not repeated. This tests calibration-seed
sensitivity without choosing a preferred seed after observing test results.

Later first-window diagnostics observed differences when toggling use_cache,
so the repeat jobs additionally check the *same selected map* against baseline
on the same 16 validation windows with use_cache=True. This isolates execution
path sensitivity; the primary replicated final evaluations remain cache-off.
The diagnostic does not select or alter tiles, seeds, or the budget.

## Calibration-budget follow-up

The second Llama-3.1-8B seed (20260907) improves fit NLL by 0.01398 but
*raises validation NLL by 0.00936 (SE 0.00391)*. This was observed before its
full test result, and prevents recommending unconditional 16-sequence maps.
The selected-map validation check correctly marks this proposal unsupported.

Next test: keep the tile criterion and 0.1 budget fixed, increase Llama fit
data from 16 to 64 sequences on BOTH seeds, and preserve each seed's original
16 fit sequences as a prefix and its validation/probe windows exactly. Extra
fit windows come only from the unused first training third. This isolates
calibration sample size instead of retuning the election threshold or changing
validation text. Final tests remain identical; no seed is discarded.

## Finite-step calibration follow-up

Follow-up after the 64-sequence validation results: seed 20260907 worsens
even FIT loss (+0.04053 NLL), and seed 20260906 has a weak fit improvement
(-0.00419), despite both predicting -0.1. Thus simply adding data cannot
repair the finite-step approximation. Test bounded discrete trust backtracking
on both 64-sequence seeds: start at 0.1, halve up to seven times, and accept
the first step whose actual fit improvement is at least 25% of the predicted
improvement and whose fit mean + 2 SE is negative. Return NVFP4 if none passes.
Only fit data controls step size; validation assesses the chosen map once.
Freeze these constants before running either backtracking job. This is an
exploratory optimization follow-up on development models, not a fresh unseen
model replication. The fixed-budget results remain reported, including failures.

## Third-seed confirmation

Declared before inspecting the first two backtracking runs' validation/test
results: run the unchanged backtracking procedure with 64 fit sequences on
Qwen3-4B and Llama-3.1-8B, both seed 20260908, without reusing scores. This
uses fresh calibration/validation draws, though final evaluation text remains
the same. All criteria, including the 0.25 acceptance ratio, eight-attempt
cap, and independent validation gate, stay fixed. This is a calibration-seed
check on development models, not a claim of transfer to unseen model families.

## Frozen eight-model extension

Requested after the preceding study completed. Before observing any extension
results, fix this panel: Llama-3.2-3B base/instruct, Llama-2-7B/13B,
Qwen3-8B/14B, Llama-3.1-8B-Instruct and Llama-3.2-1B-Instruct.
The first four are new models in this study; the latter four previously
received only the initial 16-sequence fixed-budget rule. Run BOTH seeds
20260909 and 20260910 for every model, with 64 fitting sequences and the
unchanged backtracking rule (0.1 initial budget, halving, 0.25 agreement,
mean + 2 SE < 0, at most eight attempts). Validation remains 16 separate
windows and never chooses a step size. No reordering or alpha search.

Report full WikiText and identical 64-window C4 comparisons for every
proposal, including rejected ones. Also report the deployment outcome that
returns NVFP4 when validation rejects a proposal. A zero-tile fallback is
not counted as evidence of MixFP4 improvement. Both seeds remain in the
record, even when they disagree. This broadens generations, sizes and
instruction tuning within the two supported architectures; it does not
establish transfer to other architectures. Access/implementation failures
are recorded separately, never counted as accuracy failures or silently
replaced. Slurm array concurrency is capped at two one-H100 tasks.

## Qwen3.8-27B target integration

The user prioritizes Qwen/Qwen3.8-27B as a possible final target. Its published
configuration is Qwen3_5ForConditionalGeneration with hybrid linear/full
attention. Do not load it through the old Qwen3 implementation. Use a
job-local current Transformers environment, and native model operations.
Text evaluation quantizes text nn.Linear weights and their input activations;
embedding, head, recurrent state, depthwise convolutions, normalization and
unused vision parameters retain native precision. Explicitly report these
boundaries, pin the resolved model revision, and verify native forward/STE
equality, gradient coverage, tile shapes and two-H100 memory before calibration.
No requested checkpoint is silently substituted. Fit/validation selection
constants stay frozen. Target downloads and environment caches use worker
local /tmp, not the login node's disk.

Extension array 329090 tasks 0/1 hit gated-repository 401 errors on the new
Llama-3.2-3B and Llama-2-7B models. Tasks 2/3 continue; pending tasks 4-15 were
cancelled to prioritize this target, and remain outstanding extension work.
Probe job 329099 depends on completion of tasks 2/3 and requests two H100s.
It was cancelled while pending and replaced by 329101 to keep the existing
CUDA 12.8 PyTorch build while installing only the new Transformers-side
packages in node-local scratch. No model computation ran in 329099.

Probes 329101/329104 found six mismatched weights in Qwen3.8 layer 10's
linear-attention qkv projection. Diagnostic 329107 established that the old
NVFP4 reference emitted illegal E2M1 code 8 when a subnormal E4M3 block scale
rounded down. MixFP4 already saturates at 6. The NVFP4 reference now also
saturates at ±6; the synthetic regression and complete saved target matrix
then match MixFP4's E2M1 candidate exactly. This is a legal-format correctness
fix, not a new calibration rule. The original task-sensitivity panel had
already verified exact identity on every matrix, so its weight candidates
were unaffected by this corner case. Probe 329108 retries the corrected path.

The configured `output_gate_type=swish` belongs to the GatedDeltaNet RMSNorm
gate (as implemented by SGLang), and equals native Transformers' SiLU there.
The full-attention output gate remains sigmoid. The probe checks the native
gated normalization against its explicit formula; no gate is silently changed.
