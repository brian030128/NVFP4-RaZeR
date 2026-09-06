# Calibrating 8x64 MixFP4 without a configuration sweep

Implementation: `analyze_task_sensitivity.py`. Evidence: `REPORT.md` and
`OBSERVATIONS.md`. All commands below run computation in Slurm allocations.

## Procedure

1. Load pristine weights. Compute each matrix's NVFP4 E2M1 candidate and its
   E0M3 candidate at alpha=1, with the existing FP32 tensor scale and E4M3
   16-element block scales. Keep an all-E2M1 W4A4 reference model.
2. Use 16 real-text sequences of 2048 tokens. On each, backpropagate next-token
   NLL through that quantized reference. Activation forward values use the
   original nvfp4_4over6 quantizer; its backward uses an identity STE.
3. At every linear projection, compute its weight gradient from its actual
   quantized input X and its output adjoint D: gradient_W = D^T X. Reduce
   gradient_W * (Q_E0M3 - Q_E2M1) over each 8x64 tile. Parameters remain frozen;
   this collects scores and does not train weights.
4. Across sequences, compute each tile's mean score and standard error.
   Eligible tiles satisfy mean + 2 SE < 0. Sort them by mean, most negative
   first, and keep the largest prefix whose total predicted NLL reduction is
   at most 0.1 nats/token. All other tiles remain E2M1.
5. Evaluate this one selected map and its reference on 16 separate
   calibration windows. A measured change d in NLL forecasts a relative
   perplexity change of exp(d)-1 for similar evaluation text. This is a
   one-candidate validation, not a configuration sweep. Export it for use only
   when validation mean + 2 SE < 0; otherwise retain the NVFP4 baseline.

The hardware type tile stays 8x64 throughout. The procedure changes the tile
identities selected on each model, using the same two constants and data
budget. No row or column permutation is needed.

The key change from reconstruction-error election is the reference objective.
Distance to pristine weights measures local reconstruction. The task score
asks whether a change helps the already-quantized network predict its next
tokens, including downstream effects. This supplies information absent from
weight error alone; it does not guarantee that an infinitesimal STE score
survives a finite discrete switch.

## Use the calibration-only path

```bash
sbatch --qos=normal slurm/task_sensitivity.sbatch qwen3-4b \
  --calibrate-only --validate-selected \
  --out results/my_task_calibration/qwen3-4b
```

This writes:

* `type_map.json`: validated sparse E0M3 tile indices, or an all-E2M1 fallback
  if the check does not support an improvement.
* `candidate_type_map.json`: the proposed map, preserved even when rejected.
* `report.json`: selection count, calibration timing, and optional validation
  forecast and its across-window uncertainty.
* `scores.pt`: reusable per-tile means and standard errors. These large tensor
  checkpoints are gitignored; JSON results and exported maps remain tracked.

`--validate-selected` is recommended: the second Llama calibration seed exposed
a real regression that this check predicted before final testing. Omitting it
exports an **unvalidated** map for research purposes. A new output directory is
required; prior reports are not overwritten.
`--resume-scores PATH` reuses scores only after checking calibration-data
fingerprints, model identifier, settings, and source revision when recorded.

The research-panel jobs additionally evaluate fixed hess rules and matched
random masks. Those comparisons establish the method's behavior; they are
not necessary to calibrate a new model.

## Applying an exported map

Inside a compute-node job, load the **same pristine model weights** and call:

```python
import json
from analyze_task_sensitivity import apply_type_map

with open("results/my_task_calibration/qwen3-4b/type_map.json") as f:
    type_map = json.load(f)
apply_type_map(model, type_map)
```

The model should use this repository's quantized model modules with activation
dtype `nvfp4_4over6`, 4 activation bits, and activation group size 16. Evaluation
in this study disables the KV cache. `apply_type_map` rewrites weights offline;
it must not be called on weights that were already quantized. It checks source
revision when available, module names, tile shapes, and index validity.

Each module entry gives a grid shape `(output_channels / 8, input_channels / 64)`.
Flat indices use row-major ordering of that grid. Missing module entries mean
all-E2M1 NVFP4 for that matrix; the language-model head remains unquantized, as
in the existing evaluation harness.

## What is and is not predicted

Tensor statistics alone cannot provide a universal task-level ordering. Even
for a scalar squared-error task, the same two quantized weights q0 and q1
reverse their ordering when the target changes from q0 to q1. The weight
tensor and its quantization errors are identical in both tasks. A useful
general procedure therefore needs information about inputs and downstream
loss, not just a universal preference for a weight distribution.

Here the proposed loss change is the sum of STE gradient times tile delta.
The difference between actual and proposed loss change includes finite-step
interactions AND error from differentiating through discrete activation
rounding. It is not justified to call this residual only a smooth Hessian
term. The 64-sequence Llama seed 20260907 proposal predicts -0.1 NLL but
changes fit NLL by +0.04053; extra fitting data alone does not repair it.

An exploratory `--backtrack-proposal` option calibrates the finite step on
FIT data: halve the 0.1 budget until the actual fit improvement is at least
25% of prediction and fit mean + 2 SE is negative, allowing at most eight
attempts. Independent validation still checks the resulting map once.
This adds bounded calibration forwards and is not a purely analytical
performance predictor. See the protocol and measured follow-up before
preferring it to the simpler one-proposal procedure.

The two 64-sequence Llama development runs selected budgets 0.05 and 0.025
automatically, requiring two and three fit evaluations. Both then improved
WikiText and C4, after the original 0.1 proposals had harmed both datasets.
This is useful adaptive calibration, but still evaluates a few finite
proposals; it must not be described as predicting performance from tensors
alone or eliminating all candidate evaluation.

With all constants frozen, a third calibration seed improves WikiText by
0.9432 PPL on Qwen3-4B and 0.0589 on Llama-3.1-8B. C4 changes by -0.4417
and +0.0045 respectively; the latter is inconclusive (delta NLL +0.00053,
SE 0.00124). Thus the calibration procedure repeats on both development
models, while transfer of its improvement to another text domain is not
guaranteed. The other four model results use the original 16-sequence rule,
not the later 64-sequence backtracking procedure.

To exercise this path, use a new output directory:

```bash
sbatch --qos=normal slurm/task_sensitivity.sbatch llama-3.1-8b-local \
  --fit 64 --backtrack-proposal --calibrate-only --validate-selected \
  --out results/my_task_calibration/llama-3.1-8b-backtracked
```

The raw derivative is a local approximation. It identifies strong beneficial
and harmful interventions in the measured probes, but does not predict their
finite magnitudes accurately. Selecting every individually negative tile fails.
The fixed budget limits this extrapolation empirically; 0.1 is a development
choice, not a theorem or a per-model optimum.

The optional validation check supplies a more useful numerical forecast for
the chosen map. Its uncertainty describes variation across calibration
windows; it cannot guarantee behavior on a different text distribution.
Final test sets are never used to choose tile identities or thresholds.
