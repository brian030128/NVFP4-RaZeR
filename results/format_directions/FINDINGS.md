# Results of the continued search

Six frozen studies completed; none passed its declared screen. The closest
final panel was conditional selection between complete columnwise-compensated
tile branches: it improved over FourOverSix by at least0.01PPL in seven of nine
model/domain cells and over dynamic MSE in six. The two remaining FourOverSix
comparisons were supported regressions, so the result is not promoted.

| Model | Wiki ΔPPL | Math ΔPPL | Code ΔPPL |
|---|---:|---:|---:|
| Llama-3.2-1B-Instruct | -1.393416 | +0.287529 | -0.375121 |
| OPT-350M | -2.235915 | +0.867600 | -0.603291 |
| Qwen3-0.6B | -3.262640 | -0.614861 | -0.078448 |

These are paired differences from FourOverSix within the sixth study. Negative
means lower reference-text loss/PPL. Math and code are not generation accuracy.
Both positive math differences have paired NLL intervals supporting harm. The
policy also harms one comparison against compensated E2M1. See the complete
[REPORT.md](REPORT.md) for uncertainty and all comparators; results from other
studies use different example ranges and must not be ranked by absolute PPL.

## What changed in understanding

- Exact conditional costs improved the initial held-out layer mechanism panel
  in all27 matrix/domain cases, yet this did not establish W4A4 task transfer.
- Accounting for activation mismatch improved six of six comparisons against
  dynamic MSE but still harmed cases against the stronger FourOverSix control.
- A calibration-free activation rule using consuming weight column energies
  improved only three of nine cases over FourOverSix.
- Binary interaction-aware election reduced median fit reconstruction error
  by roughly21–24% and improved six of nine downstream point comparisons, with
  two supported harms. Several matrices hit the fixed pass cap.
- Teacher-Fisher geometry improved seven of nine comparisons against uniform
  output geometry, but only four of nine against FourOverSix. Better geometry
  within this approximate objective did not solve transfer.
- Compensating inside each tile produced the seven-of-nine final panel above.
  Thus the initial coarse backend was a real limitation worth testing, but its
  replacement was not sufficient to remove domain regressions.

## Research judgment

The strongest evidence concerns mechanisms, not a validated universal rule.
Improving a local quadratic objective is insufficient evidence of task-loss
generalization. The exact update identities and numerical tests establish what
the code optimizes; they do not certify unseen-domain accuracy. No conclusion
that every possible broadly transferable rule is impossible follows.

These are discovery experiments across repeatedly inspected benchmark domains,
despite disjoint example ranges between stages. They do not meet the standard
for a final generalization claim. A defensible positive paper would still need
a locked method on new models/domains, generated task evaluations, competitive
baselines, and a contribution beyond existing GPTQ, KFAC or coordinate descent.

Tests for canonical candidates, tile alignment, exact objectives, telescoping
compensation costs, monotonic coordinate updates, isotropic reductions and
stationarity on test cases passed on Slurm. All final evaluation jobs completed.
Protocols, source hashes, pinned revisions, per-example losses and exportable
maps/weights are retained. No per-domain configuration was elected by these
evaluation results.
