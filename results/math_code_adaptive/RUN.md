# Math/code-only adaptive study: execution record

The user restricted evaluation to WikiText and C4 and excluded both sources
from calibration. The preceding unfinished three-source/five-dataset jobs
333759, 333762 and 333764 were cancelled; their partial artifacts are retained
separately and cannot update the report as complete results.

Protocol: [PROTOCOL.md](PROTOCOL.md). No seed replication. All ten settings
are derived from one shared causal OpenWebMath/CodeParrot scoring pass per
model. The fixed-256 rule is a labeled comparison using identical inputs.

Jobs:

* 333779: Qwen3-4B and Llama-3.1-8B calibration, completed successfully.
* 333786: the two smaller models evaluated only on WikiText and C4.
* 333787: Qwen3.8-27B math/code-only calibration, after smaller evaluation.
* 333788: 27B evaluation in fixed C4 / Wiki-adaptive / Wiki-fixed groups.
* 333970: validation, merge, separate PPL/count tables, and figures in MIXFP4_REPORT.md.

Continued September 9: both smaller evaluations completed successfully
(35m18s and 34m48s). Qwen3.8-27B calibration completed in 31m50s and selected
1–8 adaptive blocks across the ten settings. Check job 333951 passed the
report/merge tests and validated every completed smaller-model cell, including
PPL reconstruction, per-window contrasts, map hashes, and control provenance.
The completed subset is in [completed_small_333786/REPORT.md](completed_small_333786/REPORT.md).

The 27B evaluation array throttle was raised from one to two concurrent
dataset/policy groups to shorten waiting time without adding evaluations.
Slurm returned a nonspecific error while applying the array-wide update, but
`scontrol show job` verified ArrayTaskThrottle=2 and both C4 and Wiki-adaptive
jobs running. Each group still uses two H100s and its fixed inputs/maps.

Summary job 333790 was replaced before execution by 333970 to install
worker-local plotting dependencies and generate the figures. The replacement
depends on the active 27B array; the already-completed smaller array had
expired from Slurm's live dependency registry, and its outputs were separately
validated. Aggregation also performs a labeled post hoc calibration-only
curvature audit on existing maps and records hashes of the saved score files.
It does not select new maps or use target losses for calibration.

Calibration 333779 passed adaptive-prefix/PSD-bound and causal quantization
tests. Both models completed exactly 128 CE/KL/sampled-label score rows per
module with finite values. Frozen adaptive counts are 0–1 for Qwen3-4B and
1–2 for Llama-3.1-8B across the ten settings; fixed comparisons select 256.
These counts show a conservative penalty and are not evidence of PPL gains.
No destination results were loaded during scoring or count election.

## Completed study

All calibration and inference jobs completed successfully. The three 27B
evaluation groups finished in 54m04s (C4), 1h08m49s (Wiki adaptive), and
1h00m49s (Wiki fixed). Summary job 333970 completed in 19 seconds and validated
all 132 separate model/policy/dataset cells: 30 adaptive maps, 30 fixed-256
maps, and six controls, each evaluated on two datasets. Matching frozen maps,
source hashes, evaluation-window hashes, causal-prefix checks, PPL
reconstruction, and paired contrasts passed. Report/merge tests also passed.

The adaptive rule selected 0–8 blocks and beat fixed-256 point PPL in only
1/60 matched comparisons. Fixed-256 improved baseline point PPL in 57/60
cells, with 48 descriptive two-SE-supported improvements and no
two-SE-supported harms. These are correlated comparisons, not seed repeats
or independent confirmations. This adaptive surrogate is not a competitive
replacement; no setting or penalty was retuned from these target results.

Calibration-only diagnostic job 334102 completed in three minutes. It
reconstructed all adaptive objective terms and checked the estimated PSD
inequality. On fixed maps the sampled quadratic penalty was only about
1.9–12.2% of the conservative majorizer across the models and settings. This
post hoc diagnostic changes no maps and does not measure actual finite-switch
network loss. Score-file hashes are recorded in the audit artifact.

Summary refresh job 334142 completed successfully in 18 seconds, fixing the
figure legend layout and adding the negative result explicitly to both
reports. It reused completed measurements and the cached curvature audit;
it performed no model inference. The regenerated figure was visually checked.

See [complete tables and paired results](summary_333786_333788/REPORT.md),
[all separate-dataset cells](summary_333786_333788/cells.csv), and
[curvature audit](summary_333786_333788/CURVATURE_AUDIT.md).
