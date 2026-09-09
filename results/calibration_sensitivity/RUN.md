# Execution record

**Superseded by the user's revised scope.** Jobs 333759, 333762 and 333764
were cancelled on request before this study completed. Partial results remain
as an execution record and are not complete comparative evidence. The new
study calibrates only on math/code, evaluates only WikiText/C4, and includes
an adaptive count rule: [replacement protocol](../math_code_adaptive/PROTOCOL.md).

Protocol: [PROTOCOL.md](PROTOCOL.md). No calibration seed replication.

* Preparation: Slurm 333738. Derives all maps from saved scores; checks exact
  reproduction of the existing C4-64, mixed-64 and pooled-192 maps.
* Qwen3-4B / Llama-3.1-8B evaluation: dev array 333759, after preparation.
* Qwen3.8-27B evaluation: dev array 333762, after both smaller models succeed.
  Three sequential two-GPU jobs cover C4, WikiText, and the three transfer
  datasets respectively; each evaluates all 19 policies.
* Aggregation, figures, and MIXFP4_REPORT.md update: 333764, after evaluations.
* Report-table and syntax checks: 333749, completed successfully.

These are submitted jobs, not a claim that the evaluation is complete.
Completed reports require `status: complete` and passing integrity checks.

The initial preparation test job 333702 failed before producing maps: a
synthetic decimal tie differed by a float32 reduction-layout rounding bit.
The tie fixture now uses an exactly representable binary value, preserving
the cross-module tie test. All 17 policies pass against full-table selection,
including eligibility and cap boundaries. Actual stored-map reproduction
remains mandatory and was not relaxed. Its dependent evaluation job 333736
was cancelled. Summary job 333742 was replaced by 333744 before execution
to include the primary pooled-performance table update.

No evaluation results select or eliminate a calibration setting. The existing
256 cap and two-SE score are unchanged. All 17 settings plus two controls are
reported on all five datasets, with separate PPL values and actual E0M3 type
block counts. Sparse maps and source-score hashes are in the prepared JSONs;
per-window losses and reuse provenance are in the model reports.

Preparation 333738 completed successfully in 10m25s. All 51 calibrated maps
select 256 E0M3 type blocks; all nine historical-map reproduction checks pass.
The normal-queue evaluation jobs 333739 and 333740 were held by
`QOSMaxJobsPerUserLimit` because unrelated user jobs already occupied the
normal QoS. They and their dependent summary 333744 were cancelled before
evaluation. The dev replacements preserve every setting and input. Dataset
partitioning only accommodates the two-hour queue limit; merged reports
require disjoint complete coverage and identical model/map/data provenance.
