# Execution status — complete

All six experiment reports and stress.json have `complete: true`. The summary
job produced the complete 80-cell REPORT.md, CSVs and comparison.png.
See FINDINGS.md for conclusions and the tested selection rule. Completion is
verified from durable artifacts; the later sandbox blocked a live scheduler
socket query, which was not needed to read these completed results.

Current H100-only job chain:

| Job | Work | Resource request |
|---|---|---|
| 329519 | Qwen3.8-27B, two calibration seeds, four policies and tile probes | 2 H100 |
| 329520 | Frozen target maps on math/code text | 2 H100, after 329519 |
| 329805 | Qwen3-4B / Llama-3.1-8B transfer panel | 2 H100, one per independent model, independent of target |
| 329806 | Teacher-KL hypotheses on the two native panel models | 2 H100, one per model, after 329805 |
| 329807 | Score agreement, paired CSV tables, report and plot | 1 H100 allocation for CPU work, after 329806 and 329520 |

All caches and temporary packages use job-specific worker `/tmp`; durable
outputs go here. No heavy compute runs on the login node. No H200 requested.
The independent model panel can overlap the target experiment, allowing up to
four H100s concurrently. Its two workers share one Slurm job slot and use
separate allocated GPUs and job/worker-specific caches. This changes scheduling
only, not experimental scope. Slurm's per-user job cap has delayed its start.

The first setup attempts are documented in RUN_NOTES.md. Native baseline
reproduction succeeded; the placement optimization was rejected by an exact
gradient-score check and is disabled. No scientific result uses its scores.

Both target reports, stress.json, both panel reports, and both teacher reports
are complete, and summary generation succeeded. Raw rejected proposals are
retained; their candidate gains must not be credited to baseline exports.

User interpretation criterion: an absolute PPL reduction of 0.01 is worthwhile.
Final tables and plots emphasize absolute ΔPPL and count gains meeting that
threshold separately from paired uncertainty. This reporting preference does
not retroactively change frozen calibration or export decisions.
