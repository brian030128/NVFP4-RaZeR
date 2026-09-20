# Published measurement snapshots

These JSON reports preserve the measured PPL results, exact-compaction checks, and later development/confirmation outcomes. INDEX.json records each source path and SHA-256. No model weights, tensor caches, credentials, or job log tails are included.

- `measured_both_ppl/report.json`: measured raw, row-only, and both-axis policies under the released evaluation protocol.
- `confirmed_rows_ppl/report.json`: separately confirmed compacted row-only policy.
- `exact_both_compaction/report.json`: bitwise preservation of the best both-axis quantized weights.
- Other folders: bounded rotation and curvature-aware experiments. Their CE/KL values are calibration/development/confirmation losses, not new WikiText or C4 PPL.
- `fisher_subset_validate_v2_confirm/report.json`: the final218tile candidate failed its preset fresh two-SE gate; no PPL was run.

The accepted endpoint is the previously measured212tile both-axis result. The90% recovery target was not reached; the user accepted this endpoint on2026-09-20.
