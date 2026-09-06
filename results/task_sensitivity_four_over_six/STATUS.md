# Execution status

Completed: Slurm 329279 ran target seeds 20260912/20260913 sequentially on
two H100s; 329286 ran mechanism ablations and exact FourOverSix export replay;
array 329289 ran four smaller models at most two one-H100 tasks concurrently;
329294 ran regression tests and generated REPORT.md. Every job exited zero.
All study jobs have finished. See FINDINGS.md for interpretation and
GUARANTEE.md for the precise mathematical claims and assumptions.

Seed 20260912 completed: 226 tiles, fit and independent validation accepted at
the original 0.1 predicted-NLL budget. WikiText FourOverSix 7.289580 -> 6.973420
PPL (delta NLL -0.044340, SE 0.003813). C4 9.117679 -> 9.118835 PPL (delta
NLL +0.000127, SE 0.000954; inconclusive). Relative to plain NVFP4, the C4 point
estimate retains about 95% of FourOverSix's improvement.

The prior NVFP4-based maps also beat the matched FourOverSix WikiText baseline:
7.133067/7.156911 versus 7.289580. Combining FourOverSix with recalibrated type
switches improves further in both completed seeds. Seed 20260913 selected
327 tiles and passed validation: WikiText 7.289580 -> 6.935030; C4
9.117679 -> 9.111269 (C4 difference remains inconclusive).
All four transfer models passed validation and improved WikiText; three
improved C4 beyond their descriptive two-SE intervals, with base Llama
inconclusive. No configuration was chosen using test performance.

Guard incident: the host guard log records `codex` PID 1062432 reaching 300
seconds cumulative CPU and receiving TERM at 2026-09-06 13:58:13. It identifies
the long-lived client process, not an experiment Python process. The Slurm
experiment continued. The guard configuration and allowlist were not changed.
