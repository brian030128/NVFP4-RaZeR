# Statistical validity report

Protocol SHA-256: `bfee511543475db7fba3eca5d62e16570ca34d800831d055bfd45532492413c1`.

All paired PPL estimands are token-weighted `dlogPPL = mean_NLL(method) - mean_NLL(comparator)`; negative favors the method. Final intervals use 10,000 deterministic document/article-cluster bootstrap replicates unless an artifact explicitly labels an exploratory count. Finite Monte Carlo tests use the plus-one rule, and frozen endpoint families use Holm adjustment.

The complete machine-readable table contains 848 raw PPL rows and 486 paired dlogPPL/CI rows. Each row names its source path, full source SHA-256, and (for nested analyses) JSON pointer.

Calibration-map uncertainty is not folded into an ordinary evaluation-window CI. `CALIBRATION_ROBUSTNESS.json` separately reports each of five maps, between-draw SD/small-sample intervals, within-evaluation bootstrap variance, pairwise map overlap, and hierarchical sensitivity.

## Gate outcomes

| gate | passed | interpretation |
|---|---:|---|
| G0 | True | {"passed": true} |
| G1 | True | {"passed": true} |
| G2 | True | {"passed": true} |
| G3 | True | {"passed": true} |
| G4 | False | {"passed": false} |
| G5 | False | {"continued_to_256": false, "decision_rule_applied": true, "passed": false} |
| G6 | False | {"passed": false} |
| G7 | True | {"passed": true} |
| G8 | False | {"accuracy_passed": null, "passed": false, "quality_passed": false} |
| G9 | False | {"heldout_component_passed": false, "passed": false, "reason": "broad SOTA requires a corrected, scope-matched full-panel superiority family; held-out component alone is insufficient"} |

Development searches (k, aggregation, selector, scale/clipping) are explicitly post-selection evidence; naive winner CIs are not treated as confirmatory. Held-out P71 inference begins only after the exact global configuration and maps were hash-locked.

Raw tables: `FINAL_PPL_TABLE.csv/json`, `CALIBRATION_ROBUSTNESS.csv/json`, `MATCHED_BUDGET_CONTROLS.csv/json`, and `DOWNSTREAM_RESULTS.csv/json`.
