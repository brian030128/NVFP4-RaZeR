# First-seed observations before held-out evaluation

These are calibration findings from target seed 20260912, not final transfer
results. All four candidate maps passed their declared validation gates.

| Rule | E0M3 tiles | Final fitting budget | Wiki validation ΔNLL | C4 validation ΔNLL |
|---|---:|---:|---:|---:|
| Wiki | 226 | 0.1 | -0.039138 ± 0.023478 | +0.000578 ± 0.002368 |
| C4 | 905 | 0.0125 | -0.017404 ± 0.016945 | -0.007188 ± 0.003862 |
| Mixed | 728 | 0.1 | -0.067102 ± 0.049697 | -0.002752 ± 0.004560 |
| Consensus | 1,227 | 0.0125 | -0.026758 ± 0.023195 | -0.006144 ± 0.004327 |

Uncertainty is two paired standard errors, not simultaneous confidence bounds.
Single-domain rules use 16 validation windows per reported domain; mixed and
consensus use 8. Wiki/C4 acceptance uses the named domain, mixed acceptance the
pooled validation set, consensus each domain separately. Cross-row uncertainty
and magnitudes are therefore not a matched comparison; use final held-out
contrasts for that comparison.

## Finite-switch prediction is a concrete failure mode

C4's initial 53,859-tile proposal predicted ΔNLL -0.1000 but measured +0.09948
± 0.01330 on the very same fitting data. Halving the budget to 0.05 still
worsened fitting loss; 0.025 gave a small improvement but only 11.6% of its
forecast. Budget 0.0125 passed with -0.00611 ± 0.00224 and a 48.9% ratio.
This establishes that domain matching alone does not repair the summed
first-order forecast. It does not uniquely separate STE error, curvature and
interactions.

Consensus also needed three halvings. At budget 0.1, its 120,631 tiles had
negative stable individual scores in both domains, but measured C4 change was
+0.00741 ± 0.01642. At budget 0.05, C4 worsened by +0.02251 ± 0.01064.
The accepted 1,227-tile map measured fitting improvements in both domains,
then independently passed both validation checks.

## Score disagreement includes a large noise component

Full 64-window Wiki/C4 score cosine was 0.03794, while the two disjoint C4
32-window halves had cosine 0.21933. Estimated squared-SE / squared-mean
energy was 0.2037 for Wiki and 0.6462 for C4. These descriptive diagnostics
cannot establish incompatible optimal maps. The C4 map's independent Wiki
validation gain is itself evidence against a simple all-or-nothing conflict.

The frozen Wiki map's summed gradient forecast on C4 was -0.000252 versus
-0.09993 on Wiki. These are surrogate predictions, not measured improvements
or whole-map confidence intervals.

Fifteen unique selected tiles were probed individually on reserved 8-window
sets per domain. These small, deliberately selected probes have limited power;
final reporting must include inconclusive results and selection bias.

## Second-seed replication, still before held-out evaluation

Seed 20260913 completed map calibration. Wiki (327 tiles), C4 (864), and mixed
(360) passed their validation gates. Consensus (1,247) passed fitting but was
rejected by independent per-domain validation: Wiki ΔNLL -0.02409 ± 0.03063,
C4 -0.004236 ± 0.004602. Both means are favorable, but both intervals include
zero. Its export is the baseline; the frozen candidate remains in diagnostics.

C4 again required budget 0.0125. Its initial 52,413-tile proposal predicted
-0.1000 but measured +0.09430 ± 0.01247; the accepted fitting map measured
-0.005756 ± 0.001916. Independent C4 validation passed (-0.005253 ± 0.002877),
while Wiki validation was inconclusive (+0.000459 ± 0.020428).
Mixed required one halving to 0.05 in this seed.

The score diagnostic also replicated: cross-domain cosine 0.03334, C4
split-half cosine 0.21765, noise-scale ratios Wiki 0.1610 and C4 0.6443.

Both seeds finished 15 unique isolated probes each. Point-sign agreement was
9/15 and 8/15 on Wiki, 6/15 and 7/15 on C4. No probe showed a stable wrong
sign under the two-SE criterion. Only one and two Wiki effects, and zero C4
effects, excluded zero. Thus these probes are underpowered to establish either
reliable tile-level prediction or reliable domain-conflicting interventions;
roughly half point-sign agreement is not evidence of a 50% all-tile error rate.

Held-out Wiki/C4, math/code LM stress, the smaller-model panel and teacher-KL
follow-up remain necessary before drawing a transfer conclusion.
