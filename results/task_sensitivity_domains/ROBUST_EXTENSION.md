# Prespecified extension: domain-consensus selection

Added before running the domain-transfer protocol. The three original policies,
data budgets, and diagnostic categories remain unchanged. Add a fourth policy
using the same 32+32 fitting windows as Mixed. A tile is eligible only if its
gradient mean plus two SE is negative in BOTH domains. Rank eligible tiles by
their worse (less negative) domain mean and cap summed predicted reduction at
0.1 NLL. Backtrack at most eight times, halving that budget. The actual joint
proposal must achieve at least 25% of predicted reduction and mean+2SE < 0
in EACH fitting domain. Once-only validation must also pass in each domain
(eight independent windows each), otherwise export the unchanged baseline.

Hypotheses, before measurements:

1. Domain-specific score signs and successful C4-only calibration would explain
   weak WikiText-to-C4 transfer through distribution dependence.
2. Wrong-sign isolated interventions or poor fit actual/predicted ratios would
   identify surrogate error and finite-step interactions as additional causes.
3. Mixed calibration may recover complementary corrections in both domains.
4. Consensus eligibility plus per-domain finite-step checks may sacrifice
   in-domain gain to improve worst-domain behavior. It can also select nothing;
   a fallback is not evidence of a useful universal quantizer.

All four candidate maps, including rejected ones, receive the identical held-out
evaluations. Exported results use the baseline for rejected maps. No evaluation
result chooses thresholds, seed, or policy. Two domains on one model establish
neither broad-domain generalization nor a universal guarantee. This experiment
is type-only, with fixed FourOverSix E2M1 and alpha=1 E0M3 candidates.
