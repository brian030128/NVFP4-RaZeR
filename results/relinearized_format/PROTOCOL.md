# Frozen relinearized common-descent study

The directed-Fisher fit audit shows that a single baseline quadratic can fail
its own actual teacher-KL objective. This study changes the optimization
mechanism, rather than fitting another covariance estimator.

Use the same fixed FourOverSix/E0M3-alpha1 candidates and legal 8x64 tiles,
native BF16 nonhead-linear W4A4 scope, and one shared 64x512 C4 training set
from shard3. No dataset or checkpoint is chosen separately for a configuration.
Only binary format gates change; no scales or floating-point weights train.

At the CURRENT hard-quantized network, compute paired per-sequence gradients
for ground-truth next-token CE and KL(BF16 teacher || quantized student).
For each legal flip (including undo), multiply by its signed weight change.
Define U_j=max(mean(CE_j)+2SE(CE_j), mean(KL_j)+2SE(KL_j)). Flip exactly the
one tile with smallest U_j if negative; otherwise do nothing. All forward
passes use legal hard types. Backward uses an identity activation STE.

Recipe: 256 updates, four documents per update; cycle through the same64
documents in a fixed seeded permutation each of16 epochs (seed20260924).
No learning rate, candidate-loss backtracking, validation acceptance gate,
checkpoint selection, or adjustment after test results. The final iterate is
the exported map, including a poor one. Two SE is an exploratory noise filter,
not a multiple-testing certificate, and finite tile changes need not decrease
either actual loss. Cycles are recorded and not hidden as convergence.

First collect scores once at the unchanged baseline on all64 documents. This
shared pass supplies a stale256-tile common-descent control and a historical
CE-only fixed0.1 predicted-loss budget control. Other controls are FourOverSix
and fixed-candidate weight-MSE election. All reuse identical candidates,
training examples and evaluation inputs; iterative method has more compute,
which must be reported. This is an ablation of relinearization, not a claim
that all controls use equal compute or a new binary-optimization invention.

Freeze maps before evaluating actual fitting CE/KL (an audit, no feedback) and
fresh Wiki test windows352:384 plus GSM8K/MBPP test rows368:400,32 per domain.
These are development domains after repeated earlier studies, not sealed
confirmation. Screen: >=0.01PPL improvement vsFourOverSix in>=7/9 cells,
no supported harm; lower point NLL than stale256 and weight-MSE in>=6/9 each;
actual fitting CE and KL both below baseline on all three models. Paired2SE
intervals are descriptive. Passing requires a separate new-model/new-domain
confirmation with the recipe unchanged before claiming generalization.

Related work includes Bop (https://arxiv.org/abs/1906.02107) and differentiable
mixed-precision quantization (https://arxiv.org/abs/1905.11452). Repeated binary
updates and STE are prior art. A useful empirical result here would not by
itself establish paper-level novelty or a universal weight-only rule.
