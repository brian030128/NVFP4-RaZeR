# Frozen output-sensitivity direction

Fifth direction. Interacting reconstruction improved some domains but harmed
OPT math. Change the objective to approximate teacher-output KL sensitivity;
do not tune the reconstruction rule against those evaluation results.

One shared32-window Wiki training pass per model,512 tokens/window. Pristine
BF16 teacher; at each token draw one label from its output distribution using
seed20260917, then backpropagate the mean sampled-label NLL. Ground-truth
calibration labels are unused. Collect H=E[xq xq^T] using FourOverSix local inputs
and G=E[g g^T] in disjoint8-output-channel blocks. All comparators share the
same pass. Normalize G by its overall mean diagonal, not per-output block.

Objective sum_b tr(G_b E_b H E_b^T). This is a block-diagonal, Kronecker-factored
Fisher approximation, not the exact network KL. It omits cross-layer/cross-row
block terms and input/gradient dependence. Teacher gradients are evaluated at
pristine BF16 trajectories. These approximations must be disclosed.

Candidates remain FourOverSix / E0M3-alpha1 from original weights, legal8x64
tiles, fixed original tensor scales. Deterministic binary coordinate descent,
increasing K order, at most8passes and1e-12 relative roundoff guard. No loss
backtracking, gate, domain-specific hyperparameter, or matrix-specific policy.

Comparators: FourOverSix; uniform-output interacting reconstruction; diagonal
output Fisher; full8x8 block output Fisher (proposed). All maps frozen before
fresh evaluation. Same three models as the previous panel. Wiki test windows
128:160, GSM8K/MBPP indices144:176, at most512 tokens/example. Primary W4A4
reference-text NLL. Point gain >=0.01PPL over FourOverSix in >=7/9 cells, no
supported harm (paired delta minus2SE>0), and improvement over uniform-output
reconstruction in >=6/9 cells. No claim of task generation accuracy.

KFAC/Fisher-aware quantization is prior art, e.g.
[BRECQ](https://arxiv.org/abs/2102.05426). The experiment tests whether output
sensitivity is the missing mechanism in tile-format selection; it does not
claim the approximation or coordinate descent is novel. These remain discovery
domains despite fresh example ranges; larger unseen models/domains are required
for final confirmation if the fixed screen passes.
