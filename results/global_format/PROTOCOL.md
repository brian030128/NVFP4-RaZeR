# Frozen whole-network type objective

The preceding local objectives omit interactions between layers and do not
optimize task loss. This direction uses a single whole-network binary objective
and does not backtrack against candidate losses or choose per-domain settings.

All weights initially FourOverSix, all inputs FourOverSix with identity STE for
scoring only. Candidate alternative remains E0M3-alpha1 at8x64 weight tiles.
Keep every candidate value and scale fixed. One shared pass over64 C4 training
windows of512 tokens per model produces A_it, the directional derivative of
mean sequence NLL for tile t. Dataset revision pinned; train shard3, seed20260920,
first64 unique qualifying documents with seeded crop positions. No evaluation
labels, math/code examples, or WikiText calibration are used. All comparator
rules consume the same score matrix; there is no calibration per configuration.

Let T=511 predicted tokens, mu=mean_i A_i. The objective is

    J(s) = mu^T s + T/(2N) ||A s||²,   s in {0,1}^tiles.

T/N A^T A is the empirical Fisher of whole-sequence log likelihood, normalized
per token. It is not the exact Hessian and uses observed labels plus activation
STE. Retain all cross-tile and cross-layer terms implicitly through A. Eligibility
requires mu+2SE<0, a descriptive stability filter, not a multiplicity certificate.
Start at zero. Greedily take the most negative exact single-bit change, allowing
undo, until none remains or4096 steps. The cap is fixed and must be reported if
hit. No hand-selected predicted-NLL budget, damping, rank search or loss gate.

Controls: FourOverSix; diagonal version of the same empirical Fisher; historical
fixed0.1 predicted-NLL budget without backtracking, using the same eligibility.
All maps frozen before evaluation. Save raw scores and maps for replay.

Models Llama1B, OPT350M, Qwen0.6B. Fresh Wiki test windows256:288 and GSM8K/MBPP
indices272:304, max512 tokens. Primary screen: >=0.01PPL gain vsFourOverSix in
>=7/9 cells, no supported harm, and point improvement over the fixed-budget
control in>=6/9. Paired2SE intervals are descriptive. These remain development
domains, so passing requires a locked confirmation on additional models/domains.

Cross-layer quadratic quantization is prior art, e.g. CLADO
(https://arxiv.org/abs/2307.05657). The experiment tests fine-grained type choices
using a shared low-rank task objective. No theorem or novelty claim follows
merely from empirical Fisher or binary coordinate descent.
