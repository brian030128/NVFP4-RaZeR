# Frozen description-cost sparsity rule

Earlier sparse studies used either256 tiles or a0.1 predicted-loss budget.
Neither number follows from the number of opportunities to fit noisy scores.
Replace that choice by one fixed prior and retain all other scoring details.

For D legal8x64 tiles, set independent prior switch probability1/(D+1).
The expected prior number of switches is D/(D+1), approximately one. This is
an explicit sparse design assumption, not a uniquely correct universal prior.
Relative to all FourOverSix, a map with k switches costs k log(D) nats.

Use the existing shared64x512 C4 CE/KL score table from job332316, whose maps
and source hashes are retained. No additional calibration pass, source search,
fitting-loss acceptance, or checkpoint selection. Let U_j be the maximum of
the CE and teacher-KL directional mean+2SE. With N=64*511 scored tokens,
switch tile j iff U_j + log(D)/N < 0. No tile cap, budget coefficient or
model-specific threshold. This exactly minimizes an ADDITIVE surrogate with
the specified description penalty. TwoSE is not a simultaneous certificate.

The actual network objective is nonlinear in the type bits and the backward
pass uses identity activation STE. Neither this surrogate optimum nor a sparse
prior proves actual KL/NLL reduction or arbitrary-domain generalization.
The prior calculation is classical description-length/Bayesian regularization,
not claimed as a new theorem. Whole-window activation tensor scaling also
means current reference-text losses are not a causal sequence-code guarantee.

Controls reuse the same recorded C4 observations and fixed candidates:
FourOverSix, weight-MSE, stale256 common descent, historical fixed0.1 CE budget.
Replay source weight hashes and fitting token hashes before exporting the new
map. Audit actual CE and KL on the64 inputs only after map freezing.

Fresh development evaluation: Wiki windows416:448, GSM8K/MBPP rows432:464,
32 examples per domain, at most512tokens. Models Llama1B, OPT350M, Qwen0.6B.
The aim is conservative retention, not a claim to beat the larger map in every
cell: require >=0.01PPL gain over FourOverSix in>=7/9cells, no supported harm,
lower point NLL than weight-MSE in>=6/9, and both actual fitting losses lower
on all three models. Additionally retain at least half of the stale256 map's
positive baseline NLL gain in every cell where that gain is supported by its
own paired2SE interval. This retention condition is descriptive, not certified.
No choosing which model, seed or domain to exclude. Any failed requirement
is preserved. A pass requires unchanged-method new-model/domain confirmation.
