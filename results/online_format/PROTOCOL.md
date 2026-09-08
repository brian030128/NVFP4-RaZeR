# Frozen online correlated-error study (2026-09-08)

No calibration data. Keep every non-head Linear weight at FourOverSix.
Construct activation candidates FourOverSix and E0M3 alpha1 with the same
original per-tensor normalization, scale groups16, and legal16x64 type tiles.

Exact mechanism: initialize R=(Q0(X)-X)Wq^T at each live layer input. Visit
K tiles in increasing order, exactly once. For a candidate tile change D,
compute V=D Wq^T and accept separately per16-token packet when
2<R,V>+||V||²<0. Update R immediately. Eight float32 epsilons relative to the
absolute linear/quadratic terms form a numerical guard, not a fitted margin.
This monotonically reduces current-layer activation output error, not network
task loss. It uses the actual current inputs, without a calibration corpus.

Compact mechanism: replace Wq^T Wq with R^T R+diag(d). R is a randomized
rank-r SVD factor of Wq, two power iterations, seed20260918. d matches the
remaining diagonal energy, clipped nonnegative. Set r=min(16,O//128-1),
lower-bounded at0. Float32 metadata then consumes at most1/16 of ideal packed
W4 payload for the tested O>=128. Rank is chosen from this storage budget,
never from accuracy. Build and freeze metadata before loading evaluation data.

Evaluate all four policies: FourOverSix activation, plain activation MSE,
exact correlated-output rule, and compact rule. Same weights and examples.
Models Llama1B, OPT350M, Qwen0.6B, pinned revisions. Fresh Wiki test
windows192:224 and GSM8K/MBPP indices208:240, up to512tokens each. Reference-text
NLL; descriptive paired two-SE intervals. Primary compact rule must improve
PPL by>=0.01 in>=7/9 cells vsFourOverSix, with no supported harm, and beat
activation MSE in>=6/9. Exact mechanism uses the same screen independently;
its success does not license claiming the compact rule succeeded.

No per-domain tuning or selection between exact/compact policies. Exact dense
output projections are expensive and serve as a mechanism test. Compact
metadata/storage and unfused runtime are reported. Neither is a production
kernel demonstration or a novelty claim. Final confirmation would require
locked algorithms on additional models/domains and generation tasks.
