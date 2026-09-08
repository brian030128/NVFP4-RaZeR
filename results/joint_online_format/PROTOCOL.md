# Frozen joint weight/activation error direction

No calibration corpus. All weight values stay at canonical FourOverSix.
Choose between FourOverSix and E0M3-alpha1 activation candidates in16x64 tiles.
One increasing-K sweep, with the numerical guard from the prior online study.

Exact objective: ||Aq Wq^T - X W^T||² on the current live input X. The initial
residual is (Q0(X)-X)Wq^T + X(Wq-W)^T. The added term permits activation-format
choices to compensate existing weight error. This compares to the original
linear operation on the current input, not a pristine full-network trajectory.
For a tile change D, accept when 2<R,DWq^T>+||DWq^T||²<0 and update R.

Compact representation: output basis U from rank-r randomized SVD of Wq, two
iterations, seed20260918. Store F=U^T Wq, B=U^T(Wq-W), and residual per-channel
Gram entries d,c,e for Wq and Wq-W. Approximate joint error by
||(Aq-X)F^T+XB^T||² + sum_j[d_j E_j²+2c_j E_j X_j+e_j X_j²].
Clamp d,e nonnegative and c within sqrt(de) for numerical PSD consistency.
Select r=min(16,(O//64-3)//2), lower-bounded0; float32 metadata is capped at
1/8 of ideal packed W4 payload for the tested O>=192. No accuracy-driven rank.

Comparators: FourOverSix, activation MSE, exact activation-only output error,
exact joint output error, compact joint output error. All source weights,
metadata, parameters and comparators fixed before evaluation. The exact rules
retain expensive dense matrices and are mechanism controls, not deployments.

Models Llama1B, OPT350M, Qwen0.6B. Fresh Wiki test windows224:256 and GSM8K/MBPP
indices240:272, up to512 tokens. Primary compact screen: >=0.01PPL gain vs
FourOverSix in>=7/9 cells, no supported harm (paired mean minus2SE>0), and point
improvement over activation MSE in>=6/9. Exact joint is separately screened by
the same criteria and must beat exact activation-only in>=6/9 to support the
joint-error mechanism. No switching between policies per model/domain.

Reference-text losses, not generated task accuracy. These are discovery
domains; success must be confirmed on untouched models and additional domains.
Standard quadratic identities and low-rank approximation are not novelty claims.
