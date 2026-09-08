# Frozen predictive-curvature teacher objective

The empirical task-gradient outer product can confound a nonzero systematic
gradient with curvature. This test uses teacher-KL gradients and independent
model-sampled curvature, shared across all policies in one calibration traversal.

Same fixed C4 train shard3,64x512 windows and seed as the global-format protocol.
For each input, run pristine BF16 teacher and FourOverSix W4A4 baseline. Collect
tile derivatives A of per-token KL(teacher||baseline). From the SAME baseline
forward, sample independent labels at each predicted token from its predictive
distribution using seed20260922. A second backward yields per-token sequence
score derivatives B. T=511; T/N B^T B estimates the token-normalized Fisher
under the baseline predictive distribution, with the stated activation STE.
Ground-truth next-token labels are unused for election. No per-policy data pass.

Objective: mean(A)^T s + .5 s^T H s, s binary per8x64 weight tile, candidates
FourOverSix/E0M3-alpha1 fixed at source weights. H=(1-rho)V^T V+rho diag(V^T V),
V=sqrt(T/N)B. Estimate rho as off-diagonal covariance sampling-noise energy
divided by observed off-diagonal covariance energy, clipped to[0,1]. This is
an analytic covariance shrinkage estimator, not an accuracy-selected setting.
No sweep, damping or task-loss gate. Eligibility mean(A)+2SE(A)<0; greedy exact
single-bit updates with undo, numerical tolerance1e-12 and4096-step cap.

Controls: FourOverSix; fixed0.1 KL-gradient budget without backtracking;
diagonal predictive Fisher; unshrunk predictive Fisher. Primary: estimated
shrinkage. All policies use the same scores and are frozen before evaluation.
Report caps and estimated shrinkage, including degenerate rho=0 or1.

Fresh Wiki test windows288:320, GSM8K/MBPP rows304:336, at most512 tokens.
Models Llama1B, OPT350M, Qwen0.6B. Screen: >=0.01PPL gain vsFourOverSix in>=7/9,
no supported harm, and point improvement over fixed-budget and diagonal
controls in>=6/9 each. No winner per model/domain. Descriptive paired2SE
intervals; reference-text loss is not generation accuracy. A screen pass
requires independent new-model/domain confirmation before promotion.

This is a low-rank curvature approximation; the real KL need not decrease.
Cross-layer quadratic quantization, Fisher geometry, teacher matching and
covariance shrinkage are prior ideas, not claimed new mathematical inventions.
