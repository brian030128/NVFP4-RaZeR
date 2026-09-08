# Frozen teacher-error Fisher decomposition

Keep the prior predictive study's teacher, student, source weights, C4 inputs,
64 sequences, sampled-label seed, eligibility and legal8x64 candidates. Change
the curvature estimator to retain an analytically known component; do not
increase calibration data or search settings against evaluation loss.

For one token, teacher probabilities t and baseline probabilities p define
q(y)=t(y)/p(y)-1, chi=sum_y p(y)q(y)^2. Let b(y) be the gradient of negative log
p(y), and g the teacher-KL gradient. Then E_p[b q]=g and

    E_p[b b^T] = g g^T / chi + E_p[(b-g q/chi)(b-g q/chi)^T].

For a sequence with independent conditional label draws, use q_sum across
tokens and chi_sum across tokens. With per-token sequence gradients A_i,B_i
and T=511, retain rows D_i=A_i sqrt(T/(N chi_i)), and sample residual rows
V_i=(B_i-A_i q_i/chi_i) sqrt(T/N). The known D^T D component is retained in
full. Apply the prior analytic diagonal-shrinkage estimator only to V^T V.
For chi=0, define the analytic component as zero and retain the sampled
curvature; an identical teacher/student distribution has zero KL gradient.

Optimize mean(A)^T s + .5||D s||² + .5(1-rho)||V s||²
+ .5rho sum_t diag(V^T V)_t s_t, with binary type switches. Exact greedy bit
updates with undo, tolerance1e-12, maximum16384 steps. Primary convergence
is required for passing; a cap is not silently treated as an optimized map.

Controls: FourOverSix, historical fixed0.1 gradient budget, diagonal of the
decomposed curvature, and standard predictive curvature with all correlations
shrunk. All share the same A/B samples. Standard control gets the same16384
step cap. No actual candidate loss decides maps. Preserve every result.

Fresh Wiki test windows320:352, GSM8K/MBPP rows336:368, up to512tokens.
Models Llama1B, OPT350M, Qwen0.6B. Primary screen: >=0.01PPL gain vsFourOverSix
in>=7/9cells; no supported harm; point improvement over fixed-budget and
standard-Fisher controls in>=6/9 each; primary converged in all models.
Paired2SE intervals descriptive, reference-text loss not generation accuracy.

The decomposition is an algebraic identity under the stated probability model
and Jacobian. In code the Jacobian uses identity activation STE; sampled
curvature and its shrinkage are approximations. The identity does not prove
actual network KL or task loss decreases after discrete changes. Novelty and
new-model/domain confirmation remain separate requirements.
