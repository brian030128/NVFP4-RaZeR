# A guarantee for capturing meaningful gains

## The requested property

Let R0 be expected loss of the common baseline, Rj the loss of reference
configuration j, and R* the loss of our selected map. For a prespecified
retention fraction rho in (0, 1], the desired inequalities are

    R* <= (1-rho) R0 + rho Rj, for every reference j.

Whenever configuration j improves the baseline by Gj = R0-Rj > 0, this says
our map captures at least rho Gj. Include the baseline itself as a reference
to additionally require R* <= R0. This formalizes gain retention without
requiring the best configuration. It must be checked separately for each
declared workload distribution or metric; gains can conflict across workloads.

## A model-independent finite-sample certificate, with assumptions

Freeze the selected map and K reference configurations before inspecting an
independent certification set of n IID examples from the deployment
distribution. Suppose the declared loss is bounded in [0, B]. For each
reference calculate the per-example difference

    Zj = loss(selected) - (1-rho) loss(baseline) - rho loss(reference j).

Zj is in [-B, B]. Hoeffding's inequality and a union bound give simultaneous
upper bounds, with probability at least 1-delta,

    E[Zj] <= mean(Zj) + 2 B sqrt(log(K/delta)/(2n)).

If all these upper bounds are <= 0, all the requested gain-retention
inequalities hold simultaneously with that confidence. The statement is
independent of architecture, weights, and the algorithm used to construct the
candidate. It is conditional on independence, the loss bound, and the same
deployment distribution. Multiple certified candidates or repeated looks need
additional multiplicity control or an appropriate sequential procedure.

This is a direct paired-loss application of risk control through hypothesis
testing; see [Learn then Test](https://arxiv.org/abs/2110.01052) and
[risk-controlling prediction sets](https://arxiv.org/abs/2101.02703).
The gain-retention contrast and elementary Hoeffding bound above are our
derivation, not a claim that those papers prove MixFP4 performance.

The certificate can be conservative. Failure to certify is inconclusive,
not evidence that a gain is absent. Clipping NLL to obtain B changes the
guaranteed metric; it cannot silently be presented as a raw-perplexity guarantee.
Sample extrema are not valid universal loss bounds. The existing contiguous
WikiText windows and 2-SE estimates do not meet this certificate automatically.

## Why strict unconditional dominance cannot be universal

For any context x, let p and q be two models' normalized next-token
probabilities. If they differ, some token y has q(y|x) < p(y|x): otherwise q
would be at least p everywhere and strictly greater somewhere, contradicting
that both sum to one. On a distribution concentrated on this (x,y), q has
strictly worse log loss. Conversely, some token improves. Thus no changed
predictive distribution can strictly improve log loss for every possible
context/label distribution. Leaving predictions unchanged guarantees equality,
not a meaningful strict win.

The obstruction applies to every quantizer, including FourOverSix and MixFP4.
It does not rule out a common calibrated rule working across many models on
a specified task distribution, nor the conditional certificate above.

The same obstruction applies directly to partial gain retention. Consider
two tokens, baseline probabilities (0.5, 0.5), and reference configurations
(0.9, 0.1) and (0.1, 0.9). Retaining any fraction rho > 0 of the first
configuration's gain on token 1 requires
q1 >= 0.5^(1-rho) * 0.9^rho > 0.5. Retaining the second configuration's gain
on token 2 likewise requires q2 > 0.5. No normalized fixed predictor can
satisfy both for every possible label distribution. This illustrative
counterexample concerns the requested universal claim, not an assertion that
these exact probabilities were observed in the MixFP4 experiments.

Without a known bound or tail assumption, finite observations cannot certify
the mean of arbitrary raw NLL differences: a rare unseen event can have an
arbitrarily large adverse loss and reverse the expectation. Observed small
variance alone does not exclude that event.

## Relation to the underlying tensor rule

The task-gradient dot weight-change score estimates a configuration's marginal
benefit at the CURRENT baseline. Changing that baseline from NVFP4 to
FourOverSix can remove a previously useful correction or alter its sign.
Therefore we score complementary corrections after preserving the baseline's
existing gains. This explains why simply unioning independently good maps
does not come with an additive performance guarantee.

If an additive surrogate had a proven uniform error epsilon over a candidate
family, its exact minimizer would have actual loss at most 2 epsilon above
the best family member. Retaining fraction rho of a best gain G would follow
when 2 epsilon <= (1-rho)G. We do not currently have such a uniform error bound:
finite tile interactions and activation-rounding discontinuities invalidate
an unsupported smooth-Hessian argument. Measuring empirical agreement is
useful evidence but is not that proof.

## A genuinely distribution-free alternative with a deployment cost

There is an unconditional sequence-regret guarantee if the predictor may mix
the FULL predictive distributions of K configurations online. With prior
weights pi_j > 0 summing to one, Bayesian prediction has sequence probability

    P_mix(y_1:T) = sum_j pi_j P_j(y_1:T).

Since this is at least pi_j P_j for each j, its cumulative log loss satisfies

    L_mix <= L_j + log(1/pi_j), simultaneously for every j and every sequence.

Uniform priors give overhead log(K)/T in average token NLL. Relative to a
baseline's positive average gain G_j, it captures at least
G_j-log(K)/T; fraction rho follows when log(K)/T <= (1-rho)G_j.
This needs no IID assumption or bounded log loss. Predictions use posterior
weights based on past observations, never future labels. It is the classical
Bayesian prediction guarantee; see
[Bartlett, Online Prediction](https://www.stat.berkeley.edu/~bartlett/papers/b-op-16.pdf).

This does NOT certify a single static 8x64 type map. It generally requires
evaluating multiple full configurations to produce each predictive mixture.
Mixing weight tiles is not mixing normalized output probabilities. Distilling
or approximating the mixture would require a separate approximation bound.
It is a mathematical comparison point, not a replacement for the requested
efficient MixFP4 method or a claimed deployment result.
The sequence-probability interpretation also requires genuinely causal expert
predictions. The existing fake-quantized prefill evaluation computes dynamic
activation tensor scales over a whole window, which can include later token
positions. Stored teacher-forced loss totals alone therefore do not establish
a normalized causal sequence likelihood or a deployed online-mixture result.
Any algebraic mixture calculation on those totals is labeled diagnostic only.

## Exact local geometry explains the need for input information

For a single 8x64 tile, let E0 and Eb be its reconstruction-error matrices
relative to pristine weights. A switch improves squared linear-output error
for every input vector x exactly when

    A = E0^T E0 - Eb^T Eb is negative semidefinite.

This follows from the identity error_change(x) = x^T A x. A positive largest
eigenvalue supplies an explicit adverse input: its eigenvector. Moreover,
the condition requires ker(Eb) subset ker(E0), hence rowspace(E0) subset
rowspace(Eb). Two generic different 8x64 rounding-error matrices have different
low-dimensional row spaces, so this strong input-universal condition is rarely
available. This is a local linear-output statement, not a language-loss theorem.

If calibration gives the second moment S0 = E[x x^T], expected local error
change is tr(A S0). For any unknown second moment S satisfying
||S-S0||_op <= eta,

    tr(A S) <= tr(A S0) + eta ||A||_*.

Therefore a negative right-hand side is a sufficient local certificate over
that covariance neighborhood. The nuclear norm is the sum of absolute
eigenvalues of symmetric A. The measured safe radius -tr(A S0)/||A||_* is
informative when positive, but estimating S0 does not itself certify eta for
deployment. Downstream nonlinear layers and other changed tiles also remain
outside this single-tile reconstruction certificate.
The certificate concerns an isolated tile's output reconstruction; it excludes
cross terms with reconstruction errors in other K tiles of the same output.

The mechanism job measures these matrices on the union of the old and new
selected tiles, computes explicit adverse-input witnesses, and records the
calibration second-moment margin. It does not rename this local margin a
universal perplexity guarantee.
Both candidate geometries use the same recorded NVFP4-reference input second
moments, explicitly labeled in the output. No assumption is made that
FourOverSix's actual internal activation distribution is identical.
