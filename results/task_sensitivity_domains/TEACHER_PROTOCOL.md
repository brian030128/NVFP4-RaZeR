# Teacher-probability hypothesis

Added after the first target 32+32 fitting-score diagnostic, before any new
held-out evaluation. C4's estimated squared-SE / squared-mean norm is 0.820,
versus 0.156 on WikiText, and cross-domain cosine is 0.0305. These descriptive
numbers suggest testing sampling noise, not assuming every differing sign
represents a stable domain conflict.

For student next-token probabilities q, full-precision teacher probabilities p,
and observed next token y, the logit gradients are respectively q-p for
KL(p||q), and q-onehot(y) for observed-token NLL. The teacher objective removes
the observed-token residual p-onehot(y). Hypothesis: this gives a more stable
estimate of quantization damage and transfers better across text domains.
The teacher need not be perfectly calibrated, so improved fidelity does not
guarantee improved actual NLL; the experiments must check both.

Test Qwen3-4B and Llama-3.1-8B using the EXACT same 32 WikiText + 32 C4 fitting
windows and eight + eight validation windows as their native CE panels. Cache
unquantized full-precision teacher logits for these 80 windows, without using
held-out test text. Quantized weights, activations, candidate formulas, type
tiles and original scoring backend are unchanged. Two policies reuse the
existing pooled and per-domain consensus algorithms, replacing the fitting
objective with teacher KL in nats/token. The same 0.1 budget, eight halvings,
25% actual/predicted check, and two-SE fit checks remain fixed.

Independent acceptance still uses ACTUAL next-token NLL, with the same gate
as its CE counterpart: pooled validation for teacher_mixed and a check in each
domain for teacher_consensus. Also report validation KL, but it does not
override that acceptance decision. No NLL fitting result chooses a step.
Evaluate all candidates and baseline fallbacks on the existing four held-out
domain panels; no held-out result updates a proposal or threshold. Include
direct paired teacher-versus-CE contrasts on the same model and examples.

This is a follow-up motivated by fitting diagnostics, not an untouched-model
claim. It runs after the native CE model panel, still on at most two H100s.
Teacher logits live only in worker memory/cache and are discarded with the job.
