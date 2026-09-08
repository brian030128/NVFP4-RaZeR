# Frozen tile-branch GPTQ test

Sixth study, targeting a limitation of the initial compensation experiment:
that implementation quantized64 columns together and compensated only future
tiles. This test performs columnwise GPTQ compensation inside each64-column
tile before deciding the format. No result from this study may select a
dataset-specific policy.

Shared single32x512 Wiki training pass per model. H=E[xq xq^T] from pristine
model trajectories with local FourOverSix input quantization. Damping is fixed
at1% mean diagonal. U=chol((H+damping I)^-1), upper triangular.

At each64-column step, simulate two branches: one entirely E2M1 and one
entirely E0M3. Recompute the FP8 scale at each16-column group from the current
compensated values; E2M1 uses canonical FourOverSix normalized-MSE scale4/6
election, E0M3 uses alpha1. Within each branch, quantize one column at a time,
and compensate the remaining columns using U. All values are dequantized BF16.
Keep one format for all8rows x64columns, and propagate its chosen innovations
to future tiles. Preserve the original global tensor scale.

The conditional criterion sums squared normalized GPTQ innovations. Its sum
telescopes exactly to the final regularized weight reconstruction error.
Compare against E2M1-only columnwise compensation and dynamic MSE election
between the same two complete branches. All three policies use the same
backend. Also compare against uncompensated canonical FourOverSix.

Models: Llama1B, OPT350M, Qwen0.6B. Fresh Wiki test windows160:192 and GSM8K/MBPP
indices176:208, at most512tokens. Full-model W4A4 reference-text NLL; all weights
frozen and saved before evaluation. Same fixed procedure for all models.

Screen: >=0.01PPL improvement over FourOverSix in>=7/9cells; no supported harm
against FourOverSix or compensated E2M1; point improvement over dynamic MSE
in>=6/9cells. Paired two-SE intervals are descriptive. Passing does not establish
paper readiness. No configuration/scale/order/damping sweep is allowed.

GPTQ is prior work: https://arxiv.org/abs/2210.17323. Mixed-format GPTQ also
exists: https://arxiv.org/html/2605.31035 (Appendix C). This tests a complete
legal-tile branch decision, not a claimed new GPTQ algorithm or theorem.
