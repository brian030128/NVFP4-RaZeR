# What these experiments do and do not establish

The user requested a generalizable method, without calibrating/electing a
different configuration against each evaluation dataset. The studies here use
one frozen algorithm per hypothesis, shared inputs across all comparators, and
no per-domain selection of the best configuration. A shared calibration pass
is distinct from choosing a configuration using downstream test losses.

This is still exploratory research across successive hypotheses. Fresh example
ranges prevent direct example reuse between stages, but repeatedly inspecting
the same benchmark domains informs method development. These domains are not
a sealed final test suite. A successful screen must be followed by confirmation
on larger, previously unused models and additional domains, with the method
locked. None of the current screens alone establishes paper readiness.

## Distinct hypotheses

1. Conditional error compensation: compare formats under the exact Schur
   complement cost when future weights remain continuous. Layer reconstruction
   improved on all27 initial matrix/domain cases. Full-model W4A4 failed the
   declared screen: three of six comparisons improved over dynamic MSE. The
   reconstruction guarantee is an identity for each resulting path, not a
   guarantee that greedy conditional choices beat every competing final path.
2. Asymmetric input-error compensation: account for the mismatch between
   original and FourOverSix inputs in a common least-squares objective. All six
   point comparisons improved over dynamic MSE, but Llama math/code regressed
   against uncompensated FourOverSix. This tested version is not adopted.
3. Consumer-weighted activation election: use consuming weight column energies
   online, with no calibration. Only three of nine point comparisons improved
   over FourOverSix. Cheap diagonal output geometry did not transfer reliably.
4. Interacting fixed weight candidates: optimize exact empirical output error
   of a fully quantized layer by binary tile flips. This removes continuous
   future-weight assumptions and independent-tile scoring. See its frozen
   protocol and generated summary for results; do not infer success merely
   from its monotonic training objective.
5. Teacher-Fisher tile election: replace equal output-channel reconstruction
   weights with a block-diagonal output sensitivity metric. One teacher-label
   sampling/backward pass collects shared Fisher and input factors. No labels
   from evaluation or configuration-specific calibration are used. The
   Kronecker-factor approximation and finite samples still limit the objective.
6. Columnwise compensation within legal tile branches: the first study omitted
   compensation within each64-column tile. This follow-up uses the full
   columnwise update inside both format branches and elects a complete legal
   tile by accumulated conditional cost. It tests that backend limitation
   without varying damping, group size, activation order or calibration corpus.

The objectives in1/2/4 are empirical layer losses, not task losses. All use
pristine model trajectories to collect moments, which do not fully represent
the quantized network's upstream error. Stage3 operates on actual runtime
inputs, but drops cross-channel output-error terms. These are concrete limits
of the mechanisms, not evidence that every possible universal rule is doomed.

## Prior art and scope

Mixed-format GPTQ already appears in
[MixFP4 Appendix C](https://arxiv.org/html/2605.31035).
Asymmetric calibration is existing work, e.g.
[GPTQv2](https://arxiv.org/abs/2504.02692).
Quantization by coordinate descent is existing work, e.g.
[QuantEase](https://arxiv.org/abs/2309.01885).
Fisher-weighted reconstruction also precedes this work, e.g.
[BRECQ](https://arxiv.org/abs/2102.05426).
The standard quadratic identities, Schur complements and coordinate descent
updates are not claimed as new contributions.

A useful broadly transferable rule remains an empirical possibility. A rule
that improves task loss for every possible input/label distribution is a much
stronger requirement; these experiments do not establish such a guarantee.

The interaction/Fisher diagnostics reduced median fit objectives by roughly
21–27%, yet failed the transfer screen. Not every matrix reached single-flip
stationarity within the fixed eight-pass cap: see mechanism_summary.json for
counts. These are results for the frozen finite-budget algorithms, not a claim
that globally optimal reconstruction cannot transfer.

## Measurement limits

- W4A4 here is fake quantization of non-head linear weights and their inputs;
  other operations remain in the model's normal precision.
- WikiText uses equal512-token windows. Math/code use per-example mean
  reference-text NLL, followed by an unweighted mean across examples; their PPL
  is not corpus token-weighted PPL or generated task success.
- 32 observations per cell and paired two-SE intervals are descriptive screening
  evidence, not simultaneous confidence guarantees. Adjacent Wiki windows may
  be correlated.
- Absolute PPL differs across stages because evaluation examples change. Use
  paired within-stage contrasts, not absolute cross-stage scores.
- Compensated-weight artifacts require their own saved weights. Fixed-candidate
  type-only maps can instead be reconstructed from original weights and masks.
- Runtime measurements are unfused research implementations, not production
  mixed-FP4 kernel speed measurements.
