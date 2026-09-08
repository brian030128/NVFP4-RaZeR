# Frozen interaction-aware format selection

Fourth direction, fixed before any results. Earlier methods and their failures
remain in results/format_directions. Motivation: optimistic continuous error
compensation and independent tile scores omit the final discrete interactions.

Freeze canonical FourOverSix and E0M3-alpha1 candidates at original weights.
Use legal 8x64 weight tiles, 16-element scale blocks, original tensor global
scale. All activations use FourOverSix. No rotation or weight compensation.
Collect one shared 32-window Wiki training pass per model, on pristine model
trajectories. H=E[xq xq^T], where xq uses deployment FourOverSix quantization.
No damping, inverse, fitting labels, validation gates, or configuration sweep.

Minimize L(s)=tr((Q(s)-W) H (Q(s)-W)^T). Initialize at FourOverSix.
For a tile change D, exact change is 2< (Q-W)H,D >+tr(D H D^T).
Traverse K tiles in increasing order; independent output-row groups can update
together. Update the residual after each accepted change. At most eight passes,
or stop when a complete pass makes no changes. The cap is fixed for all models;
report nonconvergence rather than silently tuning it. Double-precision arithmetic
and a relative 1e-12 roundoff guard. Monotonicity concerns this empirical layer
objective only; it guarantees neither global optimality nor held-out task loss.

Comparators: FourOverSix; independent weight-MSE tile election; independent
exact changes evaluated against FourOverSix (no interaction updates); proposed
interacting election. Same candidates, input moments, and eval texts for all.
All maps frozen and saved before evaluation data is loaded.

Models: Llama-3.2-1B-Instruct, OPT-350M, Qwen3-0.6B. Pin model/dataset revisions
and token hashes. Fresh Wiki test windows96:128 (512 tokens), GSM8K/MBPP test
indices112:144 (up to512 tokens). Full-model W4A4 reference-text NLL; math/code
are not generated task accuracy. Screen: PPL improvement >=0.01 over FourOverSix
in at least7/9 cells, no positive paired delta minus2SE, and point improvement
over independent exact changes in at least6/9 cells. Descriptive intervals are
not familywise guarantees. Passing only licenses a larger confirmation panel.

Coordinate descent for quantization already exists (QuantEase,
https://arxiv.org/abs/2309.01885). Binary quadratic optimization is not new.
This experiment tests whether the tile interaction mechanism transfers. No
novelty or paper-readiness claim follows from the formulation alone.
