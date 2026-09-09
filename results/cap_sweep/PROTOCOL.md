# Tile-count sweep for the frozen CE/KL election

## Question

The published rule elects at most 256 8x64 type blocks. That constant was never
swept: `stale_map` has carried `count=256` as a default since the relinearized
study, and the two-SE filter leaves 35,000-200,000 eligible tiles, so the cap
binds and is the entire sparsity mechanism. Two adaptive replacements were
measured and both under-selected by orders of magnitude — a Cauchy-Schwarz
curvature penalty chose 0-8 blocks and beat fixed-256 in 1/60 comparisons, and
an MDL threshold retained 16-40% of the fixed-cap gain. Both predicted an
optimal count from a surrogate. Neither measured the count/gain curve.

This sweep measures that curve. It does not propose a new selection rule.

## What is held fixed

Everything except the count. Candidates are the same FourOverSix E2M1 baseline
and E0M3-alpha1 alternative on legal 8x64 weight tiles. Scores are the saved
192-sequence (64 C4, 64 OpenWebMath, 64 CodeParrot) CE and teacher-KL tables
from jobs 332349 and 332389; no model is re-scored. The election is the
unchanged `u_j = max(mean CE + 2 SE, mean KL + 2 SE)` ranking. Evaluation is the
unchanged held-out C4 recipe from `results/c4_frozen/PROTOCOL.md`: validation
shard 00001, 256 distinct documents after excluding all 192 calibration text
hashes, one seed-20260928 512-token crop each, causal per-token activation
factors, W4A4 on nonhead text linear weights and inputs.

Counts: 0 (FourOverSix), 16, 64, 256, 1024, 4096, 16384, 65536, and all
eligible tiles. The prefixes are nested by construction, so this is one ranking
inspected at nine depths, not nine independent selections.

## Integrity checks

- Re-election at 256 must reproduce the already-frozen `pooled192` map
  bitwise, for every module. This is the proof that the sweep's election path
  is the original one; the job asserts it and refuses to continue otherwise.
- `tests/test_cap_sweep.py` checks on synthetic tables that the inline prefix
  rule equals `stale_map` at every finite count, that prefixes nest, that an
  oversized count never elects a nonnegative tile, and that a CE-negative /
  KL-positive tile is excluded by the max.
- Source weight hashes, the map file hash and the score file hash are recorded
  and re-verified after evaluation. Prefix independence is checked at the
  baseline, at 256, and at the all-eligible extreme.

## What this sweep cannot establish

C4 is a calibration source, so these are held-out within-source measurements.
More importantly, **reading a preferred count off this table is selection on
the evaluation set.** The sweep is licensed to answer "what shape is the
curve" — flat plateau, interior optimum, or monotone — and not "which count
should the method use." Any count rule suggested by the curve, including a
rule that scales the count with model size, eligible-tile count or measured
quantization damage, must then be stated in advance and validated once on
domains untouched here: the literature (PG19), science (arXiv) and government
(GovReport) families used by the transfer panel, and WikiText-2.

Two-SE intervals are descriptive evaluation-window intervals. They do not
adjust for the nine correlated counts compared per model, for calibration-draw
variability (one pool per model), or for document dependence. The four models
here are the four with saved score tables; OPT-350M, Qwen3-0.6B,
Llama-3.2-1B and Qwen3.8-27B would need re-scoring to join.

## Motivation for the size question

Relative held-out C4 gain at the fixed 256 cap is not ordered by parameter
count: Qwen3-0.6B -8.2%, Qwen3-4B -4.7%, Llama-3.2-1B -3.1%, Pythia-1.4B
-1.5%, OPT-350M -1.0%, Llama-3.1-8B -0.8%, OLMo-1B -0.7%, Qwen3.8-27B -0.08%
(inconclusive). A fixed 256 is 0.7% of type blocks at 350M and 0.0002% at 27B,
so a shrinking budget fraction is one candidate explanation for the decay; a
second is that gain tracks recoverable quantization damage rather than size.
This sweep addresses the first. The second needs BF16 reference perplexities on
the identical windows, which do not currently exist for the seven-model panel.
