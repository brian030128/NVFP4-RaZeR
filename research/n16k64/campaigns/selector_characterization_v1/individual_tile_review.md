# Bounded mechanism review

The frozen N16 sample contains 160 interventions per model: 40 each from
selected, near-threshold rejected, strongly rejected, and the entire universe.
The first three pools use recorded combined CE/KL upper-score thresholds;
the random pool overlaps those strata. Selection is uniform without replacement
within each pool (seed 20260912), not a representative 160-tile universe sample.
`results/effect_diagnostics.csv` records realized pool sizes and conditional
inclusion probabilities. No population-weighted estimate is claimed.

The predictor and finite effect both use the same 128 seed0 calibration
sequences, the all-FourOverSix reference, and causal per-token activations.
CE predicts CE, KL predicts teacher KL. This is not held-out validation.
Each sequence contributes 511 shifted targets. The new CPU adapter verifies
the sealed raw report, recomputes every paired effect mean and sequence SE,
and retains all tiles. Per-tile 1.96-SE intervals are descriptive and
uncorrected; they do not define reliable ground-truth labels.

Across all 160 sampled tiles, CE sign agreement is 46.875% (Llama), 54.375%
(Qwen), and 49.375% (Mistral). The corresponding CE Spearman correlations
are 0.0445, 0.2393, and -0.1275; pointwise resolvable fractions are 16.25%,
6.25%, and 5%. These do not establish per-tile causal calibration.

All three raw fidelity reports have `baseline_restored_exact=false`. The
source's final check compares an eight-sequence batch with an original
16-sequence batch. This flag alone does not identify weight corruption or
stochastic noise. Separate V43 noise-control reports record identical
repeats/cycles/drift in their tested configurations. That is evidence for
repeatability under those configurations, not proof of zero numerical or
population uncertainty on every device. The new Ada pilot likewise repeats
exactly across its two attempts yet differs from historical A6000 NLL.

`results/group_interventions.csv` contains actual simultaneous selected-prefix
interventions from V43. Its predicted sums are not measured group effects.
Later mechanism and eight-bin follow-up reports support group ranking under
their frozen gates but fail universal conjunction superiority and the strict
veto/interaction gates. Boundary/corruption remains `power_limited_support`,
with endpoint-specific power labels and the hour-20 deviation unresolved for
strict protocol compliance. No individual causality, sharp kappa=3 jump,
simultaneous confidence guarantee, or reorder efficacy follows.

## Composition, breadth, and reorder boundaries

Historical Math64 and Code64 use 64 sequences; Math64+Code64 uses 128, all
512 tokens. Comparing those particular recipes confounds domain and budget.
The source also has **Math32+Code32 (64 total)** and Math16+Code16 (32 total),
so not every existing composition comparison is budget-confounded.
`results/composition_existing.csv` records 48 existing N8/N16 cells on Qwen4B
and Mistral, including these prefixes and the separate heldout draw.
Math64, Code64, and Math32+Code32 provide an equal-budget descriptive contrast
within one shared seed0 source block; they are not three independent draws or
a replicated composition factorial. Prefix nesting is preserved in the table.
The five mixed
draws estimate observed sampling variation within a fixed recipe; they do
not identify a random-domain population variance. Different numbers of
tested recipes also make raw range comparisons unsuitable causal estimators.
No new matched-budget factorial is run here.

The primary six-model panel contains development Llama-3.1-8B, Qwen3-4B,
Qwen3.8-27B and confirmation Mistral-7B-v0.3, Phi-4, OLMo-2-13B.
The later Falcon/Granite P71 tests belong to a post-hoc extension, not eight
independent primary confirmations. Neither model is added to this campaign.

The later N256 reorder program uses different layouts and numerical
conventions; it cannot be substituted into the present N8–N256 fixed-candidate
curve. Its archived target report explicitly says the 90% target was not met.
Retention is `(baseline-after)/(baseline-fine)` on a stated metric scale;
incremental recovery is `(before-after)/(before-fine)`. The latter requires
a matching pre-reorder endpoint and positive denominator. The target summary
alone is insufficient, so a further source audit uses the actual Qwen measured
raw/both report plus its published fine reference, verifying token windows,
activation/cache settings and software versions. `results/reorder_ratio_audit.csv`
separates PPL-scale retention (43.20% Wiki / 54.62% C4) from incremental
recovery of the **local raw-map remaining gap** (20.30% / 33.23%). NLL-scale
ratios are also reported separately, not substituted into the old PPL claims.

For the later Llama result, after/fine reports and window hashes are available,
but the matching 187-tile before point is only a rounded supplied reference in
the located evidence. PPL incremental recovery is therefore **approximate**
(11.32% / 15.59%), versus total retention 40.53% / 52.84%; its NLL incremental
ratio remains NA. No new paired significance or population inference is made.
These tensor-wide-activation experiments are not the primary N16/per-token
protocol, and no historical Qwen27B baseline is transferred to T3.

## Unresolved scope

- No new individual causal estimator or held-out tile validation.
- Sequence sampling uncertainty and finite-step/activation-rounding error
  are not interchangeable; these records cannot uniquely apportion causes.
- No inference from low overlap to a connected flat optimum.
- Native performance and all new reorder/model searches remain outside scope.
