# Boundary/corruption mechanism verdict

**Boundary classification:** `power_limited_support` (frozen pattern gate passed: true).

**Corruption classification:** `power_limited_support` (frozen pattern gate passed: true).

The evidence addresses aggregate, composition-matched tile groups and imposed full-map stress. It does not validate individual-tile causal signs, magnitudes, or calibration.

## Model-specific reading

- **Llama-3.1-8B:** boundary slopes -0.001201 (C4), -0.001217 (WikiText); weakest-selected minus nearest-rejected +0.000694, +0.001320; corruption slopes +0.002830, +0.004001. Power labels and confidence intervals in the primary table determine whether these directions are inferential or descriptive.
- **Qwen3-4B:** boundary slopes -0.014815 (C4), -0.030797 (WikiText); weakest-selected minus nearest-rejected -0.004591, -0.006012; corruption slopes +0.054310, +0.112805. Power labels and confidence intervals in the primary table determine whether these directions are inferential or descriptive.
- **Mistral-7B-v0.3:** boundary slopes -0.000286 (C4), -0.000451 (WikiText); weakest-selected minus nearest-rejected -0.000505, +0.000050; corruption slopes +0.002172, +0.002927. Power labels and confidence intervals in the primary table determine whether these directions are inferential or descriptive.

## Pooled sensitivity

Boundary standardized slope: all models -0.876115 [-0.922347, -0.822872]; leave-Qwen -0.822229 [-0.891130, -0.742034].

Corruption standardized slope: all models +0.959100 [+0.917997, +0.985754]; leave-Qwen +0.939942 [+0.878436, +0.979968].

## Candid interpretation

Both frozen pattern gates pass, but both classifications are power-limited. Continuous aggregate ranking and imposed corruption robustness are supported across models and corpora, including after leaving Qwen3-4B out. A clean local threshold discontinuity is not supported uniformly: the weakest-selected versus nearest-rejected contrast reverses sign for Llama and is heterogeneous across validation panels. This is compatible with useful group-level ranking plus noisy local/tile-level effects; it is not evidence that every kappa>3 tile is beneficial.

## Operational limitation

The campaign did not satisfy its hour-20 launch cutoff or 24-hour wall-clock target. Late Llama/Qwen runs were frozen required retries after co-tenancy invalidations, not outcome-selected additions. A separately frozen, outcome-blind analysis-plumbing correction was also required after two failed CPU attempts. These limitations must be disclosed with any use of the results; see `PROTOCOL_DEVIATIONS.json` and `ANALYSIS_CORRECTIONS.json`.

## Claims that remain prohibited

Reliable individual-tile causality or finite-effect calibration; a true wrong-tile percentage; every selected tile being helpful or every rejected tile harmful; k=3 as a simultaneous confidence guarantee; universal selector optimality; all-model generalization; native FP4/E0M3 execution; and latency, throughput, speedup, overhead, area, power, or Blackwell claims.
