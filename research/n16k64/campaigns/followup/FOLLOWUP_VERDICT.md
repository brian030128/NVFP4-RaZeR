# Follow-up verdict

## Fixed-gate summary

| Question | Classification | Gate passed? |
|---|---|---:|
| Group-level eight-bin dose response | supported | yes |
| Matched-budget conjunction over both single objectives | no_universal_conjunction_superiority | no |
| Mistral CE-vetoed/KL-approved symmetry | mistral_veto_symmetry_not_supported | no |

## Direct conclusions

- Dose response — frozen gate: for each predictor, all four development Spearman estimates have expected direction and at least three 95% CIs exclude zero. Gate passed: **yes**. Classification: **supported**.
- Objective ablation — frozen gate: each single-objective matched-K minus conjunction contrast is positive in all four development endpoints and Holm-significant in at least three. Gate passed: **no**. Classification: **no_universal_conjunction_superiority**.
- Mistral veto completion — frozen gate: actual add-back is worse than conjunction and matched random in both corpora, with all four mean contrasts Holm-significant in the frozen C-Mistral family. Gate passed: **no**. Classification: **mistral_veto_symmetry_not_supported**.

## Per-model evidence

- **Llama-3.1-8B:** c4: dose rho_CE=+0.429, rho_margin=-0.452; CE-matched−conjunction=+0.000637, KL-matched−conjunction=-0.000593; wiki: dose rho_CE=+0.333, rho_margin=-0.429; CE-matched−conjunction=-0.001180, KL-matched−conjunction=-0.001832.
- **Qwen3-4B:** c4: dose rho_CE=+1.000, rho_margin=-1.000; CE-matched−conjunction=-0.056791, KL-matched−conjunction=+0.022272; wiki: dose rho_CE=+0.929, rho_margin=-0.929; CE-matched−conjunction=-0.093796, KL-matched−conjunction=+0.023775.
- **Mistral-7B-v0.3:** c4: dose rho_CE=+0.571, rho_margin=-0.571; CE-matched−conjunction=+0.000472, KL-matched−conjunction=-0.000724; wiki: dose rho_CE=+0.262, rho_margin=-0.262; CE-matched−conjunction=+0.000059, KL-matched−conjunction=-0.001015.

## Qwen outlier sensitivity

All-model standardized CE slope: +0.8597 [+0.7657290, +0.9354113]; leave-Qwen3-4B-out: +0.7912 [+0.6507037, +0.9047970].
All-model standardized margin slope: -0.8631 [-0.9382667, -0.7689776]; leave-Qwen3-4B-out: -0.7974 [-0.9102675, -0.6557864].
Qwen3-4B remains in every primary per-model table; this sensitivity does not erase or downweight its result.

## Claims that remain prohibited

- reliable individual-tile causal signs, magnitudes, calibration, or causality;
- k=3 as a simultaneous per-tile confidence guarantee;
- universal CE or KL superiority from these models/corpora;
- native FP4/E0M3 Tensor Core execution;
- latency, speedup, runtime-overhead, area, power, or Blackwell-performance claims from fake quantization.

All results are fake-quantized/dequantized BF16 quality evaluation. Mistral remains one-shot analysis validation, not a fully independent confirmatory family.
