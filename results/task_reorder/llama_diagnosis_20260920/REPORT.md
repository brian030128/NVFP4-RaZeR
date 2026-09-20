# Frozen Llama mechanism diagnosis

All reported intervals are descriptive mean ±2SE across documents, without multiplicity adjustment. No search, candidate promotion, or accuracy-benchmark fitting occurred.

## Quantization context and data transfer

| Documents | Quantized arranged−raw ΔCE | Δteacher KL | BF16+delta ΔCE | BF16−delta ΔCE |
|---|---:|---:|---:|---:|
| fit | -0.003556 ± 0.000942 | -0.001635 ± 0.000461 | +0.004223 ± 0.002234 | +0.012422 ± 0.003145 |
| election | -0.001369 ± 0.001152 | -0.001055 ± 0.000406 | +0.007493 ± 0.002272 | +0.010065 ± 0.002935 |
| development | -0.001510 ± 0.000852 | -0.001503 ± 0.000555 | +0.007165 ± 0.001846 | +0.008579 ± 0.002712 |
| fresh_in_domain | -0.001434 ± 0.000915 | -0.001509 ± 0.000463 | +0.008840 ± 0.001975 | +0.009067 ± 0.002445 |
| fresh_general | -0.001627 ± 0.001278 | -0.002062 ± 0.000620 | +0.007583 ± 0.002506 | +0.011939 ± 0.004114 |

The BF16 counterfactual adds/subtracts the effective arranged−raw quantized weight change to/from original BF16 weights. This is not a deployment policy. Original fit/election and reused development documents are not independent validation. Fresh math/code64 and general-text32 documents exclude prior recorded calibration and confirmation documents.

## Projection interactions

| Fresh documents | Gate-only ΔCE | Up-only ΔCE | Down-only ΔCE | Joint minus sum of isolated ΔCE |
|---|---:|---:|---:|---:|
| fresh_in_domain | -0.000180 ± 0.000528 | -0.000121 ± 0.000300 | -0.000867 ± 0.000714 | -0.000266 ± 0.000512 |
| fresh_general | +0.000463 ± 0.000976 | +0.000126 ± 0.000501 | -0.001518 ± 0.000940 | -0.000698 ± 0.001042 |

## Depth: joint fixed-template perturbation on24 fresh documents

| Layer index | Predicted ΔCE | Actual ΔCE | Frozen residual ΔCE | Actual prediction MAE | Frozen prediction MAE | Actual / frozen sign agreement |
|---|---:|---:|---:|---:|---:|---:|
| 0 | +0.000067 ± 0.000111 | +0.000120 ± 0.007698 | +0.000200 ± 0.000725 | 0.013741 | 0.001382 | 54.2% / 54.2% |
| 15 | -0.000022 ± 0.000080 | -0.000314 ± 0.006454 | -0.000268 ± 0.000507 | 0.012564 | 0.001029 | 54.2% / 54.2% |
| 31 | -0.003047 ± 0.001245 | -0.002855 ± 0.001692 | -0.002365 ± 0.001252 | 0.002137 | 0.001260 | 91.7% / 87.5% |

The same accepted final-layer permutation/mask is transferred diagnostically to earlier layers, which have different weights and perturbation norms. This is not an optimized early-layer candidate and cannot prove earlier layers are unhelpful. The frozen control uses q0+(x−x0) at activation quantizers: its raw baseline is bitwise exact but it is not a legal FP4 execution. STE predictions use the actual BF16-rounded weight delta. BF16 rounding, nonlinear layers and interactions remain in the control. Quarter-size and isolated-projection results are in summary.json.

All416 suffix baseline audits,45 changed full-model audits, and288 depth baseline/restore/STE audits passed. Source weights, frozen files and cross-depth question identities were checked.

## Interpretation

1. **The benefit transfers and depends on quantization context.** On64 new
math/code documents, arranged−raw CE is−0.001434±0.000915 and teacher KL
−0.001509±0.000463. On32 new general-text documents, CE is−0.001627±0.001278
and teacher KL−0.002062±0.000620. Lower CE means better next-token prediction;
lower teacher KL means closer to the original BF16 output distribution. The
positive BF16 counterfactual instead worsens CE by+0.008840±0.001975 in-domain
and+0.007583±0.002506 on general text; the negative counterfactual also worsens
CE. This supports quantization-dependent repair and argues against pure
memorization of the observed documents. It does not prove there is no
task-directed component: the method still uses task-aware discrete fitting.

2. **Downstream activation quantization makes earlier-layer directional scores
unreliable in this probe.** On24 fresh documents, the joint-change prediction
MAE falls from0.013741 to0.001382 at layer0 (9.9×), and0.012564 to0.001029 at
layer15 (12.2×), when activation-quantization residuals are held fixed. At the
last layer it falls only from0.002137 to0.001260 (1.7×). Actual loss-change sign
agreement is54.2%,54.2%,91.7% for layers0,15,31. Freezing residuals reduces early
error magnitude without improving sign agreement; the early predicted effects
are tiny, and remaining BF16 rounding/nonlinearity still matters. The early
layouts were transferred rather than optimized, so this does not demonstrate
that only the last MLP can benefit.

3. **The refined down projection provides most of the fresh measured gain.**
Down-only CE changes are−0.000867±0.000714 on fresh math/code and
−0.001518±0.000940 on general text. Gate/up alone are weak and may have positive
mean CE changes on general text. Joint-minus-sum interaction intervals include
zero on both fresh groups, so this test does not establish extra interaction
benefit. These findings concern the accepted refined layout; the original
rejected layout remains rejected.

The next justified search design is to use directional scores for proposing
changes, then verify finite changes jointly through the affected suffix before
fresh validation. Do not treat the frozen-residual control as an FP4 deployment
method. No additional search was launched. Earlier accuracy findings remain:
MMLU non-STEM+0.220pp and ARC−0.085pp, neither conclusive. Loss improvements here
are not new evidence of correct-answer accuracy gains.
