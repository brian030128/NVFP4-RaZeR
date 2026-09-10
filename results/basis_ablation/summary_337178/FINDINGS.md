# Is the gain E0M3, or task-gradient selection over any legal 4-bit knob?

Job 337178, six arms: `{qwen4b, llama8b} x {e0m3, alpha, type_pure}`. Identical
data, teacher, CE/KL two-SE eligibility, tile counts and held-out sets; only the
scored direction differs (`quantize/basis.py`). 512-token windows with causal
per-token activation factors -- the protocol of `results/task_sensitivity` and
of `below_bf16_diag/FINDINGS.md` §2, **not** the 2048-token paper-aligned one.

The run reproduces two recorded numbers to the digit, which validates the path:
the qwen4b FourOverSix baseline (19.414897) and its weight-MSE election
(3,407,142 tiles, 20.108009), both quoted in `below_bf16_diag/FINDINGS.md` §2.

`adaptive_*` selects 0-15 tiles in this path and carries no signal; the matched
comparison is `fixed256_*`, 256 tiles in every arm.

## Answer: E0M3 wins. Calibrating the alpha instead is worse at matched count.

`fixed256_math_code128`, delta against each arm's own baseline:

| | qwen4b wiki | qwen4b c4 | llama8b wiki | llama8b c4 |
|---|---:|---:|---:|---:|
| e0m3 | **-1.988270** | **-0.987665** | **-0.064335** | **-0.087006** |
| alpha | -1.158278 | -0.529401 | -0.018107 | -0.035899 |
| e0m3 / alpha | 1.72x | 1.87x | 3.55x | 2.42x |

E0M3 wins all 4 cells and the paired two-SE intervals separate the two arms in
each (qwen wiki: -0.108041 +/-0.004074 against -0.061513 +/-0.003812). So the
gain is **not** basis-agnostic: it is not merely "any perturbation plus a
gradient filter". The proposed control was worth running and it lost.

**One qualification, stated because it was pre-registered.** The alpha direction
is 1.88x shorter than the E0M3 one (0.0675 against 0.1270 relative norm, both
models). Dividing the gain ratios by that gives 0.91 / 0.99 / 1.89 / 1.29 -- so
on qwen4b alpha is about as efficient *per unit of perturbation* and only loses
because it is a smaller step, while on llama8b E0M3 is genuinely 1.3-1.9x more
efficient. Gain is not linear in step length, so this is a caveat on the
margin, not a competing conclusion.

## But a third of the reported qwen4b headline is a mis-specified baseline

`type_pure` supplies what the reported arm never had: the same E0M3 alternative
scored against NVFP4 alpha=1 instead of FourOverSix. That isolates the type
election, and it also measures the baseline itself.

| model | domain | FourOverSix | NVFP4 alpha=1 | FourOverSix penalty |
|---|---|---:|---:|---:|
| qwen4b | wiki | 19.414897 | **18.758341** | **+0.656556** |
| qwen4b | c4 | 21.928243 | **21.813711** | **+0.114532** |
| llama8b | wiki | **9.092934** | 9.102980 | -0.010046 |
| llama8b | c4 | **11.600109** | 11.641787 | -0.041678 |

On qwen4b FourOverSix is **0.657 wiki PPL worse than doing nothing**, i.e.
than plain NVFP4. That penalty is **33.0% of e0m3's -1.988 headline** on wiki
and 11.6% on c4. On llama8b the sign flips and FourOverSix is the better
baseline, exactly as `CLIP_PRESETS` predicted from W4A16. So the reported
direction's advantage on the model with the spectacular result is partly
recovering ground its own baseline gave away.

## The clean claim is stronger than the reported one, on qwen4b

Absolute PPL at 256 tiles -- the comparison to use, since baselines differ:

| model | domain | e0m3 | alpha | type_pure |
|---|---|---:|---:|---:|
| qwen4b | wiki | 17.426627 | 18.256618 | **17.335883** |
| qwen4b | c4 | **20.940578** | 21.398842 | 20.947186 |
| llama8b | wiki | **9.028599** | 9.074827 | 9.028807 |
| llama8b | c4 | **11.513102** | 11.564209 | 11.609019 |

Electing E0M3 from an NVFP4 alpha=1 baseline reaches a **lower** qwen4b wiki PPL
than the reported direction (17.3359 against 17.4266) with the same 256 tiles,
and ties it on c4 and on llama8b wiki. E0M3 does not need FourOverSix underneath
it; on qwen4b it is better off without it.

## A free gain nobody has to calibrate for

The `weight_mse` control is basis-dependent, and under `alpha` it elects the
tiles where dense9 reconstructs better than FourOverSix. Since dense9's alphas
are a superset of {1, 1.5}, that is essentially every tile -- 7,095,825 of
7,096,320 -- so this row is plain `quant_nvfp4_nover6`: a per-scale-block alpha
search, no type block, no E0M3 operand, no election rule, **no metadata beyond
the ue4m3 scale NVFP4 already stores**, and no calibration at all.

| model | domain | delta vs FourOverSix | paired ΔNLL ±2SE |
|---|---|---:|---|
| qwen4b | wiki | **-0.338651** | -0.017597 ±0.004122 (supported) |
| qwen4b | c4 | -0.057580 | -0.002629 ±0.003609 (inconclusive) |
| llama8b | wiki | -0.012315 | -0.001355 ±0.002064 (inconclusive) |
| llama8b | c4 | -0.031347 | -0.002706 ±0.002611 (supported) |

The qwen4b wiki gain is 17% of what 256 calibrated E0M3 tiles buy, for zero
calibration and zero metadata. Its two llama8b values are in the same band as
the -0.0146 / -0.0218 the `quant_nvfp4_nover6` docstring records at W4A16.

Elected by MSE, the two bases behave oppositely: dense9 alpha helps, while E0M3
**hurts** (qwen4b wiki +0.693113 over 3.4M tiles; type_pure +1.496379 over 6.3M).
E0M3 is only ever useful when a task gradient picks the tiles; the alpha search
is useful without one. That is the sharpest statement of what each contributes.

## What this does and does not settle

- E0M3 earns its place. The alternative-basis control lost on all four cells,
  and `type_pure` shows the type election alone carries most of the gain.
- The reported baseline is wrong for qwen4b and should be NVFP4 alpha=1 there,
  or the alpha search should be enabled on both sides.
- The two mechanisms look independent and neither is tested together:
  dense9 alpha everywhere **plus** 256 task-gradient E0M3 tiles is the obvious
  next configuration and was not run.
- All of this is label-space PPL. No accuracy or exploitability was measured
  here, and `below_bf16_diag/FINDINGS.md` shows PPL misranks against BF16 on
  qwen4b. Nothing above compares any arm to BF16.
- Intervals are descriptive paired two-SE on 512-token windows and do not
  account for WikiText article dependence or calibration-draw variability.
