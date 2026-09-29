# A1: IF4 (Cook et al.) and MixFP4 (Zou et al.) against the papers, the official code and the repo's quantizers

Checked on 2026-09-29, before any experiment-A GPU run. The machine-readable record is `a1_check.json`, from
`check_formats.py` (CPU, 11 Llama-3.1-8B modules, 356,515,840 weights). The implementation is
`quantize/adaptive_formats.py`.

## Sources

- **IF4:** Cook et al., "Adaptive Block-Scaled Data Types", arXiv 2603.28765.
  - Official code: github.com/mit-han-lab/fouroversix @ dadfad6901d473a734fe71e0b082e70ee993e23a (2026-04-21),
    `src/fouroversix/quantize/pytorch/reference.py`: `compute_nv_scale_factors`, `fake_quantize_to_e2m1`,
    `fake_quantize_to_int4`, `select_intfloat`.
  - Constants in `kernels/constants.py`.
- **MixFP4:** Zou et al., "MixFP4: Enhancing NVFP4 with Adaptive FP4/INT4 Block Representations", arXiv 2605.31035,
  Algorithm 1. The paper gives no code.

## Definitions (implemented exactly)

| | IF4 (Cook et al.) | MixFP4 (Zou et al.) |
|---|---|---|
| tensor scale | amax / (6 · 448) | s32 = amax / 2688 (= 6 · 448) |
| block scales | one: e4m3(bmax / 6), relative to the tensor scale, for both candidates | two: e4m3(bmax / 6) for E2M1, e4m3(bmax / 7) for E1M2 |
| uniform candidate | INT4 −7..7 by the "6/7 method": round(clamp(x_b · 1.16666666, −7, 7)), dequantized × 0.8571428571 (the official truncated constants) | E1M2 with the ×2 remap, i.e. the integers −7..7 at scale bmax / 7 |
| rounding | round to nearest even (torch.round, piecewise for E2M1) | round to nearest (RNE used here; the paper does not say) |
| selection | per 16-block, the lower sum of squared errors; ties keep FP4 (strict `<`) | per 16-block, the lower MSE; ties go to E1M2 (Algorithm 1's else branch) |
| type flag | the E4M3 scale's sign bit | the E4M3 scale's sign bit |
| unspecified | — | E4M3 saturation or underflow: a zero scale gives a zero block, counted (none on the checked weights) |

## The repo's quantizers

- **`quant_nvif4`,** whose docstring cites Cook et al., is Zou's construction, not IF4: it uses a separate INT4 scale
  e4m3(bmax / 7). It also differs from Zou's Algorithm 1 in:
  - E2M1 rounding: floor(|x| + 0.5), i.e. half away from zero, not half to even;
  - tie-breaking: ties go to FP4, not E1M2;
  - a scale clamp to [2^-9, 448].
- **`mixfp4` at 1x16** equals `nvif4`, so the same holds for it.
- Neither is used in experiment A.

## Results on real weights (11 Llama-3.1-8B modules: layers 0, 15 and 31, every projection type)

| check | result |
|---|---|
| IF4, this implementation vs the official reference: per-block choice | equal on all 11 modules |
| IF4: dequantized values (the official select_intfloat formulas applied to its own outputs) | **bitwise equal** on all 11 modules |
| the coarse-tile rule at (1, 16) vs the per-block rule, every rule | equal, every module |
| MixFP4 (Zou et al.) vs the repo's quant_nvif4 | 64,019 of 356,515,840 elements differ (0.018 %): the ties, rounding and clamp above |
| MixFP4 (Zou et al.)'s E1M2 candidate vs the repo's E0M3 alpha = 1 candidate (the uniform tiles of FlipQuant (ours)) | **identical** (0 elements differ) |
| IF4's FP candidate vs MixFP4 (Zou et al.)'s E2M1 candidate (both E2M1 at scale bmax / 6) | 1,189,004 elements (0.33 %) differ in BF16 (up to 1.36 % in one module, layers.31.mlp.down_proj): the scale and dequantization are computed in a different operation order. So each rule gets its own E2M1 base, e2m1 for IF4 and e2m1z for MixFP4 (Zou et al.). |
| zero (underflowed) scales | none |
