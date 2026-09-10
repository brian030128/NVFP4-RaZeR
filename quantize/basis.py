"""Candidate bases that separate the TYPE election from the SCALE election.

The reported MixFP4 direction is

    d = quant_mix_4_6(clip='a1', elect='always') - quant_nvfp4_4over6(w)
      = (E0M3, alpha=1)                          - (E2M1, alpha in {1, 1.5} by MSE)

so it moves the element TYPE and, on every scale block where FourOverSix chose
alpha=1.5, the SCALE as well. The two mechanisms are not separated by that
direction. `CLIP_PRESETS` in quantizer.py records both halves of the problem:
that when both move "the scale search was measured as the larger of the two, so
it dominated every headline", and that a clean type comparison needs the E2M1
baseline pinned at alpha=1 -- which `tests/test_mixfp4.py::test_a1_e2m1_is_nvfp4`
shows is bit-identical to plain NVFP4.

Three bases, scored and evaluated by identical machinery:

    e0m3       base 4over6     alt E0M3 alpha1     the reported direction
    alpha      base 4over6     alt E2M1 dense9     SCALE only: no type block, no
                                                   E0M3 operand, no election rule
                                                   and no metadata beyond the
                                                   ue4m3 scale NVFP4 already has
    type_pure  base NVFP4 a1   alt E0M3 alpha1     TYPE only, both at alpha=1

`e0m3` must reproduce the original two calls bitwise; test_basis.py asserts it.
"""
import torch

from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6

BASES = {
    'e0m3': dict(baseline='FourOverSix', alternative='E0M3 alpha1',
                 isolates='element type and scale together (the reported direction)'),
    'alpha': dict(baseline='FourOverSix', alternative='E2M1 dense9 alpha',
                  isolates='scale only, E2M1 on both sides'),
    'type_pure': dict(baseline='NVFP4 alpha1', alternative='E0M3 alpha1',
                      isolates='element type only, alpha=1 on both sides'),
}


@torch.no_grad()
def build_pair(weight, basis):
    """Return (baseline, alternative) dequantized weights for one basis."""
    if basis not in BASES:
        raise ValueError(f'Unknown basis {basis!r}')
    if basis == 'e0m3':
        base = quant_nvfp4_4over6(weight, 4, 16)
        alt = quant_mix_4_6(weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
    elif basis == 'alpha':
        base = quant_nvfp4_4over6(weight, 4, 16)
        alt = quant_mix_4_6(weight, 4, 16, type_block=(8, 64), clip='dense9', elect='never')
    else:
        base = quant_mix_4_6(weight, 4, 16, type_block=(8, 64), clip='a1', elect='never')
        alt = quant_mix_4_6(weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
    if not (torch.isfinite(base).all() and torch.isfinite(alt).all()):
        raise ValueError(f'Nonfinite candidate for basis {basis}')
    return base, alt


@torch.no_grad()
def direction_statistics(weight, base, alt, type_block=(8, 64)):
    """Perturbation size of one basis, so a null result is not just a smaller step.

    A basis whose direction is an order of magnitude shorter than another's
    cannot be compared to it on gain alone: it is taking a smaller step, not a
    worse one. These are reported per module and summed by the caller.
    """
    w = weight.float()
    d = alt.float() - base.float()
    o, k = w.shape
    per_tile = d.reshape(o // type_block[0], type_block[0],
                         k // type_block[1], type_block[1]).square().sum((1, 3))
    return dict(direction_sq=float(d.square().sum()),
                baseline_error_sq=float((base.float() - w).square().sum()),
                alternative_error_sq=float((alt.float() - w).square().sum()),
                weight_sq=float(w.square().sum()),
                tiles=int(per_tile.numel()),
                moved_tiles=int((per_tile > 0).sum()))
