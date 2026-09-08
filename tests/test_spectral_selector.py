"""Numerical mechanism tests; run inside a Slurm allocation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from quantize.spectral_selector import candidates, reconstruct, select, spectral_squared
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6


def main():
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(73)
    w = torch.randn(16, 128, generator=gen).bfloat16()
    base, alt = candidates(w)
    assert torch.equal(base, quant_nvfp4_4over6(w, 4, 16))
    assert torch.equal(alt, quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always'))
    mask, info = select(w)
    result = reconstruct(base, alt, mask)
    for r in range(2):
        for c in range(2):
            sl = (slice(8*r, 8*r+8), slice(64*c, 64*c+64))
            assert torch.equal(result[sl], (alt if mask[r,c] else base)[sl])
    history = info['objective_history']
    assert all(b < a for a, b in zip(history, history[1:]))
    assert abs(spectral_squared(result.double()-w.double()) - history[-1]) < 1e-12
    assert torch.equal(select(w)[0], mask)
    scaled_mask, _ = select(w.float()*2, base.float()*2, alt.float()*2)
    assert torch.equal(scaled_mask, mask)
    zero_mask, zero_info = select(torch.zeros(8, 64))
    assert not zero_mask.any() and zero_info['objective_history'] == [0.0]
    tied_mask, _ = select(w, base, base)
    assert not tied_mask.any()
    # Variational identity: a unit-trace rank-one covariance attains the bound.
    error = result.double() - w.double()
    _, singular, vh = torch.linalg.svd(error, full_matrices=False)
    assert torch.allclose((error @ vh[0]).square().sum(), singular[0].square())
    # Smaller worst-case loss does not imply improvement in every direction.
    e_base, e_alt = torch.diag(torch.tensor([2., 0.])), torch.eye(2)
    assert spectral_squared(e_alt) < spectral_squared(e_base)
    assert e_alt[:, 1].square().sum() > e_base[:, 1].square().sum()
    for bad in (torch.zeros(7,64), torch.full((8,64), float('nan'))):
        try:
            select(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid matrix accepted')
    print('PASS: canonical candidates, tile legality, descent, reproducibility, scaling, ties, zero, variational identity, non-dominance, invalid inputs')


if __name__ == '__main__':
    main()
