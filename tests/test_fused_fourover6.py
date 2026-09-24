"""GPU tests: the fused FourOverSix kernels equal the reference quantizers bit for bit.

python tests/test_fused_fourover6.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantize.fast_act import quant_per_document  # noqa: E402
from quantize.fused_fourover6 import (LearnedScaleE2M1, STEFourOverSix, fourover6,  # noqa: E402
                                      fourover6_block_scales, scaled_e2m1)
from quantize.quantizer import quant_nvfp4_4over6  # noqa: E402


def same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def tensors(g):
    yield torch.randn(4096, 4096, generator=g, device='cuda').bfloat16()
    yield (torch.randn(1024, 14336, generator=g, device='cuda') ** 3).bfloat16()          # heavy tails
    t = torch.randn(512, 4096, generator=g, device='cuda') * 1e-3
    t[:, ::7] = 0
    t[3, :] = 0                                                                           # all-zero blocks
    t[5, :16] = 40.0                                                                      # one outlier block
    yield t.bfloat16()
    yield (torch.rand(256, 4096, generator=g, device='cuda') * torch.logspace(-8, 2, 4096, device='cuda')).bfloat16()


def test_weights():
    g = torch.Generator(device='cuda').manual_seed(0)
    for x in tensors(g):
        ref = quant_nvfp4_4over6(x, 4, 16)
        assert same(fourover6(x, 1), ref), tuple(x.shape)
        gs, pre, scales = fourover6_block_scales(x)
        assert same(scaled_e2m1(x, scales, gs), ref), tuple(x.shape)
        factor = torch.ones_like(pre, requires_grad=True)
        assert same(LearnedScaleE2M1.apply(factor, x, gs, pre), ref), tuple(x.shape)


def test_documents():
    g = torch.Generator(device='cuda').manual_seed(1)
    for b, t, k in ((1, 512, 4096), (8, 512, 4096), (3, 512, 14336), (16, 511, 1024)):
        x = (torch.randn(b, t, k, generator=g, device='cuda') * torch.logspace(-2, 1, b, device='cuda')[:, None, None]).bfloat16()
        assert same(fourover6(x, b), quant_per_document(x)), (b, t, k)


def test_gradients():
    g = torch.Generator(device='cuda').manual_seed(2)
    x = torch.randn(64, 256, generator=g, device='cuda').bfloat16().requires_grad_()
    up = torch.randn(64, 256, generator=g, device='cuda')
    y = STEFourOverSix.apply(x, 1)
    (y.float() * up).sum().backward()
    assert torch.equal(x.grad, up.bfloat16())
    # LSQ step-size gradient against a float64 reference.
    w = (torch.randn(32, 256, generator=g, device='cuda') * 0.02).bfloat16()
    gs, pre, _ = fourover6_block_scales(w)
    factor = (1 + 0.3 * torch.rand(pre.shape, generator=g, device='cuda')).requires_grad_()
    out = LearnedScaleE2M1.apply(factor, w, gs, pre)
    grad_out = torch.randn(out.shape, generator=g, device='cuda').bfloat16()
    out.backward(grad_out)
    target = factor.detach() * pre
    s = target.clamp(2 ** -9, 448).to(torch.float8_e4m3fn).float()
    # Codes are decided in FP32, as in the forward pass: low-precision inputs put many
    # ratios exactly on a rounding midpoint, where the LSQ gradient is discontinuous.
    u = (w.reshape(-1, 16).float() / gs) / s[:, None]
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device='cuda')
    code = levels[torch.bucketize(u.abs(), (levels[:-1] + levels[1:]) / 2, right=False)] * u.sign()
    d = torch.where(u.abs() <= 6, code.double() - u.double(), code.double())
    expected = (grad_out.reshape(-1, 16).double() * d).sum(-1) * gs.double() * pre.double()
    # No gradient where the E4M3 clamp is active (here: FourOverSix /4 scales above 448).
    expected = torch.where((target >= 2 ** -9) & (target <= 448), expected, 0.)
    assert ((target > 448).sum() > 0) and (factor.grad[target > 448] == 0).all()
    assert torch.allclose(factor.grad.double(), expected, rtol=1e-4, atol=1e-9), (factor.grad.double() - expected).abs().max()


if __name__ == '__main__':
    for test in (test_weights, test_documents, test_gradients):
        test()
        print(test.__name__, 'ok', flush=True)
