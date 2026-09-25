"""The fused Triton activation quantizer is bit-identical to the reference quantizer, and places
every scale byte (and zeroes every padding byte) of the kernel's scale-factor layout."""
import pytest
import torch

from mixfp4_sm120 import numerics as N
from mixfp4_sm120 import quant_act as QA
from mixfp4_sm120.lib import sf_buffer_size, sf_offset_formula

pytestmark = pytest.mark.gpu

SHAPES = [(1, 4096), (3, 2560), (127, 1024), (128, 4096), (129, 14336), (517, 9728), (64, 320), (5, 32), (2048, 5120)]


def reference(x, kind):
    nib, sb, gs = N.quantize_act(x, kind)
    t, k = x.shape
    buf = torch.zeros(sf_buffer_size(t, k), dtype=torch.uint8, device=x.device)
    r = torch.arange(t, device=x.device)[:, None]
    kb = torch.arange(k // 16, device=x.device)[None, :]
    buf[sf_offset_formula(r, kb, k).reshape(-1)] = sb.reshape(-1)
    return N.pack_nibbles(nib), buf, gs


def make_x(t, k, seed, kind='llm'):
    g = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(t, k, generator=g)
    if kind == 'llm':
        x[:, torch.randperm(k, generator=g)[: max(1, k // 512)]] *= 60.0
        if t > 3:
            x[1] = 0.0
            x[2] *= 1e-3
    elif kind == 'heavy':
        x = torch.distributions.StudentT(torch.tensor(2.0)).sample((t, k))
    return x.to('cuda', torch.bfloat16)


@pytest.mark.parametrize('kind', ['four_over_six_rows', 'nvfp4_rows'])
@pytest.mark.parametrize('shape', SHAPES)
def test_bitwise(device, kind, shape):
    t, k = shape
    for i, dist in enumerate(('llm', 'heavy', 'normal')):
        x = make_x(t, k, 10 + i, dist)
        rp, rsf, rgs = reference(x, kind)
        # garbage-filled output buffers: the kernel must write every byte the GEMM reads
        out = (torch.full_like(rp, 0xAB), torch.full_like(rsf, 0x7F), torch.full_like(rgs, float('nan')))
        p, sf, gs = QA.quantize(x, kind, out=out)
        assert torch.equal(p, rp), (kind, shape, dist, 'packed')
        assert torch.equal(sf, rsf), (kind, shape, dist, 'scale layout')
        assert torch.equal(gs, rgs), (kind, shape, dist, 'global scale')


def test_strided_rows(device):
    x = make_x(64, 8192, 3)[:, :4096]            # row stride 8192, unit column stride
    p, sf, gs = QA.quantize(x, 'four_over_six_rows')
    rp, rsf, rgs = reference(x.contiguous(), 'four_over_six_rows')
    assert torch.equal(p, rp) and torch.equal(sf, rsf) and torch.equal(gs, rgs)
