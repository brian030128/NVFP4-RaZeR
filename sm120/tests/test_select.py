"""Kernel selection: every tile width of a family computes bitwise the same output, and a
KernelSet-driven NativeLinear is batch-invariant (a token's output does not depend on T)."""
import pytest
import torch

from mixfp4_sm120 import numerics as N
from mixfp4_sm120.artifact import pack_module
from mixfp4_sm120.linear import NativeLinear
from mixfp4_sm120.select import FAMILIES, KernelSet, bucket, fallback_width

pytestmark = pytest.mark.gpu


def kset(family, **kw):
    try:
        return KernelSet(family, **kw)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f'{family} family not built: {e}')


@pytest.mark.parametrize('family', sorted(FAMILIES))
@pytest.mark.parametrize('t,n,k', [(1, 4096, 4096), (19, 1024, 4096), (77, 2560, 9728), (300, 48, 2560), (1000, 4096, 1024)])
def test_widths_bitwise_equal(device, family, t, n, k):
    ks = kset(family, table={})
    g = torch.Generator(device='cpu').manual_seed(t + n)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    mask = (torch.rand(n // 16, k // 64, generator=g) < 0.3) if family == 'mixed' else None
    pw = pack_module('m', w, None, 'map' if mask is not None else 'nvfp4', mask, (16, 64) if mask is not None else None)
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    act = 'four_over_six_rows' if family == 'mixed' else 'nvfp4_rows'
    outs = {wd: NativeLinear(pw, kern, act, 'm')(x) for wd, kern in ks.kernels.items()}
    ref = outs[128]
    for wd, o in outs.items():
        assert torch.equal(o, ref), f'width {wd} differs'


def test_batch_invariance(device):
    ks = kset('mixed')
    g = torch.Generator(device='cpu').manual_seed(3)
    w = (torch.randn(4096, 4096, generator=g) * 0.02).cuda().bfloat16()
    mask = torch.rand(256, 64, generator=g) < 0.2
    nl = NativeLinear(pack_module('m', w, None, 'map', mask, (16, 64)), ks, 'four_over_six_rows', 'm')
    x = torch.randn(512, 4096, generator=g).cuda().bfloat16()
    full = nl(x)
    for t in (1, 3, 16, 33, 100, 257):
        assert torch.equal(nl(x[:t]), full[:t])
    assert len(ks.stats) >= 2          # several widths were actually used


def test_rules():
    assert [bucket(t) for t in (1, 2, 3, 17, 129, 9000)] == [1, 2, 4, 32, 256, 8192]
    assert [fallback_width(t) for t in (1, 16, 17, 33, 64, 65, 500)] == [16, 16, 32, 64, 64, 128, 128]
