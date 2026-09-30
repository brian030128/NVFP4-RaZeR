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


def random_map(family, n, k, g, frac):
    """(mask, type_block) of a random map the family executes: 8x64 for the weights-on-B family, 16x64 for 'mixed';
    'mixed256' (kernel-opt A') gets a map uniform over 128-row panels, stored as 16x64 granules as the 256x64 artifacts
    are (a partial last panel is cut to the real rows)."""
    rows = {'mixed_wB': 8, 'mixed256': 128}.get(family, 16)
    mask = torch.rand(-(-n // rows), k // 64, generator=g) < frac
    if rows > 16:
        mask, rows = mask.repeat_interleave(rows // 16, 0)[:n // 16], 16
    return mask, (rows, 64)


@pytest.mark.parametrize('family', sorted(FAMILIES))
@pytest.mark.parametrize('t,n,k', [(1, 4096, 4096), (19, 1024, 4096), (77, 2560, 9728), (300, 48, 2560), (1000, 4096, 1024)])
def test_widths_bitwise_equal(device, family, t, n, k):
    ks = kset(family, table={})
    g = torch.Generator(device='cpu').manual_seed(t + n)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    mixed = family.startswith('mixed')
    mask, tb = random_map(family, n, k, g, 0.3) if mixed else (None, None)
    pw = pack_module('m', w, None, 'map' if mask is not None else 'nvfp4', mask, tb)
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    act = 'four_over_six_rows' if mixed else 'nvfp4_rows'
    outs = {wd: NativeLinear(pw, kern, act, 'm')(x) for wd, kern in ks.kernels.items()}
    ref = outs[128]
    for wd, o in outs.items():
        assert torch.equal(o, ref), f'width {wd} differs'


@pytest.mark.parametrize('family', ['mixed', 'mixed_wB', 'mixed256'])
def test_batch_invariance(device, family):
    ks = kset(family)
    g = torch.Generator(device='cpu').manual_seed(3)
    w = (torch.randn(4096, 4096, generator=g) * 0.02).cuda().bfloat16()
    mask, tb = random_map(family, 4096, 4096, g, 0.2)
    nl = NativeLinear(pack_module('m', w, None, 'map', mask, tb), ks, 'four_over_six_rows', 'm')
    x = torch.randn(512, 4096, generator=g).cuda().bfloat16()
    full = nl(x)
    for t in (1, 3, 16, 33, 100, 257):
        assert torch.equal(nl(x[:t]), full[:t])
    assert len(ks.stats) >= 2          # several widths were actually used


def test_rules():
    assert [bucket(t) for t in (1, 2, 3, 17, 129, 9000)] == [1, 2, 4, 32, 256, 8192]
    assert [fallback_width(t) for t in (1, 16, 17, 33, 64, 65, 500)] == [16, 16, 32, 64, 64, 128, 128]
