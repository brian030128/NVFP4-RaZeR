"""Kernel selection: every tile width of a family computes bitwise the same output, and a
KernelSet-driven NativeLinear is batch-invariant (a token's output does not depend on T)."""
import pytest
import torch

from mixfp4_sm120 import numerics as N
from mixfp4_sm120.artifact import pack_module
from mixfp4_sm120.linear import NativeLinear
from mixfp4_sm120.select import FAMILIES, TABLE_FAMILY, KernelSet, bucket, fallback_width

pytestmark = pytest.mark.gpu


def kset(family, **kw):
    try:
        return KernelSet(family, **kw)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f'{family} family not built: {e}')


def random_map(family, n, k, g, frac):
    """(mask, type_block) of a random map the family executes: 8x64 for the weights-on-B families, 16x64 for 'mixed';
    'mixed256' (kernel-opt A') gets a map uniform over 128-row panels, stored as 16x64 granules as the 256x64 artifacts
    are (a partial last panel is cut to the real rows)."""
    rows = 8 if family.startswith('mixed_wB') else 128 if family.startswith('mixed256') else 16
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


@pytest.mark.parametrize('family', ['mixed_e', 'stock_e', 'mixed_ko', 'stock_ko', 'mixed_wB_ko', 'stock_wB_ko'])
@pytest.mark.parametrize('t,n,k', [(1, 4096, 4096), (300, 1024, 4096), (1000, 4096, 1024)])
def test_schedules_bitwise_equal(device, family, t, n, k):
    """kernel-opt #4: every scheduler setting (raster 0/1/2 x swizzle 1/2/4/8) computes the default (0, 1)'s output
    bitwise at every width, through NativeLinear (sm120_linear_ex) as a table's schedule row selects it."""
    g = torch.Generator(device='cpu').manual_seed(t + n + 7)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    mixed = family.startswith('mixed')
    mask, tb = random_map(family, n, k, g, 0.3) if mixed else (None, None)
    pw = pack_module('m', w, None, 'map' if mask is not None else 'nvfp4', mask, tb)
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    act = 'four_over_six_rows' if mixed else 'nvfp4_rows'
    ref = None
    for wd in kset(family, table={}).kernels:
        ks = kset(family, table={TABLE_FAMILY[family]: {f'{n}x{k}': {str(bucket(t)): wd}}})
        nl = NativeLinear(pw, ks, act, 'm')
        for combo in [(r, s) for r in (0, 1, 2) for s in (1, 2, 4, 8)]:
            ks.schedules = {(n, k): {bucket(t): combo}}
            assert ks.width(n, k, t) == wd and ks.schedule(n, k, t) == combo
            y = nl(x)
            ref = y if ref is None else ref
            assert torch.equal(y, ref), f'width {wd} schedule {combo} differs'


@pytest.mark.parametrize('family', ['mixed', 'mixed_wB', 'mixed256', 'mixed_t0', 'mixed256_t0', 'mixed_ko', 'mixed_wB_t0',
                                    'mixed_wB_ko'])
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


def test_auto_routes_8x64_maps(device):
    """kernel-opt 8x64 plan P6 (decision b): install(kernel='auto') runs an 8x64 map on the adopted 8x64 set
    'mixed_wB_ko' when its builds are in the build directory, else on the paper kernel n8k64_wB; 'auto_wB' follows it, and
    'paper_wB' is the paper kernel. 16x64 and 256x64 maps keep their routing (test_g32.py)."""
    from mixfp4_sm120 import model as NM
    from mixfp4_sm120.lib import Kernel

    def built(*names):
        try:
            for n in names:
                Kernel.load(n)
            return True
        except Exception:  # noqa: BLE001
            return False
    adopted = built(*FAMILIES['mixed_wB_ko'].values())
    if not adopted and not built('n8k64_wB'):
        pytest.skip('neither the adopted 8x64 set nor n8k64_wB is built')
    for spec, meta in (('auto', dict(type_block=[8, 64])), ('auto', dict(type_block=[8, 64], note=dict(record=dict(unit='8x64')))),
                       ('auto_wB', None)):
        k, note = NM.resolve_kernel(spec, meta)
        if adopted:
            assert isinstance(k, KernelSet) and k.family == 'mixed_wB_ko' and '-> mixed_wB_ko' in note, (spec, note)
        else:
            assert k.cfg.name == 'n8k64_wB' and 'not built' in note, (spec, note)
    if built('n8k64_wB'):
        assert NM.resolve_kernel('paper_wB', None)[0].cfg.name == 'n8k64_wB'
    if built('stock_wB_e64'):
        ks, note = NM.resolve_kernel('auto_stock_wB', None)
        assert isinstance(ks, KernelSet) and ks.family == 'stock_wB_ko' and '-> stock_wB_ko' in note
    elif built('stock_wB'):
        k, note = NM.resolve_kernel('auto_stock_wB', None)
        assert k.cfg.name == 'stock_wB' and 'not built' in note
