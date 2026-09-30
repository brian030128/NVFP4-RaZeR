"""kernel-opt A': the 4-arm 32-row-granule builds (n16k64_wA_g32 and its narrow widths) on the maps they execute.

A 4x2 warp of the n16k64_wA CTA tile owns m-atoms w and w + 4 of each 128-row panel (weight rows 16 w .. 16 w + 15 and
64 + 16 w .. 64 + 16 w + 15). With MIXFP4_A_ATOMS_PER_GRANULE=2 both atoms take the flag of the first one, so a granule
is those two 16-row blocks: not contiguous, and a map must be uniform over whole 128-row panels (256x64 maps are).
Checked here:
1. the granule map, measured: a tag on one 16-row block alone is executed on exactly that block and the block 64 rows
   below it when the block is a warp's first atom, and on no row when it is the second (its own tag is never read); so
   every granule lies inside one 128-row panel and every 256x64 tile is a union of granules. n16k64_wA, for contrast,
   executes every block's own tag;
2. the decode probe and random GEMMs (as tests/test_gemm.py) on maps uniform over 128-row panels, stored as 16x64
   granules as the 256x64 artifacts are, a partial last panel (48 rows) included;
3. bitwise equality with n16k64_wA from the reference build directory (sm120/build, or SM120_REF_BUILD_DIR) on the
   same operands, at every width;
4. NativeLinear refuses a map that is not uniform over the panels and accepts the 16x64-granule form of one that is;
   retile() re-declares it at 256x64 only if it is uniform there.
"""
import os
from pathlib import Path

import pytest
import torch

from mixfp4_sm120 import artifact as A
from mixfp4_sm120 import numerics as N
from mixfp4_sm120.lib import Kernel
from mixfp4_sm120.linear import NativeLinear
from test_gemm import SHAPES, fp64_reference, identity_operand, linear_gemm, place, random_weight_operand, rel

pytestmark = pytest.mark.gpu

G32 = ['n16k64_wA_g32', 'n16k64_wA_g32_n64', 'n16k64_wA_g32_n32', 'n16k64_wA_g32_n16']
PANEL = 128
REF_ROOT = Path(os.environ.get('SM120_REF_BUILD_DIR', Path(__file__).resolve().parents[1] / 'build'))


def kernel(name, unpatched=False):
    try:
        k = Kernel.load(name, allow_unpatched=unpatched)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f'{name} not built: {e}')
    assert k.cfg.map_tile_rows == PANEL and tuple(k.type_block) == (32, 64)
    return k


def reference():
    try:
        return Kernel.load('n16k64_wA', build_root=REF_ROOT)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f'reference n16k64_wA not built in {REF_ROOT}: {e}')


def panel_mask(mask, n):
    """[ceil(n/128), k/64] panel mask -> the [n/16, k/64] 16x64-granule mask (a partial last panel cut to real rows)."""
    return mask.repeat_interleave(PANEL // 16, 0)[:n // 16]


def random_panel_mask(n, k, g, frac):
    return torch.rand(-(-n // PANEL), k // 64, generator=g) < frac


def probe(kern, wnib, wsb, n, k):
    """Decode probe (tests/test_gemm.py): D of an exact identity activation operand, [n, k] in FP64."""
    got = torch.empty((n, k), dtype=torch.float64, device='cuda')
    for off in range(0, k, 1024):
        t = min(1024, k - off)
        xnib, xsb = identity_operand(t, k, off)
        d = linear_gemm(kern, N.pack_nibbles(wnib), place(wsb, k), 1.0, N.pack_nibbles(xnib), place(xsb, k), None, n, t, k)
        got[:, off:off + t] = d.double().t()
    return got


def executed_blocks(kern, n, k, block, seed):
    """16-row blocks the kernel executes as E0M3 when only 16-row block `block` is tagged (over all of K)."""
    m16 = torch.zeros(n // 16, k // 64, dtype=torch.bool)
    m16[block] = True
    wnib, wsb = random_weight_operand(n, k, m16, (16, 64), seed=seed)
    got = probe(kern, wnib, wsb, n, k)
    e0m3 = (got != N.decode_exact(wnib, wsb & 0x7F, 1.0)).reshape(n // 16, 16 * k).any(1)
    return sorted(int(b) for b in torch.nonzero(e0m3).flatten())


@pytest.mark.parametrize('name', G32 + ['n16k64_wA (reference)'])
def test_granule_map(device, name):
    kern = reference() if name.startswith('n16k64_wA ') else kernel(name)
    n, k = 512, 128                                   # two 256-row tiles, four 128-row panels, two k_blocks
    got = {b: executed_blocks(kern, n, k, b, seed=b) for b in range(n // 16)}
    if kern.cfg.map_tile_rows is None:
        assert got == {b: [b] for b in got}, got
        return
    want = {b: ([b, b + 4] if b % 8 < 4 else []) for b in got}
    assert got == want, got
    granules = [set(v) for v in got.values() if v]
    assert sorted(b for gr in granules for b in gr) == list(range(n // 16))       # a partition of the rows
    for gr in granules:
        assert len({b // (PANEL // 16) for b in gr}) == 1                          # inside one 128-row panel
        assert len({b // (256 // 16) for b in gr}) == 1                            # inside one 256x64 tile


@pytest.mark.parametrize('name', G32)
@pytest.mark.parametrize('n,k', [(128, 128), (256, 512), (48, 256), (4096, 128), (1024, 320)])
@pytest.mark.parametrize('pattern', ['random', 'all', 'none', 'single'])
def test_decode_probe(device, name, n, k, pattern):
    kern = kernel(name)
    g = torch.Generator(device='cpu').manual_seed(n * 7 + k)
    grid = (-(-n // PANEL), k // 64)
    if pattern == 'random':
        mask = torch.rand(grid, generator=g) < 0.5
    elif pattern == 'all':
        mask = torch.ones(grid, dtype=torch.bool)
    elif pattern == 'none':
        mask = torch.zeros(grid, dtype=torch.bool)
    else:
        mask = torch.zeros(grid, dtype=torch.bool)
        mask[torch.randint(0, grid[0], (1,), generator=g), torch.randint(0, grid[1], (1,), generator=g)] = True
    m16 = panel_mask(mask, n)
    wnib, wsb = random_weight_operand(n, k, m16, (16, 64), seed=n + k)
    got = probe(kern, wnib, wsb, n, k)
    executed_e0m3 = (got != N.decode_exact(wnib, wsb & 0x7F, 1.0))
    tagged = N.expand_mask(m16, (16, 64), (n, k)).cuda().repeat_interleave(16, 1)
    assert torch.equal(executed_e0m3, tagged), 'format executed by the tensor core differs from the tag'
    assert torch.equal(got, N.decode_exact(wnib, wsb, 1.0))


@pytest.mark.parametrize('name', G32)
@pytest.mark.parametrize('t,n,k', SHAPES)
def test_random_gemm_and_reference_bitwise(device, name, t, n, k):
    kern, ref_kern = kernel(name), reference()
    g = torch.Generator(device='cpu').manual_seed(t * 31 + n + k)
    m16 = panel_mask(random_panel_mask(n, k, g, 0.4), n)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(t, k, generator=g)
    x[:, torch.randperm(k, generator=g)[:4]] *= 30
    x = x.cuda().bfloat16()
    wnib, wsb, gs_w = N.quantize_weight(w, 'map', m16, (16, 64))
    xnib, xsb, gs_x = N.quantize_act(x, 'four_over_six_rows')
    args = (N.pack_nibbles(wnib), place(wsb, k), float(gs_w), N.pack_nibbles(xnib), place(xsb, k), gs_x, n, t, k)
    d = linear_gemm(kern, *args)
    ref, absdot = fp64_reference(wnib, wsb, gs_w, xnib, xsb, gs_x)
    err = (d.double() - ref).abs()
    assert bool((err <= 2.0 ** -8 * ref.abs() + 2.0 ** -20 * absdot + 1e-30).all())
    assert rel(d, ref) < 3e-3
    assert torch.equal(d.view(torch.int16), linear_gemm(ref_kern, *args).view(torch.int16)), 'differs from n16k64_wA'


@pytest.mark.parametrize('name', G32)
@pytest.mark.parametrize('frac', [0.0, 0.3, 1.0])
def test_unpatched_negative_control(device, name, frac):
    kern, bad = kernel(name), kernel(name, unpatched=True)
    t, n, k = 256, 1024, 2048
    g = torch.Generator(device='cpu').manual_seed(5)
    m16 = panel_mask(random_panel_mask(n, k, g, frac), n)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    wnib, wsb, gs_w = N.quantize_weight(w, 'map', m16, (16, 64))
    xnib, xsb, gs_x = N.quantize_act(x, 'four_over_six_rows')
    args = (N.pack_nibbles(wnib), place(wsb, k), float(gs_w), N.pack_nibbles(xnib), place(xsb, k), gs_x, n, t, k)
    good, wrong = linear_gemm(kern, *args), linear_gemm(bad, *args)
    ref, _ = fp64_reference(wnib, wsb, gs_w, xnib, xsb, gs_x)
    if frac == 0.0:
        assert torch.equal(good, wrong)
    else:
        assert rel(wrong, ref) > 20 * rel(good, ref)


def test_native_linear_requires_uniform_panels(device):
    kern = kernel('n16k64_wA_g32')
    n, k = 512, 256
    g = torch.Generator(device='cpu').manual_seed(17)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    fine = torch.zeros(n // 16, k // 64, dtype=torch.bool)
    fine[1, 0] = True                                          # warp 1's first m-atom only, not its second
    with pytest.raises(ValueError):
        NativeLinear(A.pack_module('m', w, None, 'map', fine, (16, 64)), kern, 'four_over_six_rows', 'm')
    contiguous32 = torch.zeros(n // 32, k // 64, dtype=torch.bool)
    contiguous32[0, 0] = True                                  # a contiguous 32-row tile is not a granule here
    with pytest.raises(ValueError):
        NativeLinear(A.pack_module('m', w, None, 'map', contiguous32, (32, 64)), kern, 'four_over_six_rows', 'm')
    panels = torch.zeros(n // PANEL, k // 64, dtype=torch.bool)
    panels[1, 2] = True                                        # uniform over 128-row panels, not over 256-row tiles
    pw = A.pack_module('m', w, None, 'map', panel_mask(panels, n), (16, 64))
    x = torch.randn(33, k, generator=g).cuda().bfloat16()
    y = NativeLinear(pw, kern, 'four_over_six_rows', 'm')(x)
    assert torch.equal(y, NativeLinear(pw, reference(), 'four_over_six_rows', 'm')(x))
    with pytest.raises(A.ArtifactError):
        A.retile(pw, (256, 64))
    assert A.retile(pw, (128, 64)).e0m3_tiles == 1
    tiles = torch.zeros(n // 256, k // 64, dtype=torch.bool)
    tiles[1, 2] = True
    pw256 = A.retile(A.pack_module('m', w, None, 'map', tiles.repeat_interleave(16, 0), (16, 64)), (256, 64))
    assert pw256.type_block == (256, 64) and pw256.e0m3_tiles == 1


def test_auto_routes_256x64_maps(device):
    """kernel-opt A' adoption: install(kernel='auto') runs a map whose own unit covers whole 128-row panels (the 256x64
    maps, stored as 16x64 granules) on 'mixed256' when it is built, and every other map on 'mixed' as before."""
    from mixfp4_sm120 import model as NM
    k16, note16 = NM.resolve_kernel('auto', dict(type_block=[16, 64], note=dict(record=dict(unit='16x64'))))
    assert k16.family == 'mixed' and note16 is None
    k0, note0 = NM.resolve_kernel('auto', dict(type_block=None))
    assert k0.family == 'mixed' and note0 is None
    k256, note256 = NM.resolve_kernel('auto', dict(type_block=[16, 64], note=dict(record=dict(unit='256x64'))))
    try:
        Kernel.load('n16k64_wA_g32')
        built = True
    except Exception:  # noqa: BLE001
        built = False
    assert k256.family == ('mixed256' if built else 'mixed'), (k256.family, note256)
    assert ('-> mixed256' in note256) if built else ('not built' in note256)
    assert NM.resolve_kernel('auto_mixed', None)[0].family == 'mixed'
