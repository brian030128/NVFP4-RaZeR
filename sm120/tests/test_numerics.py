"""The native codes/scales decode bit for bit to the fake quantizers the selector and fake-quant
evaluation use (quantize/quantizer.py, quantize/causal_four_over_six.py on this branch)."""
import pytest
import torch

from conftest import weight_cases
from mixfp4_sm120 import numerics as N
from quantize.causal_four_over_six import quantize_rows
from quantize.quantizer import quant_mix_4_6, quant_nvfp4, quant_nvfp4_4over6, _quant_e2m1

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def nvfp4_rows_reference(x):
    """campaign.quant.nvfp4_rows (research/n16k64 branch), copied verbatim."""
    shape = x.shape
    b = x.float().reshape(-1, shape[-1] // 16, 16)
    gs = (b.abs().amax((1, 2), keepdim=True) / (6.0 * 448)).clamp_min(torch.finfo(torch.float32).tiny)
    s = b / gs
    bmax = s.abs().amax(-1, keepdim=True)
    scale = (bmax / 6.0).clamp(max=448, min=2 ** (-9)).to(torch.float8_e4m3fn).to(s.dtype)
    return (_quant_e2m1(s, scale) * gs).reshape(shape).bfloat16()


def apply_mask(base, alt, mask, type_block):
    """campaign.tiles.apply_mask."""
    bm, bk = type_block
    return torch.where(mask.to(base.device).repeat_interleave(bm, 0).repeat_interleave(bk, 1), alt, base)


CASES = weight_cases(DEV)


@pytest.mark.parametrize('name', sorted(CASES))
def test_four_over_six_weight(name):
    w = CASES[name]
    if not bool(w.abs().amax() > 0):
        pytest.skip('all zero')
    nib, sb, gs = N.quantize_weight(w, 'four_over_six')
    assert torch.equal(N.decode_fake_order(nib, sb, gs), quant_nvfp4_4over6(w, 4, 16))
    assert not bool((sb >> 7).any())


@pytest.mark.parametrize('name', sorted(CASES))
def test_e0m3_weight(name):
    w = CASES[name]
    code, scale, gs = N.weight_e0m3(w)
    nib = N.e0m3_nibbles(code)
    sb = N.scale_bytes(scale, torch.ones_like(scale, dtype=torch.bool))
    fake = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
    assert torch.equal(N.decode_fake_order(nib, sb, gs), fake)


@pytest.mark.parametrize('name', sorted(CASES))
def test_nvfp4_weight(name):
    w = CASES[name]
    nib, sb, gs = N.quantize_weight(w, 'nvfp4')
    assert torch.equal(N.decode_fake_order(nib, sb, gs), quant_nvfp4(w, 4, 16))


@pytest.mark.parametrize('name', sorted(CASES))
@pytest.mark.parametrize('pattern', ['random', 'none', 'all', 'checker'])
def test_map_weight(name, pattern):
    w = CASES[name]
    tb = (16, 64)
    n, k = w.shape
    if n % 16 or k % 64:
        pytest.skip('not N16K64-divisible')
    g = torch.Generator(device='cpu').manual_seed(1)
    grid = (n // 16, k // 64)
    mask = {'random': torch.rand(grid, generator=g) < 0.3, 'none': torch.zeros(grid, dtype=torch.bool),
            'all': torch.ones(grid, dtype=torch.bool),
            'checker': (torch.arange(grid[0])[:, None] + torch.arange(grid[1])[None]) % 2 == 0}[pattern]
    nib, sb, gs = N.quantize_weight(w, 'map', mask, tb)
    fake = apply_mask(quant_nvfp4_4over6(w, 4, 16),
                      quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always'), mask, tb)
    assert torch.equal(N.decode_fake_order(nib, sb, gs), fake)
    # the tag is set on exactly the scale blocks of the masked tiles
    assert torch.equal((sb >> 7).bool(), N.expand_mask(mask, tb, (n, k)).to(sb.device))


def activation_cases():
    g = torch.Generator(device='cpu').manual_seed(2)
    x = torch.randn(64, 4096, generator=g)
    x[:, torch.randperm(4096, generator=g)[:8]] *= 50       # outlier channels
    x[5] = 0.0                                               # an all-zero token
    x[6, :32] = 1e-30                                        # underflowing values
    x[7] *= 1e4
    y = torch.randn(3, 5, 2560, generator=g)                 # batched input
    return {'llama_like': x.to(DEV, torch.bfloat16), 'batched': y.to(DEV, torch.bfloat16)}


ACTS = activation_cases()


@pytest.mark.parametrize('name', sorted(ACTS))
def test_four_over_six_rows(name):
    x = ACTS[name]
    assert torch.equal(N.fake_quant_act_rows(x, 'four_over_six_rows'), quantize_rows(x))


@pytest.mark.parametrize('name', sorted(ACTS))
def test_nvfp4_rows(name):
    x = ACTS[name]
    assert torch.equal(N.fake_quant_act_rows(x, 'nvfp4_rows'), nvfp4_rows_reference(x))


def test_encodings_roundtrip():
    codes = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -.5, -1., -1.5, -2., -3., -4., -6.])
    nib = N.e2m1_nibbles(codes)
    assert nib.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
    e0 = torch.arange(-7, 8).float()
    assert N.e0m3_nibbles(e0).tolist() == [15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7]
    packed = N.pack_nibbles(torch.tensor([[1, 2, 3, 15]], dtype=torch.uint8))
    assert packed.tolist() == [[0x21, 0xF3]]
    assert torch.equal(N.unpack_nibbles(packed), torch.tensor([[1, 2, 3, 15]], dtype=torch.uint8))
    with pytest.raises(N.NumericsError):
        N.e2m1_nibbles(torch.tensor([2.5]))
    with pytest.raises(N.NumericsError):
        N.e0m3_nibbles(torch.tensor([8.0]))
    with pytest.raises(N.NumericsError):
        N.scale_bytes(torch.tensor([0.3]))          # not E4M3-exact


def test_tag_does_not_change_scale():
    s = torch.tensor([2 ** -9, 0.5, 1.0, 448.0])
    sb = N.scale_bytes(s, torch.tensor([True, False, True, True]))
    val, flag = N.split_scale_bytes(sb)
    assert torch.equal(val, s) and flag.tolist() == [True, False, True, True]
