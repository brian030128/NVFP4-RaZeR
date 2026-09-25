"""Kernel-level correctness of the patched SM120 mixed GEMM (through the C ABI, no model code).

1. Decode probe: with the activation operand an exact identity (code 1.0, scale 1.0) and unit
   epilogue scales, D equals codebook(nibble, format) * scale of every weight element exactly (both
   factors are exact in bf16). E2M1 and E0M3 decode every nonzero nibble differently, so this
   identifies, element by element, which format the tensor core executed -- and checks it against
   the format tag, i.e. the selector tile, the element belongs to.
2. Random GEMMs against an FP64 reference built from the same decoded operands, over model and
   boundary shapes, all-E2M1 / all-E0M3 / mixed maps.
3. Negative control: the unpatched binary of the same build is badly wrong whenever E0M3 tiles
   exist, and bit-identical to the patched one when none do.
4. Epilogue: per-token / per-channel scale vectors and bias, and async / repeated / stream use.
"""
import pytest
import torch

from mixfp4_sm120 import numerics as N
from mixfp4_sm120.lib import Kernel, sf_buffer_size, sf_offset_formula

pytestmark = pytest.mark.gpu

MIXED = ['n16k64_wA', 'n16k64_wA_8x1', 'n16k64_wA_n64', 'n16k64_wA_n32', 'n16k64_wA_n16', 'n8k64_wB']


def linear_gemm(kern, wp, wsf, gs_w, xp, xsf, gs_x, n, t, k, **kw):
    """y[t, n] = x W^T through whichever operand the configuration puts the weights on."""
    if kern.weight_operand == 0:
        return kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=gs_w, scale_n=gs_x, **kw)
    return kern.gemm(xp, xsf, wp, wsf, t, n, k, scale_m=gs_x, scale_n_default=gs_w, **kw)


def kernel(name, unpatched=False):
    try:
        return Kernel.load(name, allow_unpatched=unpatched)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f'{name} not built: {e}')


def place(sb, k):
    """Row-major [rows, k/16] scale bytes -> kernel layout buffer."""
    rows = sb.shape[0]
    buf = torch.zeros(sf_buffer_size(rows, k), dtype=torch.uint8, device=sb.device)
    r = torch.arange(rows, device=sb.device)[:, None]
    kb = torch.arange(k // 16, device=sb.device)[None, :]
    buf[sf_offset_formula(r, kb, k).reshape(-1)] = sb.reshape(-1)
    return buf


def random_weight_operand(n, k, mask, type_block, seed):
    """Random nonzero nibbles (both signs), random legal scales, tags from `mask`."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    mag = torch.randint(1, 8, (n, k), generator=g)
    sign = torch.randint(0, 2, (n, k), generator=g)
    nib = (mag | (sign << 3)).to(torch.uint8)
    # scales: random E4M3 values in [2^-6, 8]
    s = torch.exp2(torch.randint(-6, 3, (n, k // 16), generator=g).float()) * (1 + torch.randint(0, 8, (n, k // 16), generator=g).float() / 8)
    s = s.to(torch.float8_e4m3fn).float()
    flags = N.expand_mask(mask, type_block, (n, k)) if mask is not None else torch.zeros((n, k // 16), dtype=torch.bool)
    return nib.cuda(), N.scale_bytes(s, flags).cuda()


def identity_operand(t, k, offset=0):
    """[t, k] E2M1 operand with code 1.0 at (i, offset + i), zero elsewhere; scale bytes 1.0."""
    nib = torch.zeros((t, k), dtype=torch.uint8)
    i = torch.arange(t)
    nib[i, offset + i] = 2                     # E2M1 index 2 = 1.0
    sb = torch.full((t, k // 16), 0x38, dtype=torch.uint8)   # UE4M3 1.0
    return nib.cuda(), sb.cuda()


@pytest.mark.parametrize('name', MIXED)
@pytest.mark.parametrize('n,k', [(128, 128), (256, 512), (48, 256), (4096, 128), (1024, 320)])
@pytest.mark.parametrize('pattern', ['random', 'all', 'none', 'single'])
def test_decode_probe(device, name, n, k, pattern):
    kern = kernel(name)
    tb = kern.type_block
    grid = (n // tb[0], k // tb[1])
    g = torch.Generator(device='cpu').manual_seed(n * 7 + k)
    if pattern == 'random':
        mask = torch.rand(grid, generator=g) < 0.5
    elif pattern == 'all':
        mask = torch.ones(grid, dtype=torch.bool)
    elif pattern == 'none':
        mask = torch.zeros(grid, dtype=torch.bool)
    else:
        mask = torch.zeros(grid, dtype=torch.bool)
        mask[torch.randint(0, grid[0], (1,), generator=g), torch.randint(0, grid[1], (1,), generator=g)] = True
    wnib, wsb = random_weight_operand(n, k, mask, tb, seed=n + k)
    # probe K in windows of up to 1024 tokens
    got = torch.empty((n, k), dtype=torch.float64, device='cuda')
    for off in range(0, k, 1024):
        t = min(1024, k - off)
        xnib, xsb = identity_operand(t, k, off)
        d = linear_gemm(kern, N.pack_nibbles(wnib), place(wsb, k), 1.0, N.pack_nibbles(xnib), place(xsb, k), None, n, t, k)
        got[:, off:off + t] = d.double().t()     # y is [t, n] for both operand placements
    want = N.decode_exact(wnib, wsb, 1.0)
    executed_e0m3 = (got != N.decode_exact(wnib, wsb & 0x7F, 1.0))
    tagged = N.expand_mask(mask, tb, (n, k)).cuda().repeat_interleave(16, 1)
    assert torch.equal(executed_e0m3, tagged), 'format executed by the tensor core differs from the tag'
    assert torch.equal(got, want)


@pytest.mark.parametrize('name', MIXED)
def test_decode_probe_unpatched_is_e2m1(device, name):
    kern = kernel(name, unpatched=True)
    n, k = 256, 256
    tb = kern.type_block
    mask = torch.ones((n // tb[0], k // tb[1]), dtype=torch.bool)
    wnib, wsb = random_weight_operand(n, k, mask, tb, seed=3)
    xnib, xsb = identity_operand(k, k)
    d = linear_gemm(kern, N.pack_nibbles(wnib), place(wsb, k), 1.0, N.pack_nibbles(xnib), place(xsb, k), None, n, k, k)
    assert torch.equal(d.double().t(), N.decode_exact(wnib, wsb & 0x7F, 1.0))


def fp64_reference(wnib, wsb, gs_w, xnib, xsb, gs_x, bias=None):
    w = N.decode_exact(wnib, wsb, gs_w)
    x = N.decode_exact(xnib, xsb, gs_x[:, None])
    y = x @ w.t()
    if bias is not None:
        y = y + bias.double()
    return y, (x.abs() @ w.abs().t())


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))


SHAPES = [  # (tokens, out, in): Llama-3.1-8B, Qwen3-4B, Mistral, Phi-4, Qwen3.8 N=48, boundaries
    (1, 4096, 4096), (2, 1024, 4096), (3, 14336, 4096), (7, 4096, 14336), (16, 2560, 2560),
    (17, 9728, 2560), (64, 2560, 9728), (127, 6144, 5120), (128, 5120, 17920), (129, 48, 2560),
    (256, 4096, 64), (300, 256, 192), (1000, 1024, 320), (2048, 4096, 4096), (4096, 1024, 4096),
]


@pytest.mark.parametrize('name', MIXED)
@pytest.mark.parametrize('t,n,k', SHAPES)
def test_random_gemm(device, name, t, n, k):
    kern = kernel(name)
    tb = kern.type_block
    g = torch.Generator(device='cpu').manual_seed(t * 31 + n + k)
    mask = torch.rand((n // tb[0], k // tb[1]), generator=g) < 0.4
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(t, k, generator=g)
    x[:, torch.randperm(k, generator=g)[:4]] *= 30
    x = x.cuda().bfloat16()
    wnib, wsb, gs_w = N.quantize_weight(w, 'map', mask, tb)
    xnib, xsb, gs_x = N.quantize_act(x, 'four_over_six_rows')
    d = linear_gemm(kern, N.pack_nibbles(wnib), place(wsb, k), float(gs_w), N.pack_nibbles(xnib), place(xsb, k), gs_x, n, t, k)
    ref, absdot = fp64_reference(wnib, wsb, gs_w, xnib, xsb, gs_x)
    got = d.double()
    assert got.shape == ref.shape
    err = (got - ref).abs()
    # bf16 output rounding (2^-9 relative) plus FP32 accumulation (bounded by 2^-20 of sum |a b|)
    bound = 2.0 ** -8 * ref.abs() + 2.0 ** -20 * absdot + 1e-30
    assert bool((err <= bound).all()), f'max err/bound {float((err / bound).max()):.3f}'
    assert rel(got, ref) < 3e-3


@pytest.mark.parametrize('name', MIXED)
@pytest.mark.parametrize('frac', [0.0, 0.3, 1.0])
def test_unpatched_negative_control(device, name, frac):
    kern, bad = kernel(name), kernel(name, unpatched=True)
    t, n, k = 256, 1024, 2048
    tb = kern.type_block
    g = torch.Generator(device='cpu').manual_seed(5)
    mask = torch.rand((n // tb[0], k // tb[1]), generator=g) < frac
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    wnib, wsb, gs_w = N.quantize_weight(w, 'map', mask, tb)
    xnib, xsb, gs_x = N.quantize_act(x, 'four_over_six_rows')
    args = (N.pack_nibbles(wnib), place(wsb, k), float(gs_w), N.pack_nibbles(xnib), place(xsb, k), gs_x, n, t, k)
    good = linear_gemm(kern, *args)
    wrong = linear_gemm(bad, *args)
    ref, _ = fp64_reference(wnib, wsb, gs_w, xnib, xsb, gs_x)
    if frac == 0.0:
        assert torch.equal(good, wrong)
    else:
        assert rel(wrong, ref) > 20 * rel(good, ref)


@pytest.mark.parametrize('name', MIXED)
def test_epilogue_vectors_and_bias(device, name):
    """D = bf16((s_m[m] s_n[n]) acc + bias[m]) with vector scales on both axes and a bias."""
    kern = kernel(name)
    t, n, k = 77, 384, 1024
    tb = kern.type_block
    g = torch.Generator(device='cpu').manual_seed(9)
    mask = torch.rand((n // tb[0], k // tb[1]), generator=g) < 0.5
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    wnib, wsb, gs_w = N.quantize_weight(w, 'map', mask, tb)
    xnib, xsb, gs_x = N.quantize_act(x, 'four_over_six_rows')
    s_m = (torch.rand(n, generator=g) + 0.5).cuda()
    bias = (torch.randn(n, generator=g)).cuda().bfloat16()
    wp, wsf, xp, xsf = N.pack_nibbles(wnib), place(wsb, k), N.pack_nibbles(xnib), place(xsb, k)
    if kern.weight_operand == 0:       # per-channel vector on M, per-token on N, bias on M
        d = kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m=s_m * gs_w, scale_n=gs_x, bias=bias)
    else:                              # per-token vector on M, per-channel on N, bias on N
        d = kern.gemm(xp, xsf, wp, wsf, t, n, k, scale_m=gs_x, scale_n=s_m * gs_w, bias=bias)
    ref, absdot = fp64_reference(wnib, wsb, 1.0, xnib, xsb, gs_x)
    ref = ref * (s_m * gs_w).double()[None, :] + bias.double()[None, :]
    err = (d.double() - ref).abs()
    bound = 2.0 ** -8 * ref.abs() + 2.0 ** -20 * absdot * (s_m * gs_w).double()[None, :] + 1e-30
    assert bool((err <= bound).all())


@pytest.mark.parametrize('name', MIXED)
def test_async_repeat_and_streams(device, name):
    """Many launches without host synchronisation, on two streams, reusing output buffers."""
    kern = kernel(name)
    t, n, k = 96, 512, 1024
    tb = kern.type_block
    g = torch.Generator(device='cpu').manual_seed(11)
    cases = []
    for i in range(12):
        mask = torch.rand((n // tb[0], k // tb[1]), generator=g) < 0.5
        w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
        x = torch.randn(t, k, generator=g).cuda().bfloat16()
        wnib, wsb, gs_w = N.quantize_weight(w, 'map', mask, tb)
        xnib, xsb, gs_x = N.quantize_act(x, 'four_over_six_rows')
        ops = (N.pack_nibbles(wnib), place(wsb, k), float(gs_w), N.pack_nibbles(xnib), place(xsb, k), gs_x)
        cases.append((ops, linear_gemm(kern, *ops, n, t, k).clone()))
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outs = [torch.empty((t, n), dtype=torch.bfloat16, device='cuda') for _ in range(2)]
    results = []
    for rep in range(3):
        for i, (ops, want) in enumerate(cases):
            s = streams[i % 2]
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                o = linear_gemm(kern, *ops, n, t, k, out=outs[i % 2])
                results.append((i, o.clone()))
            torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    for i, o in results:
        assert torch.equal(o, cases[i][1])


DECOMP = [('heuristic', 1, 0), ('data_parallel', 1, 1), ('split2', 2, 2), ('split4', 4, 2), ('split8', 8, 2),
          ('stream_k', 1, 3)]


@pytest.mark.parametrize('dname,splits,mode', DECOMP)
@pytest.mark.parametrize('t,n,k', [(1, 4096, 4096), (7, 1024, 4096), (64, 4096, 14336), (200, 2560, 9728), (1, 48, 2560)])
def test_stream_k(device, dname, splits, mode, t, n, k):
    """Stream-K build: every decomposition is correct, deterministic, and executes the same formats."""
    kern = kernel('n16k64_wA_sk')
    tb = kern.type_block
    g = torch.Generator(device='cpu').manual_seed(t + n + k + splits)
    mask = torch.rand((n // tb[0], k // tb[1]), generator=g) < 0.4
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    wnib, wsb, gs_w = N.quantize_weight(w, 'map', mask, tb)
    xnib, xsb, gs_x = N.quantize_act(x, 'four_over_six_rows')
    args = (N.pack_nibbles(wnib), place(wsb, k), float(gs_w), N.pack_nibbles(xnib), place(xsb, k), gs_x, n, t, k)
    d = linear_gemm(kern, *args, splits=splits, decomposition=mode)
    ref, absdot = fp64_reference(wnib, wsb, gs_w, xnib, xsb, gs_x)
    err = (d.double() - ref).abs()
    assert bool((err <= 2.0 ** -8 * ref.abs() + 2.0 ** -20 * absdot + 1e-30).all())
    assert torch.equal(d, linear_gemm(kern, *args, splits=splits, decomposition=mode))   # deterministic
    # exact decode probe through the split path
    if t == 1 and n <= 4096:
        pn, psb = random_weight_operand(n, 256, torch.rand((n // tb[0], 4), generator=g) < 0.5, tb, seed=t)
        inib, isb = identity_operand(256, 256)
        pd = linear_gemm(kern, N.pack_nibbles(pn), place(psb, 256), 1.0, N.pack_nibbles(inib), place(isb, 256), None,
                         n, 256, 256, splits=splits, decomposition=mode)
        assert torch.equal(pd.double().t(), N.decode_exact(pn, psb, 1.0))


def test_non_stream_k_rejects_splits(device):
    kern = kernel('n16k64_wA')
    z = torch.zeros(128, 64, dtype=torch.uint8, device='cuda')
    with pytest.raises(ValueError):
        kern.gemm(z, z, z, z, 128, 128, 128, splits=2)
