"""Frozen map -> artifact -> reload -> NativeLinear, on a synthetic module set.

Checks export/reload identity, tamper and non-uniform-tag rejection, and the native Linear against
(a) an FP64 reference from the same decoded operands (native arithmetic error only) and
(b) the fake-quant F.linear (native vs fake-quant difference), with (c) the quantization error of
the fake-quant path against the BF16 layer for scale.
"""
import json

import pytest
import torch
import torch.nn.functional as F

from mixfp4_sm120 import artifact as A
from mixfp4_sm120 import mapio
from mixfp4_sm120 import numerics as N
from mixfp4_sm120.lib import Kernel
from mixfp4_sm120.linear import NativeLinear

pytestmark = pytest.mark.gpu
TB = (16, 64)


def make_modules(seed=0, bias=False):
    g = torch.Generator(device='cpu').manual_seed(seed)
    shapes = {'layers.0.q_proj': (1024, 512), 'layers.0.down_proj': (512, 1536), 'layers.1.v_proj': (48, 512),
              'layers.1.up_proj': (1536, 512)}
    mods = {}
    for name, (o, i) in shapes.items():
        lin = torch.nn.Linear(i, o, bias=bias, dtype=torch.bfloat16, device='cuda')
        with torch.no_grad():
            w = torch.randn(o, i, generator=g) * 0.02
            w[:, torch.randperm(i, generator=g)[:3]] *= 20
            lin.weight.copy_(w)
            if bias:
                lin.bias.copy_(torch.randn(o, generator=g) * 0.1)
        mods[name] = lin
    return mods


def write_map(tmp_path, mods, frac=0.3, seed=1):
    g = torch.Generator(device='cpu').manual_seed(seed)
    masks = {n: torch.rand(m.weight.shape[0] // TB[0], m.weight.shape[1] // TB[1], generator=g) < frac
             for n, m in mods.items()}
    header = mapio.build_header(protocol_id='unit-test', policy=dict(name='synthetic'),
                                model=dict(model_id='synthetic', revision='0'), type_block=TB, masks=masks,
                                weight_shapes={n: tuple(m.weight.shape) for n, m in mods.items()},
                                source_manifest_sha256='none', calibration_manifest_sha256='none')
    digest, path = mapio.write_map(tmp_path / 'synthetic.mixfp4map', header, masks)
    return path, digest, masks


@pytest.fixture(scope='module')
def kern():
    try:
        return Kernel.load('n16k64_wA')
    except Exception as e:  # noqa: BLE001
        pytest.skip(f'n16k64_wA not built: {e}')


def test_export_reload_roundtrip(device, tmp_path):
    mods = make_modules()
    path, digest, masks = write_map(tmp_path, mods)
    meta = A.export(tmp_path / 'art', mods, kind='map', map_path=path, map_sha256=digest)
    assert meta['map']['sha256'] == digest
    meta2, weights = A.load(tmp_path / 'art')
    assert meta2 == meta
    rec = A.masks_from_artifact(weights)
    for n, m in mods.items():
        assert torch.equal(rec[n], masks[n])
        nib, sb, gs = N.quantize_weight(m.weight, 'map', masks[n], TB)
        assert torch.equal(weights[n].packed, N.pack_nibbles(nib))
        assert torch.equal(weights[n].scales, sb)
        assert weights[n].global_scale == float(gs)
        assert torch.equal(weights[n].decode(), A.fake_quant_weight(m.weight, 'map', masks[n], TB))
    sizes = meta['sizes_bytes']
    assert sizes['packed'] * 4 == sizes['bf16_weights']            # 4 bits per weight
    assert sizes['scales'] * 32 == sizes['bf16_weights']           # 1 byte per 16 weights


def test_tamper_detection(device, tmp_path):
    mods = make_modules()
    path, digest, _ = write_map(tmp_path, mods)
    A.export(tmp_path / 'art', mods, kind='map', map_path=path, map_sha256=digest)
    w = tmp_path / 'art' / 'weights.safetensors'
    data = bytearray(w.read_bytes())
    data[-10] ^= 0x01
    w.write_bytes(bytes(data))
    with pytest.raises(A.ArtifactError):
        A.load(tmp_path / 'art')
    with pytest.raises(mapio.MapVerificationError):
        A.export(tmp_path / 'art2', mods, kind='map', map_path=path, map_sha256='0' * 64)


def test_nonuniform_tag_rejected():
    sb = torch.zeros((32, 8), dtype=torch.uint8)
    sb[0, 0] = 0x80                      # one scale block of a 16x64 tile tagged, the rest not
    with pytest.raises(A.ArtifactError):
        A.tile_flags(sb, TB)
    sb[:16, :4] = 0x80
    assert A.tile_flags(sb, TB).tolist() == [[True, False], [False, False]]


def fp64_linear(pw, x, act_kind):
    xnib, xsb, gsx = N.quantize_act(x.reshape(-1, x.shape[-1]), act_kind)
    xd = N.decode_exact(xnib, xsb, gsx[:, None])
    wd = N.decode_exact(N.unpack_nibbles(pw.packed), pw.scales, pw.global_scale)
    y = xd @ wd.t()
    if pw.bias is not None:
        y = y + pw.bias.double()
    return y.reshape(*x.shape[:-1], -1)


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('lead', [(1,), (5,), (2, 37), (4, 128), (1, 2048)])
def test_native_linear(device, kern, tmp_path, bias, lead):
    mods = make_modules(bias=bias)
    path, digest, masks = write_map(tmp_path, mods)
    A.export(tmp_path / 'art', mods, kind='map', map_path=path, map_sha256=digest)
    meta, weights = A.load(tmp_path / 'art')
    g = torch.Generator(device='cpu').manual_seed(sum(lead))
    rows = []
    for n, m in mods.items():
        nl = NativeLinear(weights[n], kern, meta['activation_quantizer'], name=n)
        x = torch.randn(*lead, m.in_features, generator=g)
        x[..., torch.randperm(m.in_features, generator=g)[:4]] *= 25
        x = x.cuda().bfloat16()
        y = nl(x)
        assert y.shape == (*lead, m.out_features) and y.dtype == torch.bfloat16 and y.is_contiguous()
        ref = fp64_linear(weights[n], x, meta['activation_quantizer'])
        fake = F.linear(N.fake_quant_act_rows(x, 'four_over_six_rows'), weights[n].decode(),
                        None if m.bias is None else m.bias)
        bf16 = m(x)
        rows.append((n, rel(y, ref), rel(fake, ref), rel(y, fake), rel(fake, bf16)))
        assert rel(y, ref) < 3e-3, rows[-1]          # native arithmetic error
        assert rel(fake, ref) < 6e-3, rows[-1]       # fake-quant's own BF16 rounding
        assert nl.calls == 1 and nl.tokens == x.numel() // m.in_features
    # the native error is not larger than the fake-quant path's rounding error, in aggregate
    assert sum(r[1] for r in rows) <= 1.2 * sum(r[2] for r in rows)


def test_empty_and_single_token(device, kern, tmp_path):
    mods = make_modules()
    path, digest, _ = write_map(tmp_path, mods)
    A.export(tmp_path / 'art', mods, kind='map', map_path=path, map_sha256=digest)
    meta, weights = A.load(tmp_path / 'art')
    n = 'layers.0.q_proj'
    nl = NativeLinear(weights[n], kern, 'four_over_six_rows', name=n)
    assert nl(torch.empty(0, 512, device='cuda', dtype=torch.bfloat16)).shape == (0, 1024)
    x = torch.randn(1, 1, 512, device='cuda', dtype=torch.bfloat16)
    assert rel(nl(x), fp64_linear(weights[n], x, 'four_over_six_rows')) < 3e-3
    with pytest.raises(AttributeError):
        _ = nl.weight


def test_e2m1_artifacts(device, kern, tmp_path):
    """NVFP4 and FourOverSix artifacts (no map) run on the same mixed kernel with all tags clear."""
    mods = make_modules()
    for kind in ('nvfp4', 'four_over_six'):
        meta = A.export(tmp_path / kind, mods, kind=kind)
        assert meta['activation_quantizer'] == A.WEIGHT_KIND_ACT[kind]
        _, weights = A.load(tmp_path / kind)
        for n, m in mods.items():
            nl = NativeLinear(weights[n], kern, meta['activation_quantizer'], name=n)
            x = torch.randn(33, m.in_features, device='cuda').bfloat16()
            assert rel(nl(x), fp64_linear(weights[n], x, meta['activation_quantizer'])) < 3e-3


def test_repeat_determinism_and_cuda_graph(device, kern, tmp_path):
    """Repeated calls are bitwise identical; a captured CUDA graph replays the same result and
    picks up new inputs written into its static input buffer."""
    mods = make_modules(bias=True)
    path, digest, _ = write_map(tmp_path, mods)
    A.export(tmp_path / 'art', mods, kind='map', map_path=path, map_sha256=digest)
    meta, weights = A.load(tmp_path / 'art')
    n = 'layers.0.down_proj'
    nl = NativeLinear(weights[n], kern, 'four_over_six_rows', name=n)
    g = torch.Generator(device='cpu').manual_seed(4)
    xs = [torch.randn(3, 1536, generator=g).cuda().bfloat16() for _ in range(3)]
    want = [nl(x) for x in xs]
    assert all(torch.equal(nl(x), w) for x, w in zip(xs, want))
    static_x = xs[0].clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            nl(static_x)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_y = nl(static_x)
    for x, w in zip(xs, want):
        static_x.copy_(x)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(static_y, w)


def test_shared_input_quantization(device, kern, tmp_path):
    """q/k/v-style calls on one input reuse its quantized activation; results are bitwise equal to
    unshared calls; an in-place update of the input, or a different tensor, is never served stale."""
    from mixfp4_sm120.linear import clear_quant_cache
    mods = make_modules()
    path, digest, _ = write_map(tmp_path, mods)
    A.export(tmp_path / 'art', mods, kind='map', map_path=path, map_sha256=digest)
    meta, weights = A.load(tmp_path / 'art')
    names = ['layers.0.q_proj', 'layers.1.v_proj', 'layers.1.up_proj']      # all take 512 inputs
    nls = [NativeLinear(weights[n], kern, 'four_over_six_rows', n) for n in names]
    if not kern.has_quant:
        pytest.skip('library without the CUDA quantizer')
    x = torch.randn(2, 9, 512, device='cuda').bfloat16()
    for nl in nls:
        nl.share_input = False
    want = [nl(x) for nl in nls]
    clear_quant_cache()
    for nl in nls:
        nl.share_input = True
    got = [nl(x) for nl in nls]
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert [nl.quant_reused for nl in nls] == [0, 1, 1]
    x.mul_(2.0)                                     # in place: the cached quantization is stale
    y = nls[1](x)
    nls[1].share_input = False
    assert torch.equal(y, nls[1](x)) and nls[1].quant_reused == 1
    nls[1].share_input = True
    x2 = x.clone()                                  # same values, different tensor: recomputed,
    assert torch.equal(nls[2](x2), nls[2](x)) and nls[2].quant_reused == 1   # never served stale


def test_shared_quantization_not_reused_across_capture(device, kern, tmp_path):
    """An eager call right before capture on the same stream must not let the graph skip quantization."""
    from mixfp4_sm120.linear import clear_quant_cache
    mods = make_modules()
    path, digest, _ = write_map(tmp_path, mods)
    A.export(tmp_path / 'art', mods, kind='map', map_path=path, map_sha256=digest)
    meta, weights = A.load(tmp_path / 'art')
    nl = NativeLinear(weights['layers.0.q_proj'], kern, 'four_over_six_rows', 'q')
    if not kern.has_quant:
        pytest.skip('library without the CUDA quantizer')
    clear_quant_cache()
    static_x = torch.randn(4, 512, device='cuda').bfloat16()
    s = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        nl(static_x)                                # eager, then capture on the same stream
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=s):
            static_y = nl(static_x)
    new = torch.randn(4, 512, device='cuda').bfloat16()
    static_x.copy_(new)
    graph.replay()
    torch.cuda.synchronize()
    nl.share_input = False
    assert torch.equal(static_y, nl(new))


@pytest.mark.parametrize('bias', [False, True])
def test_native_linear_weights_on_b(device, tmp_path, bias):
    """N8K64 map artifact on the weights-on-B build (fused path when available), vs FP64."""
    try:
        kb = Kernel.load('n8k64_wB')
    except Exception as e:  # noqa: BLE001
        pytest.skip(str(e))
    mods = make_modules(bias=bias)
    g = torch.Generator(device='cpu').manual_seed(8)
    masks = {n: torch.rand(m.weight.shape[0] // 8, m.weight.shape[1] // 64, generator=g) < 0.3 for n, m in mods.items()}
    header = mapio.build_header(protocol_id='unit-test', policy=dict(name='synthetic-n8'),
                                model=dict(model_id='synthetic', revision='0'), type_block=(8, 64), masks=masks,
                                weight_shapes={n: tuple(m.weight.shape) for n, m in mods.items()},
                                source_manifest_sha256='none', calibration_manifest_sha256='none')
    digest, path = mapio.write_map(tmp_path / 'n8.mixfp4map', header, masks)
    A.export(tmp_path / 'art', mods, kind='map', map_path=path, map_sha256=digest)
    meta, weights = A.load(tmp_path / 'art')
    for n, m in mods.items():
        nl = NativeLinear(weights[n], kb, meta['activation_quantizer'], name=n)
        x = torch.randn(3, 11, m.in_features, generator=g).cuda().bfloat16()
        assert rel(nl(x), fp64_linear(weights[n], x, meta['activation_quantizer'])) < 3e-3
