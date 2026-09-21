"""V10: vectorized fake quantizers versus independent scalar references and boundary cases."""
import json
import math
import os

import numpy as np
import pytest
import torch

from campaign import quant as Q
from campaign import refquant as R
from quantize.causal_four_over_six import quantize_rows
from quantize.quantizer import (quant_mix_4_6, quant_nvfp4, quant_nvfp4_4over6, quant_nvfp4_razer_e4m3)

FINDINGS = {}


def record(key, value):
    FINDINGS[key] = value
    out = os.environ.get('V10_FINDINGS')
    if out:
        with open(out, 'w') as f:
            json.dump(FINDINGS, f, indent=1, sort_keys=True, default=str)


def weights(kind, rows=24, cols=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    if kind == 'normal':
        w = torch.randn(rows, cols, generator=g) * 0.02
    elif kind == 'heavy':
        torch.manual_seed(seed)
        w = (torch.distributions.StudentT(2.0).sample((rows, cols)) * 0.01)
    elif kind == 'outlier_blocks':
        w = torch.randn(rows, cols, generator=g) * 0.02
        w[::5, ::16] *= 40.0
    elif kind == 'subnormal_scale':
        w = torch.randn(rows, cols, generator=g) * 1e-6
        w[0, 0] = 30.0  # one huge global maximum forces subnormal / clamped block scales elsewhere
    elif kind == 'with_zero_blocks':
        w = torch.randn(rows, cols, generator=g) * 0.02
        w[3, :16] = 0.0
        w[7, 32:48] = 0.0
    else:
        raise ValueError(kind)
    return w.to(torch.bfloat16)


KINDS = ['normal', 'heavy', 'outlier_blocks', 'subnormal_scale', 'with_zero_blocks']


def test_e4m3_reference_matches_torch_cast():
    vals = [v for v, _ in R._E4M3]
    probe = []
    for a, b in zip(vals[:-1], vals[1:]):
        probe += [a, b, (a + b) / 2, np.nextafter(np.float32((a + b) / 2), np.float32(0)), np.nextafter(np.float32((a + b) / 2), np.float32(1e9))]
    rng = np.random.default_rng(0)
    probe += list(rng.uniform(2 ** -9, 448, 20000)) + list(np.exp(rng.uniform(math.log(2 ** -9), math.log(448), 20000)))
    t = torch.tensor(probe, dtype=torch.float32)
    got = t.to(torch.float8_e4m3fn).to(torch.float64).tolist()
    ref = [R.round_e4m3(float(np.float32(x))) for x in probe]
    bad = [(p, g, r) for p, g, r in zip(probe, got, ref) if g != r]
    record('e4m3_cast', dict(values=len(vals), probes=len(probe), mismatches=len(bad), examples=bad[:5]))
    assert not bad


def test_bf16_reference_matches_torch_cast():
    rng = np.random.default_rng(1)
    probe = np.concatenate([rng.standard_normal(20000) * 10.0 ** rng.integers(-8, 3, 20000)]).astype(np.float32)
    got = torch.tensor(probe).to(torch.bfloat16).to(torch.float64).tolist()
    ref = [R.round_bf16(float(x)) for x in probe]
    bad = sum(g != r for g, r in zip(got, ref))
    record('bf16_cast', dict(probes=len(probe), mismatches=bad))
    assert bad == 0


@pytest.mark.parametrize('kind', KINDS)
def test_four_over_six_vs_reference(kind):
    w = weights(kind)
    vec = quant_nvfp4_4over6(w, 4, 16).double().numpy()
    ref, amb = R.ref_four_over_six(w.float().numpy())
    blocks_v, blocks_r = vec.reshape(-1, 16), ref.reshape(-1, 16)
    mism = [i for i in range(blocks_v.shape[0]) if not np.array_equal(blocks_v[i], blocks_r[i])]
    record(f'four_over_six_{kind}', dict(blocks=blocks_v.shape[0], ambiguous_alpha_ties=amb, mismatched_blocks=len(mism)))
    assert len(mism) <= amb


@pytest.mark.parametrize('kind', KINDS)
def test_e0m3_vs_reference(kind):
    w = weights(kind)
    vec = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always').double().numpy()
    ref = R.ref_e0m3(w.float().numpy())
    n = int((vec != ref).sum())
    record(f'e0m3_{kind}', dict(elements=vec.size, mismatches=n))
    assert n == 0


@pytest.mark.parametrize('kind', KINDS)
def test_nvfp4_vs_reference(kind):
    w = weights(kind)
    vec = quant_nvfp4(w, 4, 16).double().numpy()
    ref = R.ref_nvfp4(w.float().numpy())
    n = int((vec != ref).sum())
    record(f'nvfp4_{kind}', dict(elements=vec.size, mismatches=n))
    assert n == 0


def test_activation_rows_vs_reference():
    g = torch.Generator().manual_seed(3)
    x = (torch.randn(2, 5, 48, generator=g) * torch.tensor([0.1, 5.0, 50.0]).repeat_interleave(16)).to(torch.bfloat16)
    x[0, 1] = 0.0  # an all-zero token row must not produce NaN
    vec = quantize_rows(x).double().numpy()
    ref, amb = R.ref_four_over_six_rows(x.float().numpy())
    mism = int((vec.reshape(-1, 16) != ref.reshape(-1, 16)).any(-1).sum())
    record('four_over_six_rows', dict(blocks=vec.size // 16, ambiguous=amb, mismatched_blocks=mism,
                                      zero_row_finite=bool(np.isfinite(vec[0, 1]).all())))
    assert mism <= amb and np.isfinite(vec).all()


def _grid_membership(dq, scale_fn, grid):
    return dq


def test_representable_values_and_saturation():
    w = weights('outlier_blocks', rows=64, cols=256, seed=11)
    wf = w.float().reshape(-1, 16)
    gs = wf.abs().amax() / 2688.0
    s = wf / gs
    bmax = s.abs().amax(-1, keepdim=True)
    # FourOverSix: every output divided by (chosen scale * gs) must be an E2M1 level.
    q = quant_nvfp4_4over6(w, 4, 16).float().reshape(-1, 16) / gs
    s6 = (bmax / 6).clamp(max=448, min=2 ** -9).to(torch.float8_e4m3fn).float()
    s4 = (bmax / 4).clamp(max=448, min=2 ** -9).to(torch.float8_e4m3fn).float()
    lv = torch.tensor(R.E2M1_LEVELS)
    # BF16 output rounding has a relative step of 2**-7; allow one half step of that.
    tol = lambda r: ((r.abs().unsqueeze(-1) - lv).abs() <= lv * 2 ** -8 + 1e-6).any(-1)
    ok6, ok4 = tol(q / s6), tol(q / s4)
    assert (ok6.all(-1) | ok4.all(-1)).all()
    # E0M3: integer codes within [-7, 7].
    e = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always').float().reshape(-1, 16) / gs
    s7 = (bmax * (1.0 / 7.0)).clamp(max=448, min=2 ** -9).to(torch.float8_e4m3fn).float()
    codes = e / s7
    assert ((codes - codes.round()).abs() <= codes.round().abs() * 2 ** -8 + 1e-6).all() and codes.abs().max() <= 7 * (1 + 2 ** -8)
    # NVFP4 saturation clamp: never above 6 even with subnormal scales.
    sub = weights('subnormal_scale', rows=64, cols=256, seed=12)
    n4 = quant_nvfp4(sub, 4, 16).float().reshape(-1, 16)
    gs2 = sub.float().abs().amax() / 2688.0
    sb = (sub.float().reshape(-1, 16) / gs2).abs().amax(-1, keepdim=True)
    sc = (sb / 6).clamp(max=448, min=2 ** -9).to(torch.float8_e4m3fn).float()
    assert ((n4 / gs2 / sc).abs() <= 6 * (1 + 2 ** -8)).all()
    record('representable_values', dict(four_over_six='E2M1 levels under chosen alpha', e0m3='integers in [-7,7]',
                                        nvfp4_saturation='|code|<=6 with subnormal scales'))


def test_zero_and_sign_behaviour():
    w = weights('normal', rows=16, cols=64, seed=4)
    for fn in (Q.four_over_six, Q.e0m3, Q.nvfp4):
        a, b = fn(w), fn(-w)
        assert torch.equal(a, -b), fn.__name__
        z = w.clone()
        z[2, :16] = 0
        assert (fn(z)[2, :16] == 0).all()
    record('sign_symmetry_and_zero_blocks', 'q(-w) == -q(w) exactly; all-zero scale blocks dequantize to exact zeros')


def test_alpha_choice_prefers_lower_error_with_ties_to_six():
    # gs = 1 (global max 2688), block max 6 -> s6 = 1, s4 = 1.5 exactly.
    base = torch.zeros(2, 16, dtype=torch.float32)
    base[0, 0] = 2688.0
    base[1] = torch.tensor([6.0, 3.0, 1.5, 0.75, 4.5, 0.0, 6.0, 3.0, 1.5, 0.75, 4.5, 0.0, 6.0, 3.0, 1.5, 0.75])
    q = quant_nvfp4_4over6(base, 4, 16).float()
    ref, amb = R.ref_four_over_six(base.numpy())
    assert np.array_equal(q.double().numpy(), ref)
    record('alpha_choice', dict(example_row=q[1].tolist(), ambiguous=amb))


def test_tie_rounding_conventions_are_documented():
    """Two documented mechanisms make quant_mix_4_6(clip='base', elect='never') differ from
    quant_nvfp4_4over6 (the campaign never uses the former as the baseline):
      1. code ties: FourOverSix buckets tie toward the smaller magnitude, _quant_e2m1 ties away from zero;
      2. scale arithmetic: quant_mix_4_6 multiplies by float32(alpha/6), FourOverSix divides by 6,
         which rounds to a different UE4M3 scale for some block maxima."""
    from quantize.quantizer import _quant_e2m1
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    released = _quant_e2m1(ties, torch.tensor(1.0)).tolist()
    bucket = [R.e2m1_nearest_ties_down(float(t)) for t in ties]
    assert released == [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    assert bucket == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    bm = torch.exp(torch.linspace(math.log(2 ** -8), math.log(2600.0), 2_000_000))
    div = (bm / 6.0).clamp(max=448, min=2 ** -9).to(torch.float8_e4m3fn).float()
    mul = (bm * (1.0 / 6.0)).clamp(max=448, min=2 ** -9).to(torch.float8_e4m3fn).float()
    diff = int((div != mul).sum())
    record('tie_conventions', dict(tie_inputs=ties.tolist(), released_e2m1_ties_away=released,
                                   four_over_six_bucket_ties_down=bucket,
                                   scale_rounding_disagreements_div6_vs_mul_1over6=dict(probes=bm.numel(), differing=diff)))
    assert released != bucket


def test_mix46_never_equals_four_over_six_off_ties_and_always_equals_e0m3():
    diffs = {}
    mids = torch.tensor(R.E2M1_MIDS)
    for kind in KINDS:
        w = weights(kind, rows=48, cols=128, seed=5)
        a = quant_nvfp4_4over6(w, 4, 16)
        b = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='base', elect='never')
        wf = w.float()
        gs = wf.abs().amax() / 2688.0
        blocks = (wf / gs).reshape(-1, 16)
        bm = blocks.abs().amax(-1, keepdim=True)
        s6 = (bm / 6).clamp(max=448, min=2 ** -9).to(torch.float8_e4m3fn).float()
        s4 = (bm / 4).clamp(max=448, min=2 ** -9).to(torch.float8_e4m3fn).float()
        tie = lambda r: (r.abs().unsqueeze(-1) == mids).any(-1)
        tie_any = (tie(blocks / s6) | tie(blocks / s4)).reshape(w.shape)
        differ = a != b
        diffs[kind] = dict(elements=w.numel(), differing=int(differ.sum()), exact_tie_elements=int(tie_any.sum()),
                           differing_not_at_exact_tie=int((differ & ~tie_any).sum()))
        # Differences may occur at exact ties, or anywhere in a block whose alpha choice flips because of a tie.
        flip_blocks = differ.reshape(-1, 16).any(-1)
        assert bool((tie_any.reshape(-1, 16).any(-1) | ~flip_blocks).all()), kind
        e8 = quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')
        e16 = quant_mix_4_6(w, 4, 16, type_block=(16, 64), clip='a1', elect='always')
        assert torch.equal(e8, e16) and torch.equal(e8, Q.e0m3(w))
    record('mix46_base_never_vs_four_over_six', dict(per_kind=diffs, conclusion='every differing block contains an exact code tie; '
           'elect=always is bitwise E0M3 for 8x64 and 16x64 type blocks'))


def test_noncontiguous_and_orientation():
    w = weights('normal', rows=64, cols=128, seed=6)
    t = w.t().contiguous().t()  # same values, non-contiguous strides
    assert not t.is_contiguous()
    for fn in (quant_nvfp4_4over6, quant_nvfp4):
        assert torch.equal(fn(w, 4, 16), fn(t, 4, 16))
    assert torch.equal(Q.e0m3(w), Q.e0m3(t)) and torch.equal(Q.four_over_six(w), Q.four_over_six(t))
    x = torch.randn(3, 7, 64, dtype=torch.bfloat16)
    x4 = torch.randn(2, 3, 5, 64, dtype=torch.bfloat16)
    assert quantize_rows(x).shape == x.shape and quantize_rows(x4).shape == x4.shape
    # per-token independence (causality): each row equals its own single-row quantization
    y = quantize_rows(x)
    for i in range(3):
        for j in range(7):
            assert torch.equal(y[i, j], quantize_rows(x[i, j:j + 1])[0])
    record('noncontiguous_orientation', 'non-contiguous weights bitwise equal; 3-D/4-D activations keep shape; rows independent')


def test_alignment_and_nonfinite_rejection():
    with pytest.raises(Q.CandidateError):
        Q.four_over_six(torch.randn(32, 24, dtype=torch.bfloat16))  # 32*24 divisible by 16, K is not
    with pytest.raises(Q.CandidateError):
        Q.e0m3(torch.randn(8, 64, 2, dtype=torch.bfloat16))
    bad = torch.randn(16, 64, dtype=torch.bfloat16)
    bad[0, 0] = float('nan')
    with pytest.raises(Q.CandidateError):
        Q.four_over_six(bad)
    bad[0, 0] = float('inf')
    with pytest.raises(Q.CandidateError):
        Q.nvfp4(bad)
    with pytest.raises(Q.CandidateError):
        Q.e0m3(torch.zeros(16, 64, dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        quantize_rows(torch.randn(4, 24))
    # the unguarded released function silently straddles rows for this shape:
    w = torch.randn(32, 24)
    silently = quant_nvfp4_4over6(w, 4, 16)
    record('alignment_rejection', dict(released_function_accepts_K24=bool(silently.shape == w.shape),
                                       campaign_wrappers_reject=True,
                                       released_all_zero_tensor_output_finite=bool(torch.isfinite(quant_nvfp4_4over6(torch.zeros(16, 16), 4, 16)).all())))


def test_dtype_conversions():
    w = weights('normal', rows=16, cols=64, seed=7)
    for fn in (Q.four_over_six, Q.e0m3, Q.nvfp4, Q.nover6):
        out = fn(w)
        assert out.dtype == torch.bfloat16
        assert torch.equal(fn(w.float()), out)  # float32 and bf16 inputs of identical values agree
    d = Q.e0m3(w).float() - Q.four_over_six(w).float()
    assert d.dtype == torch.float32 and torch.isfinite(d).all()
    record('dtypes', 'candidates return bfloat16; direction D = alt.float() - base.float() in float32')


def test_rowwise_baselines_match_released_single_row():
    g = torch.Generator().manual_seed(9)
    x = (torch.randn(6, 64, generator=g) * 3).to(torch.bfloat16)
    rz = Q.razer_e4m3_rows(x)
    nv = Q.nvfp4_rows(x)
    for i in range(6):
        assert torch.equal(rz[i], quant_nvfp4_razer_e4m3(x[i:i + 1], 4, 16)[0])
        assert torch.equal(nv[i], quant_nvfp4(x[i:i + 1], 4, 16)[0])
    record('rowwise_baselines', 'razer_e4m3_rows and nvfp4_rows equal the released functions applied per token row')


def test_chunked_row_quantization_is_bitwise_identical():
    g = torch.Generator().manual_seed(21)
    for kind in Q.ROW_WISE:
        fn = Q.ACTIVATION[kind]
        x = (torch.randn(3, 37, 64, generator=g) * 4).to(torch.bfloat16)
        full = fn(x)
        for chunk in (1, 7, 16, 110, 111, 1000):
            assert torch.equal(Q.chunked_rows(fn, x, chunk), full), (kind, chunk)
    record('chunked_rows', 'row-chunked activation quantization equals the unchunked call bitwise for every row-wise quantizer')
