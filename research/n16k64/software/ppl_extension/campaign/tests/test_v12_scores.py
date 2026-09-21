"""V12: direct N16 directional scores versus per-sequence aggregation of N8 scores; SE and election identity."""
import json
import math
import os

import pytest
import torch

from campaign import tiles as T
from quantize.relinearized_format import common_descent_scores

FINDINGS = {}


def record(k, v):
    FINDINGS[k] = v
    if os.environ.get('V12_FINDINGS'):
        json.dump(FINDINGS, open(os.environ['V12_FINDINGS'], 'w'), indent=1, sort_keys=True, default=str)


def per_sequence_scores(seqs, o, k, dtype, integer=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    grads, dirs = [], []
    for _ in range(seqs):
        if integer:
            grads.append(torch.randint(-8, 9, (o, k), generator=g).to(dtype))
            dirs.append(torch.randint(-4, 5, (o, k), generator=g).to(dtype))
        else:
            grads.append((torch.randn(o, k, generator=g) * 1e-3).to(dtype))
            dirs.append((torch.randn(o, k, generator=g) * 1e-2).to(dtype))
    return grads, dirs


@pytest.mark.parametrize('o,k', [(16, 64), (48, 192), (256, 128)])
def test_direct_n16_equals_sum_of_children_exact_float64(o, k):
    grads, dirs = per_sequence_scores(5, o, k, torch.float64, integer=True)
    for g, d in zip(grads, dirs):
        n8 = T.directional_scores(g, d, (8, 64))
        n16_direct = T.directional_scores(g, d, (16, 64))
        n16_agg = T.aggregate_n8_to_n16(n8.reshape(1, -1), o, k).reshape(-1)
        assert torch.equal(n16_direct, n16_agg)
    record(f'exact_identity_float64_{o}x{k}', True)


def test_direct_n16_close_to_aggregated_float32_random():
    o, k = 512, 1024
    grads, dirs = per_sequence_scores(4, o, k, torch.float32)
    worst = 0.0
    for g, d in zip(grads, dirs):
        n8 = T.directional_scores(g, d, (8, 64))
        direct = T.directional_scores(g, d, (16, 64)).double()
        agg = T.aggregate_n8_to_n16(n8.double().reshape(1, -1), o, k).reshape(-1)
        rel = ((direct - agg).abs() / agg.abs().clamp_min(1e-12)).max().item()
        worst = max(worst, rel)
        assert torch.allclose(direct, agg, rtol=1e-4, atol=1e-10)
    record('float32_direct_vs_aggregated_max_rel_error', worst)


def test_flatten_order_groups_vertical_pairs_only():
    o, k = 48, 256
    s = 1
    labels = torch.arange((o // 8) * (k // 64), dtype=torch.float64).reshape(1, -1)
    agg = T.aggregate_n8_to_n16(labels, o, k).reshape(o // 16, k // 64)
    grid8 = labels.reshape(o // 8, k // 64)
    for r in range(o // 16):
        for c in range(k // 64):
            assert agg[r, c] == grid8[2 * r, c] + grid8[2 * r + 1, c]
    # uniquely identifiable pairs: use powers of two so every sum identifies its two children
    uniq = (2.0 ** torch.arange((o // 8) * (k // 64), dtype=torch.float64)).reshape(1, -1)
    a2 = T.aggregate_n8_to_n16(uniq, o, k).reshape(-1)
    for idx, v in enumerate(a2.tolist()):
        bits = [i for i in range(uniq.numel()) if int(v) >> i & 1]
        r, c = divmod(idx, k // 64)
        assert bits == [2 * r * (k // 64) + c, (2 * r + 1) * (k // 64) + c]
    record('flatten_order_pairs', 'N16 (r,c) = N8 (2r,c) + N8 (2r+1,c); no horizontal or cross-row mixing')


def test_election_identity_direct_vs_transformed_shards():
    o, k, seqs = 64, 256, 128
    g = torch.Generator().manual_seed(7)
    base = torch.randn(o, k, generator=g) * 1e-3
    grads = [(base + torch.randn(o, k, generator=g) * 2e-3) for _ in range(seqs)]
    d_ce = torch.randn(o, k, generator=g) * 1e-2
    ce8 = torch.stack([T.directional_scores(gr, d_ce, (8, 64)) for gr in grads])
    kl8 = torch.stack([T.directional_scores(gr * 0.5 + torch.randn(o, k, generator=g) * 1e-3, d_ce, (8, 64)) for gr in grads])
    ce16_direct = torch.stack([T.directional_scores(gr.double(), d_ce.double(), (16, 64)) for gr in grads])
    # direct KL path reconstructed from the same per-sequence N8 values to isolate aggregation
    kl16 = T.aggregate_n8_to_n16(kl8.double(), o, k)
    ce16 = T.aggregate_n8_to_n16(ce8.double(), o, k)
    m_agg = T.Moments.from_raw(ce16, kl16)
    m_dir = T.Moments.from_raw(ce16_direct, kl16)
    agree = {}
    for kk in (2, 3, 4, 5, 6):
        u1, u2 = T.upper_bound(m_agg, kk), T.upper_bound(m_dir, kk)
        assert torch.allclose(u1, u2, rtol=1e-4, atol=1e-9)
        e1, e2 = u1 < 0, u2 < 0
        margin = u1.abs() > 1e-6 * u1.abs().max()
        assert torch.equal(e1[margin], e2[margin])
        agree[f'k{kk}'] = dict(selected=int(e1.sum()), mask_equal=bool(torch.equal(e1, e2)))
    record('election_identity', agree)


def test_se_recomputed_after_summation_not_combined_from_children():
    torch.manual_seed(0)
    n = 128
    child_a = torch.randn(n, 1).double()
    child_b = (-0.9 * child_a + 0.1 * torch.randn(n, 1).double())  # strongly anti-correlated children
    parent = child_a + child_b
    m_parent = T.Moments.from_raw(parent, parent)
    _, se_true = m_parent.mean_se('ce')
    se_a = T.Moments.from_raw(child_a, child_a).mean_se('ce')[1]
    se_b = T.Moments.from_raw(child_b, child_b).mean_se('ce')[1]
    se_added = se_a + se_b
    se_quadrature = (se_a ** 2 + se_b ** 2).sqrt()
    ref = parent.std(0, unbiased=True) / math.sqrt(n)
    assert torch.allclose(se_true, ref.reshape(-1), rtol=1e-10)
    assert se_added.item() > 5 * se_true.item() and se_quadrature.item() > 3 * se_true.item()
    record('se_recomputation', dict(parent_se=se_true.item(), child_se_sum=se_added.item(), child_se_quadrature=se_quadrature.item()))


def test_or_and_are_not_n16_election():
    n = 64
    torch.manual_seed(1)
    # children: one strongly negative, one strongly positive -> parent cancels; OR elects, N16 does not
    a = (-1.0 + 0.05 * torch.randn(n, 1)).double()
    b = (1.2 + 0.05 * torch.randn(n, 1)).double()
    # children individually weak (not elected) but same sign -> parent elected; AND/OR both miss it
    c = (-0.012 + 0.05 * torch.randn(n, 1)).double()
    d = (-0.012 + 0.05 * torch.randn(n, 1)).double()
    def el(x):
        return bool(T.elect(T.Moments.from_raw(x, x), 3)[0])
    cancel = dict(child_a=el(a), child_b=el(b), OR=el(a) or el(b), AND=el(a) and el(b), N16=el(a + b))
    assert cancel['OR'] and not cancel['N16']
    torch.manual_seed(2)
    c = (-0.2 + 0.9 * torch.randn(n, 1)).double()
    d = (-0.2 + 0.9 * torch.randn(n, 1)).double()
    amplify = dict(child_c=el(c), child_d=el(d), OR=el(c) or el(d), AND=el(c) and el(d), N16=el(c + d),
                   t_c=float(c.mean() / (c.std() / math.sqrt(n))), t_parent=float((c + d).mean() / ((c + d).std() / math.sqrt(n))))
    record('or_and_counterexamples', dict(cancellation=cancel, amplification=amplify))


def test_streamed_moments_equal_raw_and_torch():
    g = torch.Generator().manual_seed(3)
    ce = torch.randn(128, 1000, generator=g)
    kl = ce * 0.3 + torch.randn(128, 1000, generator=g)
    raw = T.Moments.from_raw(ce, kl)
    streamed = T.Moments(1000)
    for i in range(128):
        streamed.update(ce[i], kl[i])
    for a in ('ce_sum', 'ce_sq', 'kl_sum', 'kl_sq', 'cross'):
        assert torch.equal(getattr(raw, a), getattr(streamed, a))
    m, se = raw.mean_se('ce')
    assert torch.allclose(m, ce.double().mean(0), rtol=1e-12)
    assert torch.allclose(se, ce.double().std(0, unbiased=True) / math.sqrt(128), rtol=1e-9)
    rho = raw.correlation()
    ref = torch.stack([torch.corrcoef(torch.stack([ce[:, j].double(), kl[:, j].double()]))[0, 1] for j in range(1000)])
    assert torch.allclose(rho, ref, atol=1e-10)
    # additivity across sequence subsets (math + code halves)
    a = T.Moments.from_raw(ce[:64], kl[:64]).add(T.Moments.from_raw(ce[64:], kl[64:]))
    assert torch.allclose(a.ce_sum, raw.ce_sum, rtol=1e-12)
    record('moments', 'streamed == raw bitwise; mean/SE/correlation match torch float64; subset moments add')


def test_archived_k2_formula_equals_common_descent_scores():
    g = torch.Generator().manual_seed(4)
    ce = torch.randn(128, 5000, generator=g) * 1e-4
    kl = torch.randn(128, 5000, generator=g) * 1e-5
    u = T.archived_upper_scores(ce, kl, 2)
    assert torch.equal(u, common_descent_scores(ce, kl, torch.zeros(5000, dtype=torch.bool)))
    record('archived_k2_identity', True)


def test_rule_nesting():
    g = torch.Generator().manual_seed(5)
    ce = torch.randn(128, 20000, generator=g) - 0.15
    kl = torch.randn(128, 20000, generator=g) - 0.15
    m = T.Moments.from_raw(ce, kl)
    prev = None
    counts = {}
    for kk in (0, 2, 3, 4, 5, 6):
        e = T.elect(m, kk, 'ce_kl')
        assert torch.equal(e, T.elect(m, kk, 'ce_only') & T.elect(m, kk, 'kl_only'))
        if prev is not None:
            assert not (e & ~prev).any()
        prev = e
        counts[kk] = int(e.sum())
    assert torch.equal(T.elect(m, 0, 'ce_kl'), T.elect(m, 3, 'mean_only'))
    record('rule_nesting', counts)
