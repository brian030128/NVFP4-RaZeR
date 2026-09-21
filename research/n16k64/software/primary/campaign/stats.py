"""Statistical machinery for V70/V72/V73 (CPU only; numpy/scipy/torch)."""
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats as ss

SEED = 20260911


# ----------------------------------------------------------------------------- PPL: paired cluster bootstrap
def window_table(ppl_report, policy, domain):
    ev = ppl_report['evaluation'][policy][domain]
    return np.array([w['nll_sum'] for w in ev['windows']], dtype=np.float64), np.array([w['tokens'] for w in ev['windows']], dtype=np.float64)


def clusters_for(windows_meta, domain, mode='natural'):
    if domain == 'c4':
        docs = windows_meta['documents']
        per = windows_meta.get('windows', len(docs)) // len(docs)
        ids = [d['document_sha256'] for d in docs for _ in range(per)]
    elif domain == 'wiki':
        if mode == 'block5':
            ids = [f'b{w["window"] // 5}' for w in windows_meta['window_articles']]
        else:
            ids = [f'a{w["first_article"]}' for w in windows_meta['window_articles']]
    elif domain in ('math_eval', 'code_eval'):  # one window per held-out document
        ids = [m['document_sha256'] for m in windows_meta['window_meta']]
    else:  # long context: books
        ids = [f'book{m["book"]}' for m in windows_meta['window_meta']]
    uniq = {c: i for i, c in enumerate(dict.fromkeys(ids))}
    return np.array([uniq[c] for c in ids])


def paired_dlogppl(nll_a, nll_b, tokens, cluster, B=10000, seed=SEED, margin=None):
    """Delta = (sum nll_a - sum nll_b) / sum tokens (token-weighted mean NLL difference = delta log PPL)."""
    d = nll_a - nll_b
    k = cluster.max() + 1
    dc = np.bincount(cluster, weights=d, minlength=k)
    nc = np.bincount(cluster, weights=tokens, minlength=k)
    est = dc.sum() / nc.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(B, k))
    boot = dc[idx].sum(1) / nc[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out = dict(estimate=float(est), ci95=[float(lo), float(hi)], clusters=int(k), windows=int(len(d)), B=B,
               ppl_ratio=float(math.exp(est)), ppl_ratio_ci95=[float(math.exp(lo)), float(math.exp(hi))],
               p_two_sided=float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))))
    if margin is not None:
        # one-sided non-inferiority: H0 delta >= margin; bootstrap p = P*(delta* >= margin) shifted to the null
        out['noninferiority'] = dict(margin=margin, p_one_sided=float(((boot - est) >= (margin - est)).mean()), upper_ci_below_margin=bool(hi < margin))
    return out


def holm(pvals, alpha=0.05):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist(), [bool(a < alpha) for a in adj]


# ----------------------------------------------------------------------------- token diagnostics
def token_arrays(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def ece(confidence, correct, bins=10):
    conf = np.asarray(confidence, dtype=np.float64)
    cor = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(cor[m].mean() - conf[m].mean())
    return float(e)


def paired_token_metric(a, b, windows_tokens, cluster, B=2000, seed=SEED):
    """Cluster bootstrap for a per-token metric difference (mean over tokens)."""
    starts = np.concatenate([[0], np.cumsum(windows_tokens)[:-1]]).astype(int)
    per_window = np.add.reduceat(a.astype(np.float64) - b.astype(np.float64), starts)
    return paired_dlogppl(per_window, np.zeros_like(per_window), windows_tokens.astype(np.float64), cluster, B=B, seed=seed)


# ----------------------------------------------------------------------------- accuracy: paired example bootstrap
def load_samples(path, metric):
    rows = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            r = json.loads(line)
            key = r.get('doc_hash') or json.dumps([r.get('doc_id'), r.get('target')], sort_keys=True)
            val = r.get(metric)
            if val is None and ',' in metric:
                val = r.get(metric.split(',')[0])
            rows[(r.get('doc_id'), key)] = float(val)
    return rows


def paired_accuracy(a_rows, b_rows, B=10000, seed=SEED):
    keys = sorted(set(a_rows) & set(b_rows), key=str)
    if len(keys) != len(a_rows) or len(keys) != len(b_rows):
        raise ValueError(f'unpaired examples: {len(a_rows)} vs {len(b_rows)} common {len(keys)}')
    a = np.array([a_rows[k] for k in keys])
    b = np.array([b_rows[k] for k in keys])
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    boot = d[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=len(d), acc_a=float(a.mean()), acc_b=float(b.mean()), diff=float(d.mean()), ci95=[float(lo), float(hi)],
                discordant=int((d != 0).sum()), mcnemar_p=float(ss.binomtest(int((d > 0).sum()), int((d != 0).sum())).pvalue) if (d != 0).any() else 1.0,
                boot=boot)


def macro_accuracy(per_task_boot):
    """Macro average over tasks with independent task bootstraps (same B)."""
    tasks = sorted(per_task_boot)
    diff = float(np.mean([per_task_boot[t]['diff'] for t in tasks]))
    boot = np.mean(np.stack([per_task_boot[t]['boot'] for t in tasks]), 0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(tasks=tasks, diff=diff, ci95=[float(lo), float(hi)])


# ----------------------------------------------------------------------------- selection multiplicity (V73)
def tile_statistics(moments):
    n = moments['n']
    out = {}
    for obj in ('ce', 'kl'):
        s, q = moments[f'{obj}_sum'].double(), moments[f'{obj}_sq'].double()
        mean = s / n
        var = ((q - s * s / n) / (n - 1)).clamp_min(0)
        se = (var / n).sqrt()
        out[f't_{obj}'] = torch.where(se > 0, mean / se, torch.where(mean < 0, torch.full_like(mean, -1e9), torch.full_like(mean, 1e9)))
        out[f'mean_{obj}'] = mean
        out[f'se_{obj}'] = se
    mc, mk = out['mean_ce'], out['mean_kl']
    cov = (moments['cross'].double() - n * mc * mk) / (n - 1)
    denom = (out['se_ce'] * out['se_kl'] * n)
    out['rho'] = torch.where(denom > 0, cov / denom, torch.zeros_like(cov)).clamp(-1, 1)
    out['n'] = n
    return out


def bh_count(p, q):
    p = np.sort(p)
    m = len(p)
    thresh = q * np.arange(1, m + 1) / m
    ok = np.nonzero(p <= thresh)[0]
    return int(ok.max() + 1) if ok.size else 0


def by_count(p, q):
    m = len(p)
    c = np.sum(1.0 / np.arange(1, m + 1))
    return bh_count(p, q / c)


def bivariate_tail(k, rho):
    """P(Z1 < -k, Z2 < -k) for standard bivariate normal with correlation rho (vectorized by interpolation)."""
    grid = np.linspace(-0.999, 0.999, 401)
    vals = np.array([ss.multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([-k, -k]) for r in grid])
    return np.interp(np.clip(rho, -0.999, 0.999), grid, vals)


def multiplicity_summary(stats_by_module, k=3, qs=(0.05, 0.10)):
    t_ce = torch.cat([v['t_ce'] for v in stats_by_module.values()]).numpy()
    t_kl = torch.cat([v['t_kl'] for v in stats_by_module.values()]).numpy()
    rho = torch.cat([v['rho'] for v in stats_by_module.values()]).numpy()
    n = next(iter(stats_by_module.values()))['n']
    m = len(t_ce)
    sel = (t_ce < -k) & (t_kl < -k)
    out = dict(tiles=m, sequences=n, k=k, selected_k=int(sel.sum()))
    for dist in ('normal', 't'):
        cdf = (lambda x: ss.norm.cdf(x)) if dist == 'normal' else (lambda x: ss.t.cdf(x, df=n - 1))
        p_iut = np.maximum(cdf(t_ce), cdf(t_kl))
        res = dict(iut_p_min=float(p_iut.min()), selected_iut_alpha_phi_minus_k=int((p_iut < cdf(-k)).sum()))
        for q in qs:
            res[f'BH_q{q}'] = bh_count(p_iut, q)
            res[f'BY_q{q}'] = by_count(p_iut, q)
        out[dist] = res
    phi = float(ss.norm.cdf(-k))
    out['null_expected_false_positive_bounds'] = dict(
        iut_any_dependence=dict(per_tile=phi, all_tiles=m * phi, note='valid for any CE/KL dependence (intersection-union)'),
        independence_phi_squared=dict(per_tile=phi ** 2, all_tiles=m * phi ** 2, note='archived argument; invalid unless CE/KL are independent'),
        measured_rho=dict(all_tiles=float(bivariate_tail(k, rho).sum()), note='sum over tiles of P(Z1<-k, Z2<-k; rho_hat) at the global null'))
    q = np.quantile(rho, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    out['ce_kl_correlation_over_sequences'] = dict(quantiles=dict(zip(['p01', 'p05', 'p25', 'p50', 'p75', 'p95', 'p99'], map(float, q))), mean=float(rho.mean()))
    finite = np.isfinite(t_ce) & np.isfinite(t_kl) & (np.abs(t_ce) < 1e8) & (np.abs(t_kl) < 1e8)
    out['across_tile_correlation_of_t'] = dict(pearson=float(np.corrcoef(t_ce[finite], t_kl[finite])[0, 1]),
                                               spearman=float(ss.spearmanr(t_ce[finite][:2_000_000], t_kl[finite][:2_000_000]).correlation))
    return out


def sign_flip(raw_ce, raw_kl, k=3, R=1000, seed=SEED):
    """raw_*: [S, T] per-sequence scores for a tile sample. Same random sign per sequence for CE, KL and all tiles."""
    ce, kl = raw_ce.double(), raw_kl.double()
    S = ce.shape[0]
    g = torch.Generator().manual_seed(seed)

    def passing(c, l):
        uc = c.mean(0) + k * c.std(0, unbiased=True) / math.sqrt(S)
        ul = l.mean(0) + k * l.std(0, unbiased=True) / math.sqrt(S)
        return int(((uc < 0) & (ul < 0)).sum())

    observed = passing(ce, kl)
    perm = []
    for _ in range(R):
        s = (torch.randint(0, 2, (S, 1), generator=g).double() * 2 - 1)
        perm.append(passing(ce * s, kl * s))
    perm = np.array(perm)
    return dict(tiles=int(ce.shape[1]), sequences=S, k=k, observed=observed, permuted_mean=float(perm.mean()),
                permuted_p95=float(np.percentile(perm, 95)), permuted_max=int(perm.max()),
                estimated_false_discovery_proportion=(float(perm.mean() / observed) if observed else None),
                p_value_global=float((1 + (perm >= observed).sum()) / (R + 1)))


# ----------------------------------------------------------------------------- N8/N16 structure (V72)
def structure(n8_masks, n16_masks, shapes, n8_stats=None, n16_stats=None):
    tot = defaultdict(int)
    per_module = {}
    child_hist = np.zeros(3, dtype=np.int64)
    for n, m16 in n16_masks.items():
        o, k = shapes[n]
        m8 = n8_masks[n].reshape(o // 8, k // 64)
        pair = m8.reshape(o // 16, 2, k // 64)
        anyc, both = pair.any(1), pair.all(1)
        m16 = m16.reshape(o // 16, k // 64)
        c = pair.sum(1)
        for v in range(3):
            child_hist[v] += int((c[m16] == v).sum())
        e = dict(n8=int(m8.sum()), n16=int(m16.sum()), n8_any=int(anyc.sum()), n8_both=int(both.sum()),
                 n16_and_any=int((m16 & anyc).sum()), n16_and_both=int((m16 & both).sum()),
                 n16_not_any=int((m16 & ~anyc).sum()), any_not_n16=int((anyc & ~m16).sum()))
        per_module[n] = e
        for kk, v in e.items():
            tot[kk] += v
    jac = lambda a, b, i: (i / (a + b - i)) if (a + b - i) else 1.0
    out = dict(totals=dict(tot), selected_weights=dict(n8=tot['n8'] * 512, n16=tot['n16'] * 1024),
               jaccard_n16_vs_n8_any=jac(tot['n16'], tot['n8_any'], tot['n16_and_any']),
               jaccard_n16_vs_n8_both=jac(tot['n16'], tot['n8_both'], tot['n16_and_both']),
               selected_children_per_selected_n16_parent={str(v): int(child_hist[v]) for v in range(3)}, per_module=per_module)
    return out
