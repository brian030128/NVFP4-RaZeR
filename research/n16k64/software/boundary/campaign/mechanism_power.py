"""Freeze a deterministic screening subset from pre-existing paired PPL evidence.

This script is intentionally outcome-blind with respect to the new mechanism maps.  It
uses only the archived N16-k3 versus FourOverSix paired window deltas to estimate the
noise distribution.  For each deterministic candidate subset, clusters are resampled
10,000 times after centering their paired NLL sums.  Power is the probability that a
shifted bootstrap draw falls outside the empirical two-sided 95% null interval.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SEED = 20260917
B = 10_000
EFFECTS = (0.001, 0.003, 0.005)
CANDIDATES = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)


def cluster_ids(meta, domain):
    if domain == 'c4':
        docs = meta['documents']
        per = meta.get('windows', len(docs)) // len(docs)
        return [d['document_sha256'] for d in docs for _ in range(per)]
    if domain == 'wiki':
        return [f'a{w["first_article"]}' for w in meta['window_articles']]
    raise ValueError(domain)


def keyed_order(corpus, ids):
    unique = list(dict.fromkeys(ids))
    return sorted(unique, key=lambda x: hashlib.sha256(
        f'{corpus}:{x}:seed={SEED}'.encode()).hexdigest())


def aggregate(rows_a, rows_b, ids):
    order = list(dict.fromkeys(ids))
    at = {x: i for i, x in enumerate(order)}
    dc = np.zeros(len(order), np.float64)
    nc = np.zeros(len(order), np.float64)
    for a, b, cid in zip(rows_a, rows_b, ids):
        i = at[cid]
        if int(a['tokens']) != int(b['tokens']):
            raise ValueError('unpaired token count')
        dc[i] += float(a['nll_sum']) - float(b['nll_sum'])
        nc[i] += float(a['tokens'])
    return order, dc, nc


def assess(dc, nc, seed):
    estimate = float(dc.sum() / nc.sum())
    centered = dc - estimate * nc
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(dc), size=(B, len(dc)))
    noise = centered[idx].sum(1) / nc[idx].sum(1)
    lo0, hi0 = np.percentile(noise, [2.5, 97.5])
    powers = {}
    for effect in EFFECTS:
        pos = (effect + noise < lo0) | (effect + noise > hi0)
        neg = (-effect + noise < lo0) | (-effect + noise > hi0)
        powers[f'{effect:.3f}'] = {
            'positive': float(pos.mean()),
            'negative': float(neg.mean()),
            'minimum_direction': float(min(pos.mean(), neg.mean())),
        }
    lo, hi = np.percentile(estimate + noise, [2.5, 97.5])
    return {
        'clusters': int(len(dc)),
        'tokens': int(nc.sum()),
        'archived_estimate_for_variance_only': estimate,
        'bootstrap_ci95': [float(lo), float(hi)],
        'ci_half_width': float((hi - lo) / 2),
        'null_interval': [float(lo0), float(hi0)],
        'power': powers,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parent-root', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    parent = Path(args.parent_root)
    output = {
        'schema': 'mixfp4-mechanism-power/v1',
        'seed': SEED,
        'bootstrap_replicates': B,
        'effects_delta_log_ppl': list(EFFECTS),
        'criterion': {'minimum_direction_power_at_0.003': 0.80,
                      'or_ci_half_width_at_most': 0.0025},
        'selection_rule': 'lowest sha256(corpus:cluster_id:seed=20260917)',
        'models': {},
    }
    for model in ('llama8b', 'qwen4b'):
        run = parent / 'runs' / f'V31_ppl_primary_{model}_attempt1' / 'ppl'
        rep = json.loads((run / 'ppl_report.json').read_text())
        mr = {'source_run': str(run.parent), 'corpora': {}}
        for di, domain in enumerate(('wiki', 'c4')):
            meta = json.loads((run / f'windows_{domain}.json').read_text())
            ids = cluster_ids(meta, domain)
            base = rep['evaluation']['four_over_six'][domain]['windows']
            alt = rep['evaluation']['n16_k3'][domain]['windows']
            ordered_ids, dc_all, nc_all = aggregate(alt, base, ids)
            lookup = {x: i for i, x in enumerate(ordered_ids)}
            keyed = keyed_order(domain, ordered_ids)
            candidates = []
            selected = None
            for n in sorted(set(min(x, len(keyed)) for x in CANDIDATES if x <= len(keyed)) | {len(keyed)}):
                chosen = keyed[:n]
                ix = np.array([lookup[x] for x in chosen], dtype=np.int64)
                result = assess(dc_all[ix], nc_all[ix], SEED + di * 100 + (0 if model == 'llama8b' else 10) + n)
                result['requested_clusters'] = n
                result['meets_power'] = result['power']['0.003']['minimum_direction'] >= 0.80
                result['meets_precision'] = result['ci_half_width'] <= 0.0025
                result['meets_criterion'] = bool(result['meets_power'] or result['meets_precision'])
                candidates.append(result)
                if selected is None and result['meets_criterion']:
                    selected = n
            if selected is None:
                selected = len(keyed)
                descriptive = True
            else:
                descriptive = False
            chosen = keyed[:selected]
            chosen_set = set(chosen)
            indices = [i for i, x in enumerate(ids) if x in chosen_set]
            mr['corpora'][domain] = {
                'available_clusters': len(keyed),
                'available_windows': len(ids),
                'candidate_results': candidates,
                'selected_cluster_count': selected,
                'selected_cluster_ids': chosen,
                'selected_window_indices': indices,
                'selected_window_count': len(indices),
                'descriptive_only': descriptive,
                'calibration_document_overlap': 0,
                'calibration_overlap_note': 'WikiText/C4 evaluation corpora are distinct from OpenWebMath/CodeParrot calibration corpora.',
            }
        output['models'][model] = mr
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(output, indent=1, sort_keys=True) + '\n'
    out.write_text(data)
    print(hashlib.sha256(data.encode()).hexdigest(), out)


if __name__ == '__main__':
    main()
