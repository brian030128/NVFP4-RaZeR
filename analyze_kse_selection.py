"""Where the k-SE rule puts its E0M3 tiles.

MIXFP4_REPORT reports how many 8x64 tiles the rule elects; it does not say which.
This rebuilds the election from the frozen calibration score shards -- the same
upper_scores() the shipped rule uses, so the k=3 count must reproduce the number
already in results/kse_paper/*/report.json -- and then records, for every elected
tile, the module that owns it, the layer, the projection kind, and the (row, col)
position of the tile in that matrix's tile grid.

Reads ~100 GB of score shards across the three models, so it runs under Slurm.
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch

from run_kse_paper import MODELS, K_VALUES, upper_scores

K = 3


def kind_of(name):
    return name.split('.')[-1]


def layer_of(name):
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p == 'layers':
            return int(parts[i + 1])
    return -1


def block_of(name):
    """Which sub-block of the transformer layer the matrix belongs to."""
    for tag in ('self_attn', 'linear_attn', 'mlp'):
        if f'.{tag}.' in name:
            return tag
    return 'other'


def analyse(model, stage_root, out_dir):
    old = Path(stage_root) / f'results/math_code_adaptive/calibration_{MODELS[model]}_{model}'
    prior = json.loads((old / 'report.json').read_text())
    assert prior['status'] == 'complete' and prior['maps_frozen']
    assert not prior['uses_c4_calibration'] and not prior['uses_wiki_calibration']
    names = list(prior['matrices'])

    r = dict(status='running', model=model, calibration=str(old),
             job_id=os.environ.get('SLURM_JOB_ID'), k=K, k_values=list(K_VALUES),
             n_calibration_sequences=None, total_tiles=0, counts={}, modules=[])
    modules = []
    counts = Counter()
    total = 0
    for i, name in enumerate(names):
        shard = torch.load(old / 'scores' / f'{i:03d}.pt', map_location='cpu', weights_only=True)
        assert shard['name'] == name
        ce, kl = shard['ce'].double(), shard['kl'].double()
        assert ce.shape == kl.shape and ce.shape[0] == 128
        r['n_calibration_sequences'] = int(ce.shape[0])
        rows, cols = prior['matrices'][name]['shape']
        rows, cols = rows // 8, cols // 64
        n_tiles = ce.shape[1]
        assert rows * cols == n_tiles, (name, rows, cols, n_tiles)
        total += n_tiles

        per_k = {}
        prev = None
        for k in sorted(K_VALUES):
            sel = upper_scores(ce, kl, k) < 0
            per_k[k] = int(sel.sum())
            counts[k] += per_k[k]
            if prev is not None:
                # the upper score is increasing in k, so the elected sets must nest
                assert bool((sel & ~prev).any()) is False, (name, k)
            prev = sel

        u_ce = ce.mean(0) + K * ce.std(0, unbiased=True) / math.sqrt(ce.shape[0])
        u_kl = kl.mean(0) + K * kl.std(0, unbiased=True) / math.sqrt(kl.shape[0])
        sel = torch.maximum(u_ce, u_kl) < 0
        idx = sel.nonzero(as_tuple=True)[0]

        entry = dict(name=name, kind=kind_of(name), layer=layer_of(name), block=block_of(name),
                     shape=prior['matrices'][name]['shape'], grid=[rows, cols], tiles=n_tiles,
                     selected=per_k[K], per_k={str(k): v for k, v in per_k.items()},
                     tiles_rows=[], tiles_cols=[], binding=[], t_ce=[], t_kl=[])
        if idx.numel():
            sr = (idx // cols).tolist()
            sc = (idx % cols).tolist()
            entry['tiles_rows'], entry['tiles_cols'] = sr, sc
            # which objective is the binding one (the max), and each tile's t stat
            entry['binding'] = ['kl' if b else 'ce' for b in (u_kl[idx] >= u_ce[idx]).tolist()]
            for obj, x in (('t_ce', ce), ('t_kl', kl)):
                m = x[:, idx].mean(0)
                se = x[:, idx].std(0, unbiased=True) / math.sqrt(x.shape[0])
                entry[obj] = [round(v, 4) for v in (m / se).tolist()]
        modules.append(entry)
        del shard, ce, kl
        if i % 25 == 0:
            print(f'{model} {i}/{len(names)} {name} sel={per_k[K]}', flush=True)

    r['total_tiles'] = total
    r['counts'] = {str(k): v for k, v in sorted(counts.items())}
    r['modules'] = modules
    r['status'] = 'complete'
    out = Path(out_dir) / f'{model}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1) + '\n')
    print(f'{model}: {counts[K]:,} of {total:,} tiles at k={K}', flush=True)
    return r


def cross_check(r, stage_root, kse_job):
    """The rebuilt election must match the count the report already publishes."""
    p = Path(stage_root) / f'results/kse_paper/job_{kse_job}/{r["model"]}/report.json'
    if not p.exists():
        return None
    ref = json.loads(p.read_text())
    assert ref['total_tiles'] == r['total_tiles'], (ref['total_tiles'], r['total_tiles'])
    for k, v in ref['election'].items():
        if k.startswith('k'):
            assert r['counts'][k[1:]] == v['selected'], (k, v['selected'], r['counts'][k[1:]])
    print(f'{r["model"]}: election reproduces job {kse_job}', flush=True)
    return True


KSE_JOBS = dict(llama8b='336566', qwen4b='336566', qwen27b='336969')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(MODELS), required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage-root', default='/home/u4320956/NVFP4-RaZeR')
    args = ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    torch.set_num_threads(4)
    r = analyse(args.model, args.stage_root, args.out)
    r['election_reproduced'] = bool(cross_check(r, args.stage_root, KSE_JOBS[args.model]))
    (Path(args.out) / f'{args.model}.json').write_text(json.dumps(r, indent=1) + '\n')


if __name__ == '__main__':
    main()
