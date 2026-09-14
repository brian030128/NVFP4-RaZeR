"""
    Does requiring BOTH CE and KL actually change the election?

    The shipped rule elects a tile when `max(mean CE + k SE, mean KL + k SE) < 0`, i.e. when
    both objectives are confidently negative. MIXFP4_REPORT.md argues for that on grounds that
    the two fail in opposite directions -- CE alone can be lowered by fitting the calibration
    corpus's particular tokens, KL alone by restoring agreement on positions the task does not
    care about -- but the argument has never been measured against the alternatives.

    This measures the first half of the question, which is free once the scores exist: how many
    tiles each objective elects and how far the sets differ. If KL alone elects nearly the same
    set, the conjunction is doing little work and the report should say so; if it elects far
    more, the conjunction is the thing holding the count down, and whether that helps is then
    an accuracy question rather than a counting one.

        python analyze_objective_ablation.py --calib <dir> --model llama8b
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch

from run_kse_paper import MODELS

OBJECTIVES = ('max', 'ce', 'kl')


def bound(x, k):
    """mean + k standard errors, per tile."""
    return x.mean(0) + k * x.std(0, unbiased=True) / math.sqrt(x.shape[0])


def scores_for(calib, prior, ks):
    """Per-objective upper scores for every k, rebuilt from the frozen shards."""
    names = list(prior['matrices'])
    parts = {(o, k): [] for o in OBJECTIVES for k in ks}
    slices, offset = {}, 0
    for i, n in enumerate(names):
        shard = torch.load(calib / 'scores' / f'{i:03d}.pt', map_location='cpu',
                           weights_only=True)
        assert shard['name'] == n
        ce, kl = shard['ce'], shard['kl']
        for k in ks:
            b_ce, b_kl = bound(ce, k), bound(kl, k)
            parts[('ce', k)].append(b_ce)
            parts[('kl', k)].append(b_kl)
            parts[('max', k)].append(torch.maximum(b_ce, b_kl))
        slices[n] = (offset, offset + ce.shape[1])
        offset += ce.shape[1]
        del shard, ce, kl
    return {key: torch.cat(v) for key, v in parts.items()}, slices, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib', required=True)
    ap.add_argument('--model', choices=sorted(MODELS), required=True)
    ap.add_argument('--ks', default='2,3,4', type=lambda s: [int(x) for x in s.split(',')])
    ap.add_argument('--save-maps', default=None,
                    help='Directory to write <model>_k<k>_<objective>.pt election maps into.')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    calib = Path(args.calib)
    prior = json.loads((calib / 'report.json').read_text())
    assert prior['status'] == 'complete'
    upper, slices, names = scores_for(calib, prior, args.ks)
    total = upper[('max', args.ks[0])].numel()

    report = dict(model=args.model, calibration=str(calib), total_tiles=total, ks=args.ks,
                  elections={}, overlaps={})
    print(f'{args.model}: {total:,} tiles\n')
    print(f'{"k":>3} {"objective":>10} {"elected":>12} {"% of tiles":>12}')
    for k in args.ks:
        for o in OBJECTIVES:
            sel = upper[(o, k)] < 0
            n = int(sel.sum())
            report['elections'][f'k{k}_{o}'] = dict(k=k, objective=o, selected=n,
                                                    fraction=n / total)
            print(f'{k:>3} {o:>10} {n:>12,} {100 * n / total:>11.4f}%')
            if args.save_maps:
                d = Path(args.save_maps)
                d.mkdir(parents=True, exist_ok=True)
                torch.save(dict(model=args.model, k=k, objective=o,
                                map={nm: sel[lo:hi].clone() for nm, (lo, hi) in slices.items()},
                                elected_tiles=n, total_tiles=total,
                                source=prior['source'], calibration=str(calib)),
                           d / f'{args.model}_k{k}_{o}.pt')
        print()

    # How much of the difference is the conjunction actually making?
    print(f'{"k":>3} {"pair":>12} {"both":>10} {"only A":>10} {"only B":>10} {"Jaccard":>9}')
    for k in args.ks:
        for a, b in (('max', 'kl'), ('max', 'ce'), ('kl', 'ce')):
            sa, sb = upper[(a, k)] < 0, upper[(b, k)] < 0
            inter = int((sa & sb).sum())
            union = int((sa | sb).sum())
            j = inter / union if union else float('nan')
            report['overlaps'][f'k{k}_{a}_vs_{b}'] = dict(
                both=inter, only_a=int((sa & ~sb).sum()), only_b=int((sb & ~sa).sum()),
                jaccard=j)
            print(f'{k:>3} {a + "/" + b:>12} {inter:>10,} {int((sa & ~sb).sum()):>10,} '
                  f'{int((sb & ~sa).sum()):>10,} {j:>9.3f}')
        print()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
        print(f'written to {args.out}')


if __name__ == '__main__':
    main()
