"""
    Does the perplexity result -- the conjunction beats KL alone at an equal budget of switches
    -- reproduce on zero-shot accuracy?

    The perplexity sweep answers this cleanly because both objectives were evaluated at five
    thresholds each, so every KL-only setting can be checked against a conjunction setting that
    elects no more tiles. Accuracy needs the same treatment, and for the same reason: at a fixed
    k the two objectives elect very different numbers of tiles, so "swap the objective, hold k"
    measures permissiveness as much as objective.

    Two things make this harder than the perplexity version and are handled here rather than
    glossed:

    - The points live in different jobs. That is safe only because §1a's control found the
      evaluation exact on one GPU model, and this script re-verifies it on the shared policies
      instead of assuming it -- a pair of jobs that disagree is reported, not averaged.
    - The unweighted task mean and the document-weighted paired delta can disagree in sign.
      Qwen3-4B's k4 is 0.6222 against the base's 0.6231 by task mean, yet +0.0061 at p = 0.005
      by paired test, because hellaswag carries 10,042 documents and openbookqa 500. Everything
      below is document-weighted; the panel mean is shown only for continuity with §1a.

        python analyze_accuracy_frontier.py
"""

import argparse
import glob
import json
import os
import re

from analyze_zeroshot_paired import compare, load
from analyze_run_reproducibility import gpu_of, job_node

MODELS = [('llama8b', 'Llama-3.1-8B'), ('qwen4b', 'Qwen3-4B'), ('qwen27b', 'Qwen3.8-27B')]
BASE = 'four_over_six'
K = 3


def objective_of(pol):
    """'max' for k3, 'kl' for k3_kl, 'ce' for k5_ce; None for bf16/nvfp4/four_over_six."""
    m = re.fullmatch(r'k(\d+)(?:_(kl|ce))?', pol)
    return (int(m.group(1)), m.group(2) or 'max') if m else None


def collect(model, root='results/zeroshot_kse'):
    """Every elected policy measured for this model, as {policy: (tiles, mean, samples, job)}.

    Restricted to ONE GPU model. Pooling points across jobs is exact within a GPU model and is
    not across them -- §1a's control measured 2.8% of documents flipping between H100 and H200 --
    so mixing them would build a frontier whose steps are partly hardware. The GPU with the most
    measured policies wins; the rest are dropped and counted.

    A policy measured twice on the winning GPU keeps its first appearance, and the duplicate
    becomes a consistency check rather than being averaged away.
    """
    found = []
    for path in sorted(glob.glob(os.path.join(root, 'job_*', '*', 'report.json'))):
        try:
            r = json.load(open(path))
        except Exception:
            continue
        if r.get('status') != 'complete' or r.get('model') != model:
            continue
        job = os.path.basename(os.path.dirname(os.path.dirname(path)))
        gpu = gpu_of(job_node(job.replace('job_', '')))
        sdir = os.path.join(os.path.dirname(os.path.dirname(path)), f'samples_{model}')
        for pol, v in r.get('accuracy', {}).items():
            if not isinstance(v, dict) or 'mean' not in v:
                continue
            if pol != BASE and not objective_of(pol):
                continue
            f = os.path.join(sdir, f'{pol}.json')
            if not os.path.isfile(f):
                continue
            tiles = r.get('election', {}).get(pol, {}).get('selected', 0)
            found.append((gpu, pol, tiles, v['mean'], f, job))

    if not found:
        return {}, [], None, 0
    per_gpu = {}
    for gpu, pol, *_ in found:
        per_gpu.setdefault(gpu, set()).add(pol)
    keep = max(per_gpu, key=lambda g: len(per_gpu[g]))
    dropped = sum(1 for g, *_ in found if g != keep)

    out, dupes = {}, []
    for gpu, pol, tiles, mean, f, job in found:
        if gpu != keep:
            continue
        if pol in out:
            dupes.append((pol, out[pol], (tiles, mean, f, job)))
        else:
            out[pol] = (tiles, mean, f, job)
    return out, dupes, keep, dropped


def build():
    """The budget-matched accuracy block, or None when too few points exist."""
    L = ['#### Does the budget-matched result hold on accuracy?', '',
         'The perplexity table above compares the two objectives at an equal budget of '
         'switches, because at a fixed k they elect very different numbers of tiles. The same '
         'comparison on the multiple-choice panel, with every delta measured against the '
         'FourOverSix base by paired McNemar over documents.', '']

    checks, any_rows, gpus = [], False, {}
    for model, label in MODELS:
        pts, dupes, gpu, dropped = collect(model)
        if BASE not in pts or len(pts) < 3:
            continue
        gpus[label] = (gpu, dropped)

        # Points come from different jobs, which §1a's control says is safe only within one GPU
        # model. Verify on the policies measured twice rather than trusting it.
        for pol, a, b in dupes:
            _, p = compare(load(a[2]), load(b[2]))
            checks.append((label, pol, a[3], b[3], gpu_of(job_node(a[3].replace('job_', ''))),
                           gpu_of(job_node(b[3].replace('job_', ''))), p['b'] + p['c']))

        base_f = pts[BASE][2]
        rows = []
        for pol, (tiles, mean, f, job) in pts.items():
            if pol == BASE:
                continue
            _, p = compare(load(base_f), load(f))
            k, objective = objective_of(pol)
            rows.append((objective, k, tiles, mean, p['delta'], p['p']))
        rows.sort(key=lambda r: (r[0] != 'max', r[2]))

        any_rows = True
        L += [f'**{label}**', '',
              '| objective | k | tiles | panel mean | d vs base (paired) | p |',
              '|---|---:|---:|---:|---:|---:|',
              f'| — | — | 0 | {pts[BASE][1]:.4f} | — | — |']
        for objective, k, tiles, mean, d, pv in rows:
            name = {'max': 'max(CE, KL)', 'kl': 'KL only', 'ce': 'CE only'}[objective]
            star = ' **(shipped)**' if objective == 'max' and k == K else ''
            L.append(f'| {name}{star} | {k} | {tiles:,} | {mean:.4f} | {d:+.4f} | {pv:.3g} |')
        L.append('')

        # For each single-objective point, the best conjunction point that elects no more tiles.
        verdicts = []
        for objective, k, tiles, mean, d, pv in rows:
            if objective == 'max':
                continue
            cand = [r for r in rows if r[0] == 'max' and r[2] <= tiles]
            if not cand:
                verdicts.append(f'`k{k}_{objective}` ({tiles:,} tiles): no conjunction setting '
                                f'was measured at or below this budget, so it is untested here.')
                continue
            best = max(cand, key=lambda r: r[4])
            mine = [p for p in pts.items() if objective_of(p[0]) == (best[1], 'max')][0][1][2]
            _, head = compare(load(mine), load(pts[f'k{k}_{objective}'][2]))
            sig = head['p'] < 0.05
            verdicts.append(
                f'`k{k}_{objective}` ({tiles:,} tiles) against `k{best[1]}` ({best[2]:,}): '
                f'{head["delta"]:+.4f}, p = {head["p"]:.3g} -- '
                + ('significantly different' if sig else 'indistinguishable'))
        if verdicts:
            L += ['Head to head against the best conjunction setting that elects no more tiles:',
                  ''] + [f'- {v}' for v in verdicts] + ['']

    if gpus:
        where = '; '.join(f'{lb} on {g}' + (f' ({d} point(s) on other hardware dropped)'
                                            if d else '')
                          for lb, (g, d) in gpus.items())
        note = (f'Each frontier is pooled across jobs but confined to one GPU model ({where}), '
                f'because §1a\'s control found the evaluation exact within a GPU model and '
                f'2.8% of documents flipped across two. ')
        if checks:
            bad = [c for c in checks if c[6]]
            note += ('; '.join(f'{lb} `{pol}` {ja} vs {jb} differ on {n} documents'
                               for lb, pol, ja, jb, _, _, n in bad) if bad else
                     f'The {len(checks)} policies measured twice on that hardware agree on every '
                     f'document, so the pooling is exact rather than assumed.')
        L += [note, '']

    return '\n'.join(L) + '\n' if any_rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/zeroshot_kse/SECTION_frontier.md')
    args = ap.parse_args()
    out = build()
    if not out:
        print('no model has enough accuracy points yet')
        return
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(out)
    print(out)
    print(f'written to {args.out}')


if __name__ == '__main__':
    main()
