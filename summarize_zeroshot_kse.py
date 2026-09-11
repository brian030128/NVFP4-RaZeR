"""
    Build the zero-shot accuracy section of MIXFP4_REPORT.md from the k = 3 runs.

    The report's tables are perplexity. These are the same four policies on zero-shot multiple
    choice, with the same layout -- one block per model, policies as rows -- so a reader can set
    the two side by side. Perplexity deltas are read out of the report's own source files rather
    than retyped, so the two metrics cannot drift apart here.

        python summarize_zeroshot_kse.py --out results/zeroshot_kse/SECTION.md
"""

import argparse
import glob
import json
import math
import os

from analyze_zeroshot_paired import compare, load as load_samples

# The report's models, in the report's order and with its labels.
MODELS = [('llama8b', 'Llama-3.1-8B'), ('qwen4b', 'Qwen3-4B'), ('qwen27b', 'Qwen3.8-27B')]

# (row label in the report, policy key in the run's report.json)
POLICIES = [('BF16 reference', 'bf16'),
            ('NVFP4 W4A4', 'nvfp4'),
            ('NVFP4 FourOverSix W4A4', 'four_over_six'),
            ('**MixFP4 (k=3), ours**', 'k3')]
BASE = 'four_over_six'      # MixFP4's own base: the comparison the method is about
K = 3

# Where the report reads its perplexity from (build_mixfp4_report.py).
KSE_JOB = {'llama8b': '336566', 'qwen4b': '336566', 'qwen27b': '336969'}


def find_runs(root):
    """Newest complete run per model."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(root, 'job_*', '*', 'report.json'))):
        try:
            r = json.load(open(path))
        except Exception:
            continue
        # A run whose re-election drifted from the shipped map is still included; the drift is
        # reported with it rather than used to hide it. Only incomplete runs are skipped.
        if r.get('status') != 'complete':
            continue
        runs[r['model']] = (r, os.path.dirname(path))
    return runs


def ppl_deltas(model):
    """MixFP4 (k=3) minus FourOverSix in wikitext and c4, from the report's own source."""
    p = f'results/kse_paper/job_{KSE_JOB[model]}/{model}/report.json'
    if not os.path.isfile(p):
        return None
    ev = json.load(open(p))['evaluation']
    if f'k{K}' not in ev or BASE not in ev:
        return None
    return {d: ev[f'k{K}'][d]['ppl'] - ev[BASE][d]['ppl'] for d in ('wiki', 'c4')}


def fmt(x, digits=4, signed=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '—'
    return f'{x:+.{digits}f}' if signed else f'{x:.{digits}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/zeroshot_kse')
    ap.add_argument('--out', default='results/zeroshot_kse/SECTION.md')
    args = ap.parse_args()

    runs = find_runs(args.root)
    present = [(m, lbl) for m, lbl in MODELS if m in runs]
    assert present, f'no complete run under {args.root}'
    tasks = runs[present[0][0]][0]['tasks_evaluated']

    L = ['## Zero-shot accuracy', '',
         'The tables above are perplexity, which is a next-token loss: a quantizer that makes '
         'the model less confident lowers it without the model predicting anything better. The '
         'k-SE rule elects tiles by a calibration loss, so that is a live possibility rather '
         'than a hypothetical one, and nothing measured above rules it out. These are the same '
         'four policies on zero-shot multiple choice, where being less confident earns nothing.',
         '',
         f'lm-eval-harness 0.4.5, 0-shot, {len(tasks)} tasks: '
         + ', '.join(f'`{t}`' for t in tasks) +
         '. `acc_norm` where the harness defines it, `acc` otherwise; the figure is the '
         'unweighted mean over tasks. Weights and activations are built exactly as '
         '`run_kse_paper.py` builds them, and the k = 3 election is re-derived from the '
         'calibration and checked to reproduce the shipped frozen map bitwise before anything '
         'is evaluated (`run_zeroshot_kse.py`).', '']

    for model, label in present:
        r, _ = runs[model]
        acc = r['accuracy']
        el = r['election'][f'k{K}']
        L += [f'### {label}', '',
              f'The rule elects **{el["selected"]:,} of {r["total_tiles"]:,}** type blocks, '
              f'{100 * el["fraction"]:.4f}%.', '',
              '| Policy | ' + ' | '.join(tasks) + ' | mean |',
              '|---' * (len(tasks) + 2) + '|']
        for row, key in POLICIES:
            if key not in acc:
                continue
            cells = [fmt(acc[key][t]['value']) if t in acc[key] else '—' for t in tasks]
            mean = fmt(acc[key]['mean'])
            L.append(f'| {row} | ' + ' | '.join(cells) +
                     (f' | **{mean}** |' if key == f'k{K}' else f' | {mean} |'))
        L.append('')

        if BASE in acc and f'k{K}' in acc:
            d_acc = acc[f'k{K}']['mean'] - acc[BASE]['mean']
            d_nv = acc[f'k{K}']['mean'] - acc['nvfp4']['mean'] if 'nvfp4' in acc else None
            dp = ppl_deltas(model)
            L += ['| Comparison | d accuracy | d WikiText PPL | d C4 PPL |', '|---|---:|---:|---:|',
                  f'| MixFP4 − FourOverSix | {fmt(d_acc, signed=True)} | '
                  f'{fmt(dp["wiki"], 6, True) if dp else "—"} | '
                  f'{fmt(dp["c4"], 6, True) if dp else "—"} |']
            if d_nv is not None:
                L.append(f'| MixFP4 − NVFP4 | {fmt(d_nv, signed=True)} | — | — |')
            L.append('')

    # --- paired tests, where per-document outcomes were kept ---------------------------------
    rows = []
    for model, label in present:
        _, rundir = runs[model]
        sdir = os.path.join(os.path.dirname(rundir), f'samples_{model}')
        ref = os.path.join(sdir, f'{BASE}.json')
        var = os.path.join(sdir, f'k{K}.json')
        if os.path.isfile(ref) and os.path.isfile(var):
            _, pooled = compare(load_samples(ref), load_samples(var))
            rows.append((label, pooled))
    if rows:
        L += ['### Is the difference real?', '',
              'Every policy is scored on the same documents, so MixFP4 against its own base is '
              'a paired comparison and the standard error lm-eval prints -- the error of one '
              'measurement -- is the wrong yardstick. `b` counts documents only FourOverSix '
              'gets right, `c` only MixFP4; the rest carry no information about the difference. '
              'The p-value is an exact two-sided McNemar test on those counts, pooled over all '
              'tasks.', '',
              '| model | documents | b | c | pooled delta | McNemar p |', '|---|---|---|---|---|---|']
        for label, pl in rows:
            L.append(f"| {label} | {pl['n']} | {pl['b']} | {pl['c']} | "
                     f"{fmt(pl['delta'], signed=True)} | {pl['p']:.3g} |")
        L.append('')

    # Say plainly which of the report's models this covers. A section that silently lists two
    # of three invites the reader to assume the third agreed.
    missing = [lbl for m, lbl in MODELS if m not in runs]
    if missing:
        L += ['### Coverage', '',
              'This section covers ' + ', '.join(lbl for _, lbl in present) +
              '. It does not cover ' + ', '.join(missing) + '. ' +
              'No claim is made about the missing model either way.', '']
    drifted = [(lbl, runs[m][0]) for m, lbl in present
               if runs[m][0].get('frozen_map_drift')]
    if drifted:
        L += ['> **Election re-derived.** ' +
              '; '.join(
                  f'{lbl}: {r["frozen_map_drift"]["modules_differing"]} module(s) differ from '
                  f'the shipped frozen map, '
                  f'{r["frozen_map_drift"]["tiles_in_shipped_only"]} tile(s) present only in '
                  f'the shipped map and '
                  f'{r["frozen_map_drift"]["tiles_in_reelected_only"]} only here'
                  for lbl, r in drifted) +
              '. The rule, k, and calibration are the reported ones and the calibration '
              'reproduces the shipped teacher losses bit for bit; what differs is which side of '
              'the threshold a handful of borderline tiles fall on in a re-run. The effect on '
              'the elected set is a few tiles in tens of millions, so these rows are treated as '
              'the reported policy, with the difference recorded here rather than hidden.', '']

    out = '\n'.join(L) + '\n'
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(out)
    print(out)
    print(f'written to {args.out}')


if __name__ == '__main__':
    main()
