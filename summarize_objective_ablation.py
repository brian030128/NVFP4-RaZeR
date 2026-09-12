"""
    Build the CE-and-KL ablation section of MIXFP4_REPORT.md.

    The report elects a tile when `max(mean CE + k SE, mean KL + k SE) < 0`, so both objectives
    must be confidently negative, and argues for that conjunction on the grounds that the two
    fail in opposite directions. That argument was never measured -- no CE-only or KL-only run
    existed anywhere in results/. This turns the runs that now do exist into a section.

        python summarize_objective_ablation.py --out results/kse_objective/SECTION.md
"""

import argparse
import glob
import json
import os

from analyze_zeroshot_paired import compare, load as load_samples

MODELS = [('llama8b', 'Llama-3.1-8B'), ('qwen4b', 'Qwen3-4B'), ('qwen27b', 'Qwen3.8-27B')]
BASE = 'four_over_six'
K = 3
PRETTY = {'k3': 'max(CE, KL) — shipped', 'k3_kl': 'KL only', 'k3_ce': 'CE only',
          'k2': 'max(CE, KL)', 'k2_kl': 'KL only', 'k2_ce': 'CE only'}


def ppl_runs(root):
    """Newest complete perplexity ablation per model."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, 'job_*', '*', 'report.json'))):
        try:
            r = json.load(open(path))
        except Exception:
            continue
        if r.get('status') == 'complete' and 'evaluation' in r:
            out[r['model']] = r
    return out


def acc_runs(root):
    """Newest complete accuracy run per model that carries a single-objective policy."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, 'job_*', '*', 'report.json'))):
        try:
            r = json.load(open(path))
        except Exception:
            continue
        if r.get('status') != 'complete':
            continue
        if any(p.startswith(f'k{K}_') for p in r.get('accuracy', {})):
            out[r['model']] = (r, os.path.dirname(path))
    return out


def fmt(x, digits=6, signed=False):
    return '—' if x is None else (f'{x:+.{digits}f}' if signed else f'{x:.{digits}f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ppl-root', default='results/kse_objective')
    ap.add_argument('--acc-root', default='results/zeroshot_kse')
    ap.add_argument('--out', default='results/kse_objective/SECTION.md')
    args = ap.parse_args()

    ppl, acc = ppl_runs(args.ppl_root), acc_runs(args.acc_root)
    assert ppl or acc, 'no ablation runs found'

    L = ['### The conjunction, measured', '',
         'The argument above is an argument, not a measurement, so here is the measurement. '
         'Dropping one objective and thresholding the other alone -- same calibration, same k, '
         'same evaluation protocol, only the quantity the `< 0` test is applied to changes.',
         '']

    if ppl:
        ks = sorted({k for m, _ in MODELS if m in ppl for k in ppl[m].get('k_values', [K])})
        L += ['#### How many tiles each objective elects', '',
              'Raising k tightens the threshold, so both objectives elect fewer tiles. The '
              f'column to compare against is the shipped rule at k = {K}.', '',
              '| model | objective | ' + ' | '.join(f'k = {k}' for k in ks) + ' |',
              '|---|---|' + '---:|' * len(ks)]
        for m, label in MODELS:
            if m not in ppl:
                continue
            el = ppl[m]['election']
            for objective, name in (('max', 'max(CE, KL)'), ('kl', 'KL only'), ('ce', 'CE only')):
                key = (lambda k: f'k{k}') if objective == 'max' else (lambda k: f'k{k}_{objective}')
                if not any(key(k) in el for k in ks):
                    continue
                cells = [f'{el[key(k)]["selected"]:,}' if key(k) in el else '—' for k in ks]
                L.append(f'| {label} | {name} | ' + ' | '.join(cells) + ' |')
        L.append('')

        # Where does KL alone have to be set to elect as few tiles as the conjunction does at
        # the shipped k? Computed, because it is the fair operating point to compare at.
        matches = []
        for m, label in MODELS:
            if m not in ppl:
                continue
            el = ppl[m]['election']
            if f'k{K}' not in el:
                continue
            want = el[f'k{K}']['selected']
            cand = [(k, el[f'k{k}_kl']['selected']) for k in ks if f'k{k}_kl' in el]
            if not cand:
                continue
            kk, cnt = min(cand, key=lambda kv: abs(kv[1] - want))
            ratio = cnt / want if want else float('inf')
            close = 0.5 <= ratio <= 2.0
            matches.append(f'{label} comes closest at k = {kk}, electing {cnt:,} against '
                           f'{want:,}' + ('' if close else f' -- still {ratio:.1f}x off, so the '
                                          f'sweep does not reach a count match'))
        if matches:
            L += [f'KL alone is far more permissive at the same k, so comparing the two at '
                  f'k = {K} compares two different numbers of switches as well as two '
                  f'objectives. Matching the count instead: ' + '; '.join(matches) + '. Every '
                  f'row of the perplexity table below therefore carries its tile count -- a row '
                  f'electing ten times as many tiles is not a like-for-like comparison.', '']

        L += ['#### Perplexity', '',
              'WikiText-2 and C4 at 2048 under the report\'s own protocol '
              '(`run_kse_paper.py`), W4A4. Deltas are against the FourOverSix base the method '
              'switches from; negative is better. k = 3 is the shipped threshold; k = 2 is '
              'carried along because the frozen-map cross-check is defined on its ranking, and '
              'it doubles as a check that the result is not an artifact of one threshold.', '',
              '| model | k | policy | tiles | WikiText | d WikiText | C4 | d C4 |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
        for m, label in MODELS:
            if m not in ppl:
                continue
            ev, el = ppl[m]['evaluation'], ppl[m]['election']
            base = ev.get(BASE)
            if not base:
                continue
            L.append(f'| {label} | — | FourOverSix | 0 | {fmt(base["wiki"]["ppl"])} | — | '
                     f'{fmt(base["c4"]["ppl"])} | — |')
            for k in sorted(ppl[m].get('k_values', [K]), reverse=True):
                for pol in (f'k{k}', f'k{k}_kl', f'k{k}_ce'):
                    if pol not in ev:
                        continue
                    w, c = ev[pol]['wiki']['ppl'], ev[pol]['c4']['ppl']
                    L.append(f'| {label} | {k} | {PRETTY.get(pol, pol[len(f"k{k}_"):] + " only")} | '
                             f'{el.get(pol, {}).get("selected", 0):,} | {fmt(w)} | '
                             f'{fmt(w - base["wiki"]["ppl"], signed=True)} | {fmt(c)} | '
                             f'{fmt(c - base["c4"]["ppl"], signed=True)} |')
        L.append('')

        # The verdict is counted, not asserted: every (model, k) cell where a single-objective
        # election was run is compared against the conjunction at the same k.
        for objective in ('kl', 'ce'):
            cells, losses, harmful = [], 0, 0
            for m, label in MODELS:
                if m not in ppl:
                    continue
                ev, base = ppl[m]['evaluation'], ppl[m]['evaluation'].get(BASE)
                for k in ppl[m].get('k_values', [K]):
                    ref, var = f'k{k}', f'k{k}_{objective}'
                    if ref not in ev or var not in ev or not base:
                        continue
                    cells.append((label, k))
                    if (ev[var]['wiki']['ppl'] > ev[ref]['wiki']['ppl']
                            and ev[var]['c4']['ppl'] > ev[ref]['c4']['ppl']):
                        losses += 1
                    if (ev[var]['wiki']['ppl'] > base['wiki']['ppl']
                            and ev[var]['c4']['ppl'] > base['c4']['ppl']):
                        harmful += 1
            if not cells:
                continue
            name = {'kl': 'KL', 'ce': 'CE'}[objective]
            models = sorted({label for label, _ in cells})
            L += [f'**{name} alone loses in {losses} of {len(cells)} cells.** Across '
                  f'{len(models)} models ({", ".join(models)}) and '
                  f'{len(sorted({k for _, k in cells}))} thresholds, dropping the other objective '
                  f'is worse on both corpora in {losses} of the {len(cells)} '
                  f'(model, k) cells measured'
                  + (f', and in {harmful} of them it is worse than not switching at all -- the '
                     f'method goes from a win to a loss.' if harmful else '.'), '']
        L.append('')

    if acc:
        rows = []
        for m, label in MODELS:
            if m not in acc:
                continue
            r, rundir = acc[m]
            sdir = os.path.join(os.path.dirname(rundir), f'samples_{m}')
            for pol in (f'k{K}', f'k{K}_kl', f'k{K}_ce'):
                if pol not in r['accuracy']:
                    continue
                mean = r['accuracy'][pol]['mean']
                d = mean - r['accuracy'][BASE]['mean'] if BASE in r['accuracy'] else None
                ref, var = (os.path.join(sdir, f'{BASE}.json'),
                            os.path.join(sdir, f'{pol}.json'))
                pooled = None
                if os.path.isfile(ref) and os.path.isfile(var):
                    _, pooled = compare(load_samples(ref), load_samples(var))
                rows.append((label, pol, mean, d, pooled))
        if rows:
            L += ['#### Zero-shot accuracy', '',
                  'The same policies on the multiple-choice panel, with the paired McNemar test '
                  'against the FourOverSix base. Positive is better here.', '',
                  '| model | policy | panel mean | d accuracy | pooled delta | McNemar p |',
                  '|---|---|---:|---:|---:|---:|']
            for label, pol, mean, d, pooled in rows:
                pd = fmt(pooled['delta'], 4, True) if pooled else '—'
                pp = f"{pooled['p']:.3g}" if pooled else '—'
                L.append(f'| {label} | {PRETTY.get(pol, pol)} | {fmt(mean, 4)} | '
                         f'{fmt(d, 4, True) if d is not None else "—"} | {pd} | {pp} |')
            L.append('')

    # Coverage, derived rather than asserted, so the note cannot drift from the runs behind it.
    covered = sorted({label for m, label in MODELS if m in ppl or m in acc})
    missing = [label for m, label in MODELS if m not in ppl and m not in acc]
    objectives = sorted({o for m, _ in MODELS if m in ppl
                         for o in ppl[m].get('objectives', [])})
    note = [f'**Coverage.** Measured on {", ".join(covered)}.']
    if missing:
        note.append(f'Not run on {", ".join(missing)}, so the conclusion is a two-model '
                    f'result, not a panel-wide one.')
    if 'ce' not in objectives:
        note.append('CE-only was not run: it is the arm that costs a second full election plus '
                    'evaluation to test the side of the conjunction the perplexity numbers '
                    'already favour, and the KL-only arm is the one the report\'s argument is '
                    'weakest on. The asymmetry of the evidence is therefore real -- this shows '
                    'that KL alone is not enough, not that CE alone would also fail.')
    L += [' '.join(note), '']

    out = '\n'.join(L) + '\n'
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(out)
    print(out)
    print(f'written to {args.out}')


if __name__ == '__main__':
    main()
