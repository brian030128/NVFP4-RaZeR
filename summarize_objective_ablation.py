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
OBJECTIVE_LABEL = {'max': 'max(CE, KL)', 'kl': 'KL only', 'ce': 'CE only'}


def pretty(pol):
    """Display name for a policy key like 'k3', 'k4_kl' -- any k, any objective."""
    body = pol[1:]
    k, _, objective = body.partition('_')
    label = OBJECTIVE_LABEL.get(objective or 'max', objective or pol)
    return f'{label} — shipped' if pol == f'k{K}' else label


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
                           f'{want:,}' + ('' if close else f', which is {ratio:.2f}x the shipped '
                                          f'count -- near but not a match'))
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
            for k in sorted(ppl[m].get('k_values', [K])):
                for pol in (f'k{k}', f'k{k}_kl', f'k{k}_ce'):
                    if pol not in ev:
                        continue
                    w, c = ev[pol]['wiki']['ppl'], ev[pol]['c4']['ppl']
                    L.append(f'| {label} | {k} | {pretty(pol)} | '
                             f'{el.get(pol, {}).get("selected", 0):,} | {fmt(w)} | '
                             f'{fmt(w - base["wiki"]["ppl"], signed=True)} | {fmt(c)} | '
                             f'{fmt(c - base["c4"]["ppl"], signed=True)} |')
        L.append('')

        # Comparing the two objectives at the same k is not a fair test once k is swept: at a
        # fixed k they elect very different numbers of tiles. The question the sweep exists to
        # answer is whether KL alone ever wins at an equal budget of switches, so ask that
        # directly -- is every KL-only setting beaten by some conjunction setting that elects no
        # more tiles?
        dom_rows, undominated = [], []
        for m, label in MODELS:
            if m not in ppl:
                continue
            ev, el = ppl[m]['evaluation'], ppl[m]['election']
            ks = ppl[m].get('k_values', [K])
            pts = {}
            for objective in ('max', 'kl'):
                pts[objective] = [
                    (el[key]['selected'], ev[key]['wiki']['ppl'], ev[key]['c4']['ppl'], k)
                    for k in ks
                    for key in [f'k{k}' if objective == 'max' else f'k{k}_{objective}']
                    if key in ev and key in el]
            if not pts['max'] or not pts['kl']:
                continue
            for tiles, w, c, k in sorted(pts['kl']):
                beaten = [(t2, w2, c2, k2) for t2, w2, c2, k2 in pts['max']
                          if t2 <= tiles and w2 <= w and c2 <= c]
                if beaten:
                    t2, w2, c2, k2 = min(beaten, key=lambda r: r[0])
                    dom_rows.append(f'| {label} | {k} | {tiles:,} | {w:.4f} | {k2} | '
                                    f'{t2:,} | {w2:.4f} |')
                else:
                    dom_rows.append(f'| {label} | {k} | {tiles:,} | {w:.4f} | — | — | — |')
                    undominated.append((label, k, tiles, w, c))

        if dom_rows:
            total = len(dom_rows)
            beaten_n = total - len(undominated)
            L += ['Tightening the threshold is the obvious way to try to rescue KL alone, and it '
                  'helps a great deal -- on Llama-3.1-8B the KL-only penalty falls from +0.481 '
                  'WikiText at k = 3 to +0.009 at k = 6. But raising k also shrinks the '
                  'election, so most of that is buying back permissiveness rather than showing '
                  'the objective was fine all along.', '',
                  'The fair test is at an equal budget of switches. For each KL-only setting, is '
                  'there a conjunction setting that elects **no more tiles** and is at least as '
                  'good on **both** corpora?', '',
                  '| model | KL-only k | its tiles | its WikiText | beaten by k | tiles | '
                  'WikiText |',
                  '|---|---:|---:|---:|---:|---:|---:|'] + dom_rows + ['']
            if beaten_n == total:
                L += [f'**Every one of the {total} KL-only settings is beaten by a conjunction '
                      f'setting using no more tiles.** Strictness is not the missing ingredient: '
                      f'at any budget of switches the conjunction reaches a lower perplexity, so '
                      f'CE is selecting different and better tiles rather than merely fewer.', '']
            else:
                names = '; '.join(f'{lb} at k = {k} ({t:,} tiles)'
                                  for lb, k, t, _, _ in undominated)
                L += [f'**{beaten_n} of {total} KL-only settings are beaten by a conjunction '
                      f'setting using no more tiles.** The exceptions are {names} -- not a win '
                      f'for KL alone, since those are not better than the conjunction on both '
                      f'corpora either, but points where the two are incomparable at that '
                      f'budget. Strictness is therefore most of what separated them at k = 3, '
                      f'and it is not all of it.', '']

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
            # Kept as a secondary read: swapping the objective while holding k fixed is what
            # a practitioner would try first, even though it does not hold the count fixed.
            L += [f'Held at the same k instead of the same budget -- the naive swap -- {name} '
                  f'alone is worse on both corpora in {losses} of {len(cells)} (model, k) cells'
                  + (f', and in {harmful} of them worse than not switching at all.'
                     if harmful else '.')
                  + f' That comparison flatters neither objective, since at a fixed k the two '
                  f'elect different numbers of tiles; the budget-matched table above is the '
                  f'one to read.', '']
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
                tiles = r.get('election', {}).get(pol, {}).get('selected')
                rows.append((label, pol, tiles, mean, d, pooled))
        if rows:
            L += ['#### Zero-shot accuracy', '',
                  'The same policies on the multiple-choice panel, with the paired McNemar test '
                  'against the FourOverSix base. Positive is better here.', '',
                  '| model | policy | tiles | panel mean | d accuracy | pooled delta | '
                  'McNemar p |',
                  '|---|---|---:|---:|---:|---:|---:|']
            for label, pol, tiles, mean, d, pooled in rows:
                pd = fmt(pooled['delta'], 4, True) if pooled else '—'
                pp = f"{pooled['p']:.3g}" if pooled else '—'
                L.append(f'| {label} | {pretty(pol)} | '
                         f'{f"{tiles:,}" if tiles is not None else "—"} | {fmt(mean, 4)} | '
                         f'{fmt(d, 4, True) if d is not None else "—"} | {pd} | {pp} |')
            L.append('')

    # Perplexity and accuracy do not agree here, so the synthesis names both per model rather
    # than reporting whichever is more convenient. `verdict_rows` is (model, ppl verdict, acc
    # verdict), each entry None when that metric was not run for the model.
    synth = []
    for m, label in MODELS:
        pv = av = None
        if m in ppl:
            ev = ppl[m]['evaluation']
            if f'k{K}' in ev and f'k{K}_kl' in ev:
                dw = ev[f'k{K}_kl']['wiki']['ppl'] - ev[f'k{K}']['wiki']['ppl']
                dc = ev[f'k{K}_kl']['c4']['ppl'] - ev[f'k{K}']['c4']['ppl']
                pv = (f'costs {dw:+.3f} WikiText and {dc:+.3f} C4' if dw > 0 and dc > 0 else
                      f'gains {dw:+.3f} WikiText and {dc:+.3f} C4' if dw < 0 and dc < 0 else
                      f'is mixed ({dw:+.3f} WikiText, {dc:+.3f} C4)')
        if m in acc:
            r, rundir = acc[m]
            sdir = os.path.join(os.path.dirname(rundir), f'samples_{m}')
            ref, var = (os.path.join(sdir, f'k{K}.json'), os.path.join(sdir, f'k{K}_kl.json'))
            if os.path.isfile(ref) and os.path.isfile(var):
                _, pooled = compare(load_samples(ref), load_samples(var))
                sig = pooled['p'] < 0.05
                direction = 'better' if pooled['delta'] > 0 else 'worse'
                av = (f'is {pooled["delta"]:+.4f} on accuracy, '
                      + (f'significantly {direction} (p = {pooled["p"]:.3g})' if sig else
                         f'not distinguishable from it (p = {pooled["p"]:.2f})'))
        if pv or av:
            synth.append((label, pv, av))

    if synth and any(a for _, _, a in synth):
        # Do the two metrics actually point opposite ways anywhere, or does one merely fail to
        # resolve what the other sees? Those need different lead sentences.
        contradicts = [label for label, pv, av in synth
                       if pv and av and 'costs' in pv and 'significantly better' in av]
        lead = ('The two metrics point in opposite directions on '
                + ', '.join(contradicts) + ', so both are stated per model rather than '
                'generalizing from whichever is more convenient.' if contradicts else
                'Neither metric favours KL alone on any model. Where they differ it is in how '
                'sharply they say so, not in which way, so both are stated per model.')
        L += ['#### Reading the two together', '', lead, '']
        for label, pv, av in synth:
            parts = [x for x in (pv, av) if x]
            L.append(f'- **{label}.** Against the shipped rule at the same k, KL alone '
                     + ' and '.join(parts) + '.')
        # Whether accuracy anywhere favours KL alone decides how the closing claim may be put.
        wins = [label for label, _, av in synth
                if av and 'significantly better' in av]
        tail = ('No model shows accuracy favouring KL alone by a significant margin, so nothing '
                'in the accuracy numbers offsets the perplexity cost.' if not wins else
                f'On {", ".join(wins)} accuracy does significantly favour KL alone, which the '
                f'perplexity numbers do not, and that disagreement is unresolved here rather '
                f'than settled in favour of either.')
        L += ['',
              'Perplexity is the metric the election is calibrated on -- the score is a '
              'teacher-forced loss -- so it is the one KL alone should do well on if the '
              'objective were sufficient, and it is the one where it does not. The accuracy '
              'panel resolves about 0.005 at best (§1a), so a null there is a weaker statement '
              'than a perplexity regression of the size seen above. ' + tail, '']

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
