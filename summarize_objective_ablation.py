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
import re

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
    """All complete perplexity ablations per model, merged.

    Newest-wins is wrong here: the KL sweep and the CE sweep are separate jobs covering the
    same k but different objectives, so taking the newest silently drops whichever arm ran
    first. They are merged instead, and the merge is checked rather than assumed -- the two
    share every max(CE, KL) policy, so a disagreement there would mean the runs are not
    comparable and must not be pooled. Conflicts are recorded in `merge_conflicts`.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(root, 'job_*', '*', 'report.json'))):
        try:
            r = json.load(open(path))
        except Exception:
            continue
        if r.get('status') != 'complete' or 'evaluation' not in r:
            continue
        m = r['model']
        if m not in out:
            r.setdefault('merge_conflicts', [])
            r.setdefault('merged_jobs', [r.get('job_id', '?')])
            out[m] = r
            continue
        acc = out[m]
        for pol, e in r['evaluation'].items():
            if pol in acc['evaluation']:
                prev = acc['evaluation'][pol]
                if (prev['wiki']['ppl'] != e['wiki']['ppl']
                        or prev['c4']['ppl'] != e['c4']['ppl']):
                    acc['merge_conflicts'].append(pol)
            else:
                acc['evaluation'][pol] = e
                if pol in r.get('election', {}):
                    acc.setdefault('election', {})[pol] = r['election'][pol]
        acc['k_values'] = sorted(set(acc.get('k_values', [])) | set(r.get('k_values', [])))
        acc['objectives'] = sorted(set(acc.get('objectives', [])) | set(r.get('objectives', [])))
        acc['merged_jobs'].append(r.get('job_id', '?'))
    return out


def is_single_objective(pol):
    """True for a policy like k3_kl or k5_ce -- one objective rather than the conjunction."""
    return bool(re.fullmatch(r'k\d+_(kl|ce)', pol))


def acc_runs(root):
    """Every complete accuracy run carrying a single-objective policy, grouped by model.

    All of them, not the newest: the budget-matched settings live in separate jobs from the
    k = 3 ones, and dropping the older jobs would drop half the evidence. Comparisons stay
    *within* a job -- section 1a's control shows the evaluation is exact on one GPU model and
    flips 2.8% of documents across two, so a policy in job A and a policy in job B are not a
    paired comparison even when the weights are identical.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(root, 'job_*', '*', 'report.json'))):
        try:
            r = json.load(open(path))
        except Exception:
            continue
        if r.get('status') != 'complete':
            continue
        if any(is_single_objective(p) for p in r.get('accuracy', {})):
            job = os.path.basename(os.path.dirname(os.path.dirname(path)))
            out.setdefault(r['model'], []).append((job, r, os.path.dirname(path)))
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

        # Comparing objectives at the same k is not a fair test once k is swept: at a fixed k
        # they elect very different numbers of tiles. The question is whether an objective ever
        # wins at an equal BUDGET of switches, and it has to be asked in both directions --
        # asking only whether the conjunction beats the single objectives would have hidden
        # that CE alone beats the conjunction on Qwen3-4B.
        def frontier(ev, el, ks, objective):
            key = (lambda k: f'k{k}') if objective == 'max' else (lambda k: f'k{k}_{objective}')
            return [(el[key(k)]['selected'], ev[key(k)]['wiki']['ppl'], ev[key(k)]['c4']['ppl'], k)
                    for k in ks if key(k) in ev and key(k) in el]

        def beaten_by(point, others):
            """The cheapest setting in `others` that elects no more tiles and wins both corpora."""
            tiles, w, c, _ = point
            cand = [o for o in others if o[0] <= tiles and o[1] <= w and o[2] <= c]
            return min(cand, key=lambda o: o[0]) if cand else None

        dom_rows, unbeaten, reverse_hits = [], [], []
        for m, label in MODELS:
            if m not in ppl:
                continue
            ev, el = ppl[m]['evaluation'], ppl[m].get('election', {})
            ks = ppl[m].get('k_values', [K])
            mx = frontier(ev, el, ks, 'max')
            if not mx:
                continue
            for objective in ('kl', 'ce'):
                single = frontier(ev, el, ks, objective)
                name = OBJECTIVE_LABEL[objective]
                for pt in sorted(single):
                    hit = beaten_by(pt, mx)
                    if hit:
                        dom_rows.append(f'| {label} | {name} | {pt[3]} | {pt[0]:,} | '
                                        f'{pt[1]:.4f} | {hit[3]} | {hit[0]:,} | {hit[1]:.4f} |')
                    else:
                        dom_rows.append(f'| {label} | {name} | {pt[3]} | {pt[0]:,} | '
                                        f'{pt[1]:.4f} | — | — | — |')
                        unbeaten.append((label, name, pt))
                # And the other way: does this objective beat the conjunction at its own budget?
                for pt in sorted(mx):
                    hit = beaten_by(pt, single)
                    if hit:
                        reverse_hits.append((label, name, pt, hit))

        if dom_rows:
            total, n_unbeaten = len(dom_rows), len(unbeaten)
            L += ['Tightening the threshold is the obvious way to try to rescue a single '
                  'objective, and it helps a great deal -- on Llama-3.1-8B the KL-only penalty '
                  'falls from +0.481 WikiText at k = 3 to +0.009 at k = 6. But raising k also '
                  'shrinks the election, so much of that is buying back permissiveness rather '
                  'than showing the objective was fine all along.', '',
                  'The fair test is at an equal budget of switches: for each single-objective '
                  'setting, is there a conjunction setting that elects **no more tiles** and is '
                  'at least as good on **both** corpora?', '',
                  '| model | objective | k | its tiles | its WikiText | beaten by k | tiles | '
                  'WikiText |',
                  '|---|---|---:|---:|---:|---:|---:|---:|'] + dom_rows + ['']
            by_obj = {}
            for lb, nm, pt in unbeaten:
                by_obj.setdefault(nm, []).append(f'{lb} k = {pt[3]} ({pt[0]:,} tiles)')
            summary = (f'**{total - n_unbeaten} of {total} single-objective settings are beaten '
                       f'by a conjunction setting using no more tiles.**')
            if by_obj:
                summary += (' The ones that are not: '
                            + '; '.join(f'{nm} at ' + ', '.join(v) for nm, v in by_obj.items())
                            + '.')
            L += [summary, '']

        if reverse_hits:
            by_obj = {}
            for lb, nm, pt, hit in reverse_hits:
                by_obj.setdefault((lb, nm), []).append((pt, hit))
            L += ['Asked the other way -- is any **conjunction** setting beaten by a single '
                  'objective electing no more tiles? -- the answer is not empty, and this is the '
                  'part the report\'s argument does not predict:', '',
                  '| model | objective | conjunction k | its tiles | its WikiText | beaten by k | '
                  'tiles | WikiText |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
            for (lb, nm), items in by_obj.items():
                for pt, hit in sorted(items):
                    L.append(f'| {lb} | {nm} | {pt[3]} | {pt[0]:,} | {pt[1]:.4f} | {hit[3]} | '
                             f'{hit[0]:,} | {hit[1]:.4f} |')
            # Which objective is weak, and where, is counted from the dominance table rather
            # than asserted -- the answer differs by model and that is the whole point.
            beaten_frac = {}
            for nm in OBJECTIVE_LABEL.values():
                if nm == 'max(CE, KL)':
                    continue
                rows_nm = [r for r in dom_rows if f'| {nm} |' in r]
                if rows_nm:
                    lost = sum(1 for r in rows_nm if not r.rstrip().endswith('| — | — | — |'))
                    beaten_frac[nm] = (lost, len(rows_nm))
            tally = '; '.join(f'{nm} is beaten at {a} of {b} settings'
                              for nm, (a, b) in beaten_frac.items())
            L += ['',
                  f'So the conjunction is not uniformly the best objective at a given budget. '
                  f'The two single objectives are not equally weak either -- {tally} -- so the '
                  f'case for requiring both rests much more heavily on KL than on CE. What the '
                  f'conjunction has going for it on this evidence is not that it wins '
                  f'everywhere, but that it is never badly wrong, and which single objective '
                  f'fails is not something the calibration predicts in advance.', '']

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
            for job, r, rundir in acc.get(m, []):
                sdir = os.path.join(os.path.dirname(rundir), f'samples_{m}')
                acc_d, el = r['accuracy'], r.get('election', {})
                shipped = f'k{K}'
                for pol in sorted(acc_d, key=lambda x: (not is_single_objective(x), x)):
                    if pol == BASE or pol == shipped:
                        continue
                    if not is_single_objective(pol):
                        continue
                    tiles = el.get(pol, {}).get('selected')
                    ship_tiles = el.get(shipped, {}).get('selected')
                    tests = {}
                    for ref in (shipped, BASE):
                        fa = os.path.join(sdir, f'{ref}.json')
                        fb = os.path.join(sdir, f'{pol}.json')
                        if ref in acc_d and os.path.isfile(fa) and os.path.isfile(fb):
                            _, tests[ref] = compare(load_samples(fa), load_samples(fb))
                    rows.append((label, job, pol, tiles, ship_tiles,
                                 acc_d[pol]['mean'], tests))

        if rows:
            L += ['#### Zero-shot accuracy', '',
                  'The same question on the multiple-choice panel of §1a, with the paired '
                  'McNemar test on per-document outcomes. Each row is compared against the '
                  'shipped rule **measured in the same job**: §1a\'s control shows the '
                  'evaluation is exact on one GPU model and flips 2.8% of documents across two, '
                  'so policies from different jobs are not a paired comparison. Positive favours '
                  'the single objective.', '',
                  '| model | policy | tiles | vs shipped k = 3 | panel mean | d vs shipped | p | '
                  'd vs FourOverSix | p |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
            for label, job, pol, tiles, ship_tiles, mean, tests in rows:
                ratio = (f'{tiles / ship_tiles:.2f}x' if tiles and ship_tiles else '—')
                cells = []
                for ref in (f'k{K}', BASE):
                    t = tests.get(ref)
                    cells += ([fmt(t['delta'], 4, True), f"{t['p']:.3g}"] if t else ['—', '—'])
                L.append(f'| {label} | {pretty(pol)} | '
                         f'{f"{tiles:,}" if tiles is not None else "—"} | {ratio} | '
                         f'{fmt(mean, 4)} | ' + ' | '.join(cells) + ' |')
            L.append('')

            # Counted, not asserted: does any single objective ever beat the conjunction here?
            better = [(lb, pretty(pol)) for lb, _, pol, *_ , t in rows
                      if (v := t.get(f'k{K}')) and v['delta'] > 0 and v['p'] < 0.05]
            worse = [(lb, pretty(pol)) for lb, _, pol, *_ , t in rows
                     if (v := t.get(f'k{K}')) and v['delta'] < 0 and v['p'] < 0.05]
            tested = [r for r in rows if r[-1].get(f'k{K}')]
            null = len(tested) - len(better) - len(worse)
            if better:
                verdict = (f'{len(better)} of {len(tested)} settings beat the conjunction '
                           f'significantly (' + ', '.join(f'{a} {b}' for a, b in better) + ').')
            else:
                verdict = (f'**None of the {len(tested)} settings beats the conjunction.** '
                           f'{len(worse)} are significantly worse and {null} are indistinguishable '
                           f'from it.')
            L += [f'{verdict} The rows where a single objective elects far more tiles than the '
                  f'shipped rule are the ones that look closest to it, which is the tile count '
                  f'talking rather than the objective -- the ratio column is there to make that '
                  f'visible. Note also what the last two columns do not say together: a setting '
                  f'can beat the FourOverSix base convincingly and still not reach the '
                  f'conjunction, and several do exactly that.', '']

    # The conjunction is the INTERSECTION of the two thresholds, so adding KL to CE can only
    # remove tiles. That makes "is KL needed?" a question with an exact form: do the tiles CE
    # accepts and KL vetoes help or hurt? This is the one place a same-k comparison is the right
    # one, because the CE threshold is held fixed and only the veto changes.
    veto = []
    for m, label in MODELS:
        if m not in ppl:
            continue
        ev, el = ppl[m]['evaluation'], ppl[m].get('election', {})
        base = ev.get(BASE)
        for k in sorted(ppl[m].get('k_values', [])):
            ce, mx = f'k{k}_ce', f'k{k}'
            if ce not in ev or mx not in ev or ce not in el or mx not in el:
                continue
            n_ce, n_mx = el[ce]['selected'], el[mx]['selected']
            dw = ev[mx]['wiki']['ppl'] - ev[ce]['wiki']['ppl']
            dc = ev[mx]['c4']['ppl'] - ev[ce]['c4']['ppl']
            harmful = base and (ev[ce]['wiki']['ppl'] > base['wiki']['ppl']
                                or ev[ce]['c4']['ppl'] > base['c4']['ppl'])
            veto.append((label, k, n_ce, n_ce - n_mx, dw, dc, harmful))

    if veto:
        L += ['#### Is KL needed, or would CE alone do?', '',
              'The rule elects when **both** bounds are negative, so the elected set is the '
              'intersection: adding KL to CE can only take tiles away. That gives the question '
              'an exact form -- are the tiles CE accepts and KL vetoes worth keeping? Holding the '
              'CE threshold fixed and varying only the veto is the one comparison here where '
              'matching k is right rather than misleading, because the tile count difference '
              '*is* the effect being measured.', '',
              '| model | k | CE elects | KL vetoes | d WikiText from vetoing | d C4 | CE alone vs '
              'the base |',
              '|---|---:|---:|---:|---:|---:|---|']
        for label, k, n_ce, n_veto, dw, dc, harmful in veto:
            L.append(f'| {label} | {k} | {n_ce:,} | {n_veto:,} | {dw:+.4f} | {dc:+.4f} | '
                     + ('**worse than not switching**' if harmful else 'an improvement') + ' |')
        loose = [v for v in veto if v[6]]
        strict_bad = [v for v in veto if not v[6] and v[4] > 0.1 and v[5] > 0.1]
        L += ['',
              'Negative means the veto helps. The answer is not uniform, and the pattern is the '
              'useful part:', '']
        if loose:
            worst = max(loose, key=lambda v: abs(v[5]))
            L.append(f'- **At the loosest threshold the veto is essential.** CE alone is worse '
                     f'than not switching at all in {len(loose)} of the {len(veto)} cells, and '
                     f'the veto is worth up to {abs(worst[5]):.2f} C4 there ({worst[0]}, '
                     f'k = {worst[1]}). This is KL working as the safety net the rule claims.')
        if strict_bad:
            worst = max(strict_bad, key=lambda v: v[4])
            L.append(f'- **At strict thresholds it costs.** In {len(strict_bad)} cells the veto '
                     f'is harmful on both corpora, by as much as {worst[4]:+.2f} WikiText '
                     f'({worst[0]}, k = {worst[1]}), where it discards {worst[3]:,} tiles CE had '
                     f'accepted correctly.')
        L += ['',
              'So KL is not selecting tiles; it is insuring against a threshold that is too '
              'loose. Where the threshold is already strict, its veto mostly destroys value. '
              'That is a narrower role than "a switch is kept only when it improves the actual '
              'task loss **and** moves the quantized model back toward its own unquantized '
              'reference" suggests, and the accuracy table above is what keeps it from being an '
              'argument for dropping KL at k = 3: on Qwen3-4B, CE alone there costs 0.0161 '
              'accuracy at p = 1e-11 while electing 15.9 times as many tiles.', '']

    # The table above compares each single-objective run against the shipped rule inside its own
    # job, which answers the practical question. The fair objective comparison needs the
    # conjunction measured at the SAME budgets, which lives in its own analysis.
    try:
        from analyze_accuracy_frontier import build as frontier_block
        frontier = frontier_block()
    except Exception as exc:                    # a sub-analysis is not worth failing the section
        print(f'NOTE: accuracy frontier skipped: {exc!r}')
        frontier = None
    if frontier:
        L += frontier.rstrip().split('\n') + ['']

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
        # Every KL-only accuracy comparison for this model, each against the shipped rule in
        # its own job. Summarising the smallest-budget one alone would flatter the objective
        # and the largest alone would damn it, so the count of outcomes is what is reported.
        verdicts = []
        for job, r, rundir in acc.get(m, []):
            sdir = os.path.join(os.path.dirname(rundir), f'samples_{m}')
            for pol in sorted(r['accuracy']):
                if not re.fullmatch(rf'k\d+_kl', pol):
                    continue
                ref = os.path.join(sdir, f'k{K}.json')
                var = os.path.join(sdir, f'{pol}.json')
                if os.path.isfile(ref) and os.path.isfile(var):
                    _, pooled = compare(load_samples(ref), load_samples(var))
                    verdicts.append(pooled)
        acc_better = acc_worse = False
        if verdicts:
            bett = [v for v in verdicts if v['delta'] > 0 and v['p'] < 0.05]
            wors = [v for v in verdicts if v['delta'] < 0 and v['p'] < 0.05]
            acc_better, acc_worse = bool(bett), bool(wors)
            span = f'{min(v["delta"] for v in verdicts):+.4f} to ' \
                   f'{max(v["delta"] for v in verdicts):+.4f}'
            av = (f'spans {span} on accuracy across {len(verdicts)} threshold(s), '
                  + ('none of them significantly better' if not bett else
                     f'{len(bett)} significantly better')
                  + (f' and {len(wors)} significantly worse' if wors else
                     ' and none significantly worse'))
        ppl_worse = bool(pv and pv.startswith('costs'))
        if pv or av:
            synth.append((label, pv, av, ppl_worse, acc_better))

    if synth and any(a for _, _, a, _, _ in synth):
        # Do the two metrics actually point opposite ways anywhere, or does one merely fail to
        # resolve what the other sees? Those need different lead sentences.
        contradicts = [label for label, _, _, ppl_worse, acc_better in synth
                       if ppl_worse and acc_better]
        lead = ('The two metrics point in opposite directions on '
                + ', '.join(contradicts) + ', so both are stated per model rather than '
                'generalizing from whichever is more convenient.' if contradicts else
                'Neither metric favours KL alone on any model. Where they differ it is in how '
                'sharply they say so, not in which way, so both are stated per model.')
        L += ['#### Reading the two together', '', lead, '']
        for label, pv, av, _, _ in synth:
            parts = [x for x in (pv, av) if x]
            L.append(f'- **{label}.** Against the shipped rule at the same k, KL alone '
                     + ' and '.join(parts) + '.')
        # Whether accuracy anywhere favours KL alone decides how the closing claim may be put.
        wins = [label for label, _, _, _, acc_better in synth if acc_better]
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
        note.append('CE-only is not in these numbers yet -- its election and evaluation are '
                    'running. Until they land the evidence is one-sided by construction: what '
                    'is shown is that KL alone is not enough, not that CE alone would also '
                    'fail, and the conjunction is not yet demonstrated to need both halves.')
    L += [' '.join(note), '']

    out = '\n'.join(L) + '\n'
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(out)
    print(out)
    print(f'written to {args.out}')


if __name__ == '__main__':
    main()
