#!/usr/bin/env python3
"""Kernel-opt amendment 18 (V) report: M1 per-forward GEMM sums with the registered rule (the candidate against today's
256x64 path), the build and table effects, the gaps to stock and the ceiling per T, the table's changes and the
decisive-margin sensitivity, and C2V.

    python experiments/kernel_opt/V_report.py [--src /home/dev/n16k64_campaign/kernel_opt/V/run] [--out-dir results/kernel_opt/V]

M1 as amendments 10-17: per model and T, the per-forward GEMM sum is the sum over projections of modules x median time; a
ratio's range is over the rounds, from each round's sums. The rule (PROTOCOL.md, amendment 18; amendment 11's form, TOL =
0.5 %): 'C vs A' with typical and with worst tags each has a negative median over the 48 (model, T) cells, and no cell
is above +TOL in every round. Cells above zero in every round by less are listed (the strict form).
"""
import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
NAMES = dict(llama8b='Llama-3.1-8B', mistral7b='Mistral-7B-v0.3', phi4='Phi-4', qwen27b='Qwen3.8-27B')
TOL = 0.5
V = ('typical', 'worst')
RATIOS = ([(f'candidate vs today, {v}', f'C_{v}', f'A_{v}') for v in V]
          + [(f'builds at today\'s widths vs today, {v}', f'B_{v}', f'A_{v}') for v in V]
          + [(f'table: candidate vs builds at today\'s widths, {v}', f'C_{v}', f'B_{v}') for v in V]
          + [(f'today vs stock_ko, {v}', f'A_{v}', 'stock_ko') for v in V]
          + [(f'candidate vs stock_ko, {v}', f'C_{v}', 'stock_ko') for v in V]
          + [('ceiling vs stock_ko', 'ceil', 'stock_ko'), ('candidate vs ceiling, typical', 'C_typical', 'ceil')])
BANDS = (('T ≤ 16', lambda t: t <= 16), ('32–128', lambda t: 32 <= t <= 128), ('256–1024', lambda t: 256 <= t <= 1024),
         ('≥ 2048', lambda t: t >= 2048))


def pct(a, b):
    return 100 * (a / b - 1)


def fmt(r):
    return f"{r['pct']:+.2f} % [{r['range'][0]:+.2f}, {r['range'][1]:+.2f}]"


def rule(rows, unit):
    v = [r[unit]['pct'] for r in rows]
    above = [f"{r['model']} T={r['tokens']} ({fmt(r[unit])})" for r in rows if r[unit]['range'][0] > 0]
    beyond = [f"{r['model']} T={r['tokens']} ({fmt(r[unit])})" for r in rows if r[unit]['range'][0] > TOL]
    return dict(cells=len(v), median=statistics.median(v) if v else None, cells_above_every_round=above,
                cells_above_tolerance=beyond, below_every_round=sum(r[unit]['range'][1] < 0 for r in rows),
                met=bool(v) and statistics.median(v) < 0 and not beyond, strict_form_met=not above)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=Path('/home/dev/n16k64_campaign/kernel_opt/V/run'))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'kernel_opt' / 'V')
    args = ap.parse_args()
    md = ['## M1: per-forward GEMM time, deviation-2\n',
          'Per model and T: the sum over projections of modules × median GEMM time. Brackets: the range over the rounds of '
          'the same ratio, from each round\'s sums. A = today\'s 256x64 path (`mixed256`, `build_A1`, the paper table); '
          'B = `mixed256_ko` (`build_V`) at today\'s widths; C = `mixed256_ko` on the candidate table (the re-tuned widths '
          'and scheduler rows).\n']
    data, allrows = {}, []
    for model in MODELS:
        path = args.src / 'gemm' / f'{model}.json'
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        if rec.get('status') != 'complete':
            md.append(f'{model}: {rec.get("status")}\n')
            continue
        projs = {p: v for p, v in rec['projections'].items() if any(r['proj'] == p for r in rec['rows'])}
        by = {(r['config'], r['proj'], r['tokens']): r for r in rec['rows']}
        tokens = sorted({r['tokens'] for r in rec['rows']})
        nr = len(rec['rows'][0]['rounds'])
        cfgs = sorted({r['config'] for r in rec['rows']})
        s, rr = {}, {}
        for t in tokens:
            for c in cfgs:
                s[(c, t)] = sum(projs[p]['modules'] * by[(c, p, t)]['gemm_us'] for p in projs)
                rr[(c, t)] = [sum(projs[p]['modules'] * by[(c, p, t)]['rounds'][i]['gemm_us'] for p in projs) for i in range(nr)]
        ok = all(c['equal'] for c in rec['checks']['bitwise'])
        md.append(f"### {NAMES[model]}\n\nChecks: {len(rec['checks']['bitwise'])} bitwise (each configuration's isolated "
                  f"path against NativeLinear, and B / C against A on the timed operands), all equal: {ok}; counts ok: "
                  f"{rec['checks']['counts_ok']}; other processes: {len(rec['checks']['other_processes'])}.\n")
        md.append('| T | ' + ' | '.join(u for u, _, _ in RATIOS[:6]) + ' |')
        md.append('|---|' + '---|' * 6)
        rows = []
        for t in tokens:
            row = dict(model=model, tokens=t)
            for u, a, b in RATIOS:
                per = [pct(x, y) for x, y in zip(rr[(a, t)], rr[(b, t)])]
                row[u] = dict(pct=pct(s[(a, t)], s[(b, t)]), range=[min(per), max(per)])
            rows.append(row)
            md.append(f'| {t} | ' + ' | '.join(fmt(row[u]) for u, _, _ in RATIOS[:6]) + ' |')
        md.append('')
        allrows += rows
        data[model] = dict(rows=rows, sums={f'{c}@{t}': v for (c, t), v in s.items()},
                           round_sums={f'{c}@{t}': v for (c, t), v in rr.items()}, bitwise_ok=ok, sampler=rec.get('sampler'))
    if allrows:
        md.append('### All models: over the (model, T) cells, and by T band (median)\n')
        md.append('| ratio | median | min | max | above 0 in every round | below 0 in every round | '
                  + ' | '.join(b for b, _ in BANDS) + ' |')
        md.append('|---|---:|---:|---:|---:|---:|' + '---:|' * len(BANDS))
        summ = {}
        for u, _, _ in RATIOS:
            v = [r[u]['pct'] for r in allrows]
            bands = [[r[u]['pct'] for r in allrows if f(r['tokens'])] for _, f in BANDS]
            summ[u] = dict(median=statistics.median(v), min=min(v), max=max(v), cells=len(v),
                           above=sum(r[u]['range'][0] > 0 for r in allrows), below=sum(r[u]['range'][1] < 0 for r in allrows),
                           bands={b: (statistics.median(x) if x else None) for (b, _), x in zip(BANDS, bands)})
            md.append(f"| {u} | {summ[u]['median']:+.2f} % | {min(v):+.2f} % | {max(v):+.2f} % | {summ[u]['above']} of "
                      f"{len(v)} | {summ[u]['below']} of {len(v)} | "
                      + ' | '.join(f'{statistics.median(b):+.2f} %' if b else '—' for b in bands) + ' |')
        md.append('')
        md.append('### Per T, median over the models\n')
        md.append('| T | ' + ' | '.join(u for u, _, _ in RATIOS) + ' |')
        md.append('|---:|' + '---:|' * len(RATIOS))
        per_t = {}
        for t in sorted({r['tokens'] for r in allrows}):
            per_t[t] = {u: statistics.median([r[u]['pct'] for r in allrows if r['tokens'] == t]) for u, _, _ in RATIOS}
            md.append(f'| {t} | ' + ' | '.join(f'{per_t[t][u]:+.2f} %' for u, _, _ in RATIOS) + ' |')
        md.append('')
        rules = {v: rule(allrows, f'candidate vs today, {v}') for v in V}
        rules['met'] = all(rules[v]['met'] for v in V)
        md.append(f'### The registered rule (TOL = {TOL} %)\n')
        md.append(f"- **Candidate vs today:** {'met' if rules['met'] else 'NOT met'}. " + '; '.join(
            f"{v}: median {rules[v]['median']:+.2f} % over {rules[v]['cells']} cells, {rules[v]['below_every_round']} "
            f"below zero in every round, {len(rules[v]['cells_above_tolerance'])} above +{TOL} % in every round" for v in V)
            + '.')
        for v in V:
            if rules[v]['cells_above_every_round']:
                md.append(f"  - {v}, above zero in every round (the strict form): "
                          + '; '.join(rules[v]['cells_above_every_round']))
        md.append('')
        data.update(summary=summ, per_t=per_t, rules=rules)
    tv = REPO / 'results' / 'kernel_opt' / 'V' / 'table_v'
    if (tv / 'table_v.json').exists():
        t1 = json.loads((tv / 'table_v.json').read_text())
        t0 = json.loads((tv / 'table_v0.json').read_text())
        cells = [(sh, b, t0['mixed256'][sh][b], w) for sh, row in t1['mixed256'].items() for b, w in row.items()]
        changed = [c for c in cells if str(c[2]) != str(c[3])]
        sched = t1['schedule']['mixed256_ko']
        moved = sum(1 for r in sched.values() for v in r.values() if tuple(v) != (0, 1))
        md.append('## The candidate table (`table_v/`)\n')
        md.append(f'- **Widths:** {len(changed)} of {len(cells)} cells differ from today\'s (the paper table\'s \'mixed\' '
                  f'rows).')
        md.append(f'- **Scheduler rows:** {moved} of {sum(len(r) for r in sched.values())} cells are not (0, 1).')
        if (tv / 'sensitivity.json').exists():
            sens = json.loads((tv / 'sensitivity.json').read_text())['summary']
            md.append('- **The decisive-margin sensitivity** (reported only): ' + '; '.join(
                f"{k}: {v['cells_changed']} cells would differ"
                + (f" (median {v['median_pct']:+.2f} %, max {v['max_pct']:+.2f} % against the fastest)" if v['cells_changed']
                   else '') for k, v in sens.items()) + '.')
        md.append('')
        data['table'] = dict(cells=len(cells), changed=[dict(shape=a, bucket=b, today=c, candidate=d) for a, b, c, d in changed],
                             scheduler_cells_moved=moved)
    c2 = args.src / 'c2_V.json'
    if c2.exists():
        rec = json.loads(c2.read_text())
        md.append('## C2V: 4096³, width 128\n')
        md.append(f"Checks: {rec['checks']}. stock_ko runs width {rec['stock_ko']['width']}, scheduler "
                  f"{rec['stock_ko']['schedule']}.\n")
        keys = list(next(iter(rec['summary'].values())))
        md.append('| ratio | ' + ' | '.join(rec['summary']) + ' |')
        md.append('|---|' + '---:|' * len(rec['summary']))
        for k in keys:
            md.append(f'| {k} | ' + ' | '.join(f"{rec['summary'][m][k]:+.2f} %" for m in rec['summary']) + ' |')
        md.append('')
        data['c2'] = rec['summary']
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'V.json').write_text(json.dumps(data, indent=1) + '\n')
    (args.out_dir / 'V_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
