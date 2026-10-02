#!/usr/bin/env python3
"""Kernel-opt amendment 11 (the 8x64 plan's P2) report: M1 per-forward GEMM sums, the registered t0 and P1 rules, and the
C2w''' 4096^3 breakdown.

    python experiments/kernel_opt/w8p2_report.py [--src /home/dev/n16k64_campaign/kernel_opt/w8p2] \
        [--out-dir results/kernel_opt/w8/p2]

M1 is summarized as amendment 10's report (w8_report.py) does. Per model and T, the per-forward GEMM sum is the sum over
projections of modules x median time, for every configuration of bench_w8p2_isolated.py. A ratio's spread is its range
over the 3 rounds, computed from each round's sums.

Two groups of ratios:
- t0 and the dispatch: P2's t0 builds against the tagged ones (1b's default and #2's), and #2's dispatch against the
  default on the t0 builds (P1).
- The gap to stock: the 8x64 path before (1b, #2) and after (t0, t0 + #2) against stock_ko (the target, weights on A)
  and stock_wB (the same placement).

Over all (model, T) cells and by T band, the report gives the median, min and max, and the cells above or below zero in
every round. It then evaluates the two registered rules (PROTOCOL.md, amendment 11):
- t0 adoption: every t0 unit has a negative median over the cells, and no cell is above zero in every round by more than
  TOL (0.5 %). The cells above zero in every round by less are listed (amendment 6's strict form).
- P1: on the t0 builds with typical tags, #2's dispatch has a negative median over the cells, and more cells below zero
  in every round than above.
C2w''': every mode's times and its summary.
"""
import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
NAMES = dict(llama8b='Llama-3.1-8B', mistral7b='Mistral-7B-v0.3', phi4='Phi-4', qwen27b='Qwen3.8-27B')
# (label, a, b): a vs b
GROUPS = (
    ('t0 and the dispatch', (
        ('t0 vs 1b, typical', 'wBt0_typical', 'wB_typical'),
        ('t0 vs 1b, worst', 'wBt0_worst', 'wB_worst'),
        ('t0 #2 vs #2, typical', 'wBt0freq_typical', 'wBfreq_typical'),
        ('t0 #2 vs #2, worst', 'wBt0freq_worst', 'wBfreq_worst'),
        ("P1: t0 #2's vs t0 default, typical", 'wBt0freq_typical', 'wBt0_typical'),
        ("P1: t0 #2's vs t0 default, worst", 'wBt0freq_worst', 'wBt0_worst'))),
    ('the gap to stock', (
        ('1b vs stock_ko, typical', 'wB_typical', 'stock_ko'),
        ('#2 vs stock_ko, typical', 'wBfreq_typical', 'stock_ko'),
        ('t0 vs stock_ko, typical', 'wBt0_typical', 'stock_ko'),
        ('t0 vs stock_ko, worst', 'wBt0_worst', 'stock_ko'),
        ('t0 #2 vs stock_ko, typical', 'wBt0freq_typical', 'stock_ko'),
        ('t0 #2 vs stock_ko, worst', 'wBt0freq_worst', 'stock_ko'),
        ('t0 vs stock_wB, typical', 'wBt0_typical', 'stock_wB'),
        ('t0 #2 vs stock_wB, typical', 'wBt0freq_typical', 'stock_wB'),
        ('stock_wB vs stock_ko', 'stock_wB', 'stock_ko'))))
RATIOS = tuple(r for _, rs in GROUPS for r in rs)
T0_UNITS = ('t0 vs 1b, typical', 't0 vs 1b, worst', 't0 #2 vs #2, typical', 't0 #2 vs #2, worst')
P1 = "P1: t0 #2's vs t0 default, typical"
TOL = 0.5   # %, the t0 adoption rule's tolerance for a cell above zero in every round
BANDS = (('T ≤ 16', lambda t: t <= 16), ('32–128', lambda t: 32 <= t <= 128), ('256–1024', lambda t: 256 <= t <= 1024),
         ('≥ 2048', lambda t: t >= 2048))


def pct(a, b):
    return 100 * (a / b - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=Path('/home/dev/n16k64_campaign/kernel_opt/w8p2'))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'kernel_opt' / 'w8' / 'p2')
    args = ap.parse_args()
    md = ['## M1: per-forward GEMM time, deviation-2\n',
          'Per model and T: the sum over projections of modules × median GEMM time. Brackets: the range over the rounds '
          'of the same ratio, from each round\'s sums. "1b" is the `mixed_wB` set (optimization 1b, the kernel-opt '
          '`build`); "#2" the same set built with #2\'s dispatch (`build_freq`); "t0" the `mixed_wB_t0` set (`build_P2`); '
          '"t0 #2" that set with #2\'s dispatch (`build_P2freq`). Every 8x64 set runs the paper table\'s `mixed_wB` rows.\n']
    data, allrows = {}, []
    for model in MODELS:
        path = args.src / 'gemm' / f'{model}.json'
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        if rec.get('status') != 'complete':
            md.append(f'{model}: {rec.get("status")}\n')
            continue
        projs = {p: v for p, v in rec['projections'].items() if any(r['proj'] == p for r in rec['rows'])}   # all, in a full run
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
        widths = {f'{p}@{t}': dict(wB=by[('wB_typical', p, t)]['width'], t0=by[('wBt0_typical', p, t)]['width'],
                                   stock_ko=by[('stock_ko', p, t)]['width'], kernel=by[('wBt0_typical', p, t)]['kernel'])
                  for p in projs for t in tokens}
        assert all(w['wB'] == w['t0'] for w in widths.values()), 'the t0 set ran another width than 1b'
        md.append(f"### {NAMES[model]}\n\nChecks: {len(rec['checks']['bitwise'])} bitwise, all equal: {ok}; counts ok: "
                  f"{rec['checks']['counts_ok']}; other processes: {len(rec['checks']['other_processes'])}.\n")
        rows = [dict(model=model, tokens=t) for t in tokens]
        for title, ratios in GROUPS:
            md.append(f'**{title}**\n')
            md.append('| T | ' + ' | '.join(f'{u} [rounds]' for u, _, _ in ratios) + ' |')
            md.append('|---|' + '---|' * len(ratios))
            for row, t in zip(rows, tokens):
                cells = []
                for u, a, b in ratios:
                    per = [pct(x, y) for x, y in zip(rr[(a, t)], rr[(b, t)])]
                    row[u] = dict(pct=pct(s[(a, t)], s[(b, t)]), range=[min(per), max(per)])
                    cells.append(f"{row[u]['pct']:+.2f} % [{min(per):+.2f}, {max(per):+.2f}]")
                md.append(f'| {t} | ' + ' | '.join(cells) + ' |')
            md.append('')
        md.append('Widths (8x64 sets / stock_ko) per projection and T:\n')
        md.append('| projection | ' + ' | '.join(str(t) for t in tokens) + ' |')
        md.append('|---|' + '---|' * len(tokens))
        for p in projs:
            md.append(f'| {p} | ' + ' | '.join(f"{widths[f'{p}@{t}']['t0']} / {widths[f'{p}@{t}']['stock_ko']}"
                                               for t in tokens) + ' |')
        md.append('')
        allrows += rows
        data[model] = dict(rows=rows, sums={f'{c}@{t}': v for (c, t), v in s.items()},
                           round_sums={f'{c}@{t}': v for (c, t), v in rr.items()}, bitwise_ok=ok, widths=widths,
                           sampler=rec.get('sampler'))
    rules = {}
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
                           above=sum(r[u]['range'][0] > 0 for r in allrows),
                           below=sum(r[u]['range'][1] < 0 for r in allrows),
                           bands={b: (statistics.median(x) if x else None) for (b, _), x in zip(BANDS, bands)})
            md.append(f"| {u} | {summ[u]['median']:+.2f} % | {min(v):+.2f} % | {max(v):+.2f} % | "
                      f"{summ[u]['above']} of {len(v)} | {summ[u]['below']} of {len(v)} | "
                      + ' | '.join(f'{statistics.median(b):+.2f} %' if b else '—' for b in bands) + ' |')
        md.append('')
        above = {u: [f"{r['model']} T={r['tokens']} ({r[u]['pct']:+.2f} % [{r[u]['range'][0]:+.2f}, {r[u]['range'][1]:+.2f}])"
                     for r in allrows if r[u]['range'][0] > 0] for u in T0_UNITS}
        beyond = {u: [r for r in allrows if r[u]['range'][0] > TOL] for u in T0_UNITS}
        rules['t0_adoption'] = dict(rule=f'every t0 unit has a negative median, and no cell is above zero in every round by more '
                                         f'than {TOL} %', medians={u: summ[u]['median'] for u in T0_UNITS},
                                    cells_above=above, cells_above_tolerance={u: len(v) for u, v in beyond.items()},
                                    met=all(summ[u]['median'] < 0 for u in T0_UNITS) and not any(beyond.values()),
                                    strict_form_met=not any(above.values()))
        p = summ[P1]
        rules['P1'] = dict(rule="#2's dispatch on the t0 builds, typical tags: median < 0 and more cells below zero in "
                                'every round than above', median=p['median'], below=p['below'], above=p['above'],
                           cells=p['cells'], met=p['median'] < 0 and p['below'] > p['above'])
        md.append('### The registered rules\n')
        md.append(f"- **t0 adoption** ({rules['t0_adoption']['rule']}): "
                  f"{'met' if rules['t0_adoption']['met'] else 'NOT met'}. Medians: "
                  + ', '.join(f'{u} {summ[u]["median"]:+.2f} %' for u in T0_UNITS) + '. Amendment 6\'s strict form (no '
                  f"cell above zero in every round): {'met' if rules['t0_adoption']['strict_form_met'] else 'not met'}.")
        for u, cells in above.items():
            if cells:
                md.append(f'  - {u}, above zero in every round: ' + '; '.join(cells))
        md.append(f"- **P1** ({rules['P1']['rule']}): median {p['median']:+.2f} %, {p['below']} of {p['cells']} cells below "
                  f"and {p['above']} above in every round: {'met' if rules['P1']['met'] else 'NOT met'}.")
        md.append('')
        data['summary'] = summ
        data['rules'] = rules
    c2 = args.src / 'c2_w8p2.json'
    if c2.exists():
        rec = json.loads(c2.read_text())
        md.append('## C2w‴: 4096³ (µs; medians of 3 rotated rounds)\n')
        md.append(f"stock_ko runs {rec['stock_ko']} (width, scheduler setting).\n")
        md.append('| configuration | ' + ' | '.join(rec['modes']) + ' |')
        md.append('|---|' + '---:|' * len(rec['modes']))
        for nm in next(iter(rec['modes'].values())):
            md.append(f'| {nm} | ' + ' | '.join(f"{rec['modes'][m][nm]['us']:.1f}" for m in rec['modes']) + ' |')
        md.append('')
        md.append('| ratio | ' + ' | '.join(rec['modes']) + ' |')
        md.append('|---|' + '---:|' * len(rec['modes']))
        for key in next(iter(rec['summary'].values())):
            md.append(f'| {key} | ' + ' | '.join(f"{rec['summary'][m][key]:+.2f} %" for m in rec['summary']) + ' |')
        md.append('')
        md.append('Checks: ' + ', '.join(f'{k} {v}' for k, v in rec['checks'].items()) + '\n')
        data['c2'] = dict(summary=rec.get('summary'), checks=rec['checks'], stock_ko=rec['stock_ko'])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'w8p2.json').write_text(json.dumps(data, indent=1) + '\n')
    (args.out_dir / 'w8p2_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
