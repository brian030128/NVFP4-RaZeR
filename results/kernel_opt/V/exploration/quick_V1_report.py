#!/usr/bin/env python3
"""Exploratory: quick_V1.py's records summarized. Per model and T, per-forward sums (modules x median GEMM time) and
each variant's ratio to today's path, with the range over the rounds (* below 0 in every round, ! above). Then per
width: each variant's ratio over the (projection, T) cells that run at that width (module-weighted sums).

    python results/kernel_opt/V/exploration/quick_V1_report.py JSON [JSON ...]
"""
import json
import statistics
import sys

VARIANTS = ['e', 'eF', 'eU', 'eFU', 'e_t0', 'eF_t0', 'eU_t0', 'eFU_t0']


def pct(a, b):
    return 100 * (a / b - 1)


for path in sys.argv[1:]:
    rec = json.loads(open(path).read())
    projs = {p: v for p, v in rec['projections'].items() if any(r['proj'] == p for r in rec['rows'])}
    by = {(r['config'], r['proj'], r['tokens']): r for r in rec['rows']}
    tokens = sorted({r['tokens'] for r in rec['rows']})
    nr = len(rec['rows'][0]['rounds'])
    ok = all(c['equal'] for c in rec['checks']['bitwise'])
    print(f"\n## {rec['model']}: {len(rec['checks']['bitwise'])} bitwise checks, all equal {ok}")
    for tag in ('typical', 'worst'):
        if (f'today_{tag}', next(iter(projs)), tokens[0]) not in by:
            continue

        def s(c, t, i=None):
            return sum(projs[p]['modules'] * (by[(c, p, t)]['gemm_us'] if i is None else by[(c, p, t)]['rounds'][i]['gemm_us'])
                       for p in projs)
        print(f'\n### {tag}: % vs today (per forward)\n')
        print('| T | ' + ' | '.join(VARIANTS) + ' | today vs stock_ko | ceil vs stock_ko |')
        print('|---:|' + '---:|' * (len(VARIANTS) + 2))
        meds = {v: [] for v in VARIANTS}
        for t in tokens:
            cells = []
            for v in VARIANTS:
                c = f'{v}_{tag}'
                per = [pct(s(c, t, i), s(f'today_{tag}', t, i)) for i in range(nr)]
                x = pct(s(c, t), s(f'today_{tag}', t))
                meds[v].append(x)
                cells.append(f"{x:+.2f}{'*' if max(per) < 0 else '!' if min(per) > 0 else ''}")
            cells.append(f"{pct(s(f'today_{tag}', t), s('stock_ko', t)):+.2f}")
            cells.append(f"{pct(s('ceil', t), s('stock_ko', t)):+.2f}")
            print(f'| {t} | ' + ' | '.join(cells) + ' |')
        print('| median | ' + ' | '.join(f'{statistics.median(meds[v]):+.2f}' for v in VARIANTS) + ' | | |')
        # per width
        widths = sorted({str(by[(f'today_{tag}', p, t)]['width']) for p in projs for t in tokens}, key=lambda w: int(w))
        print(f'\n### {tag}: per width, % vs today over the cells at that width (module-weighted)\n')
        print('| width | cells | ' + ' | '.join(VARIANTS) + ' |')
        print('|---:|---:|' + '---:|' * len(VARIANTS))
        for w in widths:
            cells = [(p, t) for p in projs for t in tokens if str(by[(f'today_{tag}', p, t)]['width']) == w]
            base = sum(projs[p]['modules'] * by[(f'today_{tag}', p, t)]['gemm_us'] for p, t in cells)
            vals = [pct(sum(projs[p]['modules'] * by[(f'{v}_{tag}', p, t)]['gemm_us'] for p, t in cells), base) for v in VARIANTS]
            print(f'| {w} | {len(cells)} | ' + ' | '.join(f'{x:+.2f}' for x in vals) + ' |')
