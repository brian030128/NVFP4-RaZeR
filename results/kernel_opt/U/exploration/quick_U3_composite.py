#!/usr/bin/env python3
"""Exploratory: the per-width composite from quick_U3.py's records. Each (projection, T) cell takes the candidate's time
when the table's width there is 64 or wider (64, '128x64', 128) and today's adopted path's time otherwise; both were
measured in the same rounds. Per-forward sums (modules x median) and their ratio ranges over the rounds, as
quick_U3_report.py.

    python results/kernel_opt/U/exploration/quick_U3_composite.py JSON [JSON ...]
"""
import json
import statistics
import sys

UNITS = (('16x64', 'k16_freq', 'k16_uni'), ('8x64', 'k8_freq', 'k8_pipe'))


def wide(w):
    return str(w) in ('64', '128', '128x64')


def pct(a, b):
    return 100 * (a / b - 1)


for path in sys.argv[1:]:
    rec = json.loads(open(path).read())
    by = {(r['config'], r['proj'], r['tokens']): r for r in rec['rows']}
    projs = {p: v for p, v in rec['projections'].items() if any(r['proj'] == p for r in rec['rows'])}
    tokens = sorted({r['tokens'] for r in rec['rows']})
    nr = len(rec['rows'][0]['rounds'])
    for unit, base, cand in UNITS:
        for tag in ('typical', 'worst'):
            b, c = f'{base}_{tag}', f'{cand}_{tag}'
            if (b, next(iter(projs)), tokens[0]) not in by or (c, next(iter(projs)), tokens[0]) not in by:
                continue
            print(f"{rec['model']} {unit} {tag}: per-width composite ({cand} at widths >= 64, {base} below)")
            print('| T | widths >= 64 (modules share) | composite vs adopted | composite vs stock_ko | adopted vs stock_ko |')
            print('|---:|---:|---:|---:|---:|')
            meds = []
            for t in tokens:
                def tsum(cfg_of, i=None):
                    tot = 0.0
                    for p in projs:
                        r = by[(cfg_of(p), p, t)]
                        tot += projs[p]['modules'] * (r['gemm_us'] if i is None else r['rounds'][i]['gemm_us'])
                    return tot
                comp = lambda p: c if wide(by[(b, p, t)]['width']) else b  # noqa: E731
                for p in projs:
                    assert by[(b, p, t)]['width'] == by[(c, p, t)]['width'], (p, t)
                share = sum(projs[p]['modules'] for p in projs if wide(by[(b, p, t)]['width'])) / sum(
                    projs[p]['modules'] for p in projs)
                s_comp, s_base, s_stock = tsum(comp), tsum(lambda p: b), tsum(lambda p: 'stock_ko')
                per = [pct(tsum(comp, i), tsum(lambda p: b, i)) for i in range(nr)]
                v = pct(s_comp, s_base)
                meds.append(v)
                mark = '*' if max(per) < 0 else ('!' if min(per) > 0 else '')
                print(f'| {t} | {100 * share:.0f} % | {v:+.2f}{mark} [{min(per):+.1f}, {max(per):+.1f}] | '
                      f'{pct(s_comp, s_stock):+.2f} | {pct(s_base, s_stock):+.2f} |')
            print(f'| median | | {statistics.median(meds):+.2f} | | |\n')
