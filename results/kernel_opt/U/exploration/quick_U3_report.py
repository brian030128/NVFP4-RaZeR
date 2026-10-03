#!/usr/bin/env python3
"""Exploratory: per-forward sums of quick_U3.py's M1-style records (sum over projections of modules x median GEMM time),
per T, with each ratio's range over the rounds (from each round's sums).

    python results/kernel_opt/U/exploration/quick_U3_report.py JSON [JSON ...]
"""
import json
import statistics
import sys

RATIOS = (('16x64 uni vs freq', 'k16_uni_typical', 'k16_freq_typical'),
          ('16x64 freq vs stock_ko', 'k16_freq_typical', 'stock_ko'),
          ('16x64 uni vs stock_ko', 'k16_uni_typical', 'stock_ko'),
          ('16x64 ceiling vs stock_ko', 'c16', 'stock_ko'),
          ('8x64 uni vs freq', 'k8_uni_typical', 'k8_freq_typical'),
          ('8x64 unip vs freq', 'k8_unip_typical', 'k8_freq_typical'),
          ('8x64 pipe vs freq', 'k8_pipe_typical', 'k8_freq_typical'),
          ('8x64 hybrid vs freq', 'k8_hyb_typical', 'k8_freq_typical'),
          ('8x64 hybrid vs stock_ko', 'k8_hyb_typical', 'stock_ko'),
          ('8x64 freq vs stock_ko', 'k8_freq_typical', 'stock_ko'),
          ('8x64 ceiling vs stock_ko', 'c8', 'stock_ko'),
          ('16x64 uni vs freq, worst', 'k16_uni_worst', 'k16_freq_worst'),
          ('16x64 freq vs stock_ko, worst', 'k16_freq_worst', 'stock_ko'),
          ('16x64 uni vs stock_ko, worst', 'k16_uni_worst', 'stock_ko'),
          ('8x64 pipe vs freq, worst', 'k8_pipe_worst', 'k8_freq_worst'),
          ('8x64 freq vs stock_ko, worst', 'k8_freq_worst', 'stock_ko'),
          ('8x64 pipe vs stock_ko, worst', 'k8_pipe_worst', 'stock_ko'))


def pct(a, b):
    return 100 * (a / b - 1)


for path in sys.argv[1:]:
    rec = json.loads(open(path).read())
    projs = {p: v for p, v in rec['projections'].items() if any(r['proj'] == p for r in rec['rows'])}
    by = {(r['config'], r['proj'], r['tokens']): r for r in rec['rows']}
    tokens = sorted({r['tokens'] for r in rec['rows']})
    cfgs = sorted({r['config'] for r in rec['rows']})
    nr = len(rec['rows'][0]['rounds'])
    ok = all(c['equal'] for c in rec['checks']['bitwise'])
    print(f"{path}: {rec['model']} status {rec['status']}; bitwise checks {len(rec['checks']['bitwise'])} all equal {ok}; "
          'projections ' + ', '.join(f"{p} x{v['modules']}" for p, v in projs.items()))
    widths = {}
    for (c, p, t), r in by.items():
        widths.setdefault((c, t), {})[p] = r['width']
    s, rr = {}, {}
    for t in tokens:
        for c in cfgs:
            s[(c, t)] = sum(projs[p]['modules'] * by[(c, p, t)]['gemm_us'] for p in projs)
            rr[(c, t)] = [sum(projs[p]['modules'] * by[(c, p, t)]['rounds'][i]['gemm_us'] for p in projs) for i in range(nr)]
    ratios = [r for r in RATIOS if (r[1], tokens[0]) in s and (r[2], tokens[0]) in s]
    print('| T | ' + ' | '.join(u for u, _, _ in ratios) + ' |')
    print('|---:|' + '---:|' * len(ratios))
    cols = {u: [] for u, _, _ in ratios}
    ratios = [r for r in RATIOS if (r[1], tokens[0]) in s and (r[2], tokens[0]) in s]
    for t in tokens:
        cells = []
        for u, a, b in ratios:
            per = [pct(x, y) for x, y in zip(rr[(a, t)], rr[(b, t)])]
            v = pct(s[(a, t)], s[(b, t)])
            cols[u].append((t, v, min(per), max(per)))
            mark = '*' if max(per) < 0 else ('!' if min(per) > 0 else '')
            cells.append(f'{v:+.2f}{mark} [{min(per):+.1f}, {max(per):+.1f}]')
        print(f'| {t} | ' + ' | '.join(cells) + ' |')
    print('| median | ' + ' | '.join(f"{statistics.median(v for _, v, _, _ in cols[u]):+.2f}" for u, _, _ in ratios) + ' |')
    print('(* below 0 in every round, ! above 0 in every round)')
