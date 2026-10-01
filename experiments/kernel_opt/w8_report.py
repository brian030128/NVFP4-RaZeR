#!/usr/bin/env python3
"""Kernel-opt amendment 10 (the 8x64 baseline) report: M1 per-forward GEMM sums and the C2w 4096^3 breakdown.

    python experiments/kernel_opt/w8_report.py [--src /home/dev/n16k64_campaign/kernel_opt/w8] [--out-dir results/kernel_opt/w8]

M1: per model and T, the per-forward GEMM sum (sum over projections of modules x median time) of every configuration of
bench_w8_isolated.py; a ratio's spread is its range over the 3 rounds, from each round's sums. The 8x64 path (mixed_wB,
default and #2's dispatch, typical and worst tags) against stock_ko (the deployment's stock, weights on A), the paper
stock and stock_wB (the same placement); stock_wB and stock_ko against the paper stock. Over all (model, T) cells and by
T band: median, min, max, and the cells above or below zero in every round. C2w: every mode's times and its summary.
"""
import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
NAMES = dict(llama8b='Llama-3.1-8B', mistral7b='Mistral-7B-v0.3', phi4='Phi-4', qwen27b='Qwen3.8-27B')
# (label, a, b): a vs b
RATIOS = (('8x64 vs stock_ko, typical', 'wB_typical', 'stock_ko'),
          ('8x64 vs stock_ko, worst', 'wB_worst', 'stock_ko'),
          ('8x64 #2 vs stock_ko, typical', 'wBfreq_typical', 'stock_ko'),
          ('8x64 vs paper stock, typical', 'wB_typical', 'stock_paper'),
          ('8x64 vs stock_wB, typical', 'wB_typical', 'stock_wB'),
          ("#2's vs default, typical", 'wBfreq_typical', 'wB_typical'),
          ("#2's vs default, worst", 'wBfreq_worst', 'wB_worst'),
          ('stock_wB vs stock_ko', 'stock_wB', 'stock_ko'),
          ('stock_ko vs paper stock', 'stock_ko', 'stock_paper'))
BANDS = (('T ≤ 16', lambda t: t <= 16), ('32–128', lambda t: 32 <= t <= 128), ('256–1024', lambda t: 256 <= t <= 1024),
         ('≥ 2048', lambda t: t >= 2048))


def pct(a, b):
    return 100 * (a / b - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=Path('/home/dev/n16k64_campaign/kernel_opt/w8'))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'kernel_opt' / 'w8')
    args = ap.parse_args()
    md = ['## M1: per-forward GEMM time of the 8x64 path against stock, deviation-2\n',
          'Per model and T: the sum over projections of modules × median GEMM time. Brackets: the range over the rounds '
          'of the same ratio, from each round\'s sums. "8x64" is the `mixed_wB` set (optimization 1b); "#2" the same set '
          'built with #2\'s dispatch.\n']
    data, allrows = {}, []
    for model in MODELS:
        path = args.src / 'gemm' / f'{model}.json'
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        if rec.get('status') != 'complete':
            md.append(f'{model}: {rec.get("status")}\n')
            continue
        projs = rec['projections']
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
        widths = {f'{p}@{t}': dict(wB=by[('wB_typical', p, t)]['width'], stock_ko=by[('stock_ko', p, t)]['width'],
                                   stock_paper=by[('stock_paper', p, t)]['width'], kernel=by[('wB_typical', p, t)]['kernel'])
                  for p in projs for t in tokens}
        md.append(f"### {NAMES[model]}\n\nChecks: {len(rec['checks']['bitwise'])} bitwise, all equal: {ok}; counts ok: "
                  f"{rec['checks']['counts_ok']}; other processes: {len(rec['checks']['other_processes'])}.\n")
        md.append('| T | ' + ' | '.join(f'{u} [rounds]' for u, _, _ in RATIOS) + ' |')
        md.append('|---|' + '---|' * len(RATIOS))
        rows = []
        for t in tokens:
            row, cells = dict(model=model, tokens=t), []
            for u, a, b in RATIOS:
                per = [pct(x, y) for x, y in zip(rr[(a, t)], rr[(b, t)])]
                row[u] = dict(pct=pct(s[(a, t)], s[(b, t)]), range=[min(per), max(per)])
                cells.append(f"{row[u]['pct']:+.2f} % [{min(per):+.2f}, {max(per):+.2f}]")
            rows.append(row)
            md.append(f'| {t} | ' + ' | '.join(cells) + ' |')
        md.append('')
        md.append('Widths (8x64 set / stock_ko / paper stock) and the 8x64 build per projection and T:\n')
        md.append('| projection | ' + ' | '.join(str(t) for t in tokens) + ' |')
        md.append('|---|' + '---|' * len(tokens))
        for p in projs:
            md.append(f'| {p} | ' + ' | '.join(f"{widths[f'{p}@{t}']['wB']} / {widths[f'{p}@{t}']['stock_ko']} / "
                                               f"{widths[f'{p}@{t}']['stock_paper']}" for t in tokens) + ' |')
        md.append('')
        allrows += rows
        data[model] = dict(rows=rows, sums={f'{c}@{t}': v for (c, t), v in s.items()},
                           round_sums={f'{c}@{t}': v for (c, t), v in rr.items()}, bitwise_ok=ok, widths=widths,
                           sampler=rec.get('sampler'))
    if allrows:
        md.append('### All models: over the (model, T) cells, and by T band (median)\n')
        md.append('| ratio | median | min | max | above 0 in every round | below 0 in every round | '
                  + ' | '.join(b for b, _ in BANDS) + ' |')
        md.append('|---|---:|---:|---:|---:|---:|' + '---:|' * len(BANDS))
        for u, _, _ in RATIOS:
            v = [r[u]['pct'] for r in allrows]
            bands = [[r[u]['pct'] for r in allrows if f(r['tokens'])] for _, f in BANDS]
            md.append(f'| {u} | {statistics.median(v):+.2f} % | {min(v):+.2f} % | {max(v):+.2f} % | '
                      f'{sum(r[u]["range"][0] > 0 for r in allrows)} of {len(v)} | '
                      f'{sum(r[u]["range"][1] < 0 for r in allrows)} of {len(v)} | '
                      + ' | '.join(f'{statistics.median(b):+.2f} %' if b else '—' for b in bands) + ' |')
        md.append('')
    c2 = args.src / 'c2_w8.json'
    if c2.exists():
        rec = json.loads(c2.read_text())
        md.append('## C2w: 4096³ (µs; medians of 3 rotated rounds)\n')
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
    (args.out_dir / 'w8.json').write_text(json.dumps(data, indent=1) + '\n')
    (args.out_dir / 'w8_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
