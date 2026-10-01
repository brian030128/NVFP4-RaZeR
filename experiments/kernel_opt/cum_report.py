#!/usr/bin/env python3
"""Kernel-opt amendment 9 (the cumulative registered run) report, M1: per-forward GEMM sums of the combined 16x64 build
and the adopted stock set against today's paper builds.

    python experiments/kernel_opt/cum_report.py [--src /home/dev/n16k64_campaign/kernel_opt/cum] [--out-dir results/kernel_opt/cum]

Per model and T, the per-forward GEMM sum (sum over projections of modules x median time) of every configuration of
bench_cum_isolated.py; a ratio's spread is its range over the 3 rounds, from each round's sums. Reported:
- the combined build vs the paper build (typical and worst tags), the adopted stock vs the paper stock, the adopted
  default dispatch vs the paper build, and #2's dispatch vs the default on the adopted path;
- each 16x64 path's gap to its own stock (paper: the paper stock; adopted: stock_ko) and to the paper stock;
- over all (model, T) cells: median, min, max, and the cells slower (or faster) in every round.
"""
import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
NAMES = dict(llama8b='Llama-3.1-8B', mistral7b='Mistral-7B-v0.3', phi4='Phi-4', qwen27b='Qwen3.8-27B')
# (label, after, before)
RATIOS = (('combined vs paper, typical', 'comb_16x64_typical', 'paper_16x64_typical'),
          ('combined vs paper, worst', 'comb_16x64_worst', 'paper_16x64_worst'),
          ('stock_ko vs paper stock', 'stock_ko', 'stock_paper'),
          ('adopted default vs paper, typical', 'ko_16x64_typical', 'paper_16x64_typical'),
          ("#2's vs default dispatch (adopted), typical", 'comb_16x64_typical', 'ko_16x64_typical'),
          ("#2's vs default dispatch (adopted), worst", 'comb_16x64_worst', 'ko_16x64_worst'))
# (label, 16x64 config, its own stock)
GAPS = (('paper', 'paper_16x64_typical', 'stock_paper'), ('combined', 'comb_16x64_typical', 'stock_ko'),
        ('combined vs paper stock', 'comb_16x64_typical', 'stock_paper'))


def pct(a, b):
    return 100 * (a / b - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=Path('/home/dev/n16k64_campaign/kernel_opt/cum'))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'kernel_opt' / 'cum')
    args = ap.parse_args()
    md = ['## M1: per-forward GEMM time, the combined build and stock_ko vs the paper builds, deviation-2\n',
          'Per model and T: the sum over projections of modules × median GEMM time. Brackets: the range over the 3 '
          'rounds of the same ratio, from each round\'s sums. Gaps: typical tags against the stock named.\n']
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
        wdiff = sorted({(p, t) for p in projs for t in tokens
                        if by[('comb_16x64_typical', p, t)]['width'] != by[('paper_16x64_typical', p, t)]['width']},
                       key=lambda x: (x[1], x[0]))
        wdiff_s = sorted({(p, t) for p in projs for t in tokens
                          if by[('stock_ko', p, t)]['width'] != by[('stock_paper', p, t)]['width']}, key=lambda x: (x[1], x[0]))
        md.append(f"### {NAMES[model]}\n\nChecks: {len(rec['checks']['bitwise'])} bitwise, all equal: {ok}; counts ok: "
                  f"{rec['checks']['counts_ok']}; other processes: {len(rec['checks']['other_processes'])}. Width differs "
                  f"from the paper table in {len(wdiff)} of {len(projs) * len(tokens)} (projection, T) cells for 16x64 "
                  f"and {len(wdiff_s)} for stock.\n")
        md.append('| T | ' + ' | '.join(f'{u} [rounds]' for u, _, _ in RATIOS) + ' | '
                  + ' | '.join(f'gap {g}' for g, _, _ in GAPS) + ' |')
        md.append('|---|' + '---|' * (len(RATIOS) + len(GAPS)))
        rows = []
        for t in tokens:
            row, cells = dict(model=model, tokens=t), []
            for u, a, b in RATIOS:
                per = [pct(x, y) for x, y in zip(rr[(a, t)], rr[(b, t)])]
                row[u] = dict(pct=pct(s[(a, t)], s[(b, t)]), range=[min(per), max(per)])
                cells.append(f"{row[u]['pct']:+.2f} % [{min(per):+.2f}, {max(per):+.2f}]")
            for g, a, b in GAPS:
                row[f'gap {g}'] = pct(s[(a, t)], s[(b, t)])
                cells.append(f"{row[f'gap {g}']:+.1f} %")
            rows.append(row)
            md.append(f'| {t} | ' + ' | '.join(cells) + ' |')
        md.append('')
        allrows += rows
        data[model] = dict(rows=rows, sums={f'{c}@{t}': v for (c, t), v in s.items()},
                           round_sums={f'{c}@{t}': v for (c, t), v in rr.items()}, bitwise_ok=ok,
                           width_cells_differ_16x64=[list(x) for x in wdiff], width_cells_differ_stock=[list(x) for x in wdiff_s],
                           sampler=rec.get('sampler'))
    if allrows:
        md.append('### All models: over the (model, T) cells\n')
        md.append('| ratio | median | min | max | slower in every round | faster in every round |')
        md.append('|---|---:|---:|---:|---:|---:|')
        for u, _, _ in RATIOS:
            v = [r[u]['pct'] for r in allrows]
            slow = sum(r[u]['range'][0] > 0 for r in allrows)
            fast = sum(r[u]['range'][1] < 0 for r in allrows)
            md.append(f'| {u} | {statistics.median(v):+.2f} % | {min(v):+.2f} % | {max(v):+.2f} % | {slow} of {len(v)} | '
                      f'{fast} of {len(v)} |')
        md.append('')
        md.append('| gap (typical tags) | median | min | max |')
        md.append('|---|---:|---:|---:|')
        for g, _, _ in GAPS:
            v = [r[f'gap {g}'] for r in allrows]
            md.append(f'| {g} | {statistics.median(v):+.2f} % | {min(v):+.2f} % | {max(v):+.2f} % |')
        md.append('')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'cum_gemm.json').write_text(json.dumps(data, indent=1) + '\n')
    (args.out_dir / 'cum_gemm_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
