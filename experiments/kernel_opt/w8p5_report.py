#!/usr/bin/env python3
"""Kernel-opt amendment 13 (the 8x64 plan's P5) report: the adopted 8x64 path against its no-dispatch ceiling at every
width (the same-placement reference), and both against stock.

    python experiments/kernel_opt/w8p5_report.py [--src /home/dev/n16k64_campaign/kernel_opt/w8p5] \
        [--out-dir results/kernel_opt/w8/p5]

M1 is summarized as in amendments 10–12. Per model and T, the per-forward GEMM sum is the sum over projections of
modules × median time. A ratio's spread is its range over the rounds, computed from each round's sums.

The report checks that the ceiling ran the 8x64 path's width and scheduler setting in every (projection, T) cell. It
gives the ratios over all cells, by T band and per T. '8x64 vs ceiling' is what dispatch work could still recover at
each width. 'ceiling vs stock_ko' is what the same-placement 1x8 / 1x4 tiles cost against the target.
"""
import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
NAMES = dict(llama8b='Llama-3.1-8B', mistral7b='Mistral-7B-v0.3', phi4='Phi-4', qwen27b='Qwen3.8-27B')
RATIOS = (('8x64 vs ceiling, typical', 'wB_typical', 'ceiling'),
          ('8x64 vs ceiling, worst', 'wB_worst', 'ceiling'),
          ('ceiling vs stock_ko', 'ceiling', 'stock_ko'),
          ('ceiling vs stock_wB tuned alike', 'ceiling', 'stock_wB_ko'),
          ('8x64 vs stock_ko, typical', 'wB_typical', 'stock_ko'),
          ('8x64 vs stock_ko, worst', 'wB_worst', 'stock_ko'),
          ('8x64 vs stock_wB tuned alike, typical', 'wB_typical', 'stock_wB_ko'),
          ('stock_wB tuned alike vs stock_ko', 'stock_wB_ko', 'stock_ko'))
BANDS = (('T ≤ 16', lambda t: t <= 16), ('32–128', lambda t: 32 <= t <= 128), ('256–1024', lambda t: 256 <= t <= 1024),
         ('≥ 2048', lambda t: t >= 2048))


def pct(a, b):
    return 100 * (a / b - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=Path('/home/dev/n16k64_campaign/kernel_opt/w8p5'))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'kernel_opt' / 'w8' / 'p5')
    args = ap.parse_args()
    md = ['## M1: per-forward GEMM time, deviation-2\n',
          'Per model and T: the sum over projections of modules × median GEMM time. Brackets: the range over the rounds '
          'of the same ratio, from each round\'s sums. "8x64" is the adopted path (`mixed_wB_ko`: t0, #2\'s dispatch, '
          'the adopted widths); "ceiling" its tiles with the dispatch compiled out (`nodisp_wB_ko`, FourOverSix weights) on '
          'the same widths and scheduler settings.\n']
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
        for p in projs:
            for t in tokens:
                a, b = by[('wB_typical', p, t)], by[('ceiling', p, t)]
                assert (a['width'], a['schedule']) == (b['width'], b['schedule']), (model, p, t, a['width'], b['width'])
        s, rr = {}, {}
        for t in tokens:
            for c in cfgs:
                s[(c, t)] = sum(projs[p]['modules'] * by[(c, p, t)]['gemm_us'] for p in projs)
                rr[(c, t)] = [sum(projs[p]['modules'] * by[(c, p, t)]['rounds'][i]['gemm_us'] for p in projs) for i in range(nr)]
        ok = all(c['equal'] for c in rec['checks']['bitwise'])
        md.append(f"### {NAMES[model]}\n\nChecks: {len(rec['checks']['bitwise'])} bitwise (each configuration's isolated "
                  f"path against NativeLinear), all equal: {ok}; counts ok: {rec['checks']['counts_ok']}; other "
                  f"processes: {len(rec['checks']['other_processes'])}; the ceiling ran the 8x64 path's width and "
                  f"scheduler setting in every cell.\n")
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
                           above=sum(r[u]['range'][0] > 0 for r in allrows),
                           below=sum(r[u]['range'][1] < 0 for r in allrows),
                           bands={b: (statistics.median(x) if x else None) for (b, _), x in zip(BANDS, bands)})
            md.append(f"| {u} | {summ[u]['median']:+.2f} % | {min(v):+.2f} % | {max(v):+.2f} % | "
                      f"{summ[u]['above']} of {len(v)} | {summ[u]['below']} of {len(v)} | "
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
        data.update(summary=summ, per_t=per_t)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'w8p5.json').write_text(json.dumps(data, indent=1) + '\n')
    (args.out_dir / 'w8p5_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
