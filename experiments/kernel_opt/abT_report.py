#!/usr/bin/env python3
"""Kernel-opt amendment 6 (t0) report: M1 per-forward GEMM sums, before (tagged) vs after (site-0 tags dropped).

    python experiments/kernel_opt/abT_report.py [--src /home/dev/n16k64_campaign/kernel_opt/T] [--out-dir results/kernel_opt/t0]

Per model and T, the per-forward GEMM sum (sum over projections of modules x median time) of every configuration of
bench_abT_isolated.py; the spread is the range over the 3 rounds of the same sum from each round's medians. Reported:
after vs before for the no-dispatch ceiling, 16x64 (default and freq dispatch, typical and worst) and 256x64 (typical
and worst); each unit's gap vs stock_wA before -> after; the C2‴ 4096^3 breakdown if present.
"""
import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
NAMES = dict(llama8b='Llama-3.1-8B', mistral7b='Mistral-7B-v0.3', phi4='Phi-4', qwen27b='Qwen3.8-27B')
UNITS = (('nodisp', 'nodisp', 'nodisp_t0'), ('16x64 typical', 'mixed_16x64_typical', 't0_16x64_typical'),
         ('16x64 worst', 'mixed_16x64_worst', 't0_16x64_worst'),
         ('16x64 freq typical', 'freq_16x64_typical', 'freqt0_16x64_typical'),
         ('16x64 freq worst', 'freq_16x64_worst', 'freqt0_16x64_worst'),
         ('256x64 typical', 'm256_typical', 'm256t0_typical'), ('256x64 worst', 'm256_worst', 'm256t0_worst'))
GAPS = (('nodisp', 'nodisp', 'nodisp_t0'), ('16x64', 'mixed_16x64_typical', 't0_16x64_typical'),
        ('16x64 freq', 'freq_16x64_typical', 'freqt0_16x64_typical'), ('256x64', 'm256_typical', 'm256t0_typical'))


def pct(a, b):
    return 100 * (a / b - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=Path('/home/dev/n16k64_campaign/kernel_opt/T'))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'kernel_opt' / 't0')
    args = ap.parse_args()
    md, data, allrows = ["## M1: per-forward GEMM time, site-0 tags dropped (after) vs tagged (before), deviation-2\n"], {}, []
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
        weq = all(by[(a, p, t)]['width'] == by[(b, p, t)]['width'] for _, b, a in UNITS[1:] for p in projs for t in tokens)
        md.append(f"### {NAMES[model]}\n\nChecks: {len(rec['checks']['bitwise'])} bitwise ok={ok}, counts ok="
                  f"{rec['checks']['counts_ok']}, other processes {len(rec['checks']['other_processes'])}; same widths "
                  f"before and after: {weq}.\n")
        md.append('| T | ' + ' | '.join(f'{u}: after vs before [rounds]' for u, _, _ in UNITS)
                  + ' | ' + ' | '.join(f'{g} vs stock_wA: before → after' for g, _, _ in GAPS) + ' |')
        md.append('|---|' + '---|' * (len(UNITS) + len(GAPS)))
        rows = []
        for t in tokens:
            row, cells = dict(model=model, tokens=t), []
            for u, b, a in UNITS:
                row[f'{u} after_vs_before_pct'] = pct(s[(a, t)], s[(b, t)])
                per = [pct(x, y) for x, y in zip(rr[(a, t)], rr[(b, t)])]
                row[f'{u} after_vs_before_range'] = [min(per), max(per)]
                cells.append(f"{row[f'{u} after_vs_before_pct']:+.2f} % [{min(per):+.2f}, {max(per):+.2f}]")
            for g, b, a in GAPS:
                row[f'{g} gap_before'] = pct(s[(b, t)], s[('stock_wA', t)])
                row[f'{g} gap_after'] = pct(s[(a, t)], s[('stock_wA', t)])
                cells.append(f"{row[f'{g} gap_before']:+.1f} → {row[f'{g} gap_after']:+.1f} %")
            rows.append(row)
            md.append(f'| {t} | ' + ' | '.join(cells) + ' |')
        md.append('')
        allrows += rows
        data[model] = dict(rows=rows, sums={f'{c}@{t}': v for (c, t), v in s.items()}, bitwise_ok=ok, widths_equal=weq,
                           sampler=rec.get('sampler'))
    if allrows:
        md.append('### All models: after vs before over the (model, T) cells\n')
        md.append('| unit | median | min | max | cells slower beyond the round range |')
        md.append('|---|---:|---:|---:|---:|')
        for u, _, _ in UNITS:
            v = [r[f'{u} after_vs_before_pct'] for r in allrows]
            slow = sum(r[f'{u} after_vs_before_range'][0] > 0 for r in allrows)
            md.append(f'| {u} | {statistics.median(v):+.2f} % | {min(v):+.2f} % | {max(v):+.2f} % | {slow} of {len(v)} |')
        md.append('')
    c2 = args.src / 'c2_t0.json'
    if c2.exists():
        rec = json.loads(c2.read_text())
        md.append('## C2‴: 4096³ (µs; medians of 3 rotated rounds)\n')
        md.append('| configuration | ' + ' | '.join(rec['modes']) + ' |')
        md.append('|---|' + '---:|' * len(rec['modes']))
        for nm in next(iter(rec['modes'].values())):
            md.append(f'| {nm} | ' + ' | '.join(f"{rec['modes'][m][nm]['us']:.1f}" for m in rec['modes']) + ' |')
        md.append('')
        md.append('Checks: ' + ', '.join(f'{k} {v}' for k, v in rec['checks'].items()) + '\n')
        data['c2'] = dict(summary=rec.get('summary'), checks=rec['checks'])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'abT.json').write_text(json.dumps(data, indent=1) + '\n')
    (args.out_dir / 'abT_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
