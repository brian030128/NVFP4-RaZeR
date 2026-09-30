#!/usr/bin/env python3
"""Kernel-opt amendment 3 (A') report: M1 per-forward GEMM sums and the C2 4096^3 breakdown.

    python experiments/kernel_opt/abA1_report.py [--src /home/dev/n16k64_campaign/kernel_opt/A1] [--out-dir results/kernel_opt/A1]

M1: per model and T, the per-forward GEMM sum (sum over projections of modules x median time) of every configuration of
bench_abA1_isolated.py; the spread is the range over the 3 rounds of the same sum from each round's medians. Reported:
g32 (after) vs today's path (before) per tag variant, and 256x64 vs stock_wA before -> after; that every call ran at the
same width before and after. C2: c2_g32.py's medians and contrasts per mode.
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
NAMES = dict(llama8b='Llama-3.1-8B', mistral7b='Mistral-7B-v0.3', phi4='Phi-4', qwen27b='Qwen3.8-27B')


def pct(a, b):
    return 100 * (a / b - 1)


def sums(rec):
    projs = rec['projections']
    by = {(r['config'], r['proj'], r['tokens']): r for r in rec['rows']}
    tokens = sorted({r['tokens'] for r in rec['rows']})
    cfgs = sorted({r['config'] for r in rec['rows']})
    nr = len(rec['rows'][0]['rounds'])
    s, rr = {}, {}
    for t in tokens:
        for c in cfgs:
            s[(c, t)] = sum(projs[p]['modules'] * by[(c, p, t)]['gemm_us'] for p in projs)
            rr[(c, t)] = [sum(projs[p]['modules'] * by[(c, p, t)]['rounds'][i]['gemm_us'] for p in projs) for i in range(nr)]
    widths_equal = all(by[(f'g32_256x64_{v}', p, t)]['width'] == by[(f'mixed_256x64_{v}', p, t)]['width']
                       for v in ('typical', 'worst') for p in projs for t in tokens)
    return s, rr, tokens, widths_equal


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=Path('/home/dev/n16k64_campaign/kernel_opt/A1'))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'kernel_opt' / 'A1')
    args = ap.parse_args()
    md, data = [], dict(gemm={}, c2=None)
    md.append("## M1: per-forward GEMM time of the 256x64 maps, A' (4-arm g32) vs today's path, deviation-2 method\n")
    for model in MODELS:
        path = args.src / 'gemm' / f'{model}.json'
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        if rec.get('status') != 'complete':
            md.append(f'{model}: {rec.get("status")}\n')
            continue
        s, rr, tokens, weq = sums(rec)
        ok = all(c['equal'] for c in rec['checks']['bitwise'])
        md.append(f"### {NAMES[model]}\n\nChecks: {len(rec['checks']['bitwise'])} bitwise ok={ok}, counts ok="
                  f"{rec['checks']['counts_ok']}, other processes {len(rec['checks']['other_processes'])}; every call at "
                  f"the same width before and after: {weq}.\n")
        md.append('| T | after vs before, typical [rounds] | after vs before, worst | 256x64 vs stock_wA, typical: '
                  'before → after | worst: before → after |')
        md.append('|---|---|---|---|---|')
        rows = []
        for t in tokens:
            row = dict(model=model, tokens=t)
            for v in ('typical', 'worst'):
                bb, aa = (f'mixed_256x64_{v}', t), (f'g32_256x64_{v}', t)
                row[f'{v}_after_vs_before_pct'] = pct(s[aa], s[bb])
                per = [pct(x, y) for x, y in zip(rr[aa], rr[bb])]
                row[f'{v}_after_vs_before_range'] = [min(per), max(per)]
                row[f'{v}_before_vs_stock_wA_pct'] = pct(s[bb], s[('stock_wA', t)])
                row[f'{v}_after_vs_stock_wA_pct'] = pct(s[aa], s[('stock_wA', t)])
            rows.append(row)
            lo, hi = row['typical_after_vs_before_range']
            md.append(f"| {t} | {row['typical_after_vs_before_pct']:+.2f} % [{lo:+.2f}, {hi:+.2f}] | "
                      f"{row['worst_after_vs_before_pct']:+.2f} % | {row['typical_before_vs_stock_wA_pct']:+.1f} → "
                      f"{row['typical_after_vs_stock_wA_pct']:+.1f} % | {row['worst_before_vs_stock_wA_pct']:+.1f} → "
                      f"{row['worst_after_vs_stock_wA_pct']:+.1f} % |")
        md.append('')
        data['gemm'][model] = dict(rows=rows, sums={f'{c}@{t}': v for (c, t), v in s.items()}, widths_equal=weq,
                                   bitwise_ok=ok, sampler=rec.get('sampler'))
    c2p = args.src / 'c2_g32.json'
    if c2p.exists():
        c2 = json.loads(c2p.read_text())
        data['c2'] = c2
        md.append("## C2: 4096³ breakdown (µs, medians of 3 rotated rounds)\n")
        md.append(f"Checks: {c2['checks']}\n")
        names = list(next(iter(c2['modes'].values())).keys())
        md.append('| configuration | ' + ' | '.join(c2['modes']) + ' |')
        md.append('|---|' + '---:|' * len(c2['modes']))
        for nm in names:
            md.append(f'| {nm} | ' + ' | '.join(f"{c2['modes'][m][nm]['us']:.1f}" for m in c2['modes']) + ' |')
        md.append('')
        for m, sm in c2.get('summary', {}).items():
            md.append(f'- **{m}:** ' + '; '.join(f'{k} {v:+.2f} %' for k, v in sm.items()))
        md.append('')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'abA1.json').write_text(json.dumps(data, indent=1) + '\n')
    (args.out_dir / 'abA1_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
