#!/usr/bin/env python3
"""Kernel-opt amendment 5 (#4) report: M1 per-forward GEMM sums, before vs after for the mixed 16x64 and the stock family.

    python experiments/kernel_opt/ab4_report.py [--src /home/dev/n16k64_campaign/kernel_opt/4] [--out-dir results/kernel_opt/4]

Per model and T, the per-forward GEMM sum (sum over projections of modules x median time) of every configuration of
bench_ab4_isolated.py; the spread is the range over the 3 rounds of the same sum from each round's medians. Reported:
after vs before for stock_wA and 16x64 (typical, worst); the 16x64 gap vs stock_wA before (both before) -> after (both
after); the 256x64 path (A', unchanged by #4) vs stock_wA before and after; #2's freq path vs stock (before) for
reference; the scheduler settings the calls used.
"""
import argparse
import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
NAMES = dict(llama8b='Llama-3.1-8B', mistral7b='Mistral-7B-v0.3', phi4='Phi-4', qwen27b='Qwen3.8-27B')
UNITS = (('stock_wA', 'stock_wA', 'stock_e'), ('16x64 typical', 'mixed_16x64_typical', 'e_16x64_typical'),
         ('16x64 worst', 'mixed_16x64_worst', 'e_16x64_worst'))


def pct(a, b):
    return 100 * (a / b - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=Path('/home/dev/n16k64_campaign/kernel_opt/4'))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'kernel_opt' / '4')
    args = ap.parse_args()
    md, data = ["## M1: per-forward GEMM time, #4 (after) vs today's paths (before), deviation-2 method\n"], {}
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
        weq = all(by[(a, p, t)]['width'] == by[(b, p, t)]['width'] for _, b, a in UNITS for p in projs for t in tokens)
        md.append(f"### {NAMES[model]}\n\nChecks: {len(rec['checks']['bitwise'])} bitwise ok={ok}, counts ok="
                  f"{rec['checks']['counts_ok']}, other processes {len(rec['checks']['other_processes'])}; same widths "
                  f"before and after: {weq}.\n")
        md.append('| T | ' + ' | '.join(f'{u}: after vs before [rounds]' for u, _, _ in UNITS)
                  + ' | 16x64 vs stock_wA: before → after | 256x64 (unchanged) vs stock_wA: before → after | freq (#2) vs stock_wA |')
        md.append('|---|' + '---|' * len(UNITS) + '---|---|---|')
        rows = []
        for t in tokens:
            row, cells = dict(model=model, tokens=t), []
            for u, b, a in UNITS:
                row[f'{u} after_vs_before_pct'] = pct(s[(a, t)], s[(b, t)])
                per = [pct(x, y) for x, y in zip(rr[(a, t)], rr[(b, t)])]
                row[f'{u} after_vs_before_range'] = [min(per), max(per)]
                cells.append(f"{row[f'{u} after_vs_before_pct']:+.2f} % [{min(per):+.2f}, {max(per):+.2f}]")
            row['gap_before'] = pct(s[('mixed_16x64_typical', t)], s[('stock_wA', t)])
            row['gap_after'] = pct(s[('e_16x64_typical', t)], s[('stock_e', t)])
            row['gap_freq'] = pct(s[('freq_16x64_typical', t)], s[('stock_wA', t)])
            row['gap256_before'] = pct(s[('m256_typical', t)], s[('stock_wA', t)])
            row['gap256_after'] = pct(s[('m256_typical', t)], s[('stock_e', t)])
            rows.append(row)
            md.append(f'| {t} | ' + ' | '.join(cells) + f" | {row['gap_before']:+.1f} → {row['gap_after']:+.1f} % | "
                      f"{row['gap256_before']:+.1f} → {row['gap256_after']:+.1f} % | {row['gap_freq']:+.1f} % |")
        md.append('')
        used = collections.Counter()
        for r in rec['rows']:
            if r['config'] in ('stock_e', 'e_16x64_typical'):
                used[(r['config'], tuple(r.get('schedule') or ()))] += 1
        md.append('Scheduler settings used (calls per (configuration, [raster, swizzle])): '
                  + ', '.join(f'{c} {list(sc)}: {n_}' for (c, sc), n_ in sorted(used.items())) + '\n')
        data[model] = dict(rows=rows, sums={f'{c}@{t}': v for (c, t), v in s.items()}, bitwise_ok=ok, widths_equal=weq,
                           schedules_used={f'{c} {list(sc)}': n_ for (c, sc), n_ in used.items()}, sampler=rec.get('sampler'))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'ab4.json').write_text(json.dumps(data, indent=1) + '\n')
    (args.out_dir / 'ab4_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
