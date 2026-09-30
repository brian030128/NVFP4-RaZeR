#!/usr/bin/env python3
"""Kernel-opt amendment 2 (#2) report: M1' per-forward GEMM sums and the C2' 4096^3 breakdown.

    python experiments/kernel_opt/ab2_report.py [--src /home/dev/n16k64_campaign/kernel_opt/opt2] [--out-dir results/kernel_opt/opt2]

M1': per model and T, the per-forward GEMM sum (sum over projections of modules x median time) of every configuration of
bench_ab2_isolated.py; the spread is the range over the 3 rounds of the same sum from each round's medians. Reported:
freq vs default (after vs before) per unit and tags, and 16x64 / 256x64 / 8x64 vs stock_wA (and 8x64 vs stock_wB),
before -> after. C2': c2_freq.py's medians and contrasts per mode.
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
NAMES = dict(llama8b='Llama-3.1-8B', mistral7b='Mistral-7B-v0.3', phi4='Phi-4', qwen27b='Qwen3.8-27B')
UNITS = (('16x64', 'mixed_16x64', 'freq_16x64'), ('256x64', 'mixed_256x64', 'freq_256x64'), ('8x64', 'wB', 'wBfreq'))


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
    return s, rr, tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=Path('/home/dev/n16k64_campaign/kernel_opt/opt2'))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'kernel_opt' / 'opt2')
    args = ap.parse_args()
    md, data = [], dict(gemm={}, c2=None)
    md.append("## M1': per-forward GEMM time, freq (after) vs default (before), deviation-2 method\n")
    for model in MODELS:
        path = args.src / 'gemm' / f'{model}.json'
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        if rec.get('status') != 'complete':
            md.append(f'{model}: {rec.get("status")}\n')
            continue
        s, rr, tokens = sums(rec)
        ok = all(c['equal'] for c in rec['checks']['bitwise'])
        md.append(f"### {NAMES[model]}\n\nChecks: {len(rec['checks']['bitwise'])} bitwise ok={ok}, counts ok={rec['checks']['counts_ok']}, "
                  f"other processes {len(rec['checks']['other_processes'])}.\n")
        md.append('| T | ' + ' | '.join(f'{u} after vs before (typ.) [rounds] | {u} worst' for u, _, _ in UNITS)
                  + ' | 16x64 vs stock_wA: before → after | 256x64 vs stock_wA | 8x64 vs stock_wA | 8x64 vs stock_wB |')
        md.append('|---|' + '---|---|' * len(UNITS) + '---|---|---|---|')
        rows = []
        for t in tokens:
            row = dict(model=model, tokens=t)
            cells = []
            for u, b, a in UNITS:
                for v in ('typical', 'worst'):
                    bb, aa = (f'{b}_{v}', t), (f'{a}_{v}', t)
                    row[f'{u}_{v}_after_vs_before_pct'] = pct(s[aa], s[bb])
                    per = [pct(x, y) for x, y in zip(rr[aa], rr[bb])]
                    row[f'{u}_{v}_after_vs_before_range'] = [min(per), max(per)]
                lo, hi = row[f'{u}_typical_after_vs_before_range']
                cells.append(f"{row[f'{u}_typical_after_vs_before_pct']:+.2f} % [{lo:+.2f}, {hi:+.2f}] | {row[f'{u}_worst_after_vs_before_pct']:+.2f} %")
            for u, b, a in UNITS:
                for ref in ('stock_wA', 'stock_wB'):
                    row[f'{u}_before_vs_{ref}_pct'] = pct(s[(f'{b}_typical', t)], s[(ref, t)])
                    row[f'{u}_after_vs_{ref}_pct'] = pct(s[(f'{a}_typical', t)], s[(ref, t)])
            rows.append(row)
            md.append(f"| {t} | " + ' | '.join(cells) + ' | '
                      + ' | '.join(f"{row[f'{u}_before_vs_{ref}_pct']:+.1f} → {row[f'{u}_after_vs_{ref}_pct']:+.1f} %"
                                   for u, ref in (('16x64', 'stock_wA'), ('256x64', 'stock_wA'), ('8x64', 'stock_wA'), ('8x64', 'stock_wB')))
                      + ' |')
        md.append('')
        data['gemm'][model] = dict(rows=rows, sums={f'{c}@{t}': v for (c, t), v in s.items()}, sampler=rec.get('sampler'))
    c2p = args.src / 'c2_freq.json'
    if c2p.exists():
        c2 = json.loads(c2p.read_text())
        data['c2'] = c2
        md.append("## C2': 4096³ breakdown (µs, medians of 3 rotated rounds)\n")
        md.append(f"Checks: {c2['checks']}\n")
        names = list(next(iter(c2['modes'].values())).keys())
        md.append('| configuration | ' + ' | '.join(c2['modes']) + ' |')
        md.append('|---|' + '---:|' * len(c2['modes']))
        for nm in names:
            md.append(f'| {nm} | ' + ' | '.join(f"{c2['modes'][m][nm]['us']:.1f}" for m in c2['modes']) + ' |')
        md.append('')
        for m, sm in c2.get('summary', {}).items():
            md.append(f'- **{m}:** ' + '; '.join(f'{k} {v:+.2f} %' for k, v in sm.items() if 'freq vs default' in k or 'vs nodisp' in k))
        md.append('')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'ab2.json').write_text(json.dumps(data, indent=1) + '\n')
    (args.out_dir / 'ab2_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
