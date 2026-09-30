#!/usr/bin/env python3
"""Kernel-opt amendment 4 (tile-table re-tune) report: M1 under the current vs the re-tuned table.

    python experiments/kernel_opt/abR_report.py [--src /home/dev/n16k64_campaign/kernel_opt/retune] [--out-dir results/kernel_opt/retune]

Per model and T, the per-forward GEMM sum (sum over projections of modules x median time) of every configuration of
bench_ab_retune.py; the spread is the range over the 3 rounds of the same sum from each round's medians. Reported:
new vs current table per unit (stock_wA, 16x64 typical / worst, 256x64 typical / worst); 16x64 and 256x64 vs stock_wA
with both on the current table -> both on the new table; and every (projection, T) cell whose width changed, with the
cell's own time change.
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
NAMES = dict(llama8b='Llama-3.1-8B', mistral7b='Mistral-7B-v0.3', phi4='Phi-4', qwen27b='Qwen3.8-27B')
UNITS = (('stock_wA', 'stock_wA'), ('16x64 typical', 'm16_typical'), ('16x64 worst', 'm16_worst'),
         ('256x64 typical', 'm256_typical'), ('256x64 worst', 'm256_worst'))


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
    return s, rr, tokens, by, projs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=Path('/home/dev/n16k64_campaign/kernel_opt/retune'))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'kernel_opt' / 'retune')
    args = ap.parse_args()
    md, data = ["## M1: per-forward GEMM time, re-tuned table (new) vs current table (cur), deviation-2 method\n"], {}
    changed_all = []
    for model in MODELS:
        path = args.src / 'gemm' / f'{model}.json'
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        if rec.get('status') != 'complete':
            md.append(f'{model}: {rec.get("status")}\n')
            continue
        s, rr, tokens, by, projs = sums(rec)
        ok = all(c['equal'] for c in rec['checks']['bitwise'])
        md.append(f"### {NAMES[model]}\n\nChecks: {len(rec['checks']['bitwise'])} bitwise ok={ok} (new = cur on the timed "
                  f"operands included), counts ok={rec['checks']['counts_ok']}, other processes "
                  f"{len(rec['checks']['other_processes'])}.\n")
        md.append('| T | ' + ' | '.join(f'{u}: new vs cur [rounds]' for u, _ in UNITS)
                  + ' | 16x64 vs stock_wA: cur → new | 256x64 vs stock_wA: cur → new |')
        md.append('|---|' + '---|' * len(UNITS) + '---|---|')
        rows = []
        for t in tokens:
            row, cells = dict(model=model, tokens=t), []
            for u, c in UNITS:
                a, b = (f'{c}_new', t), (f'{c}_cur', t)
                row[f'{u} new_vs_cur_pct'] = pct(s[a], s[b])
                per = [pct(x, y) for x, y in zip(rr[a], rr[b])]
                row[f'{u} new_vs_cur_range'] = [min(per), max(per)]
                cells.append(f"{row[f'{u} new_vs_cur_pct']:+.2f} % [{min(per):+.2f}, {max(per):+.2f}]")
            for u, c in (('16x64', 'm16_typical'), ('256x64', 'm256_typical')):
                row[f'{u} vs stock cur'] = pct(s[(f'{c}_cur', t)], s[('stock_wA_cur', t)])
                row[f'{u} vs stock new'] = pct(s[(f'{c}_new', t)], s[('stock_wA_new', t)])
            rows.append(row)
            md.append(f'| {t} | ' + ' | '.join(cells) + f" | {row['16x64 vs stock cur']:+.1f} → {row['16x64 vs stock new']:+.1f} % | "
                      f"{row['256x64 vs stock cur']:+.1f} → {row['256x64 vs stock new']:+.1f} % |")
        md.append('')
        changed = []
        for p in projs:
            for t in tokens:
                for u, c in UNITS:
                    a, b = by[(f'{c}_new', p, t)], by[(f'{c}_cur', p, t)]
                    if a['width'] != b['width']:
                        changed.append(dict(model=model, proj=p, shape=projs[p]['shape'], modules=projs[p]['modules'],
                                            tokens=t, unit=u, width_cur=b['width'], width_new=a['width'],
                                            us_cur=b['gemm_us'], us_new=a['gemm_us'], pct=pct(a['gemm_us'], b['gemm_us'])))
        changed_all += changed
        data[model] = dict(rows=rows, sums={f'{c}@{t}': v for (c, t), v in s.items()}, changed=changed, bitwise_ok=ok,
                           sampler=rec.get('sampler'))
    md.append('## Cells whose width changed (per projection, the cell\'s own median GEMM time)\n')
    md.append('| model | projection (shape) | T | unit | width cur → new | µs cur → new | change |')
    md.append('|---|---|---|---|---|---|---|')
    for c in changed_all:
        md.append(f"| {c['model']} | {c['proj']} ({c['shape'][0]}x{c['shape'][1]}) | {c['tokens']} | {c['unit']} | "
                  f"{c['width_cur']} → {c['width_new']} | {c['us_cur']:.1f} → {c['us_new']:.1f} | {c['pct']:+.1f} % |")
    md.append('')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'abR.json').write_text(json.dumps(dict(models=data, changed=changed_all), indent=1) + '\n')
    (args.out_dir / 'abR_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
