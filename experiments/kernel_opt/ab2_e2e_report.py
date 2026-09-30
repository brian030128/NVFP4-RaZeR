#!/usr/bin/env python3
"""Kernel-opt amendment 2b (#2's M2') report: end-to-end CUDA-graph prefill, default vs MIXFP4_DISPATCH_FREQ.

    python experiments/kernel_opt/ab2_e2e_report.py [--src /home/dev/n16k64_campaign/kernel_opt/opt2/e2e] [--out-dir results/kernel_opt/opt2]

Per (model, shape, policy): the median over rounds of each process's median (ab_report.py's M2 convention); a ratio's
spread is its range over rounds, pairing round r of both policies. Per unit: after (freq) vs before, and both vs fo6.
"""
import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
NAMES = dict(llama8b='Llama-3.1-8B', mistral7b='Mistral-7B-v0.3', phi4='Phi-4', qwen27b='Qwen3.8-27B')
UNITS = (('16x64', 'ours-16x64', 'ours-16x64-freq'), ('256x64', 'ours-256x64', 'ours-256x64-freq'),
         ('8x64', 'ours-8x64-opt', 'ours-8x64-freq'))
POLICIES = ('fo6',) + tuple(p for _, b, a in UNITS for p in (b, a))


def pct(a, b):
    return 100 * (a / b - 1)


def section(base):
    res = {}
    for pol in POLICIES:
        for f in sorted((base / pol).glob('round*.json')) if (base / pol).exists() else []:
            rec = json.loads(f.read_text())
            if rec.get('status') != 'complete':
                continue
            r = int(f.stem[5:])
            for shape, e in rec['graph'].items():
                if e.get('ms') is not None:
                    res.setdefault(shape, {}).setdefault(pol, dict(rounds={}))['rounds'][r] = e['ms']
    for pols in res.values():
        for d in pols.values():
            d['median'] = statistics.median(d['rounds'].values())
    return res


def ratio_range(a, b):
    common = sorted(set(a['rounds']) & set(b['rounds']))
    per = [pct(a['rounds'][r], b['rounds'][r]) for r in common]
    return [min(per), max(per)] if per else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=Path('/home/dev/n16k64_campaign/kernel_opt/opt2/e2e'))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'kernel_opt' / 'opt2')
    args = ap.parse_args()
    md, data = ["## M2': end-to-end prefill (ms per forward, CUDA graph; median over rounds)\n"], {}
    for model in MODELS:
        sec = section(args.src / 'prefill' / model)
        if not sec:
            continue
        md.append(f'### {NAMES[model]}\n')
        md.append('| shape | fo6 | ' + ' | '.join(f'{u} before | {u} after | {u} after vs before [rounds] | {u} vs fo6: before → after'
                                                for u, _, _ in UNITS) + ' |')
        md.append('|---|---|' + '---|---|---|---|' * len(UNITS))
        rows = []
        for shape, pols in sec.items():
            if not all(p in pols for p in POLICIES):
                continue
            f6 = pols['fo6']['median']
            row, cells = dict(model=model, shape=shape, fo6=f6, rounds={p: len(pols[p]['rounds']) for p in POLICIES}), []
            for u, b, a in UNITS:
                bb, aa = pols[b], pols[a]
                rng = ratio_range(aa, bb)
                row[u] = dict(before=bb['median'], after=aa['median'], after_vs_before_pct=pct(aa['median'], bb['median']),
                              after_vs_before_range=rng, before_vs_fo6_pct=pct(bb['median'], f6),
                              after_vs_fo6_pct=pct(aa['median'], f6))
                cells.append(f"{bb['median']:.2f} | {aa['median']:.2f} | {row[u]['after_vs_before_pct']:+.2f} % "
                             f"[{rng[0]:+.2f}, {rng[1]:+.2f}] | {row[u]['before_vs_fo6_pct']:+.1f} → {row[u]['after_vs_fo6_pct']:+.1f} %")
            rows.append(row)
            md.append(f'| {shape} | {f6:.2f} | ' + ' | '.join(cells) + ' |')
        md.append('')
        data[model] = rows
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'ab2_e2e.json').write_text(json.dumps(data, indent=1) + '\n')
    (args.out_dir / 'ab2_e2e_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
