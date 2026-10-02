#!/usr/bin/env python3
"""Kernel-opt amendment 15 (the 8x64 plan's P7) report: end-to-end CUDA-graph prefill and decode, the adopted 8x64 path
against the paper's 8x64 kernel and the tuned stocks with the weights on A (stock_ko) and on B (stock_wB_ko). It is
amendment 9's report (cum_e2e_report.py) with cum8_e2e.py's policies.

    python experiments/kernel_opt/cum8_e2e_report.py [--src /home/dev/n16k64_campaign/kernel_opt/w8p7/e2e] \
        [--out-dir results/kernel_opt/w8/p7]

Per (model, shape or setting, policy): the median over rounds of each process's value (ab_report.py's M2 convention:
prefill ms per forward, the process's median over its repetitions; decode tokens per second, Experiment D's). A ratio's
spread is its range over rounds, pairing round r of both policies. Prefill changes are in time (negative = faster), decode
changes in tokens per second (positive = faster), as in step 05 and Experiment D.
"""
import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
NAMES = dict(llama8b='Llama-3.1-8B', mistral7b='Mistral-7B-v0.3', phi4='Phi-4', qwen27b='Qwen3.8-27B')
POLICIES = ('ours-8x64-paper', 'ours-8x64-adopted', 'fo6-ko', 'fo6-wB-ko')
# (label, a, b): a vs b
RATIOS = (('adopted vs paper (8x64)', 'ours-8x64-adopted', 'ours-8x64-paper'),
          ('adopted vs fo6-ko', 'ours-8x64-adopted', 'fo6-ko'),
          ('adopted vs fo6-wB-ko', 'ours-8x64-adopted', 'fo6-wB-ko'),
          ('paper 8x64 vs fo6-ko', 'ours-8x64-paper', 'fo6-ko'),
          ('fo6-wB-ko vs fo6-ko', 'fo6-wB-ko', 'fo6-ko'))
FIELD = dict(prefill=('graph', 'ms'), decode=('decode', 'tokens_per_s'))


def pct(a, b):
    return 100 * (a / b - 1)


def section(base, what):
    key, field = FIELD[what]
    res = {}
    for pol in POLICIES:
        for f in sorted((base / pol).glob('round*.json')) if (base / pol).exists() else []:
            rec = json.loads(f.read_text())
            if rec.get('status') != 'complete':
                continue
            r = int(f.stem[5:])
            for shape, e in rec[key].items():
                if e.get(field) is not None:
                    res.setdefault(shape, {}).setdefault(pol, dict(rounds={}))['rounds'][r] = e[field]
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
    ap.add_argument('--src', type=Path, default=Path('/home/dev/n16k64_campaign/kernel_opt/w8p7/e2e'))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'kernel_opt' / 'w8' / 'p7')
    args = ap.parse_args()
    md, data = [], {}
    for what, title, unit in (('prefill', 'prefill, ms per forward (CUDA graph); changes in time, negative = faster', 'ms'),
                              ('decode', 'decode, tokens per second (CUDA graph); changes in throughput, positive = faster',
                               'tok/s')):
        md.append(f'## End-to-end {title}\n')
        md.append('The median over rounds of each process\'s value; brackets: the range over rounds, pairing round r.\n')
        allrows = []
        for model in MODELS:
            sec = section(args.src / what / model, what)
            if not sec:
                continue
            md.append(f'### {NAMES[model]}\n')
            md.append('| shape | ' + ' | '.join(f'{p} ({unit})' for p in POLICIES) + ' | '
                      + ' | '.join(f'{u} [rounds]' for u, _, _ in RATIOS) + ' |')
            md.append('|---|' + '---:|' * len(POLICIES) + '---|' * len(RATIOS))
            rows = []
            for shape, pols in sec.items():
                if not all(p in pols for p in POLICIES):
                    continue
                row = dict(model=model, shape=shape, **{p: pols[p]['median'] for p in POLICIES},
                           rounds={p: len(pols[p]['rounds']) for p in POLICIES})
                cells = []
                for u, a, b in RATIOS:
                    rng = ratio_range(pols[a], pols[b])
                    row[u] = dict(pct=pct(pols[a]['median'], pols[b]['median']), range=rng)
                    cells.append(f"{row[u]['pct']:+.2f} % [{rng[0]:+.2f}, {rng[1]:+.2f}]")
                rows.append(row)
                md.append(f'| {shape} | ' + ' | '.join(f'{pols[p]["median"]:.2f}' for p in POLICIES) + ' | '
                          + ' | '.join(cells) + ' |')
            md.append('')
            data.setdefault(what, {})[model] = rows
            allrows += rows
        if allrows:
            better = (lambda rng: rng[1] < 0) if what == 'prefill' else (lambda rng: rng[0] > 0)
            worse = (lambda rng: rng[0] > 0) if what == 'prefill' else (lambda rng: rng[1] < 0)
            md.append(f'### All models ({what}): over the (model, shape) cells\n')
            md.append('| ratio | median | min | max | better in every round | worse in every round |')
            md.append('|---|---:|---:|---:|---:|---:|')
            for u, _, _ in RATIOS:
                v = [r[u]['pct'] for r in allrows]
                md.append(f'| {u} | {statistics.median(v):+.2f} % | {min(v):+.2f} % | {max(v):+.2f} % | '
                          f'{sum(better(r[u]["range"]) for r in allrows)} of {len(v)} | '
                          f'{sum(worse(r[u]["range"]) for r in allrows)} of {len(v)} |')
            md.append('')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'cum8_e2e.json').write_text(json.dumps(data, indent=1) + '\n')
    (args.out_dir / 'cum8_e2e_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
