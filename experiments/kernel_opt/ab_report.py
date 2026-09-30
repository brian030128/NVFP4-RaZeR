#!/usr/bin/env python3
"""Kernel-opt report for optimization 1: before vs after from M1 (GEMM), M2 (prefill) and M3 (decode).

    python experiments/kernel_opt/ab_report.py [--out-dir results/kernel_opt]

Protocol: results/kernel_opt/PROTOCOL.md. Reads KERNEL_OPT_OUT (/home/dev/n16k64_campaign/kernel_opt):
gemm/<model>.json (bench_ab_isolated.py), e2e/prefill/<model>/<policy>/round<r>.json (bench_prefill.py) and
e2e/decode/<model>/<policy>/round<r>.json (bench_decode.py). Writes <out-dir>/ab1.json, ab1_gemm.csv, ab1_prefill.csv,
ab1_decode.csv and ab1_tables.md.
- M1: per-forward GEMM sums as step 06 / deviation 2: Σ over projections of (modules × median GEMM time). The spread
  is the range over the 3 rounds of the same quantity computed from each round's medians.
- M2 / M3: per (model, shape or setting, policy), the median over rounds of each process's median; the spread of a
  ratio is its range over rounds, pairing round r of both policies (the same rotated order).
"""
import argparse
import csv
import json
import os
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get('KERNEL_OPT_OUT', '/home/dev/n16k64_campaign/kernel_opt'))
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
NAMES = dict(llama8b='Llama-3.1-8B', mistral7b='Mistral-7B-v0.3', phi4='Phi-4', qwen27b='Qwen3.8-27B')
POLICIES = ('fo6', 'fo6-wB', 'ours-8x64', 'ours-8x64-opt')


def pct(a, b):
    return 100 * (a / b - 1)


def fmt(v, digits=1):
    return '—' if v is None else f'{v:+.{digits}f} %'


def gemm_section(model):
    path = SRC / 'gemm' / f'{model}.json'
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    assert rec['status'] == 'complete', path
    projs = rec['projections']
    by = {(r['config'], r['proj'], r['tokens']): r for r in rec['rows']}
    tokens = sorted({r['tokens'] for r in rec['rows']})
    cfgs = sorted({r['config'] for r in rec['rows']})
    rounds = len(rec['rows'][0]['rounds'])
    out = dict(tokens=tokens, sums={}, rounds={}, widths={}, checks=dict(
        bitwise=all(c['equal'] for c in rec['checks']['bitwise']), n_bitwise=len(rec['checks']['bitwise']),
        counts_ok=rec['checks']['counts_ok'], other_processes=rec['checks']['other_processes']),
        telemetry=rec.get('sampler'), power_limit_w=rec.get('power_limit_w'), kernels=rec['kernels'])
    for t in tokens:
        for c in cfgs:
            out['sums'][f'{c}@{t}'] = sum(projs[p]['modules'] * by[(c, p, t)]['gemm_us'] for p in projs)
            out['rounds'][f'{c}@{t}'] = [sum(projs[p]['modules'] * by[(c, p, t)]['rounds'][i]['gemm_us'] for p in projs)
                                         for i in range(rounds)]
        out['widths'][t] = {p: by[('auto_wB_typical', p, t)]['kernel'] for p in projs}
    rows = []
    for t in tokens:
        s, rr = out['sums'], out['rounds']
        row = dict(model=model, tokens=t, **{f'{c}_us': round(s[f'{c}@{t}'], 2) for c in cfgs})
        for v in ('typical', 'worst'):
            b, a = f'n8k64_wB_{v}@{t}', f'auto_wB_{v}@{t}'
            per = [pct(x, y) for x, y in zip(rr[a], rr[b])]
            row[f'after_vs_before_{v}_pct'] = pct(s[a], s[b])
            row[f'after_vs_before_{v}_range'] = [min(per), max(per)]
            for ref in ('stock_wA', 'stock_wB'):
                row[f'before_vs_{ref}_{v}_pct'] = pct(s[b], s[f'{ref}@{t}'])
                row[f'after_vs_{ref}_{v}_pct'] = pct(s[a], s[f'{ref}@{t}'])
        rows.append(row)
    out['rows'] = rows
    return out


def e2e_section(what, model, key):
    """{shape: {policy: dict(median, rounds=[...])}} of the per-process medians."""
    base = SRC / 'e2e' / what / model
    if not base.exists():
        return None
    res = {}
    for pol in POLICIES:
        for f in sorted((base / pol).glob('round*.json')) if (base / pol).exists() else []:
            rec = json.loads(f.read_text())
            if rec.get('status') != 'complete':
                continue
            r = int(f.stem[5:])
            block = rec['graph'] if what == 'prefill' else rec['decode']
            for shape, e in block.items():
                if e.get(key) is None:
                    continue
                d = res.setdefault(shape, {}).setdefault(pol, dict(rounds={}))
                d['rounds'][r] = e[key]
                if what == 'prefill':
                    d.setdefault('eager_rounds', {})[r] = rec['eager'][shape]['ms']
                if what == 'decode' and e.get('decode_widths') is not None:
                    d['widths'] = e['decode_widths']
    for shape, pols in res.items():
        for pol, d in pols.items():
            d['median'] = statistics.median(d['rounds'].values())
            if 'eager_rounds' in d:
                d['eager_median'] = statistics.median(d['eager_rounds'].values())
    return res


def ratio_range(a, b):
    common = sorted(set(a['rounds']) & set(b['rounds']))
    per = [pct(a['rounds'][r], b['rounds'][r]) for r in common]
    return [min(per), max(per)] if per else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'kernel_opt')
    args = ap.parse_args()
    data = dict(source=str(SRC), gemm={}, prefill={}, decode={})
    md = []
    gcsv, pcsv, dcsv = [], [], []
    md.append('## M1: per-forward GEMM time (isolated launches, cold weights; µs)\n')
    for model in MODELS:
        g = gemm_section(model)
        if g is None:
            continue
        data['gemm'][model] = g
        md.append(f'### {NAMES[model]}\n')
        md.append(f"Checks: bitwise {g['checks']['n_bitwise']} ok={g['checks']['bitwise']}, launch counts ok="
                  f"{g['checks']['counts_ok']}, other processes {len(g['checks']['other_processes'])}.\n")
        md.append('| T | stock_wA | stock_wB | 8x64 before (typ.) | 8x64 after (typ.) | after vs before (typ.) [round range] '
                  '| after vs before (worst) | vs stock_wA: before → after (typ.) | vs stock_wB: before → after (typ.) |')
        md.append('|---|---|---|---|---|---|---|---|---|')
        for r in g['rows']:
            lo, hi = r['after_vs_before_typical_range']
            md.append(f"| {r['tokens']} | {r['stock_wA_us']:.1f} | {r['stock_wB_us']:.1f} | {r['n8k64_wB_typical_us']:.1f} | "
                      f"{r['auto_wB_typical_us']:.1f} | {fmt(r['after_vs_before_typical_pct'])} [{lo:+.1f}, {hi:+.1f}] | "
                      f"{fmt(r['after_vs_before_worst_pct'])} | {fmt(r['before_vs_stock_wA_typical_pct'])} → "
                      f"{fmt(r['after_vs_stock_wA_typical_pct'])} | {fmt(r['before_vs_stock_wB_typical_pct'])} → "
                      f"{fmt(r['after_vs_stock_wB_typical_pct'])} |")
            gcsv.append(r)
        md.append('')
        md.append('Width chosen (build per projection, typical tags):\n')
        for t, w in g['widths'].items():
            md.append(f"- T = {t}: " + ', '.join(f'{p} {k}' for p, k in w.items()))
        md.append('')
    for what, key, unit in (('prefill', 'ms', 'ms per forward, CUDA graph'), ('decode', 'ms_per_token', 'ms per token')):
        md.append(f'## {"M2: prefill" if what == "prefill" else "M3: decode"} ({unit}; median over rounds)\n')
        for model in MODELS:
            sec = e2e_section(what, model, key)
            if not sec:
                continue
            data[what][model] = sec
            md.append(f'### {NAMES[model]}\n')
            md.append('| shape | fo6 | fo6-wB | ours-8x64 before | ours-8x64 after | after vs before [round range] | '
                      'vs fo6: before → after | vs fo6-wB: before → after |')
            md.append('|---|---|---|---|---|---|---|---|')
            for shape, pols in sec.items():
                if not all(p in pols for p in POLICIES):
                    continue
                b, a, f6, f6b = (pols[p] for p in ('ours-8x64', 'ours-8x64-opt', 'fo6', 'fo6-wB'))
                rng = ratio_range(a, b)
                row = dict(model=model, shape=shape, fo6=f6['median'], fo6_wB=f6b['median'], before=b['median'],
                           after=a['median'], after_vs_before_pct=pct(a['median'], b['median']), after_vs_before_range=rng,
                           before_vs_fo6_pct=pct(b['median'], f6['median']), after_vs_fo6_pct=pct(a['median'], f6['median']),
                           before_vs_fo6wB_pct=pct(b['median'], f6b['median']), after_vs_fo6wB_pct=pct(a['median'], f6b['median']),
                           rounds={p: len(pols[p]['rounds']) for p in POLICIES})
                if what == 'prefill':
                    row.update(eager_before=b['eager_median'], eager_after=a['eager_median'],
                               eager_after_vs_before_pct=pct(a['eager_median'], b['eager_median']))
                else:
                    row['widths_after'] = a.get('widths')
                md.append(f"| {shape} | {f6['median']:.2f} | {f6b['median']:.2f} | {b['median']:.2f} | {a['median']:.2f} | "
                          f"{fmt(row['after_vs_before_pct'])} [{rng[0]:+.1f}, {rng[1]:+.1f}] | {fmt(row['before_vs_fo6_pct'])} → "
                          f"{fmt(row['after_vs_fo6_pct'])} | {fmt(row['before_vs_fo6wB_pct'])} → {fmt(row['after_vs_fo6wB_pct'])} |")
                (pcsv if what == 'prefill' else dcsv).append(row)
            md.append('')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'ab1.json').write_text(json.dumps(data, indent=1) + '\n')
    for name, rows in (('ab1_gemm.csv', gcsv), ('ab1_prefill.csv', pcsv), ('ab1_decode.csv', dcsv)):
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(args.out_dir / name, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    (args.out_dir / 'ab1_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
