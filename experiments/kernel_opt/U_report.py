#!/usr/bin/env python3
"""Kernel-opt amendment 17 (U) report: M1 per-forward GEMM sums with the registered rules of part A (amendment 17's builds)
and part B (the re-tuned 16x64 width cells), the gaps to stock and the ceilings per T, and C2U.

    python experiments/kernel_opt/U_report.py [--src /home/dev/n16k64_campaign/kernel_opt/U/run] [--out-dir results/kernel_opt/U]

M1 as amendments 10-13: per model and T, the per-forward GEMM sum is the sum over projections of modules x median time; a
ratio's range is over the rounds, from each round's sums. Rules (PROTOCOL.md, amendment 17), with TOL = 0.5 %:
- Part A, per family: 'U vs A' with typical and with worst tags each has a negative median over the 48 (model, T) cells,
  and no cell is above +TOL in every round. Cells above zero in every round by less are listed (the strict form).
- Part B: 'UB16 vs U16' over the affected (model, T) cells (some projection runs a changed width or scheduler row),
  typical and worst: negative median, no affected cell above +TOL in every round. Unaffected cells run identical
  computations (reported as an A/A check).
"""
import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
NAMES = dict(llama8b='Llama-3.1-8B', mistral7b='Mistral-7B-v0.3', phi4='Phi-4', qwen27b='Qwen3.8-27B')
TOL = 0.5
V = ('typical', 'worst')
RATIOS = ([(f'16x64 part A, {v}', f'U16_{v}', f'A16_{v}') for v in V]
          + [(f'8x64 part A, {v}', f'U8_{v}', f'A8_{v}') for v in V]
          + [(f'16x64 part B, {v}', f'UB16_{v}', f'U16_{v}') for v in V]
          + [(f'16x64 A+B, {v}', f'UB16_{v}', f'A16_{v}') for v in V]
          + [(f'16x64 today vs stock_ko, {v}', f'A16_{v}', 'stock_ko') for v in V]
          + [(f'16x64 part A vs stock_ko, {v}', f'U16_{v}', 'stock_ko') for v in V]
          + [(f'16x64 A+B vs stock_ko, {v}', f'UB16_{v}', 'stock_ko') for v in V]
          + [(f'8x64 today vs stock_ko, {v}', f'A8_{v}', 'stock_ko') for v in V]
          + [(f'8x64 part A vs stock_ko, {v}', f'U8_{v}', 'stock_ko') for v in V]
          + [('16x64 ceiling vs stock_ko', 'c16', 'stock_ko'), ('8x64 ceiling vs stock_ko', 'c8', 'stock_ko'),
             ('16x64 part A vs ceiling, typical', 'U16_typical', 'c16'), ('8x64 part A vs ceiling, typical', 'U8_typical', 'c8')])
BANDS = (('T ≤ 16', lambda t: t <= 16), ('32–128', lambda t: 32 <= t <= 128), ('256–1024', lambda t: 256 <= t <= 1024),
         ('≥ 2048', lambda t: t >= 2048))


def pct(a, b):
    return 100 * (a / b - 1)


def fmt(r):
    return f"{r['pct']:+.2f} % [{r['range'][0]:+.2f}, {r['range'][1]:+.2f}]"


def rule(rows, unit):
    v = [r[unit]['pct'] for r in rows]
    above = [f"{r['model']} T={r['tokens']} ({fmt(r[unit])})" for r in rows if r[unit]['range'][0] > 0]
    beyond = [f"{r['model']} T={r['tokens']} ({fmt(r[unit])})" for r in rows if r[unit]['range'][0] > TOL]
    return dict(cells=len(v), median=statistics.median(v) if v else None, cells_above_every_round=above,
                cells_above_tolerance=beyond, below_every_round=sum(r[unit]['range'][1] < 0 for r in rows),
                met=bool(v) and statistics.median(v) < 0 and not beyond, strict_form_met=not above)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=Path('/home/dev/n16k64_campaign/kernel_opt/U/run'))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'kernel_opt' / 'U')
    args = ap.parse_args()
    md = ['## M1: per-forward GEMM time, deviation-2\n',
          'Per model and T: the sum over projections of modules × median GEMM time. Brackets: the range over the rounds of '
          'the same ratio, from each round\'s sums. A = today\'s adopted path (16x64 `build_7freq`, 8x64 `build_P3freq`); '
          'U = amendment 17\'s builds (part A) on the adopted table; UB = U on part B\'s table.\n']
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
        s, rr = {}, {}
        for t in tokens:
            for c in cfgs:
                s[(c, t)] = sum(projs[p]['modules'] * by[(c, p, t)]['gemm_us'] for p in projs)
                rr[(c, t)] = [sum(projs[p]['modules'] * by[(c, p, t)]['rounds'][i]['gemm_us'] for p in projs) for i in range(nr)]
        ok = all(c['equal'] for c in rec['checks']['bitwise'])
        md.append(f"### {NAMES[model]}\n\nChecks: {len(rec['checks']['bitwise'])} bitwise (each configuration's isolated "
                  f"path against NativeLinear, and U / UB against A on the timed operands), all equal: {ok}; counts ok: "
                  f"{rec['checks']['counts_ok']}; other processes: {len(rec['checks']['other_processes'])}.\n")
        md.append('| T | part B: changed cells | ' + ' | '.join(u for u, _, _ in RATIOS[:8]) + ' |')
        md.append('|---|---|' + '---|' * 8)
        rows = []
        for t in tokens:
            changed = sorted(p for p in projs if (by[('UB16_typical', p, t)]['width'], by[('UB16_typical', p, t)]['schedule'])
                             != (by[('U16_typical', p, t)]['width'], by[('U16_typical', p, t)]['schedule']))
            row = dict(model=model, tokens=t, part_b_changed=changed)
            for u, a, b in RATIOS:
                per = [pct(x, y) for x, y in zip(rr[(a, t)], rr[(b, t)])]
                row[u] = dict(pct=pct(s[(a, t)], s[(b, t)]), range=[min(per), max(per)])
            rows.append(row)
            md.append(f"| {t} | {', '.join(changed) or '—'} | " + ' | '.join(fmt(row[u]) for u, _, _ in RATIOS[:8]) + ' |')
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
                           above=sum(r[u]['range'][0] > 0 for r in allrows), below=sum(r[u]['range'][1] < 0 for r in allrows),
                           bands={b: (statistics.median(x) if x else None) for (b, _), x in zip(BANDS, bands)})
            md.append(f"| {u} | {summ[u]['median']:+.2f} % | {min(v):+.2f} % | {max(v):+.2f} % | {summ[u]['above']} of "
                      f"{len(v)} | {summ[u]['below']} of {len(v)} | "
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
        affected = [r for r in allrows if r['part_b_changed']]
        unaffected = [r for r in allrows if not r['part_b_changed']]
        rules = dict(part_a_16x64={v: rule(allrows, f'16x64 part A, {v}') for v in V},
                     part_a_8x64={v: rule(allrows, f'8x64 part A, {v}') for v in V},
                     part_b={v: rule(affected, f'16x64 part B, {v}') for v in V},
                     part_b_unaffected_aa={v: rule(unaffected, f'16x64 part B, {v}') for v in V},
                     affected_cells=[f"{r['model']} T={r['tokens']}: {', '.join(r['part_b_changed'])}" for r in affected])
        for key in ('part_a_16x64', 'part_a_8x64', 'part_b'):
            rules[key]['met'] = all(rules[key][v]['met'] for v in V)
        md.append(f'### The registered rules (TOL = {TOL} %)\n')
        for key, title in (('part_a_16x64', 'Part A, 16x64 (U16 vs A16)'), ('part_a_8x64', 'Part A, 8x64 (U8 vs A8)'),
                           ('part_b', 'Part B, 16x64 (UB16 vs U16, affected cells)')):
            rr_ = rules[key]
            md.append(f"- **{title}:** {'met' if rr_['met'] else 'NOT met'}. " + '; '.join(
                f"{v}: median {rr_[v]['median']:+.2f} % over {rr_[v]['cells']} cells, {rr_[v]['below_every_round']} below "
                f"zero in every round, {len(rr_[v]['cells_above_tolerance'])} above +{TOL} % in every round"
                if rr_[v]['median'] is not None else f'{v}: no cells' for v in V) + '.')
            for v in V:
                if rr_[v]['cells_above_every_round']:
                    md.append(f"  - {v}, above zero in every round (the strict form): "
                              + '; '.join(rr_[v]['cells_above_every_round']))
        aa = rules['part_b_unaffected_aa']
        md.append(f"- Part B's unaffected cells (identical computations, an A/A check): " + '; '.join(
            f"{v}: median {aa[v]['median']:+.2f} % over {aa[v]['cells']} cells" for v in V if aa[v]['median'] is not None) + '.')
        md.append('- Affected cells: ' + ('; '.join(rules['affected_cells']) or 'none (no width or scheduler row changed)') + '.')
        md.append('')
        data.update(summary=summ, per_t=per_t, rules=rules)
    c2 = args.src / 'c2_U.json'
    if c2.exists():
        rec = json.loads(c2.read_text())
        md.append('## C2U: 4096³, width 128\n')
        md.append(f"Checks: {rec['checks']}. stock_ko runs width {rec['stock_ko']['width']}, scheduler "
                  f"{rec['stock_ko']['schedule']}.\n")
        keys = [k for k in next(iter(rec['summary'].values()))]
        md.append('| ratio | ' + ' | '.join(rec['summary']) + ' |')
        md.append('|---|' + '---:|' * len(rec['summary']))
        for k in keys:
            md.append(f'| {k} | ' + ' | '.join(f"{rec['summary'][m][k]:+.2f} %" for m in rec['summary']) + ' |')
        md.append('')
        data['c2'] = rec['summary']
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'U.json').write_text(json.dumps(data, indent=1) + '\n')
    (args.out_dir / 'U_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
