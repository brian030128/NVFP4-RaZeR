#!/usr/bin/env python3
"""Kernel-opt amendment 12 (the 8x64 plan's P3, with P4) report: M1 per-forward GEMM sums, the registered adoption rule,
the cells whose width or scheduler setting changed, and the decisive-margin sensitivity.

    python experiments/kernel_opt/w8p3_report.py [--src /home/dev/n16k64_campaign/kernel_opt/w8p3] \
        [--out-dir results/kernel_opt/w8/p3]

M1 is summarized as in amendments 10 and 11. Per model and T, the per-forward GEMM sum is the sum over projections of
modules × median time. A ratio's spread is its range over the rounds, computed from each round's sums.

Ratio groups:
- The re-tune: the P3 table against P2's rows, typical and worst tags. The widths alone, and the scheduler rows on top
  of them. stock_wB tuned alike against stock_wB.
- The sensitivity: the decisive-margin widths (default 128, and default the fallback width) against P3's fastest widths.
  Neither has scheduler rows.
- The gap to stock: before and after against stock_ko, and after against stock_wB tuned alike.

The registered rule (PROTOCOL.md, amendment 12), evaluated on two groups of units:
- The 8x64 units (after vs before, typical and worst): each has a negative median over the cells, and no cell is above
  zero in every round by more than TOL (0.5 %).
- stock_wB tuned alike: the same rule for its unit.
The cells above zero in every round by less than TOL are listed.

Width and scheduler changes: every (model, projection, T) whose width or scheduler setting differs between wBp2_typical
and wBko_typical, with the GEMM time change. The sensitivity cells and the tuner's numbers come from --sensitivity.
"""
import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
NAMES = dict(llama8b='Llama-3.1-8B', mistral7b='Mistral-7B-v0.3', phi4='Phi-4', qwen27b='Qwen3.8-27B')
GROUPS = (
    ('the re-tune', (
        ('after vs before, typical', 'wBko_typical', 'wBp2_typical'),
        ('after vs before, worst', 'wBko_worst', 'wBp2_worst'),
        ('widths only vs before, typical', 'wBw_typical', 'wBp2_typical'),
        ('scheduler rows on the widths, typical', 'wBko_typical', 'wBw_typical'),
        ('stock_wB tuned alike vs stock_wB', 'stock_wB_ko', 'stock_wB'))),
    ('the decisive-margin sensitivity (no scheduler rows)', (
        ('dm128 vs fastest widths, typical', 'wBdm128_typical', 'wBw_typical'),
        ('dmfb vs fastest widths, typical', 'wBdmfb_typical', 'wBw_typical'))),
    ('the gap to stock', (
        ('before vs stock_ko, typical', 'wBp2_typical', 'stock_ko'),
        ('after vs stock_ko, typical', 'wBko_typical', 'stock_ko'),
        ('after vs stock_ko, worst', 'wBko_worst', 'stock_ko'),
        ('after vs stock_wB tuned alike, typical', 'wBko_typical', 'stock_wB_ko'),
        ('stock_wB tuned alike vs stock_ko', 'stock_wB_ko', 'stock_ko'))))
RATIOS = tuple(r for _, rs in GROUPS for r in rs)
UNITS_8X64 = ('after vs before, typical', 'after vs before, worst')
UNIT_STOCK = 'stock_wB tuned alike vs stock_wB'
TOL = 0.5   # %
BANDS = (('T ≤ 16', lambda t: t <= 16), ('32–128', lambda t: 32 <= t <= 128), ('256–1024', lambda t: 256 <= t <= 1024),
         ('≥ 2048', lambda t: t >= 2048))


def pct(a, b):
    return 100 * (a / b - 1)


def rule(summ, allrows, units):
    above = {u: [f"{r['model']} T={r['tokens']} ({r[u]['pct']:+.2f} % [{r[u]['range'][0]:+.2f}, {r[u]['range'][1]:+.2f}])"
                 for r in allrows if r[u]['range'][0] > 0] for u in units}
    beyond = {u: sum(r[u]['range'][0] > TOL for r in allrows) for u in units}
    return dict(medians={u: summ[u]['median'] for u in units}, cells_above=above, cells_above_tolerance=beyond,
                met=all(summ[u]['median'] < 0 for u in units) and not any(beyond.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=Path('/home/dev/n16k64_campaign/kernel_opt/w8p3'))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'kernel_opt' / 'w8' / 'p3')
    ap.add_argument('--sensitivity', type=Path, default=None)
    args = ap.parse_args()
    md = ['## M1: per-forward GEMM time, deviation-2\n',
          'Per model and T: the sum over projections of modules × median GEMM time. Brackets: the range over the rounds '
          'of the same ratio, from each round\'s sums. Every 8x64 configuration runs the same libraries (the t0 builds '
          'with #2\'s dispatch); "before" is P2\'s adopted path on 1b\'s rows, "after" the P3 table (widths and '
          'scheduler rows).\n']
    data, allrows, changes = {}, [], []
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
        md.append(f"### {NAMES[model]}\n\nChecks: {len(rec['checks']['bitwise'])} bitwise, all equal: {ok}; counts ok: "
                  f"{rec['checks']['counts_ok']}; other processes: {len(rec['checks']['other_processes'])}.\n")
        rows = [dict(model=model, tokens=t) for t in tokens]
        for title, ratios in GROUPS:
            md.append(f'**{title}**\n')
            md.append('| T | ' + ' | '.join(f'{u} [rounds]' for u, _, _ in ratios) + ' |')
            md.append('|---|' + '---|' * len(ratios))
            for row, t in zip(rows, tokens):
                cells = []
                for u, a, b in ratios:
                    per = [pct(x, y) for x, y in zip(rr[(a, t)], rr[(b, t)])]
                    row[u] = dict(pct=pct(s[(a, t)], s[(b, t)]), range=[min(per), max(per)])
                    cells.append(f"{row[u]['pct']:+.2f} % [{min(per):+.2f}, {max(per):+.2f}]")
                md.append(f'| {t} | ' + ' | '.join(cells) + ' |')
            md.append('')
        md.append('Widths (before → after; scheduler setting after) per projection and T:\n')
        md.append('| projection | ' + ' | '.join(str(t) for t in tokens) + ' |')
        md.append('|---|' + '---|' * len(tokens))
        for p in projs:
            cells = []
            for t in tokens:
                a, b = by[('wBp2_typical', p, t)], by[('wBko_typical', p, t)]
                cells.append(f"{a['width']} → {b['width']} {tuple(b['schedule'])}" if (a['width'], tuple(a['schedule'] or (0, 1)))
                             != (b['width'], tuple(b['schedule'] or (0, 1))) else f"{a['width']}")
                if (a['width'], tuple(a['schedule'] or (0, 1))) != (b['width'], tuple(b['schedule'] or (0, 1))):
                    changes.append(dict(model=model, proj=p, tokens=t, width_before=a['width'], width_after=b['width'],
                                        schedule_after=b['schedule'], modules=projs[p]['modules'],
                                        us_before=a['gemm_us'], us_after=b['gemm_us'],
                                        pct=pct(b['gemm_us'], a['gemm_us'])))
            md.append(f'| {p} | ' + ' | '.join(cells) + ' |')
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
        tokens_all = sorted({r['tokens'] for r in allrows})
        md.append('### Per T, median over the models\n')
        md.append('| T | ' + ' | '.join(u for _, rs in GROUPS for u, _, _ in rs) + ' |')
        md.append('|---:|' + '---:|' * len(RATIOS))
        per_t = {}
        for t in tokens_all:
            vals = {u: statistics.median([r[u]['pct'] for r in allrows if r['tokens'] == t]) for u, _, _ in RATIOS}
            per_t[t] = vals
            md.append(f'| {t} | ' + ' | '.join(f'{vals[u]:+.2f} %' for u, _, _ in RATIOS) + ' |')
        md.append('')
        rules = dict(p3_8x64=rule(summ, allrows, UNITS_8X64), stock_wB=rule(summ, allrows, (UNIT_STOCK,)))
        md.append('### The registered rule\n')
        for name, r in rules.items():
            md.append(f"- **{name}** (negative medians; no cell above zero in every round by more than {TOL} %): "
                      f"{'met' if r['met'] else 'NOT met'}. Medians: "
                      + ', '.join(f'{u} {v:+.2f} %' for u, v in r['medians'].items()) + '.')
            for u, cells in r['cells_above'].items():
                if cells:
                    md.append(f'  - {u}, above zero in every round: ' + '; '.join(cells))
        md.append('')
        md.append(f'### Width and scheduler changes (typical tags): {len(changes)} (model, projection, T) cells\n')
        if changes:
            md.append('| model | projection | T | width | scheduler | GEMM time |')
            md.append('|---|---|---:|---|---|---:|')
            for c in changes:
                md.append(f"| {c['model']} | {c['proj']} | {c['tokens']} | {c['width_before']} → {c['width_after']} | "
                          f"{tuple(c['schedule_after'])} | {c['pct']:+.2f} % |")
            md.append('')
        data.update(summary=summ, per_t=per_t, rules=rules, changes=changes)
    if args.sensitivity and args.sensitivity.exists():
        sens = json.loads(args.sensitivity.read_text())
        md.append('## The decisive-margin sensitivity (from the width tuning)\n')
        md.append(f"Rule: {sens['rule']}. Defaults: " + '; '.join(f'{k}: {v}' for k, v in sens['defaults'].items())
                  + f". {sens['cells']} (shape, bucket) cells tuned.\n")
        for k, v in sens['summary'].items():
            md.append(f"- **{k}:** {v['cells_changed']} cells would change"
                      + (f"; the rule's width is slower in the tuning by {v['median_pct']:+.2f} % (median), up to "
                         f"{v['max_pct']:+.2f} %." if v['cells_changed'] else '.'))
        md.append('')
        for k, cells in sens['changed'].items():
            if not cells:
                continue
            md.append(f'**{k}:**\n')
            md.append('| shape | bucket | fastest | rule | fastest µs | rule µs | rule vs fastest |')
            md.append('|---|---:|---|---|---:|---:|---:|')
            for c in cells:
                md.append(f"| {c['shape']} | {c['bucket']} | {c['fastest']} | {c['rule']} | {c['fastest_us']:.2f} | "
                          f"{c['rule_us']:.2f} | {c['rule_vs_fastest_pct']:+.2f} % |")
            md.append('')
        data['sensitivity'] = dict(summary=sens['summary'], cells=sens['cells'])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'w8p3.json').write_text(json.dumps(data, indent=1) + '\n')
    (args.out_dir / 'w8p3_tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
