"""#3 tables (PROTOCOL_ITEMS.md): GEMM-only and full-model prefill cost of E0M3-dense maps on the SM120 kernel.

python results/tm_opt/gemm/analyze_gemm.py MODEL [MODEL ...]   -> results/tm_opt/gemm/summary.json, tables.md
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = Path('/home/dev/n16k64_campaign/sm120_bench/results')
UNITS = {'8x64': ('n8k64_wB', 'stock_wB'), '16x64': ('n16k64_wA', 'stock_wA')}
PATTERNS = ('e2m1', 'mropt', 'tmopt', 'all_e0m3')


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def gemm(model):
    res = json.loads((O / f'gemm_{model}.json').read_text())
    table, lines = {}, []
    for unit, (mixed, stock) in UNITS.items():
        rows = [r for r in res['rows'] if r['unit'] == unit]
        for proj in dict.fromkeys(r['proj'] for r in rows):
            for t in sorted({r['tokens'] for r in rows if r['proj'] == proj}):
                sel = {(r['config'], r['pattern']): r for r in rows if r['proj'] == proj and r['tokens'] == t}
                base = sel[(stock, 'e2m1')]['kernel_us']
                entry = dict(out=sel[(stock, 'e2m1')]['out'], inp=sel[(stock, 'e2m1')]['inp'], stock_us=base,
                             patterns={p: dict(us=sel[(mixed, p)]['kernel_us'], vs_stock_pct=pct(sel[(mixed, p)]['kernel_us'], base),
                                               e0m3_fraction=sel[(mixed, p)]['e0m3_tiles'] / sel[(mixed, p)]['tiles']) for p in PATTERNS})
                entry['tmopt_vs_mropt_pct'] = pct(entry['patterns']['tmopt']['us'], entry['patterns']['mropt']['us'])
                table[f'{unit} {proj} T={t}'] = entry
        lines += [f'#### {model}, {unit} ({mixed} vs {stock}; kernel µs, CUPTI median)', '',
                  '| projection (out × in) | T | stock NVFP4 | mixed, all E2M1 | MR-OPT (E0M3 share) | TM-OPT (E0M3 share) | all E0M3 | TM-OPT vs MR-OPT |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for key, e in table.items():
            if not key.startswith(unit + ' '):
                continue
            proj, t = key.split(' ')[1], key.split('=')[1]
            p = e['patterns']
            lines.append(f"| {proj} ({e['out']} × {e['inp']}) | {t} | {e['stock_us']:.1f} | {p['e2m1']['us']:.1f} ({p['e2m1']['vs_stock_pct']:+.1f} %) | "
                         f"{p['mropt']['us']:.1f} ({p['mropt']['vs_stock_pct']:+.1f} %; {100 * p['mropt']['e0m3_fraction']:.1f} %) | "
                         f"{p['tmopt']['us']:.1f} ({p['tmopt']['vs_stock_pct']:+.1f} %; {100 * p['tmopt']['e0m3_fraction']:.1f} %) | "
                         f"{p['all_e0m3']['us']:.1f} ({p['all_e0m3']['vs_stock_pct']:+.1f} %) | {e['tmopt_vs_mropt_pct']:+.1f} % |")
        lines.append('')
    return table, lines


def prefill(model):
    out, lines = {}, []
    for unit in UNITS:
        runs = {lab: json.loads((O / f'prefill_{model}_{lab}-{unit}.json').read_text()) for lab in ('stock', 'fo6', 'mropt', 'tmopt')}
        specs = list(runs['stock']['prefill'])
        out[unit] = {lab: {s: r['prefill'][s]['ms'] for s in specs} for lab, r in runs.items()}
        out[unit]['coverage'] = {lab: r.get('coverage') for lab, r in runs.items()}
        lines += [f'#### {model}, full-model prefill, {unit} (ms, median of 5)', '',
                  '| prefill | stock NVFP4 | mixed, FourOverSix | MR-OPT | TM-OPT | TM-OPT vs MR-OPT |', '|---|---:|---:|---:|---:|---:|']
        for s in specs:
            base = out[unit]['stock'][s]
            cells = [f"{out[unit][lab][s]:.2f} ({pct(out[unit][lab][s], base):+.1f} %)" for lab in ('fo6', 'mropt', 'tmopt')]
            lines.append(f"| {s} | {base:.2f} | {' | '.join(cells)} | {pct(out[unit]['tmopt'][s], out[unit]['mropt'][s]):+.1f} % |")
        lines.append('')
    return out, lines


def main():
    summary, lines = {}, []
    for model in sys.argv[1:]:
        if not (O / f'gemm_{model}.json').exists():
            continue
        g, gl = gemm(model)
        p, pl = prefill(model)
        summary[model] = dict(gemm=g, prefill=p)
        lines += gl + pl
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (HERE / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
