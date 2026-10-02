#!/usr/bin/env python3
"""The main W4A4 perplexity table's outputs (results/main_ppl/PROTOCOL.md §5), from the run_ppl_deploy.py records.

    python experiments/main_ppl/analyze.py [--out results/main_ppl]

Inputs: MAIN_PPL_OUT/ppl/<model>/<row>/report.json. For phi4, qwen27b and mistral7b the native and BF16 rows are the
paper's step 03 records (PAPER_OUT/ppl/<model>/<row>/report.json), reused after experiments/main_ppl/run.py's check.
Within a model, every row must have evaluated the same windows (token hashes), or the script stops.

Outputs (in --out):
- main_ppl.json, main_ppl.csv: every PPL, and every paired ΔNLL with its SE;
- table_main_instruct.tex / table_main_base.tex: the main table with Mistral-7B-Instruct-v0.3 / Mistral-7B-v0.3;
- table_dnll_appendix.tex: Appendix D, ΔNLL vs FourOverSix ± 2 SE;
- REPORT.md.
Definitions:
- **NLL and PPL.** NLL is a window's mean cross-entropy (nats per token; the records' per-window values). PPL is the
  record's exp(mean NLL); log PPL = mean NLL.
- **Paired ΔNLL(row − ref)** for one model and corpus: the mean over windows of the per-window differences.
  SE = sd / √n (sd with n − 1).
  - "Significantly better" means ΔNLL + 2 SE < 0, "significantly worse" means ΔNLL − 2 SE > 0.
  - The native rows are compared with the native FourOverSix row. The simulated rows are compared with both the native
    and the simulated FourOverSix (fo6-fake).
- **Loss recovered(row)** = 1 − Σ_p [logPPL(row, p) − logPPL(BF16, p)] / Σ_p [logPPL(NVFP4, p) − logPPL(BF16, p)].
  It is summed over the 12 model–corpus pairs p of a table variant. NVFP4 scores 0 %, BF16 100 %.
- **†:** a FlipQuant cell that is not significantly better than FourOverSix.
- **Bold:** the lowest PPL per column among the natively executable rows (NVFP4, FourOverSix, FlipQuant 8x64 / 16x64 /
  256x64). BF16 and the simulated rows are never bold.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

OUT_RUNS = Path(os.environ.get('MAIN_PPL_OUT', '/home/dev/n16k64_campaign/main_ppl'))
PAPER_PPL = Path('/home/dev/n16k64_campaign/paper/ppl')
REUSED = ('phi4', 'qwen27b', 'mistral7b')
REUSED_ROWS = ('bf16', 'nvfp4', 'fo6', 'ours-8x64', 'ours-16x64', 'ours-256x64')
TITLES = {'qwen3_1p7b': 'Qwen3-1.7B', 'qwen3_8b': 'Qwen3-8B', 'mistral7b_ins': 'Mistral-7B', 'mistral7b': 'Mistral-7B',
          'nemotron9b': 'Nemotron-Nano-9B-v2', 'phi4': 'Phi-4', 'qwen27b': 'Qwen3.8-27B'}
VARIANTS = {'instruct': ('qwen3_1p7b', 'qwen3_8b', 'mistral7b_ins', 'nemotron9b', 'phi4', 'qwen27b'),
            'base': ('qwen3_1p7b', 'qwen3_8b', 'mistral7b', 'nemotron9b', 'phi4', 'qwen27b')}
ROWS = [('bf16', 'BF16'), ('nvfp4', 'NVFP4'), ('fo6', 'FourOverSix'), ('if4', 'IF4 (Cook et al.) 1x16$^\\ast$'),
        ('zou', 'MixFP4 (Zou et al.) 1x16$^\\ast$'), ('ours-8x64', 'FlipQuant (ours) 8x64'),
        ('ours-16x64', 'FlipQuant (ours) 16x64'), ('ours-256x64', 'FlipQuant (ours) 256x64')]
NATIVE = ('nvfp4', 'fo6', 'ours-8x64', 'ours-16x64', 'ours-256x64')
OURS = ('ours-8x64', 'ours-16x64', 'ours-256x64')
SIMULATED = ('if4', 'zou')
CORPORA = ('wiki', 'c4')


def record(model, row):
    path = (PAPER_PPL if model in REUSED and row in REUSED_ROWS else OUT_RUNS / 'ppl') / model / row / 'report.json'
    if not path.exists():
        return None, path
    r = json.loads(path.read_text())
    assert r['status'] == 'complete', path
    return r, path


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    return dict(dnll=mean, se=sd / math.sqrt(n), n=n)


def load(models):
    data = {}
    for model in models:
        m = data[model] = dict(rows={}, sources={}, windows=None)
        for row in [r for r, _ in ROWS] + ['fo6-fake']:
            r, path = record(model, row)
            if r is None:
                continue
            ev = r['evaluations'][row]['evaluation']
            win = {c: (r['data']['wiki' if c == 'wiki' else 'c4_paper']['token_sha256']) for c in CORPORA}
            if m['windows'] is None:
                m['windows'] = win
            assert win == m['windows'], f'{model} {row}: other windows than the model\'s other rows'
            m['rows'][row] = {c: dict(ppl=ev[c]['ppl'], nll=ev[c]['nll'], log_ppl=sum(ev[c]['nll']) / len(ev[c]['nll']))
                              for c in CORPORA}
            m['sources'][row] = str(path)
    return data


def comparisons(data):
    out = {}
    for model, m in data.items():
        rows = m['rows']
        for row in rows:
            for ref in ('fo6', 'fo6-fake', 'nvfp4'):
                if ref == row or ref not in rows:
                    continue
                if ref == 'fo6-fake' and row not in SIMULATED:
                    continue
                for c in CORPORA:
                    out[(model, row, ref, c)] = paired(rows[row][c]['nll'], rows[ref][c]['nll'])
    return out


def loss_recovered(data, models, row):
    num = den = 0.0
    for model in models:
        rows = data[model]['rows']
        if not all(k in rows for k in (row, 'bf16', 'nvfp4')):
            return None
        for c in CORPORA:
            num += rows[row][c]['log_ppl'] - rows['bf16'][c]['log_ppl']
            den += rows['nvfp4'][c]['log_ppl'] - rows['bf16'][c]['log_ppl']
    return 1 - num / den


def latex_main(data, cmp, models, variant):
    best = {}
    for model in models:
        for c in CORPORA:
            vals = [(data[model]['rows'][r][c]['ppl'], r) for r in NATIVE if r in data[model]['rows']]
            best[(model, c)] = min(vals)[1] if vals else None
    head = ' & '.join(f'\\multicolumn{{2}}{{c}}{{{TITLES[m]}}}' for m in models)
    sub = ' & '.join('Wiki & C4' for _ in models)
    lines = ['% generated by experiments/main_ppl/analyze.py (' + variant + '); * simulated (fake quant), W4A4 under the '
             'method\'s own per-block rules; bold: best natively executable result per column; '
             '\\dag: not significantly better than FourOverSix (paired, 2 SE)',
             '\\begin{tabular}{l' + 'cc' * len(models) + 'c}', '\\toprule',
             f'Method & {head} & Loss rec. \\\\', f' & {sub} & \\\\', '\\midrule']
    for row, title in ROWS:
        cells = []
        for model in models:
            for c in CORPORA:
                r = data[model]['rows'].get(row)
                if r is None:
                    cells.append('--')
                    continue
                s = f'{r[c]["ppl"]:.2f}'
                if row in OURS:
                    k = cmp.get((model, row, 'fo6', c))
                    if k is not None and not (k['dnll'] + 2 * k['se'] < 0):
                        s += '$^\\dag$'
                if best[(model, c)] == row:
                    s = f'\\textbf{{{s}}}'
                cells.append(s)
        lr = loss_recovered(data, models, row)
        cells.append('--' if lr is None else f'{100 * lr:.1f}\\%')
        lines.append(f'{title} & ' + ' & '.join(cells) + ' \\\\')
        if row in ('fo6', 'zou'):
            lines.append('\\midrule')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines) + '\n'


def latex_dnll(data, cmp, models):
    rows = [r for r, _ in ROWS if r not in ('bf16', 'fo6')]
    titles = dict(ROWS)
    head = ' & '.join(f'\\multicolumn{{2}}{{c}}{{{TITLES[m]}}}' for m in models)
    lines = ['% generated by experiments/main_ppl/analyze.py: paired ΔNLL (nats/token) vs native FourOverSix, ± 2 SE over '
             'windows; for the simulated rows (*) also vs simulated FourOverSix, in brackets',
             '\\begin{tabular}{l' + 'cc' * len(models) + '}', '\\toprule', f'Method & {head} \\\\',
             ' & ' + ' & '.join('Wiki & C4' for _ in models) + ' \\\\', '\\midrule']
    for row in rows:
        cells = []
        for model in models:
            for c in CORPORA:
                k = cmp.get((model, row, 'fo6', c))
                if k is None:
                    cells.append('--')
                    continue
                s = f'{1000 * k["dnll"]:+.1f} $\\pm$ {2000 * k["se"]:.1f}'
                kf = cmp.get((model, row, 'fo6-fake', c))
                if kf is not None:
                    s += f' [{1000 * kf["dnll"]:+.1f} $\\pm$ {2000 * kf["se"]:.1f}]'
                cells.append(s)
        lines.append(f'{titles[row]} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '% values in milli-nats per token (ΔNLL × 1000)']
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / 'results' / 'main_ppl')
    args = ap.parse_args()
    models = sorted({m for v in VARIANTS.values() for m in v}, key=list(TITLES).index)
    data = load(models)
    cmp = comparisons(data)
    res = dict(models={m: dict(title=TITLES[m], rows={r: {c: data[m]['rows'][r][c]['ppl'] for c in CORPORA}
                                                       for r in data[m]['rows']}, sources=data[m]['sources'],
                               windows={c: len(data[m]['windows'][c]) for c in CORPORA} if data[m]['windows'] else None)
                       for m in models},
               dnll=[dict(model=m, row=r, ref=ref, corpus=c, **v) for (m, r, ref, c), v in sorted(cmp.items())],
               loss_recovered={v: {r: loss_recovered(data, ms, r) for r, _ in ROWS if r != 'bf16'} for v, ms in VARIANTS.items()})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'main_ppl.json').write_text(json.dumps(res, indent=1) + '\n')
    with open(args.out / 'main_ppl.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'row', 'corpus', 'ppl', 'dnll_vs_fo6', 'se_vs_fo6', 'dnll_vs_fo6_fake', 'se_vs_fo6_fake',
                    'dnll_vs_nvfp4', 'se_vs_nvfp4', 'windows'])
        for m in models:
            for r in data[m]['rows']:
                for c in CORPORA:
                    a, b, n = (cmp.get((m, r, ref, c), {}) for ref in ('fo6', 'fo6-fake', 'nvfp4'))
                    w.writerow([m, r, c, data[m]['rows'][r][c]['ppl'], a.get('dnll'), a.get('se'), b.get('dnll'), b.get('se'),
                                n.get('dnll'), n.get('se'), len(data[m]['rows'][r][c]['nll'])])
    for v, ms in VARIANTS.items():
        if all(data[m]['rows'] for m in ms):
            (args.out / f'table_main_{v}.tex').write_text(latex_main(data, cmp, ms, v))
    (args.out / 'table_dnll_appendix.tex').write_text(latex_dnll(data, cmp, [m for m in models if data[m]['rows']]))
    print(json.dumps(res['loss_recovered'], indent=1))


if __name__ == '__main__':
    main()
