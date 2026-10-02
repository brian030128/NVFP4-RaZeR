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
ROWS = [('bf16', 'BF16'), ('nvfp4', 'NVFP4'), ('fo6', 'FourOverSix'), ('if4', 'IF4 (Cook et al.) 1x16 (W+A)$^\\ast$'),
        ('zou', 'MixFP4 (Zou et al.) 1x16 (W+A)$^\\ast$'),
        ('if4w', 'IF4 (Cook et al.) 1x16 (W) + FO6 act$^\\ast$'), ('zouw', 'MixFP4 (Zou et al.) 1x16 (W) + FO6 act$^\\ast$'),
        ('ours-8x64', 'FlipQuant (ours) 8x64'),
        ('ours-16x64', 'FlipQuant (ours) 16x64'), ('ours-256x64', 'FlipQuant (ours) 256x64')]
NATIVE = ('nvfp4', 'fo6', 'ours-8x64', 'ours-16x64', 'ours-256x64')
OURS = ('ours-8x64', 'ours-16x64', 'ours-256x64')
SIMULATED = ('if4', 'zou', 'if4w', 'zouw')
CORPORA = ('wiki', 'c4')
INCOMPLETE = []
SCOPE = 'restored'
# deviation 3 cut the table to BF16 / NVFP4 / FourOverSix; amendment 2 restored the IF4 / MixFP4 rows (both variants)
SCOPE_ROWS = {'baselines': ('bf16', 'nvfp4', 'fo6'),
              'restored': ('bf16', 'nvfp4', 'fo6', 'if4', 'zou', 'if4w', 'zouw'),
              'full': None}


def set_scope(scope):
    """The rows of the table: 'restored' (amendment 2: the baselines and the simulated IF4 / MixFP4 rows; FlipQuant
    deferred), 'baselines' (deviation 3) or 'full' (the registered table)."""
    global SCOPE, ROWS, NATIVE, OURS, SIMULATED
    SCOPE = scope
    keep = SCOPE_ROWS[scope]
    if keep is not None:
        ROWS = [r for r in ROWS if r[0] in keep]
        NATIVE, OURS = ('nvfp4', 'fo6'), ()
        SIMULATED = tuple(r for r in SIMULATED if r in keep)


def with_fake_reference():
    """fo6-fake, the simulated rows' like-for-like reference (appendix), is loaded whenever simulated rows are in scope."""
    return bool(SIMULATED)


def deferred(models):
    """Every record outside the table's rows: listed, never tabulated (deviation 3)."""
    out = []
    for sub, kind in (('ppl', 'row'), ('diagnostics', 'diagnostic'), ('smoke', 'smoke (2 windows per corpus)')):
        for rep in sorted((OUT_RUNS / sub).glob('*/*/report.json')):
            model, row = rep.parent.parent.name, rep.parent.name
            in_scope = SCOPE_ROWS[SCOPE] is None or row in SCOPE_ROWS[SCOPE] or (row == 'fo6-fake' and with_fake_reference())
            if sub == 'ppl' and (in_scope or row.endswith('-recheck')):
                continue
            status = json.loads(rep.read_text()).get('status')
            out.append(dict(model=model, row=row, kind=kind, status=status, record=str(rep)))
    return out


def record(model, row):
    path = (PAPER_PPL if model in REUSED and row in REUSED_ROWS else OUT_RUNS / 'ppl') / model / row / 'report.json'
    if not path.exists():
        return None, path
    r = json.loads(path.read_text())
    if r['status'] != 'complete':
        print(f'skipped (status {r["status"]}): {path}')
        INCOMPLETE.append(str(path))
        return None, path
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
        for row in [r for r, _ in ROWS] + (['fo6-fake'] if with_fake_reference() else []):
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


def latex_main(data, cmp, models, variant, decimals=2):
    best = {}
    for model in models:
        for c in CORPORA:
            vals = [(data[model]['rows'][r][c]['ppl'], r) for r in NATIVE if r in data[model]['rows']]
            best[(model, c)] = min(vals)[1] if vals else None
    head = ' & '.join(f'\\multicolumn{{2}}{{c}}{{{TITLES[m]}}}' for m in models)
    sub = ' & '.join('Wiki & C4' for _ in models)
    note = {'full': '* simulated (fake quant), W4A4 under the method\'s own per-block rules; bold: best natively executable '
                    'result per column; \\dag: not significantly better than FourOverSix (paired, 2 SE)',
            'baselines': 'BF16 / NVFP4 / FourOverSix only (deviation 3); bold: the lower PPL of NVFP4 and FourOverSix per '
                         'column; loss rec.: share of NVFP4\'s log-PPL loss vs BF16 removed, summed over the 12 pairs',
            'restored': '* simulated (fake quant, BF16 GEMM): (W+A) the rule on weights and activations (per-token tensor '
                        'scale), (W) + FO6 act the rule on the weights with per-token FourOverSix activations; FlipQuant '
                        'deferred (deviation 3); bold: the lower PPL of the native NVFP4 and FourOverSix per column; loss '
                        'rec.: share of NVFP4\'s log-PPL loss vs BF16 removed, summed over the 12 model-corpus pairs'}[SCOPE]
    lines = ['% generated by experiments/main_ppl/analyze.py (' + variant + '); ' + note,
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
                s = f'{r[c]["ppl"]:.{decimals}f}'
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
        if row in ('fo6', 'zouw') and row != ROWS[-1][0]:
            lines.append('\\midrule')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines) + '\n'


def latex_dnll(data, cmp, models):
    rows = [r for r, _ in ROWS if r not in ('bf16', 'fo6')]
    titles = dict(ROWS)
    both = {'mistral7b': ' (base)', 'mistral7b_ins': ' (Instruct)'} if {'mistral7b', 'mistral7b_ins'} <= set(models) else {}
    head = ' & '.join(f'\\multicolumn{{2}}{{c}}{{{TITLES[m]}{both.get(m, "")}}}' for m in models)
    lines = ['% generated by experiments/main_ppl/analyze.py: paired ΔNLL (nats per token = Δ log PPL) vs native '
             'FourOverSix, ± 2 SE over windows' + ('; for the simulated rows (*) also vs simulated FourOverSix, in brackets'
                                                   if with_fake_reference() else ''),
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
                s = f'{k["dnll"]:+.4f} $\\pm$ {2 * k["se"]:.4f}'
                kf = cmp.get((model, row, 'fo6-fake', c))
                if kf is not None:
                    s += f' [{kf["dnll"]:+.4f} $\\pm$ {2 * kf["se"]:.4f}]'
                cells.append(s)
        lines.append(f'{titles[row]} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines) + '\n'


def markdown(data, cmp, res):
    """tables.md: the PPLs (4 decimals), the paired ΔNLL vs FourOverSix (± 2 SE, * = |Δ| > 2 SE) and loss recovered."""
    titles = {r: t.replace('$^\\ast$', '*') for r, t in dict(ROWS, **{'fo6-fake': 'FourOverSix (simulated, appendix)'}).items()}
    out = ['# Main PPL table: all numbers', '', '### Perplexity (native rows: NativeLinear (c); * simulated: fake (c))', '']
    models = [m for m in data if data[m]['rows']]
    out.append('| row | ' + ' | '.join(f'{TITLES[m]}{" (base)" if m == "mistral7b" else ""} {c}' for m in models for c in ('Wiki', 'C4')) + ' |')
    out.append('|---|' + '---:|' * (2 * len(models)))
    for row in [r for r, _ in ROWS] + (['fo6-fake'] if with_fake_reference() else []):
        cells = [f'{data[m]["rows"][row][c]["ppl"]:.4f}' if row in data[m]['rows'] else '--' for m in models for c in CORPORA]
        out.append(f'| {titles[row]} | ' + ' | '.join(cells) + ' |')
    out += ['', '### Paired ΔNLL vs native FourOverSix (nats per token), ± 2 SE; * = |Δ| > 2 SE', '']
    out.append('| row | ' + ' | '.join(f'{TITLES[m]}{" (base)" if m == "mistral7b" else ""} {c}' for m in models for c in ('Wiki', 'C4')) + ' |')
    out.append('|---|' + '---:|' * (2 * len(models)))
    for row in [r for r, _ in ROWS if r not in ('bf16', 'fo6')] + (['fo6-fake'] if with_fake_reference() else []):
        cells = []
        for m in models:
            for c in CORPORA:
                k = cmp.get((m, row, 'fo6', c))
                cells.append('--' if k is None else f'{k["dnll"]:+.4f} ± {2 * k["se"]:.4f}' + (' *' if abs(k['dnll']) > 2 * k['se'] else ''))
        out.append(f'| {titles[row]} | ' + ' | '.join(cells) + ' |')
    out += ['', '### Loss recovered (12 model-corpus pairs; NVFP4 0 %, BF16 100 %)', '', '| row | Instruct variant | base variant |',
            '|---|---:|---:|']
    for row, _ in ROWS[1:]:
        v = [res['loss_recovered'][k].get(row) for k in ('instruct', 'base')]
        out.append(f'| {titles[row]} | ' + ' | '.join('--' if x is None else f'{100 * x:.1f} %' for x in v) + ' |')
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / 'results' / 'main_ppl')
    ap.add_argument('--decimals', type=int, default=2, help='PPL decimals in the main LaTeX tables')
    ap.add_argument('--scope', choices=('restored', 'baselines', 'full'), default='restored',
                    help='restored (amendment 2): baselines + IF4 / MixFP4 in both variants; baselines: BF16 / NVFP4 / '
                         'FourOverSix (deviation 3); full: every registered row')
    args = ap.parse_args()
    set_scope(args.scope)
    models = sorted({m for v in VARIANTS.values() for m in v}, key=list(TITLES).index)
    data = load(models)
    cmp = comparisons(data)
    res = dict(models={m: dict(title=TITLES[m], rows={r: {c: data[m]['rows'][r][c]['ppl'] for c in CORPORA}
                                                       for r in data[m]['rows']}, sources=data[m]['sources'],
                               windows={c: len(data[m]['windows'][c]) for c in CORPORA} if data[m]['windows'] else None)
                       for m in models},
               dnll=[dict(model=m, row=r, ref=ref, corpus=c, **v) for (m, r, ref, c), v in sorted(cmp.items())],
               loss_recovered={v: {r: loss_recovered(data, ms, r) for r, _ in ROWS if r != 'bf16'} for v, ms in VARIANTS.items()},
               incomplete_records_skipped=INCOMPLETE, scope=SCOPE,
               deferred=deferred(models) if SCOPE != 'full' else None)
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
            (args.out / f'table_main_{v}.tex').write_text(latex_main(data, cmp, ms, v, args.decimals))
    (args.out / 'table_dnll_appendix.tex').write_text(latex_dnll(data, cmp, [m for m in models if data[m]['rows']]))
    (args.out / 'tables.md').write_text(markdown(data, cmp, res))
    if SCOPE != 'full':
        lines = ['# Deferred / partial records: not part of this table', '',
                 'Kept as recorded (results/main_ppl/runs, diagnostics, smoke). Deviation 3 deferred the FlipQuant rows '
                 '(amendment 2 restored the IF4 / MixFP4 rows); they will be superseded by the new calibration.', '',
                 '| model | row | kind | status |', '|---|---|---|---|']
        lines += [f"| {d['model']} | {d['row']} | {d['kind']} | {d['status']} |" for d in res['deferred']]
        (args.out / 'deferred.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(res['loss_recovered'], indent=1))


if __name__ == '__main__':
    main()
