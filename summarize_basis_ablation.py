"""Compare candidate bases at matched selection machinery.

e0m3 and alpha share the FourOverSix baseline, so their deltas are directly
comparable. type_pure has the NVFP4 alpha=1 baseline, so it is compared on
absolute PPL. Direction lengths are reported first, because a basis with a
shorter direction is taking a smaller step and a null on it is ambiguous.
"""
import argparse
import json
import math
from pathlib import Path

HEAD = ('fixed256_math_code128', 'adaptive_math_code128')
BASIS_ORDER = ('e0m3', 'alpha', 'type_pure')


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    mean = sum(d) / len(d)
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1) / len(d))
    return mean, 2 * se


def load(stem):
    cal = json.loads((stem / 'calibration' / 'report.json').read_text())
    ev = json.loads((stem / 'evaluation' / 'report.json').read_text())
    assert cal['status'] == 'complete' and ev['status'] == 'complete', stem
    assert cal['basis'] == ev['basis']
    return cal, ev


def direction_row(cal):
    d = cal['direction']
    tot = {k: sum(v[k] for v in d.values())
           for k in ('direction_sq', 'baseline_error_sq', 'alternative_error_sq',
                     'weight_sq', 'tiles', 'moved_tiles')}
    return dict(rel_direction=(tot['direction_sq'] / tot['weight_sq']) ** .5,
                rel_base_err=(tot['baseline_error_sq'] / tot['weight_sq']) ** .5,
                rel_alt_err=(tot['alternative_error_sq'] / tot['weight_sq']) ** .5,
                moved=tot['moved_tiles'], tiles=tot['tiles'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='results/basis_ablation')
    ap.add_argument('--job', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    arms = {}
    for stem in sorted(root.glob(f'{args.job}_*')):
        _, model, basis = stem.name.split('_', 2)
        arms[(model, basis)] = load(stem)
    assert arms, 'no arms found'
    models = sorted({m for m, _ in arms})

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = {'job': args.job, 'arms': {}}
    lines = ['# Basis ablation: E0M3, or task-gradient selection over any legal knob?', '',
             'Identical data, teacher, CE/KL two-SE rule, counts and held-out sets across arms;',
             'only the scored direction differs (`quantize/basis.py`). `e0m3` is the reported',
             'direction, re-run through the same code path so the arms are comparable.', '']

    lines += ['## Direction length', '',
              'A shorter direction is a smaller step, not necessarily a worse one.', '',
              '| model | basis | baseline | alternative | \\|d\\|/\\|W\\| | moved tiles | base err | alt err |',
              '|---|---|---|---|---:|---:|---:|---:|']
    for model in models:
        for basis in BASIS_ORDER:
            if (model, basis) not in arms:
                continue
            cal, _ = arms[(model, basis)]
            row = direction_row(cal)
            bd = cal['basis_definition']
            lines.append(f"| {model} | {basis} | {bd['baseline']} | {bd['alternative']} | "
                         f"{row['rel_direction']:.6f} | {row['moved']:,}/{row['tiles']:,} | "
                         f"{row['rel_base_err']:.6f} | {row['rel_alt_err']:.6f} |")
            data['arms'].setdefault(model, {}).setdefault(basis, {})['direction'] = row
    lines.append('')

    for model in models:
        lines += [f'## {model}', '',
                  '`Δ` is against **that arm\'s own baseline**; paired ΔNLL ± 2 SE is descriptive.', '',
                  '| basis | policy | blocks | wiki PPL | Δ wiki | wiki ΔNLL ±2SE | c4 PPL | Δ c4 | c4 ΔNLL ±2SE |',
                  '|---|---|---:|---:|---:|---|---:|---:|---|']
        for basis in BASIS_ORDER:
            if (model, basis) not in arms:
                continue
            cal, ev = arms[(model, basis)]
            e = ev['evaluation']
            store = data['arms'][model][basis]
            store['baseline'] = {d: e['four_over_six'][d]['ppl'] for d in e['four_over_six']}
            store['policies'] = {}
            for policy in ['four_over_six', *HEAD, 'weight_mse']:
                if policy not in e:
                    continue
                blocks = ev['block_statistics'][policy]['selected_blocks']
                cells, rec = [], {'blocks': blocks}
                for domain in ('wiki', 'c4'):
                    if domain not in e[policy]:
                        cells += ['—', '—', '—']; continue
                    ppl = e[policy][domain]['ppl']
                    ref = e['four_over_six'][domain]['ppl']
                    rec[domain] = ppl; rec[f'{domain}_delta'] = ppl - ref
                    if policy == 'four_over_six':
                        cells += [f'{ppl:.6f}', '—', '—']
                    else:
                        m, s = paired(e[policy][domain]['nll'], e['four_over_six'][domain]['nll'])
                        rec[f'{domain}_nll'] = [m, s]
                        cells += [f'{ppl:.6f}', f'{ppl-ref:+.6f}', f'{m:+.6f} ±{s:.6f}']
                store['policies'][policy] = rec
                lines.append(f'| {basis} | {policy} | {blocks:,} | ' + ' | '.join(cells) + ' |')
        lines.append('')

        lines += ['### Same number, absolute PPL (baselines differ across arms)', '',
                  '| policy | domain | ' + ' | '.join(BASIS_ORDER) + ' |',
                  '|---|---|' + '---:|' * len(BASIS_ORDER)]
        for policy in ['four_over_six', *HEAD, 'weight_mse']:
            for domain in ('wiki', 'c4'):
                cells = []
                for basis in BASIS_ORDER:
                    v = data['arms'].get(model, {}).get(basis, {}).get('policies', {}).get(policy, {})
                    cells.append(f"{v[domain]:.6f}" if domain in v else '—')
                lines.append(f'| {policy} | {domain} | ' + ' | '.join(cells) + ' |')
        lines.append('')

    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (out / 'summary.json').write_text(json.dumps(data, indent=2) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
