"""Potential ladder: every evaluated arm paired against FourOverSix and 8x64 k=3 per window."""
import json
import math
import sys
from pathlib import Path

PPL_ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '/work/u4320956/mixfp4_potential/ppl')
REFERENCE = Path('/work/u4320956/mapped_gptq/llama8b_20260923')


def pair(a, b):
    x = [i - j for i, j in zip(a, b)]
    n = len(x); m = sum(x) / n
    se = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1) / n)
    return dict(delta_nll=m, two_se=2 * se)


def label(report, directory):
    if 'objective' in report and 'unit' in report:
        return f'multiround {report["unit"]} {report["objective"]}'
    if report['policy'] == 'rtn_rule':
        objective, k, rows = report['rule'].split(':')
        return f'{rows}x64 {objective} k={k}'
    if report['policy'] == 'rtn_fine1x16':
        rule = report['fine_rule']; k = rule.split('_k')[1] if '_k' in rule else '3'
        return f'1x16 {rule.split("_k")[0]} k={k}'
    return directory.name


def main():
    arms = {}
    for path in (sorted(PPL_ROOT.glob('*/report.json')) + sorted(REFERENCE.glob('rtn_*/report.json'))
                 + sorted(PPL_ROOT.parent.glob('multiround_256x64_*/report.json'))):
        r = json.loads(path.read_text())
        if r['status'] != 'complete' or 'evaluation' not in r or 'c4' not in r['evaluation']:
            continue
        arms[label(r, path.parent)] = r
    base = arms['rtn_four_over_six']['evaluation']
    rows = []
    for name, r in arms.items():
        e = r['evaluation']
        row = dict(arm=name, e0m3_units=r.get('elected_tiles', r.get('final_e0m3_units')),
                   wiki=e['wiki']['ppl'], c4=e['c4']['ppl'],
                   d_wiki=e['wiki']['ppl'] - base['wiki']['ppl'], d_c4=e['c4']['ppl'] - base['c4']['ppl'],
                   paired_vs_four_over_six={d: pair(e[d]['nll'], base[d]['nll']) for d in ('wiki', 'c4')})
        rows.append(row)
    rows.sort(key=lambda x: x['d_wiki'] + x['d_c4'])
    (PPL_ROOT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'{"arm":24s} {"units":>9s} {"wiki":>9s} {"dwiki":>9s} {"c4":>9s} {"dc4":>9s}  paired dNLL±2SE wiki | c4')
    for x in rows:
        p = x['paired_vs_four_over_six']
        print(f'{x["arm"]:24s} {str(x["e0m3_units"]):>9s} {x["wiki"]:9.6f} {x["d_wiki"]:+9.5f} {x["c4"]:9.6f} {x["d_c4"]:+9.5f}  '
              f'{p["wiki"]["delta_nll"]:+.5f}±{p["wiki"]["two_se"]:.5f} | {p["c4"]["delta_nll"]:+.5f}±{p["c4"]["two_se"]:.5f}')


if __name__ == '__main__':
    main()
