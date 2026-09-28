"""The registered checks on one model's evaluation (PROTOCOL.md), run by the queue right after it; exit 1 on a failure.

- Regression: BF16, NVFP4, FourOverSix and OURS 8x64 / 16x64 equal the Parts 2-3 evaluation window for window (and, for
  Llama, SCALE equals Task 2's).
- Check 4: QAT NativeLinear (c) against fake (c) of the QAT weights, B2's criterion |mean ΔNLL| <= 2 SE per corpus.

    python results/unified_baselines/check_eval.py MODEL EVAL_REPORT
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tm_opt'))
from analyze_items import paired  # noqa: E402

D = ('wiki', 'c4')


def main():
    model, report = sys.argv[1], Path(sys.argv[2])
    ev = json.loads(report.read_text())['evaluations']
    ref = json.loads((HERE.parent / 'deploy_eval' / 'runs' / model / 'report.json').read_text())['evaluations']
    pairs = [('BF16', ref['BF16']), ('NVFP4', ref['NVFP4']), ('FourOverSix', ref['FourOverSix']), ('ours-8x64', ref['tc-8x64']),
             ('ours-16x64', ref['tc-16x64'])]
    if model == 'llama8b':
        t2 = json.loads((HERE.parent / 'scale_additivity' / 'runs' / 'llama8b' / 'report.json').read_text())['evaluations']
        pairs.append(('scale', t2['scale']))
    failures = []
    for label, r in pairs:
        same = all(ev[label]['evaluation'][d]['nll'] == r['evaluation'][d]['nll'] for d in D)
        print(f'regression {label}: {"equal" if same else "DIFFERENT"}')
        if not same:
            failures.append(f'regression {label}')
    for d in D:
        p = paired(ev['qat']['evaluation'][d]['nll'], ev['qat-fake']['evaluation'][d]['nll'])
        ok = abs(p['mean']) <= p['two_se']
        print(f"check 4 {d}: native - fake {p['mean']:+.6f} +- {p['two_se']:.6f} -> {'negligible' if ok else 'SIGNIFICANT'}")
        if not ok:
            failures.append(f'check 4 {d}')
    print('CHECKS', 'PASS' if not failures else f'FAIL {failures}')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
