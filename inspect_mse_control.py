"""How many tiles would a pure weight-MSE criterion elect, and what did it score?

The gradient rule and the weight-MSE rule pick tiles for different reasons: one
minimises cross-entropy on text, the other minimises reconstruction error of the
weights. Their selected counts and measured perplexities separate "better number
format" from "trained on the loss".
"""
import json
from pathlib import Path

ROOT = Path('/home/u4320956/NVFP4-RaZeR')
CAL = ROOT / 'results/math_code_adaptive/calibration_333779_qwen4b/report.json'


def main():
    r = json.loads(CAL.read_text())
    bs = r.get('block_statistics', {})
    total = None
    for policy, s in sorted(bs.items()):
        total = s.get('total_type_blocks', total)
        print(f'{policy:24s} selected={s.get("selected_blocks"):>8} '
              f'eligible={s.get("eligible_negative_score_blocks", "-"):>8} '
              f'fraction={s.get("selected_fraction", 0):.6f}')
    print(f'\ntotal type blocks: {total}')
    ev = r.get('evaluation', {})
    print('\nevaluated policies:', ', '.join(sorted(ev)))
    for policy in sorted(ev):
        cells = {d: round(v['ppl'], 6) for d, v in ev[policy].items() if isinstance(v, dict)}
        print(f'  {policy:24s} {cells}')


if __name__ == '__main__':
    main()
