"""Paired per-window comparison of trained MixFP4 maps (run_train_map.py) against
FourOverSix and the optimized multi-round KL maps on the released Llama windows."""
import argparse
import json
import math
from pathlib import Path

MAIN = Path('/home/u4320956/NVFP4-RaZeR')
FOUR_OVER_SIX = MAIN / 'results/kse_paper/job_336566/llama8b/report.json'
MULTIROUND = {'256x64': MAIN / 'results/mixfp4_potential/optimized/multiround_256x64_kl_batched/report.json',
              '8x64': MAIN / 'results/mixfp4_potential/optimized/opt_llama8b_8x64_kl/report.json'}


def paired(a, b):
    assert len(a) == len(b)
    d = [x - y for x, y in zip(a, b)]
    mean = sum(d) / len(d)
    return mean, 2 * math.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1) / len(d))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='+', type=Path)
    args = ap.parse_args()
    fo = json.loads(FOUR_OVER_SIX.read_text())['evaluation']['four_over_six']
    multi = {u: json.loads(p.read_text()) for u, p in MULTIROUND.items()}
    print('| Run | E0M3 tiles | final dev KL | WikiText | C4 | ΔPPL vs FourOverSix (wiki / c4) | '
          'ΔNLL vs FourOverSix ±2SE (wiki / c4) | ΔNLL vs multi-round ±2SE (wiki / c4) |')
    print('|---|---:|---:|---:|---:|---|---|---|')
    for unit, r in multi.items():
        ev = r['evaluation']
        cells = [f'{paired(ev[d]["nll"], fo[d]["nll"])[0]:+.5f}±{paired(ev[d]["nll"], fo[d]["nll"])[1]:.5f}' for d in ('wiki', 'c4')]
        print(f'| multi-round {unit} | {r["final_e0m3_units"]:,} | {r["final_dev"]["kl"]:.5f} | {ev["wiki"]["ppl"]:.6f} | '
              f'{ev["c4"]["ppl"]:.6f} | {ev["wiki"]["ppl"] - fo["wiki"]["ppl"]:+.4f} / {ev["c4"]["ppl"] - fo["c4"]["ppl"]:+.4f} | '
              f'{" / ".join(cells)} | — |')
    for run in args.runs:
        r = json.loads((run / 'report.json').read_text())
        assert r['status'] == 'complete', run
        ev, unit = r['evaluation'], r['args']['unit']
        vs_fo = [paired(ev[d]['nll'], fo[d]['nll']) for d in ('wiki', 'c4')]
        vs_mr = [paired(ev[d]['nll'], multi[unit]['evaluation'][d]['nll']) for d in ('wiki', 'c4')]
        print(f'| {run.name} | {r["final_e0m3_units"]:,} | {r["final_dev"]["kl"]:.5f} | {ev["wiki"]["ppl"]:.6f} | '
              f'{ev["c4"]["ppl"]:.6f} | {ev["wiki"]["ppl"] - fo["wiki"]["ppl"]:+.4f} / {ev["c4"]["ppl"] - fo["c4"]["ppl"]:+.4f} | '
              + ' / '.join(f'{m:+.5f}±{s:.5f}' for m, s in vs_fo) + ' | '
              + ' / '.join(f'{m:+.5f}±{s:.5f}' for m, s in vs_mr) + ' |')


if __name__ == '__main__':
    main()
