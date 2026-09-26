"""Paired per-window comparison of trained MixFP4 maps (run_train_map.py) against
FourOverSix and the optimized multi-round KL maps on the released Llama windows."""
import argparse
import json
import math
from pathlib import Path

MAIN = Path('/home/u4320956/NVFP4-RaZeR')
OPT = MAIN / 'results/mixfp4_potential/optimized'
TRAIN = Path('/work/u4320956/mixfp4_potential/train_map')
WORK = Path('/work/u4320956/mixfp4_potential')
FOUR_OVER_SIX = {'llama8b': MAIN / 'results/kse_paper/job_336566/llama8b/report.json',
                 'qwen27b': MAIN / 'results/kse_paper/job_336969/qwen27b/report.json',
                 # Instruct: a zero-epoch run_train_map.py job, i.e. the all-E2M1 map.
                 'llama8b_ins': TRAIN / 'ins_four_over_six/report.json',
                 'llama1b_ins': TRAIN / 'ins1b_four_over_six/report.json'}
MULTIROUND = {'llama8b': {'256x64': OPT / 'multiround_256x64_kl_batched/report.json',
                          '8x64': OPT / 'opt_llama8b_8x64_kl/report.json'},
              'qwen27b': {'256x64': OPT / 'opt_qwen27b_256x64_kl/report.json',
                          '8x64': OPT / 'opt_qwen27b_8x64_kl/report.json'},
              'llama8b_ins': {'256x64': WORK / 'ins_llama8b_256x64_kl/report.json',
                              '8x64': WORK / 'ins_llama8b_8x64_kl/report.json'}}


def four_over_six(model):
    r = json.loads(FOUR_OVER_SIX[model].read_text())
    if 'four_over_six' in r['evaluation']:
        return r['evaluation']['four_over_six']
    assert r['status'] == 'complete' and r['final_e0m3_units'] == 0 and r['args']['epochs'] == 0
    return r['evaluation']


def paired(a, b):
    assert len(a) == len(b)
    d = [x - y for x, y in zip(a, b)]
    mean = sum(d) / len(d)
    return mean, 2 * math.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1) / len(d))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=tuple(FOUR_OVER_SIX), default='llama8b')
    ap.add_argument('runs', nargs='+', type=Path)
    args = ap.parse_args()
    fo = four_over_six(args.model)
    multi = {u: json.loads(p.read_text()) for u, p in MULTIROUND.get(args.model, {}).items()}
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
        assert r['args']['model'] == args.model, run
        ev, unit = r['evaluation'], r['args']['unit']
        vs_fo = [paired(ev[d]['nll'], fo[d]['nll']) for d in ('wiki', 'c4')]
        vs_mr = ' / '.join(f'{m:+.5f}±{s:.5f}' for m, s in (
            paired(ev[d]['nll'], multi[unit]['evaluation'][d]['nll']) for d in ('wiki', 'c4'))) if unit in multi else '—'
        print(f'| {run.name} | {r["final_e0m3_units"]:,} | {r["final_dev"]["kl"]:.5f} | {ev["wiki"]["ppl"]:.6f} | '
              f'{ev["c4"]["ppl"]:.6f} | {ev["wiki"]["ppl"] - fo["wiki"]["ppl"]:+.4f} / {ev["c4"]["ppl"] - fo["c4"]["ppl"]:+.4f} | '
              + ' / '.join(f'{m:+.5f}±{s:.5f}' for m, s in vs_fo) + f' | {vs_mr} |')


if __name__ == '__main__':
    main()
