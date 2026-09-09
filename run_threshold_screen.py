"""Screen count rules that need no per-model search, using saved score tables.

Every monotone threshold on the election score u_j selects a prefix of the same
ranking the cap truncates, so a rule can be screened by the count it produces
without evaluating a model. This writes, per model, the count each candidate
rule selects, next to the counts already measured in results/cap_sweep.

Candidates:
  kSE    max(mean_CE + k SE_CE, mean_KL + k SE_KL) < 0  -- the shipped rule with
         the constant 2 replaced by k; adapts to each model's score noise.
  relmin u_j < c * min_j u_j                            -- keep scores within a
         fraction of the strongest tile; scale free.
  mass   smallest prefix whose summed u reaches p of the total eligible mass.
"""
import argparse
import json
import math
from pathlib import Path

import torch

POOLED = {
    'olmo1b': 'results/pooled_confirmation/model_332349_olmo1b',
    'pythia14b': 'results/pooled_confirmation/model_332349_pythia14b',
    'qwen4b': 'results/pooled_scale/model_332389_qwen4b',
    'llama8b': 'results/pooled_scale/model_332389_llama8b',
}
K_VALUES = (2, 3, 4, 5, 6, 8, 10, 12)
REL_VALUES = (0.5, 0.3, 0.2, 0.1, 0.05, 0.02)
MASS_VALUES = (0.3, 0.5, 0.7, 0.9)


def stats(path):
    """Per-tile mean and standard error for both objectives."""
    record = torch.load(path, map_location='cpu', weights_only=True)
    out = {}
    for key in ('ce', 'kl'):
        x = record[key].flatten(0, 1)
        assert x.shape[0] == 192
        out[key] = (x.mean(0), x.std(0, unbiased=True) / math.sqrt(x.shape[0]))
        del x
    del record
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage-root', default='/home/u4320956/NVFP4-RaZeR')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.stage_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    for model, rel in POOLED.items():
        s = stats(root / rel / 'scores.pt')
        base = torch.maximum(s['ce'][0] + 2 * s['ce'][1], s['kl'][0] + 2 * s['kl'][1])
        total = int(base.numel())
        eligible = int((base < 0).sum())
        entry = dict(total_tiles=total, eligible_at_2se=eligible, rules={})

        for k in K_VALUES:
            u = torch.maximum(s['ce'][0] + k * s['ce'][1], s['kl'][0] + k * s['kl'][1])
            entry['rules'][f'{k}SE'] = int((u < 0).sum())
            del u
        neg = base[base < 0]
        floor = float(neg.min()) if neg.numel() else 0.0
        for c in REL_VALUES:
            entry['rules'][f'relmin{c}'] = int((base < c * floor).sum())
        if neg.numel():
            ordered = torch.sort(neg).values
            cumulative = torch.cumsum(ordered, 0)
            for p in MASS_VALUES:
                entry['rules'][f'mass{p}'] = int((cumulative <= p * float(cumulative[-1])).sum()) + 1
        report[model] = entry
        print(f'{model}: {total:,} tiles, {eligible:,} eligible at 2SE', flush=True)
        for rule, n in entry['rules'].items():
            print(f'   {rule:<10} {n:>10,}', flush=True)
        del s, base

    (out / 'counts.json').write_text(json.dumps(report, indent=2) + '\n')

    measured = {}
    for model in POOLED:
        p = root / f'results/cap_sweep/model_335887_{model}/report.json'
        if p.exists():
            d = json.loads(p.read_text())
            measured[model] = {d['election'][k]['selected']: d['contrasts'][k]['ppl_delta']
                               for k in d['counts'] if k != 'four_over_six'}

    lines = ['# Search-free count rules screened against the measured curve', '',
             'Any monotone threshold on the election score selects a prefix of the same ranking a',
             'cap truncates, so each rule is screened here by the count it produces. Counts come',
             'from the saved 192-sequence pooled score tables; no model is evaluated. The measured',
             'ΔPPL column is the held-out C4 value at the nearest measured count in',
             '[cap_sweep](../cap_sweep/REPORT.md) and is indicative only.', '']
    for model, entry in report.items():
        lines += [f'## {model}', '',
                  f'{entry["total_tiles"]:,} type blocks; {entry["eligible_at_2se"]:,} eligible at '
                  f'the shipped 2SE threshold.', '',
                  '| Rule | Selected | Nearest measured count | ΔPPL there |', '|---|---:|---:|---:|']
        for rule, n in entry['rules'].items():
            if model in measured and measured[model]:
                near = min(measured[model], key=lambda c: abs(c - n))
                lines.append(f'| {rule} | {n:,} | {near:,} | {measured[model][near]:+.6f} |')
            else:
                lines.append(f'| {rule} | {n:,} | — | — |')
        lines.append('')
    lines += ['## Reading', '',
              'A usable search-free rule must land near each model\'s measured optimum without',
              'being told the optimum. Counts far above it are the harmful regime; counts far',
              'below it leave most of the gain unclaimed. Selecting a rule by looking at this',
              'table would be selection on the evaluation set, so any rule chosen here must be',
              'fixed in advance and validated on models and protocols not screened.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('WROTE ' + str(out / 'REPORT.md'), flush=True)


if __name__ == '__main__':
    main()
