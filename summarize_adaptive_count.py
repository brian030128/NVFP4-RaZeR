"""Render the calibration-chosen count tables against the measured sweep curve."""
import argparse
import glob
import json
from pathlib import Path

ORDER = ('olmo1b', 'pythia14b', 'qwen4b', 'llama8b')
LABEL = {'olmo1b': 'OLMo-1B', 'pythia14b': 'Pythia-1.4B',
         'qwen4b': 'Qwen3-4B', 'llama8b': 'Llama-3.1-8B'}


def load(pattern):
    out = {}
    for f in glob.glob(pattern):
        d = json.loads(Path(f).read_text())
        assert d['status'] == 'complete' and d['frozen_map_reproduced']
        out[d['model']] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='results/adaptive_count')
    ap.add_argument('--job', required=True)
    ap.add_argument('--sweep-dir', default='results/cap_sweep')
    ap.add_argument('--sweep-job', required=True)
    args = ap.parse_args()
    root = Path(args.dir)
    rep = load(str(root / f'model_{args.job}_*/report.json'))
    sweep = load(str(Path(args.sweep_dir) / f'model_{args.sweep_job}_*/report.json'))
    models = [m for m in ORDER if m in rep]

    # every policy both jobs evaluated must agree exactly
    checked = 0
    for m in models:
        for p, ev in rep[m]['evaluation'].items():
            assert ev['ppl'] == sweep[m]['evaluation'][p]['ppl'], (m, p)
            checked += 1

    L = [f'# Calibration-chosen tile count (job {args.job})', '',
         'The count is chosen by the lowest actual next-token loss over calibration-source',
         'documents, across the same nine nested prefixes measured in',
         f'[cap_sweep](../cap_sweep/REPORT.md) (job {args.sweep_job}). The held-out C4 set is never',
         'consulted for the choice; its perplexities below are a readout of an already-made',
         f'decision. All {checked} policies evaluated by both jobs agree to the last digit.', '',
         '`sel` chooses on 192 documents drawn from the same three calibration sources but disjoint',
         'from the 192 scoring documents. `fit` chooses on the scoring documents themselves and is',
         'reported only to show the overfitting that invites.', '',
         '## Adaptive versus the fixed 256 cap', '',
         '| Model | chosen (sel) | tiles | ΔPPL chosen | ΔPPL at 256 | ΔPPL best possible | vs fixed 256 |',
         '|---|---:|---:|---:|---:|---:|---|']
    for m in models:
        d, s = rep[m], sweep[m]
        ps = [p for p in s['counts'] if p != 'four_over_six']
        best = min(ps, key=lambda p: s['contrasts'][p]['ppl_delta'])
        sp = d['chosen']['sel']['policy']
        a = d['adaptive_vs_fixed']['sel']
        sig = ('supported' if a['mean_nll'] + a['two_se'] < 0
               else 'harm' if a['mean_nll'] - a['two_se'] > 0 else 'inconclusive')
        L.append(f"| {LABEL[m]} | {sp} | {d['chosen']['sel']['selected']:,} | "
                 f"{s['contrasts'][sp]['ppl_delta']:+.6f} | {s['contrasts']['n256']['ppl_delta']:+.6f} | "
                 f"{s['contrasts'][best]['ppl_delta']:+.6f} | {a['ppl_delta']:+.6f} ({sig}) |")
    L += ['', 'The calibration rule improves on the fixed cap in 4/4 point comparisons, three with',
          'supporting descriptive paired two-SE intervals. It lands on the exact argmin of the',
          'measured curve for OLMo-1B, Qwen3-4B and Llama-3.1-8B; on Pythia-1.4B it takes 4,096',
          'where 1,024 was best, keeping most of the gain and staying clear of the harm region that',
          'begins at 16,384.', '',
          '## Why the selection set must be disjoint from the scoring set', '',
          '| Model | chosen on `fit` | ΔPPL | chosen on `sel` | ΔPPL |', '|---|---:|---:|---:|---:|']
    for m in models:
        d, s = rep[m], sweep[m]
        fp, sp = d['chosen']['fit']['policy'], d['chosen']['sel']['policy']
        L.append(f"| {LABEL[m]} | {fp} | {s['contrasts'][fp]['ppl_delta']:+.6f} | "
                 f"{sp} | {s['contrasts'][sp]['ppl_delta']:+.6f} |")
    L += ['', 'Choosing on the scoring documents overshoots on two of four models, and on',
          'Llama-3.1-8B it selects every eligible tile, which is a supported regression of',
          '+0.047524 PPL. The scores were fitted on those documents, so their loss keeps falling',
          'past the point where held-out loss turns. A disjoint selection set from the same sources',
          'costs one extra forward pass per candidate count and removes the failure.', '',
          '## Limits', '',
          'This is the 512-token held-out C4 protocol of `results/c4_frozen`, with causal per-token',
          'activation factors and eager attention. It is not the 2048-token paper-aligned protocol',
          'used by `results/fixed256_paper_eval`, and these numbers are not comparable to the',
          'published RaZeR table. The maps are protocol-independent, so the shape of the finding is',
          'expected to carry, but its magnitude and the location of the harm onset must be',
          're-measured under tensor-wide activation factors and 2048-token windows before any',
          'paper claim rests on them.', '',
          'Four models, one calibration draw each, one selection draw each. Two-SE intervals are',
          'descriptive evaluation-window intervals; they do not cover calibration-draw or',
          'selection-draw variability, and the nine counts per model are correlated comparisons.',
          'C4 is a calibration source for these maps, so this remains held-out within-source',
          'evaluation. Confirmation requires the untouched literature, science, government and',
          'WikiText families with the count rule fixed in advance.']
    (root / 'REPORT.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
