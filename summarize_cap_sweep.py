"""Render the tile-count sweep tables from the per-model reports."""
import argparse
import glob
import json
from pathlib import Path

ORDER = ('olmo1b', 'pythia14b', 'qwen4b', 'llama8b')
LABEL = {'olmo1b': 'OLMo-1B', 'pythia14b': 'Pythia-1.4B',
         'qwen4b': 'Qwen3-4B', 'llama8b': 'Llama-3.1-8B'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='results/cap_sweep')
    ap.add_argument('--job', required=True)
    args = ap.parse_args()
    root = Path(args.dir)
    rep = {}
    for f in glob.glob(str(root / f'model_{args.job}_*/report.json')):
        d = json.loads(Path(f).read_text())
        assert d['status'] == 'complete' and d['frozen_map_reproduced']
        rep[d['model']] = d
    models = [m for m in ORDER if m in rep]

    L = [f'# Tile-count sweep on held-out C4 (job {args.job})', '',
         "Re-election of the unchanged CE/KL two-SE rule from each model's saved 192-sequence score",
         'table at nine nested counts. No model is re-scored. Re-election at 256 reproduces the frozen',
         '`pooled192` map bitwise for every model, and each 256 row reproduces the published',
         '`results/c4_frozen` perplexity to six decimals. Evaluation is the unchanged held-out C4 recipe.',
         '', '[Protocol](PROTOCOL.md).', '', '## Result', '',
         '**256 is not the best count for three of the four models, and the curve shape is strongly**',
         '**model-dependent.** Two models leave most of the available gain unclaimed at 256; two others',
         'are harmed at large counts. No single constant is simultaneously good for all four.', '',
         '| Model | eligible / total tiles | best count | best ΔPPL | ΔPPL at 256 | supported harm from |',
         '|---|---:|---:|---:|---:|---|']
    for m in models:
        d = rep[m]
        ps = [p for p in d['counts'] if p != 'four_over_six']
        best = min(ps, key=lambda p: d['contrasts'][p]['ppl_delta'])
        harm = [p for p in ps if d['contrasts'][p]['mean_nll'] - d['contrasts'][p]['two_se'] > 0]
        onset = f"{d['election'][harm[0]]['selected']:,} tiles" if harm else 'none measured'
        L.append(f"| {LABEL[m]} | {d['eligible_tiles']:,} / {d['total_tiles']:,} | "
                 f"{d['election'][best]['selected']:,} | {d['contrasts'][best]['ppl_delta']:+.6f} | "
                 f"{d['contrasts']['n256']['ppl_delta']:+.6f} | {onset} |")
    L += ['', '## Per-model curves', '',
          'ΔPPL and ΔNLL are against the matched FourOverSix baseline; negative is better. `sig` marks',
          'a descriptive paired two-SE interval excluding zero (YES a gain, HARM a regression).', '']
    for m in models:
        d = rep[m]
        L += [f'### {LABEL[m]}', '',
              f"Baseline C4 PPL {d['evaluation']['four_over_six']['ppl']:.6f}; "
              f"{d['eligible_tiles']:,} of {d['total_tiles']:,} tiles have a negative two-SE score.", '',
              '| Count | Selected | C4 PPL | ΔPPL | ΔNLL ±2SE | sig |', '|---|---:|---:|---:|---:|---|']
        for p in d['counts']:
            e, ev = d['election'][p], d['evaluation'][p]
            if p == 'four_over_six':
                L.append(f"| baseline | 0 | {ev['ppl']:.6f} | — | — | — |")
                continue
            c = d['contrasts'][p]
            sig = ('YES' if c['mean_nll'] + c['two_se'] < 0
                   else 'HARM' if c['mean_nll'] - c['two_se'] > 0 else '·')
            name = 'all eligible' if e['requested'] is None else f"{e['requested']:,}"
            L.append(f"| {name} | {e['selected']:,} | {ev['ppl']:.6f} | {c['ppl_delta']:+.6f} | "
                     f"{c['mean_nll']:+.6f} ±{c['two_se']:.6f} | {sig} |")
        L.append('')
    L += ['## Reading', '',
          'Qwen3-4B improves monotonically to the end of its eligible set, reaching -2.863637 PPL',
          'against -1.025501 at 256: the fixed cap claims about a third of the available gain.',
          'Pythia-1.4B doubles its gain at 1,024 and then reverses into supported harm beyond 16,384,',
          'ending +2.206832 worse than baseline. OLMo-1B peaks near 4,096. Llama-3.1-8B is flat',
          'between 256 and 1,024 and decays to supported harm only when every eligible tile is taken.',
          '',
          'The two previously measured adaptive rules therefore failed at the wrong end. The',
          'curvature-penalised selector chose 0-8 blocks and the description-cost threshold',
          'under-selected; both shrank a count that, for three of these four models, should have',
          'grown. The existing backtracking rule is one-sided in the same way -- it halves its budget',
          'on failure and never raises it -- so it cannot reach the optima measured here either.',
          '',
          'Nothing in this directory selects a count. Reading a preferred count off these tables would',
          'be selection on the evaluation set, and the counts are nine correlated comparisons per model',
          'against a single calibration draw. The sweep establishes only that the headroom above 256 is',
          'large and that the penalty for overshooting is also large, so a count rule must be',
          'predictive rather than fixed. Whether a rule that never sees this evaluation set can find',
          'these optima is measured separately in [adaptive_count](../adaptive_count/).']
    (root / 'REPORT.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L[:24]))


if __name__ == '__main__':
    main()
