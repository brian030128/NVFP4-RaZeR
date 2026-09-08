"""Validate all requested frozen-map WikiText results and emit the full table."""
import argparse
import json
import math
from pathlib import Path
from run_conditional_model import paired


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--small-job', type=int, required=True)
    ap.add_argument('--large-job', type=int, required=True); args = ap.parse_args()
    root = Path('results/wiki_frozen')
    paths = [root/f'model_{args.small_job}_{m}/report.json' for m in ('qwen4b', 'llama8b')]
    paths.append(root/f'model_{args.large_job}_qwen27b/report.json')
    reports = [json.loads(p.read_text()) for p in paths]
    for r in reports:
        assert r['status'] == 'complete' and r['source_weights_verified'] and r['frozen_map_unchanged']
        assert not r['wiki_data']['calibration_exact_row_overlap']
        assert all(c['equal'] and c['max_logit_difference'] == 0 for c in r['suffix_intervention'].values())
        n = r['wiki_data']['windows']; assert n*512+r['wiki_data']['omitted_tail_tokens'] == r['wiki_data']['total_tokens']
        assert r['wiki_data']['offsets'] == [i*512 for i in range(n)]
        assert len(r['wiki_data']['token_sha256']) == n
        for p, e in r['evaluation'].items():
            assert len(e['nll']) == n and e['scored_tokens'] == n*511
            assert math.isclose(e['ppl'], math.exp(sum(e['nll'])/n), rel_tol=1e-12)
            if p != 'pooled192':
                assert paired(r['evaluation']['pooled192']['nll'], e['nll']) == r['contrasts'][p]
    cs = [r['contrasts']['four_over_six'] for r in reports]
    s = dict(models=3, point_gains=sum(c['mean_nll'] < 0 for c in cs),
             supported_gains=sum(c['mean_nll']+c['two_se'] < 0 for c in cs),
             supported_harms=sum(c['mean_nll']-c['two_se'] > 0 for c in cs),
             gains_at_least_0p01_ppl=sum(c['ppl_delta'] <= -.01 for c in cs),
             beats_weight_mse=sum(r['contrasts']['weight_mse']['mean_nll'] < 0 for r in reports),
             beats_c4_only=sum(r['contrasts']['c4_64']['mean_nll'] < 0 for r in reports),
             source_reports=[str(p) for p in paths])
    lines = ['# Current causal WikiText-2: three frozen pooled maps', '',
             'Pinned WikiText-2 raw test split, concatenated in dataset order and partitioned into '
             'nonoverlapping512-token windows. Only the final incomplete window is omitted. '
             'One fixed pooled192 map per model, reused from the preceding studies; no calibration '
             'or candidate-loss election. All policies use causal per-token activation factors.', '',
             '| Model | Windows | FourOverSix PPL | Pooled192 PPL | ΔPPL | Relative ΔPPL | ΔNLL ±2SE |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r, c in zip(reports, cs):
        b, p = r['evaluation']['four_over_six']['ppl'], r['evaluation']['pooled192']['ppl']
        lines.append(f'| {r["model"]} | {r["wiki_data"]["windows"]} | {b:.6f} | {p:.6f} | '
                     f'{p-b:+.6f} | {100*(p/b-1):+.3f}% | {c["mean_nll"]:+.6f} ±{c["two_se"]:.6f} |')
    lines += ['', '| Model | C4-only64 PPL | Mixed64 PPL | Weight-MSE PPL | Omitted tail tokens |',
              '|---|---:|---:|---:|---:|']
    for r in reports:
        lines.append('| '+r['model']+' | '+' | '.join(f'{r["evaluation"][p]["ppl"]:.6f}'
                     for p in ('c4_64', 'mixed64', 'weight_mse'))+f' | {r["wiki_data"]["omitted_tail_tokens"]} |')
    lines += ['', 'All source-weight, quantizer-source, frozen-map, exact-overlap and prefix-independence '
              'checks pass. PPL is exp(mean window NLL), equal to token-weighted PPL across scored labels. '
              'Adjacent windows may share an article; the paired two-SE intervals are descriptive and '
              'do not account for article dependence or simultaneous testing. Dataset coverage and '
              'tokenizers differ across model comparisons. The earlier inconclusive27B C4 result '
              'and failed source-diversity screen are unchanged.', '',
              'These are simulated nonhead-text-linear W4A4 reference-text results, not generation '
              'accuracy or native FP4 throughput.27B uses native Transformers5.16.1 hybrid text support; '
              'Qwen4B and Llama8B use4.57.3.', '', '```json', json.dumps(s, indent=2), '```']
    suffix = f'{args.small_job}_{args.large_job}'
    (root/f'REPORT_{suffix}.md').write_text('\n'.join(lines)+'\n')
    (root/f'summary_{suffix}.json').write_text(json.dumps(s, indent=2)+'\n')
    print(json.dumps(s, indent=2))


if __name__ == '__main__': main()
