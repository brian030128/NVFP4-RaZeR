"""Summarize every prespecified C4 model without result-dependent selection."""
import argparse
import json
import math
from pathlib import Path

MODELS = ('opt350m', 'qwen06b', 'llama1b', 'olmo1b', 'pythia14b', 'qwen4b', 'llama8b')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', type=int, required=True)
    args = ap.parse_args()
    root = Path('results/c4_frozen')
    paths = [root / f'model_{args.job}_{m}' / 'report.json' for m in MODELS]
    reports = [json.loads(p.read_text()) for p in paths]
    for model, r in zip(MODELS, reports):
        assert r['status'] == 'complete' and r['model'] == model
        assert r['source_weights_verified'] and r['frozen_map_unchanged']
        assert r['c4_data']['calibration_hash_overlap'] == 0
        assert all(v['equal'] for v in r['suffix_intervention'].values())
        assert len(r['c4_data']['documents']) == 256
        assert len({d['document_sha256'] for d in r['c4_data']['documents']}) == 256
        for e in r['evaluation'].values():
            assert len(e['nll']) == 256 and e['scored_tokens'] == 256 * 511
            assert math.isclose(e['ppl'], math.exp(sum(e['nll']) / 256), rel_tol=1e-12)
    cs = [r['contrasts']['four_over_six'] for r in reports]
    summary = dict(models=len(reports), documents_per_model=256, tokens_per_document=512,
                   point_gains=sum(c['mean_nll'] < 0 for c in cs),
                   gains_at_least_0p01_ppl=sum(c['ppl_delta'] <= -.01 for c in cs),
                   supported_gains=sum(c['mean_nll'] + c['two_se'] < 0 for c in cs),
                   supported_harms=sum(c['mean_nll'] - c['two_se'] > 0 for c in cs),
                   beats_c4_only=sum(r['contrasts']['c4_64']['mean_nll'] < 0 for r in reports),
                   beats_weight_mse=sum(r['contrasts']['weight_mse']['mean_nll'] < 0 for r in reports),
                   exact_prefix_independence_models=len(reports),
                   source_reports=[str(p) for p in paths])
    lines = ['# Held-out C4: seven frozen maps', '',
             'All maps were frozen in the preceding studies. This evaluation uses no calibration, '
             'configuration selection or loss backtracking. C4 is a calibration source; these are '
             'held-out within-source results, not new-domain confirmation.', '',
             'Each model uses 256 distinct C4 validation documents, one seeded 512-token crop each. '
             'Each policy scores the same 511 next-token labels per document. PPL is exp(mean NLL). '
             'All policies use simulated nonhead-linear W4A4 with causal per-token activation factors.', '',
             '| Model | FourOverSix PPL | Pooled192 PPL | ΔPPL | Relative ΔPPL | ΔNLL ±2SE |',
             '|---|---:|---:|---:|---:|---:|']
    for r, c in zip(reports, cs):
        b = r['evaluation']['four_over_six']['ppl']
        p = r['evaluation']['pooled192']['ppl']
        lines.append(f'| {r["model"]} | {b:.6f} | {p:.6f} | {p-b:+.6f} | '
                     f'{100*(p/b-1):+.3f}% | {c["mean_nll"]:+.6f} ±{c["two_se"]:.6f} |')
    lines += ['', '## Fixed control maps', '',
              'C4-64 labels a map calibrated earlier on 64 C4 training sequences. '
              'All numbers in this table are evaluated on held-out C4 validation documents.', '',
              '| Model | Weight MSE PPL | C4-64 PPL | Mixed64 PPL | Pooled192 PPL |',
              '|---|---:|---:|---:|---:|']
    for r in reports:
        lines.append('| ' + r['model'] + ' | ' + ' | '.join(
            f'{r["evaluation"][p]["ppl"]:.6f}' for p in ('weight_mse', 'c4_64', 'mixed64', 'pooled192')) + ' |')
    lines += ['', '## Checks and interpretation', '',
              f'Point gains: {summary["point_gains"]}/7; descriptive paired two-SE supported gains: '
              f'{summary["supported_gains"]}/7; supported harms: {summary["supported_harms"]}/7. '
              f'Pooled192 beats C4-only in {summary["beats_c4_only"]}/7 point comparisons and '
              f'weight-MSE in {summary["beats_weight_mse"]}/7.', '',
              'Model revisions and source linear weights match the frozen records. Map file hashes '
              'are unchanged after evaluation. All seven models pass exact prefix independence '
              'for baseline and pooled192. Evaluation text hashes are disjoint from all 192 '
              'calibration document hashes per model. Token hashes, document hashes, crop offsets '
              'and raw per-document losses are retained in the source reports.', '',
              'Two-SE intervals are descriptive, not simultaneous inference across models. '
              'One calibration pool and one held-out crop seed per model are measured. '
              'Tokenizer-dependent eligibility may yield different documents across models. '
              'Exact hash exclusion does not establish near-duplicate or pretraining-data independence. '
              'These measurements do not alter the earlier failed source-diversity screen.', '',
              '```json', json.dumps(summary, indent=2), '```']
    (root / f'REPORT_{args.job}.md').write_text('\n'.join(lines) + '\n')
    (root / f'summary_{args.job}.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
