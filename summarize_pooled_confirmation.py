import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--job', type=int, required=True); args = ap.parse_args()
    root = Path('results/pooled_confirmation')
    models = ('llama1b', 'opt350m', 'qwen06b', 'pythia14b', 'olmo1b')
    reports = [json.loads((root/f'model_{args.job}_{m}'/'report.json').read_text()) for m in models]
    assert all(r['status'] == 'complete' for r in reports)
    for d in ('web', 'math', 'code'):
        assert len({r['fit'][d]['revision'] for r in reports}) == 1
        assert len({r['fit'][d]['path'] for r in reports}) == 1
    for d in ('literature', 'science', 'government'):
        assert len({r['confirmation_data'][d]['revision'] for r in reports}) == 1
    cells = [(r, d, c) for r in reports for d, c in r['contrasts'].items()]
    new = [r for r in reports if r['model'] in ('pythia14b', 'olmo1b')]
    fit = {r['model']: {p: {k: sum(a[k] for a in groups.values())/len(groups) for k in ('ce', 'kl')}
                       for p, groups in r['fit_audit'].items()} for r in reports}
    s = dict(cells=len(cells), gains=sum(c['four_over_six']['ppl_delta'] <= -.01 for _, _, c in cells),
             supported_gains=sum(c['four_over_six']['mean_nll']+c['four_over_six']['two_se'] < 0 for _, _, c in cells),
             supported_harms=sum(c['four_over_six']['mean_nll']-c['four_over_six']['two_se'] > 0 for _, _, c in cells),
             beats_mse=sum(c['weight_mse']['mean_nll'] < 0 for _, _, c in cells),
             beats_c4=sum(c['c4_64']['mean_nll'] < 0 for _, _, c in cells),
             beats_mixed64=sum(c['mixed64']['mean_nll'] < 0 for _, _, c in cells),
             new_models_fit_both_improve=sum(all(fit[r['model']]['pooled192'][k] < fit[r['model']]['four_over_six'][k] for k in ('ce', 'kl')) for r in new),
             diversity_point_gains=sum(c['mean_nll'] < 0 for r in reports for c in r['diversity_contrasts'].values()),
             diversity_supported_gains=sum(c['mean_nll']+c['two_se'] < 0 for r in reports for c in r['diversity_contrasts'].values()),
             diversity_supported_harms=sum(c['mean_nll']-c['two_se'] > 0 for r in reports for c in r['diversity_contrasts'].values()))
    s['passes'] = s['gains'] >= 12 and s['supported_harms'] == 0 and s['beats_mse'] >= 9 and s['beats_c4'] >= 9 and s['new_models_fit_both_improve'] == 2
    lines = ['# Pooled-source independent confirmation', '', '```json', json.dumps(s, indent=2), '```', '',
             '| Model / domain | FourOverSix | Pooled192 | ΔPPL | ΔNLL ±2SE | ΔPPL vs MSE | ΔPPL vs C4-64 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r, d, c in cells:
        b = c['four_over_six']; e = r['evaluation']
        lines.append(f'| {r["model"]} / {d} | {e["four_over_six"][d]["ppl"]:.6f} | {e["pooled192"][d]["ppl"]:.6f} | {b["ppl_delta"]:+.6f} | {b["mean_nll"]:+.6f} ±{b["two_se"]:.6f} | {c["weight_mse"]["ppl_delta"]:+.6f} | {c["c4_64"]["ppl_delta"]:+.6f} |')
    lines += ['', '## Equal-token diversity ablation', '',
              'Mixed64 minus C4-only64; both reuse the same scoring table, election code and256 cap.', '',
              '| Model / domain | ΔPPL | ΔNLL ±2SE |', '|---|---:|---:|']
    for r in reports:
        for d, c in r['diversity_contrasts'].items():
            lines.append(f'| {r["model"]} / {d} | {c["ppl_delta"]:+.6f} | {c["mean_nll"]:+.6f} ±{c["two_se"]:.6f} |')
    lines += ['', '## Fitting audit', '', '```json', json.dumps(fit, indent=2), '```', '',
              'All maps freeze before confirmation data loading. The three older primary maps replay exactly. '
              'Reference-text fake-quantized scores; no causal likelihood, generation accuracy, kernel-speed or universal calibration-free claim.']
    (root/f'summary_{args.job}.json').write_text(json.dumps(s, indent=2)+'\n')
    (root/f'REPORT_{args.job}.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(s, indent=2))


if __name__ == '__main__': main()
