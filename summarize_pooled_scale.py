import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--job', type=int, required=True); args = ap.parse_args()
    root = Path('results/pooled_scale')
    reports = [json.loads((root/f'model_{args.job}_{m}'/'report.json').read_text()) for m in ('qwen4b', 'llama8b')]
    assert all(r['status'] == 'complete' for r in reports)
    cells = [(r, d, c) for r in reports for d, c in r['contrasts'].items()]
    fit = {r['model']: {p: {k: sum(a[k] for a in groups.values())/len(groups) for k in ('ce', 'kl')}
                       for p, groups in r['fit_audit'].items()} for r in reports}
    s = dict(cells=len(cells), gains=sum(c['four_over_six']['ppl_delta'] <= -.01 for _, _, c in cells),
             supported_gains=sum(c['four_over_six']['mean_nll']+c['four_over_six']['two_se'] < 0 for _, _, c in cells),
             supported_harms=sum(c['four_over_six']['mean_nll']-c['four_over_six']['two_se'] > 0 for _, _, c in cells),
             beats_mse=sum(c['weight_mse']['mean_nll'] < 0 for _, _, c in cells),
             beats_c4=sum(c['c4_64']['mean_nll'] < 0 for _, _, c in cells),
             prefix_independent_models=sum(all(a['equal'] for a in r['suffix_intervention'].values()) for r in reports),
             fit_both_improve=sum(all(fit[r['model']]['pooled192'][k] < fit[r['model']]['four_over_six'][k] for k in ('ce', 'kl')) for r in reports))
    s['passes_scale_transfer'] = s['gains'] >= 5 and s['supported_harms'] == 0 and s['beats_mse'] >= 4 and s['prefix_independent_models'] == 2 and s['fit_both_improve'] == 2
    lines = ['# Fixed-recipe scale transfer at4B and8B', '', '```json', json.dumps(s, indent=2), '```', '',
             '| Model / domain | FourOverSix | Pooled192 | ΔPPL | ΔNLL ±2SE | ΔPPL vs MSE | ΔPPL vs C4-64 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r, d, c in cells:
        b = c['four_over_six']; e = r['evaluation']
        lines.append(f'| {r["model"]} / {d} | {e["four_over_six"][d]["ppl"]:.6f} | {e["pooled192"][d]["ppl"]:.6f} | {b["ppl_delta"]:+.6f} | {b["mean_nll"]:+.6f} ±{b["two_se"]:.6f} | {c["weight_mse"]["ppl_delta"]:+.6f} | {c["c4_64"]["ppl_delta"]:+.6f} |')
    lines += ['', 'Fitting audit under the same per-token activation convention:', '', '```json', json.dumps(fit, indent=2), '```', '',
              'Same256 cap, scoring rule and calibration recipe. No candidate-loss selection. '
              'These are new-size results on previously inspected data families, not new untouched-domain confirmation. '
              'No reversal of the earlier failed source-diversity comparison is implied.']
    (root/f'summary_{args.job}.json').write_text(json.dumps(s, indent=2)+'\n')
    (root/f'REPORT_{args.job}.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(s, indent=2))


if __name__ == '__main__': main()
