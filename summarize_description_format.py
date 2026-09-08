import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--job', type=int, required=True); args = ap.parse_args()
    root = Path('results/description_format')
    reports = [json.loads((root/f'model_{args.job}_{m}'/'report.json').read_text()) for m in ('llama1b', 'opt350m', 'qwen06b')]
    assert all(r['status'] == 'complete' for r in reports)
    cells = [(r, d, c) for r in reports for d, c in r['contrasts'].items()]
    retention = []
    for r, d, c in cells:
        old = r['stale_contrasts'][d]
        if old['mean_nll']+old['two_se'] < 0:
            retention.append(dict(model=r['model'], domain=d, retained=c['four_over_six']['mean_nll']/old['mean_nll'],
                                  passes=c['four_over_six']['mean_nll'] <= .5*old['mean_nll']))
    s = dict(cells=len(cells), gains=sum(c['four_over_six']['ppl_delta'] <= -.01 for _, _, c in cells),
             supported_gains=sum(c['four_over_six']['mean_nll']+c['four_over_six']['two_se'] < 0 for _, _, c in cells),
             supported_harms=sum(c['four_over_six']['mean_nll']-c['four_over_six']['two_se'] > 0 for _, _, c in cells),
             beats_mse=sum(c['weight_mse']['mean_nll'] < 0 for _, _, c in cells), retention=retention,
             fit_both_improve=sum(all(r['fit_audit']['description'][k] < r['fit_audit']['four_over_six'][k] for k in ('ce', 'kl')) for r in reports))
    s['passes'] = s['gains'] >= 7 and s['supported_harms'] == 0 and s['beats_mse'] >= 6 and s['fit_both_improve'] == 3 and all(a['passes'] for a in retention)
    lines = ['# Description-cost sparsity screen', '', '```json', json.dumps(s, indent=2), '```', '',
             '| Model / domain | ΔPPL vs FourOverSix | ΔNLL ±2SE | ΔPPL vs stale256 | ΔPPL vs weight MSE |',
             '|---|---:|---:|---:|---:|']
    for r, d, c in cells:
        b = c['four_over_six']
        lines.append(f'| {r["model"]} / {d} | {b["ppl_delta"]:+.6f} | {b["mean_nll"]:+.6f} ±{b["two_se"]:.6f} | {c["stale256"]["ppl_delta"]:+.6f} | {c["weight_mse"]["ppl_delta"]:+.6f} |')
    lines += ['', 'Tile counts follow the same prior and calibration-token formula:', '']
    for r in reports: lines += [r['model'], '', '```json', json.dumps(r['election'], indent=2), '```', '']
    (root/f'summary_{args.job}.json').write_text(json.dumps(s, indent=2)+'\n')
    (root/f'REPORT_{args.job}.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(s, indent=2))


if __name__ == '__main__': main()
