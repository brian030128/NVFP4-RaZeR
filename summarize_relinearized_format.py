import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--job', type=int, required=True); args = ap.parse_args()
    root = Path('results/relinearized_format')
    reports = [json.loads((root/f'model_{args.job}_{m}'/'report.json').read_text()) for m in ('llama1b', 'opt350m', 'qwen06b')]
    assert all(r['status'] == 'complete' for r in reports)
    cells = [(r, d, c) for r in reports for d, c in r['contrasts'].items()]
    s = dict(cells=len(cells), gains=sum(c['four_over_six']['ppl_delta'] <= -.01 for _, _, c in cells),
             supported_gains=sum(c['four_over_six']['mean_nll']+c['four_over_six']['two_se'] < 0 for _, _, c in cells),
             supported_harms=sum(c['four_over_six']['mean_nll']-c['four_over_six']['two_se'] > 0 for _, _, c in cells),
             beats_stale=sum(c['stale256']['mean_nll'] < 0 for _, _, c in cells),
             beats_mse=sum(c['weight_mse']['mean_nll'] < 0 for _, _, c in cells),
             beats_budget=sum(c['fixed_budget']['mean_nll'] < 0 for _, _, c in cells),
             fit_both_improve=sum(all(r['fit_audit']['relinearized'][k] < r['fit_audit']['four_over_six'][k] for k in ('ce', 'kl')) for r in reports))
    s['passes'] = s['gains'] >= 7 and s['supported_harms'] == 0 and s['beats_stale'] >= 6 and s['beats_mse'] >= 6 and s['fit_both_improve'] == 3
    lines = ['# Relinearized common-descent screen', '', '```json', json.dumps(s, indent=2), '```', '',
             '| Model / domain | ΔPPL vs FourOverSix | ΔNLL ±2SE | ΔPPL vs stale | ΔPPL vs weight MSE |',
             '|---|---:|---:|---:|---:|']
    for r, d, c in cells:
        b = c['four_over_six']
        lines.append(f'| {r["model"]} / {d} | {b["ppl_delta"]:+.6f} | {b["mean_nll"]:+.6f} ±{b["two_se"]:.6f} | {c["stale256"]["ppl_delta"]:+.6f} | {c["weight_mse"]["ppl_delta"]:+.6f} |')
    lines += ['', 'Fitting audit and update diagnostics:', '']
    for r in reports:
        diagnostic = dict(selected=r['selected_tiles'], revisits=sum(e.get('revisited', False) for e in r['updates']),
                          undo=sum(e.get('undo', False) for e in r['updates']),
                          initial_score_seconds=r['initial_score_seconds'], optimization_seconds=r['optimization_seconds'],
                          fit={p: {k: v for k, v in a.items() if k != 'examples'} for p, a in r['fit_audit'].items()})
        lines += [r['model'], '', '```json', json.dumps(diagnostic, indent=2), '```', '']
    lines += ['Reference-text losses on development domains; no generation accuracy, sealed confirmation, or inference speed claim.']
    (root/f'summary_{args.job}.json').write_text(json.dumps(s, indent=2)+'\n')
    (root/f'REPORT_{args.job}.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(s, indent=2))


if __name__ == '__main__': main()
