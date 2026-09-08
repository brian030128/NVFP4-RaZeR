import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--job', type=int, required=True); args = ap.parse_args()
    root = Path('results/consensus_format')
    reports = [json.loads((root/f'model_{args.job}_{m}'/'report.json').read_text()) for m in ('llama1b', 'opt350m', 'qwen06b')]
    assert all(r['status'] == 'complete' for r in reports)
    cells = [(r, d, c) for r in reports for d, c in r['contrasts'].items()]
    fit_checks = []
    for r in reports:
        for p in r['primary_policies'].values():
            for source, audit in r['fit_audit'][p].items():
                fit_checks.append(all(audit[k] < r['fit_audit']['four_over_six'][source][k] for k in ('ce', 'kl')))
    s = dict(cells=len(cells), gains=sum(c['four_over_six']['ppl_delta'] <= -.01 for _, _, c in cells),
             supported_gains=sum(c['four_over_six']['mean_nll']+c['four_over_six']['two_se'] < 0 for _, _, c in cells),
             supported_harms=sum(c['four_over_six']['mean_nll']-c['four_over_six']['two_se'] > 0 for _, _, c in cells),
             beats_pooled=sum(c['pooled_'+r['primary_policies'][d]]['mean_nll'] < 0 for r, d, c in cells),
             beats_mse=sum(c['weight_mse']['mean_nll'] < 0 for _, _, c in cells),
             fit_both_improve=sum(fit_checks), fit_source_checks=len(fit_checks))
    s['passes'] = s['gains'] >= 7 and s['supported_harms'] == 0 and s['beats_pooled'] >= 6 and s['beats_mse'] >= 6 and all(fit_checks)
    lines = ['# Leave-source-out consensus screen', '', '```json', json.dumps(s, indent=2), '```', '',
             '| Model / domain | Excluded source | ΔPPL vs FourOverSix | ΔNLL ±2SE | ΔPPL vs pooled | ΔPPL vs weight MSE |',
             '|---|---|---:|---:|---:|---:|']
    for r, d, c in cells:
        b = c['four_over_six']; primary = r['primary_policies'][d]
        lines.append(f'| {r["model"]} / {d} | {primary.removeprefix("without_")} | {b["ppl_delta"]:+.6f} | {b["mean_nll"]:+.6f} ±{b["two_se"]:.6f} | {c["pooled_"+primary]["ppl_delta"]:+.6f} | {c["weight_mse"]["ppl_delta"]:+.6f} |')
    lines += ['', 'All-source policies are secondary; they cannot rescue failure of leave-source-out transfer.', '',
              '| Model / domain | All-consensus ΔPPL | All-pooled ΔPPL |', '|---|---:|---:|']
    for r, d, _ in cells:
        e = r['evaluation']; baseline = e['four_over_six'][d]['ppl']
        lines.append(f'| {r["model"]} / {d} | {e["all_consensus"][d]["ppl"]-baseline:+.6f} | {e["all_pooled"][d]["ppl"]-baseline:+.6f} |')
    (root/f'summary_{args.job}.json').write_text(json.dumps(s, indent=2)+'\n')
    (root/f'REPORT_{args.job}.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(s, indent=2))


if __name__ == '__main__': main()
