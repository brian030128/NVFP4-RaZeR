import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=int,required=True)
    ap.add_argument('--joint',action='store_true');args=ap.parse_args()
    root=Path('results/joint_online_format' if args.joint else 'results/online_format')
    reports=[json.loads((root/f'model_{args.job}_{m}'/'report.json').read_text()) for m in ('llama1b','opt350m','qwen06b')]
    assert all(r['status']=='complete' for r in reports)
    summary={};lines=['# Calibration-free correlated activation election','',
        'All policies frozen before fresh evaluation; no calibration dataset.',
        'Exact and compact rules are separate hypotheses. No per-model winner is elected.','']
    for policy,key in [('exact','exact_contrasts'),('compact','contrasts')]:
        cells=[(r,d,c) for r in reports for d,c in r[key].items()]
        s=dict(cells=len(cells),gains=sum(c['four_over_six']['ppl_delta']<=-.01 for _,_,c in cells),
            supported_gains=sum(c['four_over_six']['mean_nll']+c['four_over_six']['two_se']<0 for _,_,c in cells),
            supported_harms=sum(c['four_over_six']['mean_nll']-c['four_over_six']['two_se']>0 for _,_,c in cells),
            beats_activation_mse=sum(c['activation_mse']['mean_nll']<0 for _,_,c in cells))
        s['passes']=s['gains']>=7 and s['supported_harms']==0 and s['beats_activation_mse']>=6
        if args.joint and policy=='exact':
            s['beats_activation_only']=sum(c['activation_only']['mean_nll']<0 for _,_,c in cells)
            s['passes'] &= s['beats_activation_only']>=6
        summary[policy]=s
        lines +=[f'## {policy}','', '```json',json.dumps(s,indent=2),'```','',
            '| Model / domain | ΔPPL vs FourOverSix | ΔNLL ±2SE | ΔPPL vs activation MSE |',
            '|---|---:|---:|---:|']
        for r,d,c in cells:
            b=c['four_over_six']
            lines.append(f'| {r["model"]} / {d} | {b["ppl_delta"]:+.6f} | {b["mean_nll"]:+.6f} ±{b["two_se"]:.6f} | {c["activation_mse"]["ppl_delta"]:+.6f} |')
        lines+=['']
    lines +=['## Unfused implementation diagnostics','','| Model | Compact metadata / ideal W4 bytes | Exact seconds | Compact seconds | Baseline seconds |','|---|---:|---:|---:|---:|']
    for r in reports:
        ratio=sum(v['bytes'] for v in r['metadata'].values())/sum(v['ideal_w4_bytes'] for v in r['metadata'].values())
        times=[r['evaluation'][p]['runtime_diagnostics']['seconds'] for p in ('exact','compact','four_over_six')]
        lines.append(f'| {r["model"]} | {ratio:.6f} | '+' | '.join(f'{t:.2f}' for t in times)+' |')
    (root/f'summary_{args.job}.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/f'REPORT_{args.job}.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
