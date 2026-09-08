import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=int,required=True)
    ap.add_argument('--predictive',action='store_true');args=ap.parse_args()
    stage='predictive_format' if args.predictive else 'global_format';root=Path('results')/stage
    reports=[json.loads((root/f'model_{args.job}_{m}'/'report.json').read_text()) for m in ('llama1b','opt350m','qwen06b')]
    assert all(r['status']=='complete' for r in reports)
    cells=[(r,d,c) for r in reports for d,c in r['contrasts'].items()]
    s=dict(cells=len(cells),gains=sum(c['four_over_six']['ppl_delta']<=-.01 for _,_,c in cells),
        supported_gains=sum(c['four_over_six']['mean_nll']+c['four_over_six']['two_se']<0 for _,_,c in cells),
        supported_harms=sum(c['four_over_six']['mean_nll']-c['four_over_six']['two_se']>0 for _,_,c in cells),
        beats_fixed_budget=sum(c['fixed_budget']['mean_nll']<0 for _,_,c in cells),
        beats_diagonal=sum(c['diagonal_fisher']['mean_nll']<0 for _,_,c in cells))
    s['passes']=s['gains']>=7 and s['supported_harms']==0 and s['beats_fixed_budget']>=6
    if args.predictive:s['passes'] &= s['beats_diagonal']>=6
    lines=[f'# {stage} frozen transfer screen','', '```json',json.dumps(s,indent=2),'```','',
        'Shared calibration observations for all controls; no candidate-loss gate or per-domain policy election.','',
        '| Model / domain | ΔPPL vs FourOverSix | ΔNLL ±2SE | ΔPPL vs fixed budget | ΔPPL vs diagonal |',
        '|---|---:|---:|---:|---:|']
    for r,d,c in cells:
        b=c['four_over_six']
        lines.append(f'| {r["model"]} / {d} | {b["ppl_delta"]:+.6f} | {b["mean_nll"]:+.6f} ±{b["two_se"]:.6f} | {c["fixed_budget"]["ppl_delta"]:+.6f} | {c["diagonal_fisher"]["ppl_delta"]:+.6f} |')
    lines +=['','## Optimizer diagnostics','']
    for r in reports:
        stats=r['optimizer']['predictive'] if args.predictive else r['optimizer']
        diagnostic={k:v for k,v in stats.items() if k not in ('trace','sequence_predicted_nll')}
        lines +=[r['model'],'','```json',json.dumps(diagnostic,indent=2),'```','']
    (root/f'summary_{args.job}.json').write_text(json.dumps(s,indent=2)+'\n')
    (root/f'REPORT_{args.job}.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(s,indent=2))


if __name__=='__main__':main()
