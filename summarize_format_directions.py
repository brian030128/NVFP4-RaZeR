"""Summarize successive frozen studies, retaining unsuccessful directions."""
import json
from pathlib import Path


def main():
    root=Path('results')
    stages=[('conditional_format',331873),('asymmetric_format',331893)]
    summary={}
    lines=['# Successive format-selection directions','',
           'Each stage froze a method and evaluated every named comparator. No evaluation loss selected a per-model or per-domain configuration.',
           'Different stages use different evaluation examples; absolute PPL across stages must not be compared.','']
    for stage,job in stages:
        reports=[json.loads((root/stage/f'model_{job}_{m}'/'report.json').read_text()) for m in ('llama1b','opt350m')]
        assert all(r['status']=='complete' for r in reports)
        cells=[(r['model'],d,cs) for r in reports for d,cs in r['contrasts']['a4'].items()]
        stats=dict(cells=len(cells),
            conditional_beats_dynamic=sum(c['dynamic_mse']['mean_nll']<0 for _,_,c in cells),
            supported_vs_dynamic=sum(c['dynamic_mse']['mean_nll']+c['dynamic_mse']['two_se']<0 for _,_,c in cells),
            conditional_beats_e2m1=sum(c['e2m1']['mean_nll']<0 for _,_,c in cells),
            supported_harm_vs_e2m1=sum(c['e2m1']['mean_nll']-c['e2m1']['two_se']>0 for _,_,c in cells),
            conditional_beats_four_over_six=sum(c['four_over_six']['mean_nll']<0 for _,_,c in cells),
            supported_harm_vs_four_over_six=sum(c['four_over_six']['mean_nll']-c['four_over_six']['two_se']>0 for _,_,c in cells))
        stats['passes_declared_screen']=stats['conditional_beats_dynamic']==6 and stats['supported_vs_dynamic']>=4 and stats['supported_harm_vs_e2m1']==0
        summary[stage]=stats
        lines +=[f'## {stage}', '', '```json',json.dumps(stats,indent=2),'```','',
                 '| Model / domain | ΔPPL vs dynamic MSE | ΔPPL vs compensated E2M1 | ΔPPL vs FourOverSix |',
                 '|---|---:|---:|---:|']
        for m,d,c in cells:
            lines.append(f'| {m} / {d} | {c["dynamic_mse"]["ppl_delta"]:+.6f} | {c["e2m1"]["ppl_delta"]:+.6f} | {c["four_over_six"]["ppl_delta"]:+.6f} |')
        lines+=['']
    for stage,job,proposed,other in [('consumer_activation',331910,'consumer_energy','activation_mse'),
                                    ('interacting_format',331949,'interacting','independent'),
                                    ('fisher_format',331969,'block_fisher','uniform'),
                                    ('branched_format',331984,'conditional','dynamic_mse')]:
        paths=[root/stage/f'model_{job}_{m}'/'report.json' for m in ('llama1b','opt350m','qwen06b')]
        if not all(p.exists() for p in paths):continue
        reports=[json.loads(p.read_text()) for p in paths]
        if not all(r['status']=='complete' for r in reports):continue
        cells=[(r['model'],d,cs) for r in reports for d,cs in r['contrasts'].items()]
        stats=dict(cells=len(cells),
            beats_four_over_six=sum(c['four_over_six']['mean_nll']<0 for _,_,c in cells),
            ppl_gain_at_least_point01=sum(c['four_over_six']['ppl_delta']<=-.01 for _,_,c in cells),
            supported_vs_four_over_six=sum(c['four_over_six']['mean_nll']+c['four_over_six']['two_se']<0 for _,_,c in cells),
            supported_harm_vs_four_over_six=sum(c['four_over_six']['mean_nll']-c['four_over_six']['two_se']>0 for _,_,c in cells),
            beats_mechanism_ablation=sum(c[other]['mean_nll']<0 for _,_,c in cells))
        stats['passes_declared_screen']=stats['ppl_gain_at_least_point01']>=7 and stats['supported_harm_vs_four_over_six']==0 and stats['beats_mechanism_ablation']>=6
        if stage=='branched_format':
            stats['supported_harm_vs_e2m1']=sum(c['e2m1']['mean_nll']-c['e2m1']['two_se']>0 for _,_,c in cells)
            stats['passes_declared_screen'] &= stats['supported_harm_vs_e2m1']==0
        summary[stage]=stats
        lines +=[f'## {stage}', '', '```json',json.dumps(stats,indent=2),'```','',
                 f'| Model / domain | ΔPPL vs FourOverSix | ΔPPL vs {other} |','|---|---:|---:|']
        for m,d,c in cells:
            lines.append(f'| {m} / {d} | {c["four_over_six"]["ppl_delta"]:+.6f} | {c[other]["ppl_delta"]:+.6f} |')
        lines+=['']
    out=root/'format_directions';out.mkdir(exist_ok=True)
    passed=[name for name,s in summary.items() if s['passes_declared_screen']]
    lines[1:1]=['',f'Completed frozen studies: {len(summary)}. Declared screens passed: {len(passed)}.',
                'This is an exploratory research record, not a claim of a validated universal selector.']
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
