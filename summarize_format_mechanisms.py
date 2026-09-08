"""Diagnostic summary of fit-objective gains versus held-out task gains."""
import json
import statistics
from pathlib import Path


def main():
    summary={}
    for stage,job in [('interacting_format',331949),('fisher_format',331969)]:
        for model in ('llama1b','opt350m','qwen06b'):
            p=Path('results')/stage/f'model_{job}_{model}'/'report.json'
            if not p.exists():continue
            r=json.loads(p.read_text())
            if r['status']!='complete':continue
            matrices=list(r['matrices'].values())
            policies=['interacting'] if stage=='interacting_format' else ['uniform','diagonal_fisher','block_fisher']
            result={}
            for policy in policies:
                stats=[m['stats'] if stage=='interacting_format' else m['stats'][policy] for m in matrices]
                ratios=[s['trace'][-1]/s['trace'][0] for s in stats if s['trace'][0]>0]
                result[policy]=dict(matrices=len(stats),converged=sum(s['converged'] for s in stats),
                    median_fit_objective_ratio=statistics.median(ratios),max_fit_objective_ratio=max(ratios),
                    selected_tiles=sum(m['selected_tiles'][policy] for m in matrices),
                    tile_count=sum(m['tile_count'] for m in matrices))
            summary[f'{stage}/{model}']=result
    out=Path('results/format_directions');out.mkdir(exist_ok=True)
    (out/'mechanism_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
