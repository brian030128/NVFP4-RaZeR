"""Post hoc calibration-only audit of the fixed curvature interaction penalty."""
import argparse
import json
import os
from pathlib import Path
import torch
from run_c4_frozen import digest_file


def audit(calibration_paths,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    reports={str(Path(p)/'report.json'):digest_file(Path(p)/'report.json') for p in calibration_paths}
    source_hash=digest_file(__file__)
    if (out/'curvature_audit.json').exists() and (out/'CURVATURE_AUDIT.md').exists():
        cached=json.loads((out/'curvature_audit.json').read_text())
        if cached.get('calibration_report_sha256')==reports and cached.get('audit_source_sha256')==source_hash:
            return cached['rows']
    rows=[]; artifacts={}
    for cal_path in calibration_paths:
        cal=Path(cal_path); r=json.loads((cal/'report.json').read_text())
        assert r['status']=='complete' and not r['uses_c4_calibration'] and not r['uses_wiki_calibration']
        assert digest_file(cal/'maps.json')==r['map_sha256']
        maps=json.loads((cal/'maps.json').read_text())['maps']
        totals={p:dict(linear=0.,radius=0.,direction=torch.zeros(r['block_statistics'][p]['calibration_sequences'],dtype=torch.float64))
                for p in maps}
        for i,(name,matrix) in enumerate(r['matrices'].items()):
            path=cal/'scores'/f'{i:03d}.pt'; artifacts[str(path)]=digest_file(path)
            table=torch.load(path,map_location='cpu',weights_only=True)
            assert table['name']==name
            for p,sparse in maps.items():
                indices=sparse[name]
                if not indices: continue
                setting=p.split('_',1)[1]; ids=r['subsets'][setting]; n=len(ids)
                # Slice selected columns before sample rows to bound temporary memory.
                ce=table['ce'][:,indices][ids].double(); kl=table['kl'][:,indices][ids].double()
                f=table['fisher'][:,indices][ids].double()
                upper=torch.maximum(ce.mean(0)+2*ce.std(0,unbiased=True)/n**.5,
                                    kl.mean(0)+2*kl.std(0,unbiased=True)/n**.5)
                totals[p]['linear']+=float(upper.sum())
                totals[p]['radius']+=float((511*f.square().mean(0)).sqrt().sum())
                totals[p]['direction']+=f.sum(1)
            del table
        for p,t in totals.items():
            stats=r['block_statistics'][p]; n=stats['calibration_sequences']
            major=.5*t['radius']**2
            exact=.5*511/n*float(t['direction'].square().sum())
            assert exact<=major+1e-10
            if p.startswith('adaptive_'):
                assert abs(t['linear']-stats['predicted_linear'])<1e-10
                assert abs(major-stats['predicted_curvature_penalty'])<1e-10
            rows.append(dict(model=r['model'],policy=p,blocks=stats['selected_blocks'],
                predicted_linear=t['linear'],cauchy_penalty=major,sampled_ggn_penalty=exact,
                alignment_fraction=exact/major if major else None,
                cauchy_objective=t['linear']+major,sampled_ggn_objective=t['linear']+exact))
    lines=['# Post hoc curvature-penalty audit','',
        'This diagnostic was added after inspecting the completed smaller-model evaluations. '
        'It uses only the saved math/code calibration scores and already-frozen maps. '
        'It performs no model forward passes, map changes, new count selection, or target-data fitting.','',
        'For each existing map, compare the Cauchy–Schwarz interaction penalty with the exact '
        'quadratic form of the **same sampled GGN matrix**. This is not the true model loss '
        'or Hessian. Alignment fraction is sampled-GGN penalty / Cauchy penalty; lower '
        'values indicate a looser majorizer for that fixed map. The audit cannot establish '
        'that a less conservative selector would generalize.','',
        '| Model | Policy | Blocks | Linear term | Cauchy penalty | Sampled GGN penalty | Alignment fraction | Cauchy objective | Sampled GGN objective |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in rows:
        align='—' if v['alignment_fraction'] is None else f'{v["alignment_fraction"]:.6f}'
        lines.append(f'| {v["model"]} | {v["policy"]} | {v["blocks"]} | {v["predicted_linear"]:+.6g} | '
            f'{v["cauchy_penalty"]:.6g} | {v["sampled_ggn_penalty"]:.6g} | {align} | '
            f'{v["cauchy_objective"]:+.6g} | {v["sampled_ggn_objective"]:+.6g} |')
    (out/'curvature_audit.json').write_text(json.dumps(dict(rows=rows,score_artifact_sha256=artifacts,
        calibration_report_sha256=reports,audit_source_sha256=source_hash,
        post_hoc=True,changes_maps=False),indent=2)+'\n')
    (out/'CURVATURE_AUDIT.md').write_text('\n'.join(lines)+'\n')
    return rows


if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap=argparse.ArgumentParser(); ap.add_argument('--calibrations',nargs=3,required=True); ap.add_argument('--out',required=True)
    args=ap.parse_args(); torch.set_num_threads(2)
    rows=audit(args.calibrations,args.out)
    for model in ('qwen4b','llama8b','qwen27b'):
        selected=[v for v in rows if v['model']==model and v['policy'].startswith('fixed256_')]
        print(json.dumps(dict(model=model,fixed_count_maps=len(selected),
            alignment_min=min(v['alignment_fraction'] for v in selected),
            alignment_max=max(v['alignment_fraction'] for v in selected),
            sampled_ggn_negative_objectives=sum(v['sampled_ggn_objective']<0 for v in selected))),flush=True)
