"""CPU diagnosis of authorized calibration outcomes; no gate relaxation."""
from common import *

def main():
    start=now();inputs=load(OUT/'results/INPUTS.json');rows=[]
    runtime=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z'
    accepted=load(OUT/'results/PARENT_MOMENT_REUSE.json')
    rejections=load(OUT/'results/PARENT_MOMENT_REJECTIONS.json')
    blocks=load(OUT/'results/GPU_RESOURCE_BLOCKERS.json')
    for model in ['mistral7b','llama8b']:
        run=runtime/'runs'/f'calibration_{model}_attempt4_authorized'
        if not (run/'launch_record.json').exists():continue
        lr=load(run/'launch_record.json')
        if lr['status']=='running':continue
        report=run/'calibration/calibration_report.json'
        item=next(x for x in inputs if x['model']==model and x['draw']=='seed0')
        historical=REPO/item['run'];old=load(historical/'calibration/calibration_report.json')
        row=dict(model=model,run=str(run.relative_to(REPO)),launch_sha256=sha(run/'launch_record.json'),
            operational_status=lr['status'],gpu_hours=lr['gpu_hours'],additional_attempt_used=True,
            accepted=any(r['model']==model and r['run']==str(run.relative_to(REPO)) for r in accepted))
        if report.exists():
            new=load(report);a=old.get('score_stream_sha256',{});b=new.get('score_stream_sha256',{})
            row.update(report_sha256=sha(report),historical_report_sha256=sha(historical/'calibration/calibration_report.json'),
                matched_score_modules=sum(b.get(k)==v for k,v in a.items()),total_modules=len(a),
                matching_forward_teacher=old.get('bf16_fit_nll')==new.get('bf16_fit_nll'),
                identity={k:old.get(k)==new.get(k) for k in ['sequence_token_sha256','candidate_weight_sha256','module_manifest_sha256','activation_quantizer']})
            losses={}
            for objective in ['ce','kl']:
                x=[r[objective] for r in old.get('fit_losses',[])];y=[r[objective] for r in new.get('fit_losses',[])]
                if len(x)==len(y) and x:
                    d=np.asarray(y)-np.asarray(x)
                    losses[objective]=dict(count=len(d),equal=int((d==0).sum()),max_abs=float(abs(d).max()))
            row['fit_losses']=losses;row['maps']=[]
            for N in [8,16]:
                pa=historical/f'maps/{model}_seed0_n{N}_k3.mixfp4map';pb=run/f'maps/{model}_seed0_n{N}_k3.mixfp4map'
                if not pb.exists():continue
                _,ma=mapread(pa);_,mb=mapread(pb)
                x=np.concatenate([v.ravel() for v in ma.values()]);y=np.concatenate([mb[n].ravel() for n in ma])
                row['maps'].append(dict(N=N,**pair(x,y),xor_tiles=int((x!=y).sum()),historical_sha256=sha(pa),regenerated_sha256=sha(pb)))
        row['rejection']=[r for r in rejections if r['run']==str(run.relative_to(REPO))]
        row['qualification']='Exact score identity is required; matching forward losses do not establish identical backward scores. Cause not established. No fifth attempt authorized.'
        rows.append(row)
        if not row['accepted']:
            for key in [model+'_calibration',model+'_granularity']:
                if key not in blocks:continue
                blocks[key].update(status='blocked_retry_limit',last_run=run.name,
                    source=str((run/'launch_record.json').relative_to(REPO)),gpu_hours=lr['gpu_hours'],
                    reason='The one explicitly authorized additional calibration attempt is terminal but not scientifically admitted; see AUTHORIZED_CALIBRATION_AUDIT.json.',
                    required_external_resolution='Verified historical full seed0 per-sequence scores or complete child covariance, or an externally resolved exact-reproduction discrepancy and a new explicit user decision. No fifth calibration attempt is authorized; do not use mismatching parents.')
    target=OUT/'results/AUTHORIZED_CALIBRATION_AUDIT.json';jsonout(target,dict(checked_utc=start,runs=rows))
    jsonout(OUT/'results/GPU_RESOURCE_BLOCKERS.json',blocks)
    log('authorized-calibration-audit','python scripts/calibration_retry_audit.py',start,[target,OUT/'results/GPU_RESOURCE_BLOCKERS.json'])
    print(rows)

if __name__=='__main__':main()
