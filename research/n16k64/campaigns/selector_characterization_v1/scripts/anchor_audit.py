"""Read-only numerical/identity audit of completed pilot attempts."""
from common import *

def main():
    start=now();runtime=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z'
    rows=[];gates={};hours={}
    for p in sorted((runtime/'runs').glob('pilot_*/ppl/ppl_report.json')):
        r=load(p);run=p.parent.parent;lr=load(run/'launch_record.json')
        if lr['status']!='complete':continue  # partial/invalid reports are never scientific anchors
        model=r['model'];prefix='V62' if model=='mistral7b' else 'V31'
        old=next((PRIMARY/'runs').glob(f'{prefix}_ppl_primary_{model}_attempt1/ppl/ppl_report.json'));ref=load(old)
        env=load(run/'job_result.json')['environment'];oldenv=load(old.parent.parent/'job_result.json')['environment']
        passed=lr['status']=='complete';deltas=[]
        for pol in ['four_over_six','n16_k3']:
            a=next(i for i in r['installs'] if i['name']==pol);b=next(i for i in ref['installs'] if i['name']==pol)
            weights=a['installed_weight_sha256']==b['installed_weight_sha256'];passed &= weights
            for corpus in ['wiki','c4']:
                wm=load(p.parent/f'windows_{corpus}.json');wo=load(old.parent/f'windows_{corpus}.json')
                tokens=wm['token_sha256']==wo['token_sha256'][:len(wm['token_sha256'])];passed &= tokens
                for i,(x,y) in enumerate(zip(r['evaluation'][pol][corpus]['windows'],ref['evaluation'][pol][corpus]['windows'])):
                    d=x['nll_sum']/x['tokens']-y['nll_sum']/y['tokens'];deltas.append(d)
                    rows.append(dict(run=run.name,model=model,policy=pol,corpus=corpus,window=i,delta_nll=d,
                        tokens_equal=tokens,weights_equal=weights,absolute_tolerance=1e-10,passed=abs(d)<=1e-10,
                        reference=str(old.relative_to(REPO)),reference_sha256=sha(old),new_report_sha256=sha(p)))
        passed &= all(abs(d)<=1e-10 for d in deltas)
        gates[model]=dict(passed=bool(passed),attempt=run.name,max_abs_delta_nll=max(map(abs,deltas)),
            new_environment=env,reference_environment=oldenv,
            qualification='Small prefix numerical anchor only; full evaluation identities still mandatory',
            reason=None if passed else 'Historical numerical anchor differs; not eligible to merge with historical baseline arrays')
    # Account for all terminal launches, including invalid attempts that never
    # produced a PPL report. Missing outcomes must not erase resource use.
    attempts=[]
    for lp in sorted((runtime/'runs').glob('*/launch_record.json')):
        launch=load(lp)
        if launch['status']=='running':continue
        hours[launch['gpu_model']]=hours.get(launch['gpu_model'],0)+launch['gpu_hours']
        attempts.append(dict(run=lp.parent.name,status=launch['status'],
            gpu_hours=launch['gpu_hours'],gpu_model=launch['gpu_model'],
            invalid_reasons=launch.get('invalid_reasons'),source_sha256=sha(lp)))
    csvout(OUT/'results/anchor_windows.csv',rows);jsonout(OUT/'results/ANCHOR_GATE.json',gates)
    jsonout(OUT/'results/GPU_COST.json',dict(gpu_hours_by_type=hours,total_gpu_hours=sum(hours.values()),scope='all terminal launch records, including failed/invalid attempts; excludes active and prelaunch waiting',attempts=attempts))
    s=load(OUT/'TASK_STATUS.json');s['gpu_hours']=sum(hours.values());jsonout(OUT/'TASK_STATUS.json',s)
    log('anchor_audit','python scripts/anchor_audit.py',start,[OUT/'results/anchor_windows.csv',OUT/'results/ANCHOR_GATE.json'])
    print({m:(g['passed'],g['max_abs_delta_nll']) for m,g in gates.items()},hours)

if __name__=='__main__':main()
