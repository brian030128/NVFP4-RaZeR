"""Audit existing full8 coverage without implying missing-policy safety."""
from common import *
from mechanism_review import sealed

TASKS=['arc_easy','arc_challenge','hellaswag','openbookqa','boolq','winogrande','piqa','mmlu']
def main():
    start=now();rows=[];manifest=load(OUT/'results/MAP_MANIFEST.json')
    for model in MODELS:
        for policy in ['baseline','joint','ce_matched','kl_matched']:
            mm=next((m for m in manifest if m['model']==model and m['draw']=='seed0' and m['policy']==policy),None)
            found=[]
            for p in (PRIMARY/'runs').glob(f'*/lmeval/lmeval_report.json'):
                # Filename narrows I/O only; metadata decides eligibility.
                if model not in str(p):continue
                r=load(p)
                if r.get('model')!=model or r.get('status')!='complete' or r.get('suite')!='full8' or r.get('limit'):continue
                lr=load(p.parent.parent/'launch_record.json')
                if lr.get('status')!='complete' or lr.get('invalid_gpu_cotenancy'):continue
                for ins in r['installs']:
                    if ins.get('activation')!='four_over_six_rows':continue
                    match=policy=='baseline' and ins.get('weight')=='four_over_six'
                    if mm and ins.get('map_path'):
                        mp=Path(ins['map_path']);assert sha(mp)==ins['map_sha256'];_,mask=mapread(mp)
                        match=maskhash(mask)==mm['payload_sha256']
                    if match:found.append((p,r,ins))
            assert len(found)<=1,(model,policy,'ambiguous duplicate')
            if not found:
                for task in TASKS:rows.append(dict(model=model,draw='seed0',policy=policy,task=task,status='coverage_gap',reason='No matching full8 map result; secondary evaluation not yet scheduled. PPL anchor success is not accuracy validation.',map_hash=mm['sha256'] if mm else None))
                continue
            p,r,ins=found[0];sealed(p);result=next(iter(r['results'][ins['name']]['lm_eval'].values()))
            assert r['suite_spec']==[dict(num_fewshot=0,tasks=TASKS)]
            for task in TASKS:
                metric='acc' if task in ['boolq','winogrande','mmlu'] else 'acc_norm'
                rows.append(dict(model=model,draw='seed0',policy=policy,task=task,metric=metric,value=result['results'][task][metric+',none'],
                    status='sealed_report_reuse',source=str(p.relative_to(REPO)),source_sha256=sha(p),map_hash=ins.get('map_sha256'),
                    num_fewshot=0,lm_eval_version=r['lm_eval_version'],paired_ci='not recomputed; no CI fabricated from aggregates'))
    csvout(OUT/'results/secondary_accuracy_coverage.csv',rows)
    plans=[]
    for model in MODELS:
        baseline=next(r for r in rows if r['model']==model and r['policy']=='baseline' and r['status']=='sealed_report_reuse')
        source=REPO/baseline['source'];report=load(source);launch=load(source.parent.parent/'launch_record.json')
        plan=[dict(name='four_over_six',kind='four_over_six')]
        for policy in ['ce_matched','kl_matched']:
            m=next(m for m in manifest if m['model']==model and m['draw']=='seed0' and m['policy']==policy)
            assert sha(OUT/m['path'])==m['sha256']
            plan.append(dict(name='seed0_'+policy,kind='map',map_path=str(OUT/m['path']),map_sha256=m['sha256'],
                map_policy=policy,type_block=[16,64],protocol_id='selector_characterization_v1'))
        path=OUT/'plans'/f'{model}_accuracy_seed0_anchored.json'
        if path.exists():assert load(path)==plan,'Do not overwrite a frozen secondary plan'
        else:jsonout(path,plan)
        plans.append(dict(model=model,draw='seed0',plan=str(path.relative_to(OUT)),plan_sha256=sha(path),
            suite='full8',suite_spec=report['suite_spec'],batch_size=report['batch_size'],lm_eval_version=report['lm_eval_version'],
            source=str(source.relative_to(REPO)),source_sha256=sha(source),
            source_launch_sha256=sha(source.parent.parent/'launch_record.json'),
            source_gpu_hours=launch['gpu_hours'],estimated_gpu_hours_for_baseline_and_two_matched=3*launch['gpu_hours'],
            estimate_qualification='Historical one-policy cost, not a new measurement; excludes numerical pilots, retries and waiting',
            prerequisites=['Use unchanged primary evaluator and exact task configs/sample identities',
                'Verify full per-example baseline identity before historical paired reuse',
                'Exclusive homogeneous A6000 via audited launcher, three physical GPUs total',
                'GPU execution of accuracy adapter and baseline prompt/sample identity still require validation'],
            status='frozen_plan_not_launched'))
    jsonout(OUT/'results/SECONDARY_ACCURACY_PLAN.json',dict(plans=plans,
        tasks=TASKS,missing_matched_task_cells=48,
        estimated_gpu_hours=sum(r['estimated_gpu_hours_for_baseline_and_two_matched'] for r in plans),
        priority='After required PPL and exact granularity work; cannot infer accuracy safety from PPL',
        cache_audit=dict(path='results/ACCURACY_CACHE_AUDIT.json',sha256=sha(OUT/'results/ACCURACY_CACHE_AUDIT.json')) if (OUT/'results/ACCURACY_CACHE_AUDIT.json').exists() else None,
        launcher_status='accuracy_run.py adapter prepared and statically tested; no accuracy GPU attempt yet. Do not pass accuracy plans to gpu_run.py PPL mode.'))
    log('T2-secondary','python scripts/secondary_accuracy.py',start,[OUT/'results/secondary_accuracy_coverage.csv'])
    print('secondary task cells reused',sum(r['status']=='sealed_report_reuse' for r in rows),'/',len(rows))

if __name__=='__main__':main()
