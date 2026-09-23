"""Missing-cell specs and historical-rate estimates; never launch work."""
from common import *

def main():
    start=now();maps=load(OUT/'results/GRANULARITY_MAP_MANIFEST.json');inputs=load(OUT/'results/INPUTS.json');rows=[];cost=[]
    bp=OUT/'results/GPU_RESOURCE_BLOCKERS.json';resource_blockers=load(bp) if bp.exists() else {}
    for model in MODELS:
        inp=next(x for x in inputs if x['model']==model and x['draw']=='seed0')
        run=next((PRIMARY/'runs').glob(f'V*_ppl_primary_{model}_attempt1'))
        lr=load(run/'launch_record.json');ppl=load(run/'ppl/ppl_report.json')
        n=len(ppl['evaluation']);per_map=lr['gpu_hours']/n
        cost.append(dict(model=model,reference=str(run.relative_to(REPO)),reference_sha256=sha(run/'launch_record.json'),
            historical_ppl_policies=n,historical_ppl_gpu_hours=lr['gpu_hours'],rough_gpu_hours_per_two_corpus_map=per_map,
            missing_T2_maps=16,rough_T2_gpu_hours=16*per_map,missing_T3_maps=4,rough_T3_gpu_hours=4*per_map,
            full_baseline_validation_maps=5,rough_baseline_validation_gpu_hours=5*per_map,
            n8_reproduction_maps=1,rough_n8_reproduction_gpu_hours=per_map,
            n16_reproduction_maps=1,rough_n16_reproduction_gpu_hours=per_map,
            calibration_gpu_hours=inp['original_calibration_gpu_hours'] if model!='qwen4b' else 0,
            qualifier='Historical rate only, includes mixed baseline costs; pilot/retries/device/wait time and accuracy extra; not a measured new cost'))
        plan=[]
        cachearg='' if model=='llama8b' else ' --hub-cache /share3/saves/JAAAAAA/mixfp4_n16k64_followup_20260917T090438Z/cache/hf/hub'
        for N in [32,64,128,256]:
            mm=next((m for m in maps if m['model']==model and m['N']==N),None)
            if mm:
                plan.append(dict(name=mm['policy'],kind='map',map_path=str(OUT/mm['path']),map_sha256=mm['sha256'],map_policy=mm['policy'],
                    type_block=[N,64],protocol_id='selector_characterization_v1'))
            for corpus in ['wiki','c4']:
                rows.append(dict(task='T3',model=model,draw='seed0',N=N,corpus=corpus,map_hash=mm['sha256'] if mm else None,
                    status='pending_evaluation' if mm else 'blocked_exact_parent_moments',
                    prerequisite='Numerical A6000 anchor; no mixed-hardware baseline reuse',
                    command=(f'python scripts/gpu_run.py --name granularity_{model}_attempt{3 if model=="qwen4b" else 1} --model {model} --plan plans/{model}_granularity_anchored_v2.json --teacher instance --device-type a6000'+cachearg) if mm else (f'python scripts/gpu_run.py --name calibration_{model}_attempt{3 if model=="llama8b" else 1} --model {model} --calibrate-stream --device-type a6000'+cachearg),
                    next_validation='Exact N8/N16 calibration anchor then parent derivation required' if not mm else 'Hash/map/token/runtime identity; paired natural-cluster inference'))
        if plan or model in ['llama8b','mistral7b']:
            n8=next(m for m in maps if m['model']==model and m['N']==8)
            anchor=dict(name='n8_reproduction',kind='map',map_path=str(OUT/n8['path']),map_sha256=n8['sha256'],
                map_policy=n8['policy'],type_block=[8,64],protocol_id='selector_characterization_v1')
            n16=next(m for m in maps if m['model']==model and m['N']==16)
            anchor16=dict(name='n16_reproduction',kind='map',map_path=str(OUT/n16['path']),map_sha256=n16['sha256'],
                map_policy=n16['policy'],type_block=[16,64],protocol_id='selector_characterization_v1')
            # v1 is immutable evidence of the failed label-mismatch attempt.
            # Logical reproduction names are not the map header's policy name.
            # Independently required Llama/Mistral full N8/N16 reproduction
            # remains executable when exact parent regeneration is blocked.
            # Prefer the combined coarse plan if calibration is admitted;
            # standalone Mistral is a fallback, not a duplicate pending job.
            # This is not another calibration/coarse-evaluation retry.
            suffix='granularity_anchored_v2' if plan else 'reproduction_only_v1'
            path=OUT/'plans'/f'{model}_{suffix}.json'
            proposed=[dict(name='four_over_six',kind='four_over_six'),anchor,anchor16]+plan
            if path.exists():assert load(path)==proposed,'Never overwrite an existing frozen granularity plan'
            else:jsonout(path,proposed)
    measured_path=OUT/'results/granularity_quality.csv'
    measured={(r['model'],int(r['N']),r['corpus']):r for r in csv.DictReader(measured_path.open())
              if r['status'] in ['validated_new','validated_reuse']} if measured_path.exists() else {}
    for row in rows:
        done=measured.get((row['model'],row['N'],row['corpus']))
        if done:
            row.update(status=done['status'],command='',prerequisite='Complete and admitted; do not rerun',source=done['source'],source_sha256=done['source_sha256'])
            continue
        block=resource_blockers.get(row['model']+'_granularity',{})
        if block.get('status')=='blocked_retry_limit' and row['N'] in block.get('N',[]):
            row.update(status='blocked_retry_limit',command='',prerequisite=block['required_external_resolution'])
        if block.get('status')=='additional_attempt_authorized':
            row.update(status='waiting_named_authorized_attempt',command='',
                prerequisite='Inspect live/terminal state for '+block['authorized_run']+'; already submitted. No duplicate or attempt5. Exact score/map and full baseline admission remain required.')
        calibration=resource_blockers.get(row['model']+'_calibration',{})
        if row['status']=='blocked_exact_parent_moments' and calibration.get('status')=='queued_final_reasoned_repair':
            row.update(status='waiting_existing_calibration_repair',command='',
                prerequisite='Inspect results/GPU_QUEUE.json: '+calibration['next_run']+' is already queued; do not resubmit. Exact score/map/ownership admission required before deriving these maps.')
    csvout(OUT/'granularity_job_plan.csv',rows);csvout(OUT/'results/RESOURCE_ESTIMATE.csv',cost)
    # Read-only archive inventory: search nested raw ZIP directories, no extraction.
    import zipfile
    archives=[]
    for z in sorted((REPO/'research_artifacts').glob('*raw_part*.zip')):
        with zipfile.ZipFile(z) as archive:
            names=[x.filename for x in archive.infolist() if 'raw_scores_full.pt' in x.filename]
        archives.append(dict(archive=str(z.relative_to(REPO)),full_score_members=names,
            verification='ZIP directory inspected only; no extracted member used as evidence'))
    jsonout(OUT/'results/RAW_ARCHIVE_SEARCH.json',archives)
    log('remaining-plan','python scripts/remaining_plan.py',start,[OUT/'granularity_job_plan.csv',OUT/'results/RESOURCE_ESTIMATE.csv',OUT/'results/RAW_ARCHIVE_SEARCH.json'])
    print('Historical-rate rough GPU-hours for initial missing PPL + calibration + full baseline/N8/N16 checks',sum(r['rough_T2_gpu_hours']+r['rough_T3_gpu_hours']+r['calibration_gpu_hours']+r['rough_baseline_validation_gpu_hours']+r['rough_n8_reproduction_gpu_hours']+r['rough_n16_reproduction_gpu_hours'] for r in cost))
    print(archives)

if __name__=='__main__':main()
