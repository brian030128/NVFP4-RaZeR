"""Admit new fixed-primary-draw accuracy only after exact baseline pairing.

No GPU execution. Historical raw arrays remain unchanged. Missing new runs
produce explicit coverage gaps, never evidence for accuracy preservation.
"""
from common import *
from accuracy_samples import read_policy, metric, TASKS

RUNTIME=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z'
IDENTITY_KEYS=['batch_size','lm_eval_version','suite_spec','module_manifest_sha256','environment']

def merge_coverage(historical, fresh):
    def key(r):return tuple(r[k] for k in ['model','draw','policy','task'])
    merged={key(r):dict(r) for r in historical}
    assert len(merged)==len(historical),'duplicate historical coverage'
    seen=set()
    for row in fresh:
        k=key(row);assert k not in seen,'duplicate new coverage';seen.add(k)
        assert k in merged,'unexpected accuracy endpoint'
        if row['status']!='validated_new':continue
        assert merged[k]['status']=='coverage_gap','do not overwrite verified historical evidence'
        merged[k]=dict(row)
    return list(merged.values())

def require_baseline_identity(old, new, old_identity, new_identity):
    for key in IDENTITY_KEYS:
        assert old_identity[key]==new_identity[key],('accuracy baseline identity',key)
    assert set(old)==set(new)==set(TASKS)
    for task in TASKS:
        for key in ['ids','correct','response_sha256']:
            assert np.array_equal(old[task][key],new[task][key]),('accuracy baseline arrays',task,key)

def require_gpu_record(run):
    external=run/'external_invalidation.json'
    assert not external.exists(), 'Externally invalidated: '+load(external)['reason']
    lr=load(run/'launch_record.json')
    assert lr['status']=='complete' and not lr.get('invalid_reasons'), ('GPU launch not complete/valid',run.name,lr['status'],lr.get('invalid_reasons'))
    pre=[json.loads(x) for x in (run/'preflight_candidates.jsonl').read_text().splitlines()]
    during=[json.loads(x) for x in (run/'gpu_monitor.jsonl').read_text().splitlines()]
    post=load(run/'postflight.json')
    assert pre[-1]['passed'] and during and post['passed']
    assert all(r['passed'] for r in during)
    times=[datetime.fromisoformat(lr['started_utc'])]+[datetime.fromisoformat(r['timestamp_utc'].replace('Z','+00:00')) for r in during]+[datetime.fromisoformat(lr['finished_utc'])]
    assert all(0<=(b-a).total_seconds()<=60 for a,b in zip(times,times[1:])), 'accuracy monitoring gap'
    assert lr['protocol_sha256']==sha(OUT/'FROZEN_PROTOCOL.yaml')
    assert lr['actual_source_hashes_sha256']==sha(run/'source_hashes.json')
    return lr

def main():
    start=now();plan=load(OUT/'results/SECONDARY_ACCURACY_PLAN.json');accepted=[];excluded=[];rows=[]
    for item in plan['plans']:
        model=item['model'];reference=REPO/item['source']
        assert sha(reference)==item['source_sha256']
        ref_report=load(reference);old,oldid=read_policy(reference,'four_over_six')
        expected_plan=OUT/item['plan'];assert sha(expected_plan)==item['plan_sha256']
        candidates=[]
        for path in sorted((RUNTIME/'runs').glob('*/lmeval/lmeval_report.json')):
            r=load(path)
            if r.get('model')!=model:continue
            run=path.parent.parent
            try:
                lr=require_gpu_record(run)
                assert lr['launcher_sha256']==sha(OUT/'scripts/accuracy_run.py'), 'unverified accuracy adapter revision'
                assert lr['inventory_wait_sha256']==sha(OUT/'scripts/gpu_run_wait.py'), 'unverified inventory helper revision'
                # Pin all visible campaign/quantize/models source files to an
                # already accepted new PPL run, not just a directory name.
                prior=list(csv.DictReader((OUT/'results/quality_results.csv').open()))
                sources={x['source'] for x in prior if x['model']==model and x['status']=='validated_new'}
                assert sources, 'no validated new PPL source anchor'
                anchors=[REPO/x for x in sorted(sources)]
                assert any(load(p.parent.parent/'source_hashes.json')==load(run/'source_hashes.json') for p in anchors), 'unverified source drift'
                assert r['plan']==load(expected_plan), 'frozen accuracy plan mismatch'
                assert r['model_class']==ref_report['model_class'] and r['attn_implementation']==ref_report['attn_implementation']
                fresh,freshid=read_policy(path,'four_over_six')
                require_baseline_identity(old,fresh,oldid,freshid)
                bi=next(x for x in r['installs'] if x['name']=='four_over_six')
                assert bi==ref_report['installs'][0], 'baseline installation mismatch'
                datasets={};identities={}
                for policy in ['ce_matched','kl_matched']:
                    name='seed0_'+policy;entry=next(x for x in load(expected_plan) if x['name']==name)
                    install=next(x for x in r['installs'] if x['name']==name)
                    assert install['activation']=='four_over_six_rows'
                    assert install['map_sha256']==entry['map_sha256']==sha(Path(entry['map_path']))
                    datasets[policy],identities[policy]=read_policy(path,name)
                    for key in IDENTITY_KEYS:assert identities[policy][key]==freshid[key],(policy,key)
                    for task in TASKS:assert np.array_equal(old[task]['ids'],datasets[policy][task]['ids']),(policy,task,'unpaired prompts')
                candidates.append((path,datasets,identities,freshid))
            except (AssertionError,KeyError,OSError,ValueError) as exc:
                excluded.append(dict(model=model,source=str(path.relative_to(REPO)),reason=str(exc)))
        assert len(candidates)<=1,(model,'multiple accepted runs; resolve explicitly, do not pick best outcome')
        if not candidates:
            for policy in ['ce_matched','kl_matched']:
                for task in TASKS:rows.append(dict(model=model,draw='seed0',policy=policy,task=task,status='coverage_gap'))
            continue
        path,data,identities,baseline_identity=candidates[0];files=[]
        for task in TASKS:
            dest=OUT/'results/accuracy_new_arrays'/f'{model}_{task}.npz';dest.parent.mkdir(exist_ok=True)
            policies=['baseline','ce_matched','kl_matched'];correct=np.stack([old[task]['correct']]+[data[p][task]['correct'] for p in policies[1:]])
            np.savez_compressed(dest,ids=old[task]['ids'],policies=np.array(policies),correct=correct)
            files.append(dict(path=str(dest.relative_to(OUT)),sha256=sha(dest),task=task))
            for i,policy in enumerate(policies[1:],1):
                rows.append(dict(model=model,draw='seed0',policy=policy,task=task,metric=metric(task),status='validated_new',samples=len(correct[i]),
                    value=float(correct[i].mean()),delta_vs_baseline=float((correct[i]-correct[0]).mean()),
                    map_hash=next(x['map_sha256'] for x in load(expected_plan) if x['name']=='seed0_'+policy),
                    source=str(path.relative_to(REPO)),source_sha256=sha(path),uncertainty='Points only; no accuracy safety or non-inferiority claim'))
        accepted.append(dict(model=model,baseline_identity=baseline_identity,policy_identities=identities,arrays=files))
    csvout(OUT/'results/accuracy_new_results.csv',rows)
    historical=list(csv.DictReader((OUT/'results/secondary_accuracy_coverage.csv').open()))
    coverage=merge_coverage(historical,rows);assert len(coverage)==96
    current=OUT/'results/secondary_accuracy_current.csv';csvout(current,coverage)
    output=OUT/'results/ACCURACY_NEW_AUDIT.json'
    jsonout(output,dict(checked_utc=start,accepted=accepted,excluded=excluded,
        coverage=dict(required=96,historical=sum(x['status']=='sealed_report_reuse' for x in coverage),
            validated_new=sum(x['status']=='validated_new' for x in coverage),missing=sum(x['status']=='coverage_gap' for x in coverage),
            path=str(current.relative_to(OUT)),sha256=sha(current)),
        qualification='Exact baseline responses and prompt IDs required; no new GPU execution by this script. Historical arrays/coverage are not overwritten.'))
    log('accuracy-new-ingestion','python scripts/accuracy_new_results.py',start,[output,OUT/'results/accuracy_new_results.csv',current])
    print('New accuracy accepted models',len(accepted),'excluded runs',len(excluded))

if __name__=='__main__':main()
