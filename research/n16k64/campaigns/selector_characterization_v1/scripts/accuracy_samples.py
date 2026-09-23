"""Audit frozen full8 raw predictions; no inference and no invented intervals."""
import gzip
import hashlib
from common import *
from mechanism_review import sealed
from secondary_accuracy import TASKS

def metric(task):
    return 'acc' if task in ['boolq','winogrande','mmlu'] else 'acc_norm'

def read_policy(path, policy):
    report=load(path);run=path.parent.parent;launch=load(run/'launch_record.json')
    assert report['status']=='complete' and launch['status']=='complete' and not launch.get('invalid_gpu_cotenancy')
    assert report['suite']=='full8' and not report['limit'] and not report['failures']
    data=report['results'][policy];summary=next(iter(data['lm_eval'].values()))
    grouped={t:[] for t in TASKS};files=[]
    for task,meta in sorted(data['samples'].items()):
        group='mmlu' if task.startswith('mmlu_') else task;assert group in grouped
        source=Path(meta['path']);assert sha(source)==meta['file_sha256']
        digest_content=hashlib.sha256();records=[];seen=set()
        with gzip.open(source,'rt') as f:
            for line in f:
                digest_content.update(line.rstrip('\n').encode());r=json.loads(line)
                key=digest(dict(task=task,**{k:r[k] for k in ['doc_id','doc_hash','prompt_hash','target_hash','filter']}))
                assert key not in seen;seen.add(key)
                value=float(r[metric(group)]);assert value in [0.,1.]
                response=digest(r['filtered_resps'])
                records.append((key,value,response))
        assert len(records)==meta['rows'] and digest_content.hexdigest()==meta['content_sha256']
        raw_mean=float(np.mean([r[1] for r in records]))
        assert abs(raw_mean-summary['results'][task][metric(group)+',none'])<1e-12,(task,'raw metric mismatch')
        grouped[group]+=records
        files.append(dict(task=task,path=str(source.relative_to(REPO)),sha256=meta['file_sha256'],
            content_sha256=meta['content_sha256'],rows=len(records)))
    arrays={}
    for task,records in grouped.items():
        assert records,task
        records.sort();keys=[r[0] for r in records];assert len(set(keys))==len(keys)
        values=np.array([r[1] for r in records]);reported=summary['results'][task][metric(task)+',none']
        # MMLU frozen full8 aggregation is question-weighted over subjects.
        assert abs(float(values.mean())-reported)<1e-12,(task,'group aggregation mismatch')
        arrays[task]=dict(ids=np.array(keys),correct=values,response_sha256=np.array([r[2] for r in records]))
    environment=load(run/'job_result.json')['environment']
    identity={k:environment.get(k) for k in ['torch','transformers','cuda','cudnn','compute_capability','gpu_names','lock_sha256']}
    return arrays,dict(source=str(path.relative_to(REPO)),source_sha256=sha(path),policy=policy,
        batch_size=report['batch_size'],lm_eval_version=report['lm_eval_version'],suite_spec=report['suite_spec'],
        module_manifest_sha256=report['module_manifest_sha256'],environment=identity,files=files)

def main():
    start=now();coverage=list(csv.DictReader((OUT/'results/secondary_accuracy_coverage.csv').open()))
    audit=[];metrics=[];paired=[]
    for model in MODELS:
        datasets={};identities={}
        for policy in ['baseline','joint']:
            candidates={r['source'] for r in coverage if r['model']==model and r['policy']==policy and r['status']=='sealed_report_reuse'}
            assert len(candidates)==1
            path=REPO/candidates.pop();sealed(path);report=load(path)
            name='four_over_six' if policy=='baseline' else 'n16_k3'
            datasets[policy],identities[policy]=read_policy(path,name)
            audit.append(dict(model=model,logical_policy=policy,**identities[policy]))
            print('Raw accuracy audited',model,policy,flush=True)
        for key in ['batch_size','lm_eval_version','suite_spec','module_manifest_sha256','environment']:
            assert identities['baseline'][key]==identities['joint'][key],(model,key)
        for task in TASKS:
            b=datasets['baseline'][task];j=datasets['joint'][task]
            assert np.array_equal(b['ids'],j['ids']),(model,task,'prompt/target identities differ')
            path=OUT/'results/accuracy_arrays'/f'{model}_{task}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,ids=b['ids'],policies=np.array(['baseline','joint']),
                correct=np.stack([b['correct'],j['correct']]),
                response_sha256=np.stack([b['response_sha256'],j['response_sha256']]))
            paired.append(dict(model=model,task=task,path=str(path.relative_to(OUT)),sha256=sha(path),samples=len(b['ids'])))
            for policy in ['baseline','joint']:
                values=datasets[policy][task]['correct']
                metrics.append(dict(model=model,draw='seed0',policy=policy,task=task,metric=metric(task),samples=len(values),
                    value=float(values.mean()),status='raw_samples_verified',
                    delta_vs_baseline=float(values.mean()-b['correct'].mean()),
                    uncertainty='Not newly estimated; points only. No accuracy safety/non-inferiority claim from this audit.'))
    csvout(OUT/'results/accuracy_sample_metrics.csv',metrics)
    jsonout(OUT/'results/ACCURACY_SAMPLE_AUDIT.json',dict(status='verified_historical_samples',runs=audit,paired_arrays=paired,
        qualification='Hash/identity/metric recomputation only. Historical results not relabeled as new experiments; matched-policy coverage remains missing.'))
    log('accuracy-samples','python scripts/accuracy_samples.py',start,[OUT/'results/ACCURACY_SAMPLE_AUDIT.json',OUT/'results/accuracy_sample_metrics.csv'])
    print('Paired model/task arrays',len(paired),'metrics',len(metrics))

if __name__=='__main__':main()
