"""Validate/reuse actual per-token outcomes; create every missing logical cell.

No model inference. Report points/paired cluster CI separately from across-draw
descriptive variation. Identical masks retain distinct model/draw/policy rows.
"""
import itertools
import math
import subprocess
from common import *

RUNTIME=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z'

def result_status(record):
    return 'validated_new' if RUNTIME.name in Path(record['source']).parts else 'validated_reuse'

def baseline_agrees(candidate, reference):
    """Full-window anchor: paired identity plus original 1e-10 NLL tolerance."""
    if candidate['compatibility_key'] != reference['compatibility_key']:
        return False
    for key in ['cluster_ids', 'weight_sha256']:
        if candidate[key] != reference[key]:
            return False
    if not np.array_equal(candidate['window_tokens'], reference['window_tokens']):
        return False
    return bool(np.allclose(candidate['window_nll'], reference['window_nll'], atol=1e-10, rtol=0))

def validated_records(records, excluded):
    """Never splice fresh outcomes into old baselines on a two-window pilot alone."""
    accepted=[]
    for record in records:
        if result_status(record)=='validated_reuse':
            accepted.append(record);continue
        reference=[x for x in records if x['model']==record['model'] and x['corpus']==record['corpus']
                   and x['policy']=='four_over_six' and ('V31_ppl_primary' in x['source'] or 'V62_ppl_primary' in x['source'])]
        current=[x for x in records if x['source']==record['source'] and x['corpus']==record['corpus'] and x['policy']=='four_over_six']
        if len(reference)==1 and len(current)==1 and baseline_agrees(current[0],reference[0]):
            accepted.append(record)
        else:
            excluded.append(dict(path=record['source'],policy=record['policy'],corpus=record['corpus'],
                                 reason='Fresh run lacks matching full-window baseline anchor; retained but not admitted'))
    return accepted,excluded

def cluster(meta,dom):
    keys=([x['document_sha256'] for x in meta['documents']] if dom=='c4' else [f"article-{x['first_article']}" for x in meta['window_articles']])
    unique=sorted(set(keys));lookup={v:i for i,v in enumerate(unique)}
    return np.array([lookup[x] for x in keys]),unique

def collect():
    records=[];excluded=[];hashcache={};payloadcache={}
    for campaign in [PRIMARY,FOLLOWUP,RUNTIME]:
        for p in sorted((campaign/'runs').glob('*/ppl/ppl_report.json')):
            r=load(p);run=p.parent.parent;lr=load(run/'launch_record.json')
            if r.get('model') not in MODELS:continue
            if r.get('status')!='complete' or lr.get('status')!='complete' or lr.get('invalid_gpu_cotenancy'):
                excluded.append(dict(path=str(p.relative_to(REPO)),reason='not valid complete attempt'));continue
            if r.get('length')!=2048 or r.get('attn_implementation')!='sdpa':continue
            if r.get('evaluation_stage') not in [None,'full']:continue
            for dom in ['wiki','c4']:
                wp=p.parent/f'windows_{dom}.json'
                if not wp.exists():continue
                meta=load(wp)
                if meta.get('limited_to') or r['windows'].get(dom)!=len(meta['token_sha256']):continue
                cl,ids=cluster(meta,dom)
                if len(cl)!=len(meta['token_sha256']):continue
                environment=load(run/'job_result.json')['environment']
                runtime_identity={k:environment.get(k) for k in ['torch','transformers','cuda','cudnn','compute_capability','gpu_names','lock_sha256']}
                identity=digest(dict(model=r['spec']['model_id'],revision=r['spec']['revision'],module=r['module_manifest_sha256'],runtime=runtime_identity,
                                     dataset=[meta['repo'],meta['revision'],meta['path']],tokens=meta['token_sha256'],clusters=ids,cluster_assignment=cl.tolist()))
                installs={x['name']:x for x in r['installs']}
                for policy,ev in r['evaluation'].items():
                    if dom not in ev or policy not in installs:continue
                    ins=installs[policy]
                    if ins.get('activation')!='four_over_six_rows' or ins.get('weight') not in ['map','four_over_six']:continue
                    e=ev[dom];payload=None;cal=None;N=None
                    if ins.get('map_path'):
                        f=Path(ins['map_path'])
                        if not f.exists():excluded.append(dict(path=str(p.relative_to(REPO)),policy=policy,reason='map absent'));continue
                        if str(f) not in payloadcache:
                            assert sha(f)==ins['map_sha256'];h,m=mapread(f);payloadcache[str(f)]=(maskhash(m),h['calibration_manifest_sha256'],h['type_block'][0])
                        payload,cal,N=payloadcache[str(f)]
                    toks=np.array([w['tokens'] for w in e['windows']],np.int64)
                    assert len(toks)==len(cl)
                    sums=np.array([w['nll_sum'] for w in e['windows']],np.float64)
                    ap=None;ah=None
                    if 'token_arrays' in e:
                        ap=Path(e['token_arrays']['path'])
                        assert ap.exists(),ap
                        if str(ap) not in hashcache:hashcache[str(ap)]=sha(ap)
                        ah=hashcache[str(ap)];assert ah==e['token_arrays']['sha256'],ap
                        with np.load(ap,allow_pickle=False) as z:values=z['nll'].astype(np.float64)
                        assert toks.sum()==len(values)
                        offset=np.r_[0,toks.cumsum()];raw=np.array([values[offset[i]:offset[i+1]].sum() for i in range(len(toks))])
                        assert np.allclose(raw,sums,atol=1e-9,rtol=0)
                    else:
                        # Per-window sufficient statistics support cluster inference;
                        # no invented token-level detail or unnecessary GPU rerun.
                        seal=PRIMARY/'runs/V83_validate_artifacts_attempt7/artifact_validation/ARTIFACT_MANIFEST.sha256'
                        entries={line.split(maxsplit=1)[1].strip().lstrip('*'):line.split()[0] for line in seal.read_text().splitlines()}
                        runseal=run/'SHA256SUMS_run.txt'
                        assert campaign==PRIMARY and entries[runseal.relative_to(PRIMARY).as_posix()]==sha(runseal)
                        members={line.split(maxsplit=1)[1].strip().lstrip('*'):line.split()[0] for line in runseal.read_text().splitlines()}
                        assert members[p.relative_to(run).as_posix()]==sha(p)
                    assert abs(sums.sum()/toks.sum()-e['mean_nll'])<1e-10
                    loss=np.bincount(cl,weights=sums);ct=np.bincount(cl,weights=toks).astype(np.int64)
                    records.append(dict(model=r['model'],corpus=dom,N=N,policy=policy,payload=payload,calibration_header_hash=cal,
                        compatibility_key=identity,cluster_ids=ids,cluster_tokens=ct,cluster_loss=loss,
                        window_tokens=toks,window_nll=sums/toks,
                        source=str(p.relative_to(REPO)),source_sha256=sha(p),token_array=str(ap.relative_to(REPO)) if ap else None,token_array_sha256=ah,
                        data_level='token_array_checked' if ap else 'sealed_per_window_loss_sums_and_counts',
                        map_sha256=ins.get('map_sha256'),weight_sha256=ins.get('installed_weight_sha256'),
                        source_freeze_sha256=r['freeze_sha256'],historical_protocol=r['protocol_id']))
            print('audited evaluation',run.name,flush=True)
    return validated_records(records,excluded)

def main():
    start=now();print('TASK_STATUS',[(x['id'],x['status']) for x in load(OUT/'TASK_STATUS.json')['tasks']],flush=True)
    records,excluded=collect();maps=load(OUT/'results/MAP_MANIFEST.json');inputs=load(OUT/'results/INPUTS.json')
    # Baseline anchored to the primary evaluation, not latest or best outcome.
    baselines={}
    for m in MODELS:
        for c in ['wiki','c4']:
            candidates=[r for r in records if r['model']==m and r['corpus']==c and r['policy']=='four_over_six' and ('V31_ppl_primary' in r['source'] or 'V62_ppl_primary' in r['source'])]
            assert len(candidates)==1,(m,c,len(candidates));baselines[m,c]=candidates[0]
    quality=[];jobs=[];reuse=[];panels={};contrasts=[]
    for m in MODELS:
        for c in ['wiki','c4']:
            b=baselines[m,c];items=[('baseline',b,None)]
            for mm in [x for x in maps if x['model']==m]:
                label=mm['draw']+'/'+mm['policy'];matches=[r for r in records if r['model']==m and r['corpus']==c and r['payload']==mm['payload_sha256'] and r['compatibility_key']==b['compatibility_key']]
                row=dict(run_id=m+'/'+label+'/'+c,task='T2',model=m,draw=mm['draw'],N=16,policy=mm['policy'],corpus=c,map_hash=mm['sha256'],map_payload_sha256=mm['payload_sha256'],
                         config_hash=digest(dict(map=mm['sha256'],eval=b['compatibility_key'])),status='pending',reuse_source='',estimated_gpu_hours='',command='',output_path='',blocker='')
                if matches:
                    # Prioritize primary records; reject disagreeing duplicates rather than pick favorable ones.
                    matches.sort(key=lambda r:('followup' in r['source'],r['source']))
                    rr=matches[0]
                    for other in matches[1:]:
                        assert np.array_equal(rr['cluster_tokens'],other['cluster_tokens'])
                        assert np.allclose(rr['cluster_loss'],other['cluster_loss'],rtol=0,atol=1e-6),(rr['source'],other['source'],'same-map duplicate mismatch')
                    items.append((label,rr,mm));row.update(status=result_status(rr),reuse_source=rr['source'])
                    reuse.append({k:v for k,v in rr.items() if k not in ['cluster_loss','cluster_tokens','window_tokens','window_nll']}|dict(logical_run_id=row['run_id'],logical_map=mm['sha256'],status=result_status(rr)))
                else:
                    cachearg='' if m=='llama8b' else ' --hub-cache /share3/saves/JAAAAAA/mixfp4_n16k64_followup_20260917T090438Z/cache/hf/hub'
                    row.update(status='pending_evaluation',blocker='No hash/protocol/window-matched completed evaluation; requires model anchor gate',
                               command=f'python scripts/gpu_run.py --name eval_{m}_{mm["draw"]}_attempt1 --model {m} --plan plans/{m}_{mm["draw"]}_anchored.json --teacher instance --device-type a6000'+cachearg)
                    quality.append(dict(run_id=row['run_id'],model=m,draw=mm['draw'],N=16,policy=mm['policy'],corpus=c,map_hash=mm['sha256'],status='pending_evaluation'))
                jobs.append(row)
            losses=np.stack([x[1]['cluster_loss'] for x in items]);tokens=b['cluster_tokens']
            assert all(np.array_equal(x[1]['cluster_tokens'],tokens) for x in items)
            estimate,bs=paired_bootstrap(losses,tokens,2000,20260921)
            path=OUT/'results/arrays'/f'{m}_{c}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,policies=np.array([x[0] for x in items]),cluster_ids=np.array(b['cluster_ids']),cluster_tokens=tokens,cluster_nll_sum=losses)
            for i,(label,r,mm) in enumerate(items):
                lo,hi=np.quantile(bs[:,i]-bs[:,0],[.025,.975]);draw,pol=label.split('/') if '/' in label else ('shared','baseline')
                quality.append(dict(run_id=f'{m}/{label}/{c}',model=m,draw=draw,N=16,policy=pol,corpus=c,baseline_hash=b['source_sha256'],
                    map_hash=mm['sha256'] if mm else '',nll=float(estimate[i]),ppl=math.exp(estimate[i]),delta_nll=float(estimate[i]-estimate[0]),
                    relative_ppl=math.expm1(estimate[i]-estimate[0]),ci_low=float(lo),ci_high=float(hi),valid_tokens=int(tokens.sum()),doc_count=len(tokens),
                    status=result_status(r),bootstrap_repeats=2000,ci_status='pointwise exploratory; no simultaneous or significance claim',source=r['source']))
            indexes={label:i for i,(label,_,_) in enumerate(items)}
            for draw in DRAWS:
                for left,right in [('ce_matched','joint'),('kl_matched','joint'),('ce_natural','joint'),('kl_natural','joint'),('ce_natural','kl_natural')]:
                    a=draw+'/'+left;z=draw+'/'+right
                    row=dict(model=m,draw=draw,corpus=c,contrast=left+'-minus-'+right,role='primary' if 'matched' in left else 'secondary')
                    if a in indexes and z in indexes:
                        i,j=indexes[a],indexes[z];lo,hi=np.quantile(bs[:,i]-bs[:,j],[.025,.975]);row.update(status='validated_new' if any(result_status(items[k][1])=='validated_new' for k in [i,j]) else 'validated_reuse',delta_nll=float(estimate[i]-estimate[j]),ci_low=float(lo),ci_high=float(hi))
                    else:row['status']='pending_evaluation'
                    contrasts.append(row)
    # One plan per model/draw; only missing maps, deduplicate corpus requests.
    for m,draw in itertools.product(MODELS,DRAWS):
        missing={j['policy'] for j in jobs if j['model']==m and j['draw']==draw and j['status']=='pending_evaluation'}
        plan=[]
        for mm in maps:
            if mm['model']==m and mm['draw']==draw and mm['policy'] in missing:
                plan.append(dict(name=draw+'_'+mm['policy'],kind='map',map_path=str(OUT/mm['path']),map_sha256=mm['sha256'],map_policy=mm['policy'],
                                 type_block=[16,64],protocol_id='selector_characterization_v1'))
        if plan:
            path=OUT/'plans'/f'{m}_{draw}_anchored.json'
            proposed=[dict(name='four_over_six',kind='four_over_six')]+plan
            if path.exists():
                # Never replace a plan potentially read by a live launcher.
                # Previously frozen extra arms may now be reusable; keep the
                # original plan intact and record missing cells in job_plan.
                frozen=load(path)
                assert all(entry in frozen for entry in proposed),(path,'new arm requires a separately frozen plan')
            else:jsonout(path,proposed)
    desc=[]
    for m,c,p in itertools.product(MODELS,['wiki','c4'],['ce_natural','kl_natural','joint','ce_matched','kl_matched']):
        values=[r['delta_nll'] for r in quality if r['model']==m and r['corpus']==c and r['policy']==p and r['status'] in ['validated_reuse','validated_new']]
        desc.append(dict(model=m,corpus=c,policy=p,draws_available=len(values),coverage_complete=len(values)==5,
                         mean=np.mean(values).item() if values else None,sample_sd=np.std(values,ddof=1).item() if len(values)>1 else None,
                         minimum=min(values) if values else None,observed_worst=max(values) if values else None,regressions=sum(v>0 for v in values)))
    csvout(OUT/'results/quality_results.csv',quality);csvout(OUT/'job_plan.csv',jobs);csvout(OUT/'results/quality_contrasts.csv',contrasts);csvout(OUT/'results/draw_variation.csv',desc)
    jsonout(OUT/'results/EVALUATION_REUSE.json',reuse);jsonout(OUT/'results/EXCLUDED_EVALUATIONS.json',excluded)
    n=sum(j['status']=='pending_evaluation' for j in jobs);print('T2 logical cells',len(jobs),'verified',len(jobs)-n,'pending',n,flush=True)
    log('T2-reuse','python scripts/quality.py',start,[OUT/'results/quality_results.csv',OUT/'job_plan.csv',OUT/'results/EVALUATION_REUSE.json'])
    accuracy_audit=OUT/'results/ACCURACY_NEW_AUDIT.json'
    accuracy_missing=load(accuracy_audit)['coverage']['missing'] if accuracy_audit.exists() else 48
    checkpoint('T2','running' if n or accuracy_missing else 'done',['results/quality_results.csv','results/quality_contrasts.csv','results/veto.csv','job_plan.csv','results/ACCURACY_NEW_AUDIT.json'],
               next_action=f'T2 primary PPL 尚缺 {n} model/corpus cells；secondary accuracy 尚缺 {accuracy_missing} task cells。確認 live runs 後依 frozen plans 補缺；T3 仍須模型層級驗證。')

if __name__=='__main__':
    started=now()
    try:main()
    except Exception as e:log('T2-reuse','scripts/quality.py',started,[],status='failed',error=repr(e));raise
