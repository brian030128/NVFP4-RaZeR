"""Exact nested moments where available; otherwise mean-only bounds, not fake SE."""
import torch
from common import *

def main():
    start=now();inputs=load(OUT/'results/INPUTS.json');protocol=load(OUT/'FROZEN_PROTOCOL.yaml')
    print('TASK_STATUS',[(x['id'],x['status']) for x in load(OUT/'TASK_STATUS.json')['tasks']],flush=True)
    rows=[];maps=[];checks=[];blocked=[]
    for model in MODELS:
        r=next(r for r in inputs if r['model']==model and r['draw']=='seed0')
        state=torch.load(REPO/r['moments'],map_location='cpu',weights_only=True,mmap=True)
        head,reference=mapread(REPO/next(m['path'] for m in r['maps'] if m['rule']=='ce_kl'))
        raw=None;regenerated={};regenerated_source=None
        if r['full_n8_scores_available']:
            f=REPO/r['full_n8_scores_path'];assert sha(f)==r['raw_expected_sha256']
            raw=torch.load(f,map_location='cpu',weights_only=True,mmap=True)
            assert raw['names']==state['names'] and raw['shapes']==state['shapes']
        approved=OUT/'results/PARENT_MOMENT_REUSE.json'
        if raw is None and approved.exists():
            candidates=[x for x in load(approved) if x['model']==model]
            if candidates:
                chosen=candidates[0];regenerated_source=chosen['run']+'/parent_moments/manifest.json'
                assert sha(REPO/regenerated_source)==chosen['manifest_sha256']
                for N,item in chosen['files'].items():
                    assert sha(REPO/item['path'])==item['sha256']
                    regenerated[int(N)]=torch.load(REPO/item['path'],weights_only=True,mmap=True,map_location='cpu')
        names=state['names'];totalweights=sum(np.prod(state['shapes'][n]) for n in names)
        for N in [8,16,32,64,128,256]:
            exact=N in [8,16] or raw is not None or N in regenerated;masks={};stats=[];account=[]
            for name in names:
                shape=state['shapes'][name];ro,co=shape
                assert ro%N==0 and co%64==0
                base={o:moments(state['n8'][name],o) for o in ['ce','kl']}
                parent={};children={}
                for o in ['ce','kl']:
                    mu8,se8=base[o]
                    children[o]=mu8.reshape(ro//N,N//8,co//64).transpose(0,2,1).reshape(-1,N//8)
                    mu=children[o].sum(-1);se=None
                    if N in [8,16]:
                        oldmu,se=moments(state[f'n{N}'][name],o)
                        assert np.allclose(mu,oldmu,atol=1e-12,rtol=1e-10)
                        mu=oldmu
                    if raw is not None:
                        x=raw['scores'][name][o].numpy().astype(np.float64)
                        y=aggregate_scores(x,ro,co,N);rm,rs=mean_se_array(y)
                        assert np.allclose(rm,mu,atol=1e-12,rtol=1e-10)
                        if se is not None:assert np.allclose(rs,se,atol=1e-12,rtol=1e-8)
                        mu,se=rm,rs
                        # Compare exact direct aggregation with hierarchical pairing.
                        if N>=16:
                            half=aggregate_scores(x,ro,co,N//2).reshape(128,ro//N,2,co//64).sum(2).reshape(128,-1)
                            assert np.allclose(half,y,atol=1e-12,rtol=1e-10)
                    elif N in regenerated:
                        rm,se=moments(regenerated[N][name],o)
                        assert np.allclose(rm,mu,atol=1e-12,rtol=1e-10)
                        mu=rm
                    parent[o]=(mu,se)
                    if se is not None:
                        child_se=se8.reshape(ro//N,N//8,co//64).sum(1).ravel()
                        assert (se<=child_se+1e-12).all()
                        every=(mu8+3*se8<0).reshape(ro//N,N//8,co//64).all(1).ravel()
                        assert ((mu+3*se<0)|~every).all()
                if exact:
                    cp=parent['ce'][0]+3*parent['ce'][1]<0;kp=parent['kl'][0]+3*parent['kl'][1]<0
                    joint=cp&kp;masks[name]=joint.reshape(ro//N,co//64)
                    if N==16:assert np.array_equal(masks[name],reference[name]),(model,name,'N16 anchor mismatch')
                    fine=(base['ce'][0]+3*base['ce'][1]<0)&(base['kl'][0]+3*base['kl'][1]<0)
                    expanded=np.repeat(masks[name][:,None,:],N//8,axis=1).reshape(-1)
                    account.append(dict(fine_selected_overridden=int((fine&~expanded).sum()),fine_unselected_promoted=int((~fine&expanded).sum())))
                for o in ['ce','kl']:
                    mu,se=parent[o];ch=children[o];fine=np.minimum(ch,0).sum(-1);coarse=np.minimum(mu,0);own=coarse-fine
                    mass=np.abs(ch).sum(-1);c=np.full_like(mu,np.nan);np.divide(2*own,mass,out=c,where=mass>0)
                    assert (own>=-1e-12).all()
                    row=dict(model=model,draw='seed0',N=N,module=name,objective=o,backend_status='analytical_only',exact_parent_SE=exact,
                        eligible_weights=int(ro*co),eligible_parents=len(mu),ownership_regret=float(own.sum()),sum_abs_child_mean=float(mass.sum()),
                        weighted_cancellation=float(2*own.sum()/mass.sum()) if mass.sum()>0 else None,
                        unweighted_cancellation_mean=float(np.nanmean(c)) if np.isfinite(c).any() else None,zero_score_parents=int((mass==0).sum()),
                        minority_sign_fraction=float(np.minimum((ch<0).sum(-1),(ch>0).sum(-1)).sum()/ch.size),
                        mean_favorable_parents=int((mu<0).sum()))
                    if exact:
                        other=kp if o=='ce' else cp;reg=regret(ch,mu,se,other)
                        row.update(selector_gap=float((reg['threshold']+reg['veto']).sum()),threshold_increment=float(reg['threshold'].sum()),
                            veto_increment=float(reg['veto'].sum()),total_surrogate_gap=float(reg['total'].sum()),
                            selected_tiles=int(joint.sum()),selected_weights=int(joint.sum())*N*64,
                            mean_favorable_threshold_fail=int(((mu<0)&(mu+3*se>=0)).sum()),
                            mean_favorable_threshold_fail_fraction=float(((mu<0)&(mu+3*se>=0)).sum()/(mu<0).sum()) if (mu<0).any() else None)
                    stats.append(row)
            rows.extend(stats)
            if exact:
                path=OUT/'maps'/f'{model}_seed0_n{N}_joint.mixfp4map'
                h=mapwrite(path,head,masks,dict(name=f'n{N}_joint',rule='ce_kl',k=3,draw='seed0'),N)
                maps.append(dict(model=model,draw='seed0',N=N,policy=f'n{N}_joint',path=str(path.relative_to(OUT)),sha256=h,payload_sha256=maskhash(masks),
                                 selected_weights=sum(int(a.sum()) for a in masks.values())*N*64,total_weights=int(totalweights),
                                 selected_weight_fraction=sum(int(a.sum()) for a in masks.values())*N*64/totalweights,
                                 fine_selected_overridden_weights=sum(a['fine_selected_overridden'] for a in account)*512,
                                 fine_unselected_promoted_weights=sum(a['fine_unselected_promoted'] for a in account)*512,
                                 covariance_source=r['full_n8_scores_path'] if raw is not None else (regenerated_source if N in regenerated else r['moments'])))
            else:blocked.append(dict(model=model,N=N,reason='No full N8 per-sequence scores or complete child covariance; only mean ownership regret valid; exact parent SE and joint map unavailable'))
            print('T3 derived',model,N,'exact SE',exact,flush=True)
        del state,raw
    csvout(OUT/'results/granularity_modules.csv',rows);jsonout(OUT/'results/GRANULARITY_MAP_MANIFEST.json',maps)
    jsonout(OUT/'results/GRANULARITY_BLOCKERS.json',blocked)
    summaries=[]
    for model in MODELS:
        for N in [8,16,32,64,128,256]:
            for o in ['ce','kl']:
                subset=[r for r in rows if r['model']==model and r['N']==N and r['objective']==o]
                entry=dict(model=model,draw='seed0',N=N,objective=o,backend_status='analytical_only',exact_parent_SE=subset[0]['exact_parent_SE'])
                for key in ['ownership_regret','selector_gap','threshold_increment','veto_increment','total_surrogate_gap','selected_weights','eligible_weights']:
                    entry[key]=sum(r[key] for r in subset) if key in subset[0] else None
                mass=sum(r['sum_abs_child_mean'] for r in subset);entry['weighted_cancellation']=2*entry['ownership_regret']/mass if mass else None
                entry['selected_weight_fraction']=entry['selected_weights']/entry['eligible_weights'] if entry['selected_weights'] is not None else None
                entry['measured_nll_gap_vs_n8']=None;entry['gain_retention']=None;entry['retention_status']='Not computed by surrogate analysis; actual evaluation required'
                summaries.append(entry)
    csvout(OUT/'results/granularity_results.csv',summaries)
    files=[OUT/'results/granularity_results.csv',OUT/'results/granularity_modules.csv',OUT/'results/GRANULARITY_MAP_MANIFEST.json',OUT/'results/GRANULARITY_BLOCKERS.json']
    log('T3-CPU','python scripts/granularity.py',start,files)
    gatefile=OUT/'results/ANCHOR_GATE.json';gates=load(gatefile) if gatefile.exists() else {}
    gate_blockers=[m+': numerical prefix anchor not passed' for m in MODELS if not gates.get(m,{}).get('passed')]
    checkpoint('T3','running',[str(p.relative_to(OUT)) for p in files],gate_blockers+[r['model']+f'/N{r["N"]}: '+r['reason'] for r in blocked],
               next_action='依 GRANULARITY_BLOCKERS 補 exact parent moments；先驗證 N8/N16 score identity，再用 frozen maps 補 measured quality。檢查 live launchers，勿重複排程或更改 tolerance。')

if __name__=='__main__':
    started=now()
    try:main()
    except Exception as e:log('T3-CPU','scripts/granularity.py',started,[],status='failed',error=repr(e));raise
