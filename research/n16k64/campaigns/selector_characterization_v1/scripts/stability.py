"""T1 plus outcome-blind T2 policy construction. Bounded, one CPU thread."""
import itertools
import time
import torch
from scipy.stats import spearmanr
from common import *

def main():
    start=now();print('TASK_STATUS',[(r['id'],r['status']) for r in load(OUT/'TASK_STATUS.json')['tasks']],flush=True)
    protocol=load(OUT/'FROZEN_PROTOCOL.yaml');inputs=load(OUT/'results/INPUTS.json')
    counts=[];pairs=[];freq=[];ranks=[];quotas=[];veto=[];mapmanifest=[];summaries=[]
    for model in MODELS:
        print('T1 model',model,flush=True)
        rows=[next(r for r in inputs if r['model']==model and r['draw']==d) for d in DRAWS]
        states=[torch.load(REPO/r['moments'],map_location='cpu',weights_only=True,mmap=True) for r in rows]
        maps=[];heads=[]
        for r in rows:
            h,a=mapread(REPO/next(x['path'] for x in r['maps'] if x['rule']=='ce_kl'));heads.append(h);maps.append(a)
        names=sorted(maps[0]);assert all(set(m)==set(names) for m in maps)
        assert all(states[i]['shapes']==states[0]['shapes'] for i in range(5))
        universe=[maps[0][n].size for n in names]
        ka=[[int(m[n].sum()) for n in names] for m in maps]
        vectors=[np.concatenate([m[n].ravel() for n in names]) for m in maps]
        ranking={key:[{} for _ in DRAWS] for key in ['joint_upper_primary','joint_z_diagnostic']}
        global_scores=[[] for _ in DRAWS]
        policies=[{p:{} for p in ['ce_natural','kl_natural','joint','ce_matched','kl_matched']} for _ in DRAWS]
        for ix,name in enumerate(names):
            shape=maps[0][name].shape;u=maps[0][name].size;quota=min(k[ix] for k in ka)
            quotas.append(dict(model=model,module=name,U=u,K_min=quota,zero_quota=quota==0))
            frequencies=sum(m[name].astype(np.uint8) for m in maps)
            for f in range(6):freq.append(dict(model=model,module=name,frequency=f,tiles=int((frequencies==f).sum()),U=u))
            for d,draw in enumerate(DRAWS):
                st=states[d]['n16'][name];mc,sc=moments(st,'ce');mk,sk=moments(st,'kl');uce=mc+3*sc;ukl=mk+3*sk
                cp=uce<0;kp=ukl<0;j=cp&kp;assert np.array_equal(j,maps[d][name].ravel())
                us=np.maximum(uce,ukl);zs=np.maximum(zscore(mc,sc),zscore(mk,sk));global_scores[d].append(us)
                for key,score in [('joint_upper_primary',us),('joint_z_diagnostic',zs)]:
                    a,tie=topmask(score,quota);ranking[key][d][name]=a.reshape(shape)
                    quotas.append(dict(model=model,draw=draw,module=name,ranking=key,U=u,K_min=quota,cutoff_tie_fraction=tie,
                                       retained_quota_fraction=quota/int(j.sum()) if j.any() else None))
                q=int(j.sum());ce,ct=topmask(uce,q);kl,kt=topmask(ukl,q)
                assert int(ce.sum())==int(kl.sum())==q and not (ce&~cp).any() and not (kl&~kp).any()
                for p,a in [('ce_natural',cp),('kl_natural',kp),('joint',j),('ce_matched',ce),('kl_matched',kl)]:policies[d][p][name]=a.reshape(shape)
                counts.append(dict(model=model,draw=draw,module=name,U=u,K=q,density=q/u,density_ratio_vs_seed0=q/ka[0][ix] if ka[0][ix] else None))
                for approving,pass_a,pass_b,mean_b,upper_b in [('ce',cp,kp,mk,ukl),('kl',kp,cp,mc,uce)]:
                    vv=pass_a&~pass_b
                    veto.append(dict(model=model,draw=draw,module=name,approving=approving,veto_total=int(vv.sum()),
                        predicted_harmful=int((vv&(mean_b>0)).sum()),favorable_uncertainty_fail=int((vv&(mean_b<0)&(upper_b>=0)).sum()),
                        neutral=int((vv&(mean_b==0)).sum()),invalid=0,near_zero_upper_abs_le_1e_10=int((vv&(np.abs(upper_b)<=1e-10)).sum()),
                        both_means_negative=int(((mc<0)&(mk<0)).sum()),ce_negative_kl_nonnegative=int(((mc<0)&(mk>=0)).sum()),
                        ce_nonnegative_kl_negative=int(((mc>=0)&(mk<0)).sum()),both_nonnegative=int(((mc>=0)&(mk>=0)).sum()),
                        both_pass=int(j.sum()),only_ce_pass=int((cp&~kp).sum()),only_kl_pass=int((kp&~cp).sum()),neither_pass=int((~cp&~kp).sum())))
        U=sum(universe)
        for d,draw in enumerate(DRAWS):
            K=sum(ka[d]);counts.append(dict(model=model,draw=draw,module='ALL',U=U,K=K,density=K/U,density_ratio_vs_seed0=K/sum(ka[0])))
            for p,m in policies[d].items():
                # Canonical map writer preserves original per-module serialization order.
                ordered={x['name']:m[x['name']] for x in heads[d]['modules']}
                path=OUT/'maps'/f'{model}_{draw}_{p}.mixfp4map'
                h=mapwrite(path,heads[d],ordered,dict(name=p,rule=p,k=3,draw=draw),16)
                mapmanifest.append(dict(task='T2',model=model,draw=draw,N=16,policy=p,path=str(path.relative_to(OUT)),sha256=h,payload_sha256=maskhash(ordered),
                                        K=sum(int(v.sum()) for v in m.values()),U=U,compatibility_key=rows[d]['compatibility_key'],
                                        calibration_manifest_sha256=rows[d]['calibration_manifest_sha256']))
        def add_pairs(target,masks,kind,scope):
            flat=[np.concatenate([m[n].ravel() for n in names]) for m in masks]
            alloc=[[int(m[n].sum()) for n in names] for m in masks]
            for i,j in itertools.combinations(range(5),2):
                row=dict(model=model,draw_a=DRAWS[i],draw_b=DRAWS[j],ranking=kind,scope=scope,U=U,**pair(flat[i],flat[j]))
                seed=20260922+int(digest([model,kind,scope,i,j])[:8],16)
                row.update(null_pairs(universe,alloc[i],alloc[j],seed))
                aa=np.array(alloc[i],float);bb=np.array(alloc[j],float)
                row['allocation_share_L1']=float(np.abs(aa/aa.sum()-bb/bb.sum()).sum()) if aa.sum() and bb.sum() else None
                row['allocation_spearman']=float(spearmanr(aa,bb).statistic) if aa.std()>0 and bb.std()>0 else None
                target.append(row)
        add_pairs(pairs,maps,'natural_joint','natural')
        for kind,m in ranking.items():add_pairs(ranks,m,kind,'fixed_module_min_quota')
        # Global fixed-fraction diagnostic deliberately does not preserve allocations.
        gs=[np.concatenate(x) for x in global_scores];orders=[np.argsort(x,kind='stable') for x in gs]
        offsets=np.cumsum([0]+universe)
        for fraction in protocol['stability']['global_fractions']:
            k=int(np.floor(U*fraction));ms=[]
            for d in range(5):
                a=np.zeros(U,bool);a[orders[d][:k]]=True
                ms.append({n:a[offsets[i]:offsets[i+1]].reshape(maps[d][n].shape) for i,n in enumerate(names)})
            add_pairs(ranks,ms,'joint_upper_primary',f'global_fraction_{fraction}')
        k=np.array([sum(x) for x in ka]);summaries.append(dict(model=model,U=U,K=k.tolist(),mean_K=float(k.mean()),sample_sd_K=float(k.std(ddof=1)),
            min_K=int(k.min()),max_K=int(k.max()),common_core=int(np.stack(vectors).all(0).sum()),
            selected_union=int(np.stack(vectors).any(0).sum()),fixed_module_min_quota=sum(min(x) for x in zip(*ka)),
            zero_quota_modules=sum(min(x)==0 for x in zip(*ka)),eligible_modules=len(names),excluded_eligibility=0))
        print('T1 finished',model,summaries[-1],flush=True)
        del states,maps,policies,ranking,global_scores,gs,orders,vectors
    files=[]
    for name,data in [('map_counts',counts),('map_pairs',pairs),('selection_frequency',freq),('ranking_overlap',ranks),('quota_diagnostics',quotas),('veto',veto)]:
        p=OUT/'results'/f'{name}.csv';csvout(p,data);files.append(p)
    jsonout(OUT/'results/MAP_MANIFEST.json',mapmanifest);jsonout(OUT/'results/STABILITY_SUMMARY.json',summaries)
    files += [OUT/'results/MAP_MANIFEST.json',OUT/'results/STABILITY_SUMMARY.json']
    log('T1','OMP_NUM_THREADS=1 python scripts/stability.py',start,files)
    checkpoint('T1','done',[str(x.relative_to(OUT)) for x in files],next_action='T2：驗證每 evaluation 身份與 SHA，重用相同 map，列出其餘 cells。T3 尋找完整 N8 scores；不以 marginal SE 冒充精確 parent SE。')

if __name__=='__main__':
    started=now()
    try:main()
    except Exception as e:
        log('T1','scripts/stability.py',started,[],status='failed',error=repr(e));raise
