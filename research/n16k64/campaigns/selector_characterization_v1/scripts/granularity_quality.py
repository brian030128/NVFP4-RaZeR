"""Measured NLL curve, kept separate from analytical ownership surrogates."""
from common import *
from quality import collect, result_status

def main():
    start=now();records,_=collect();maps=load(OUT/'results/GRANULARITY_MAP_MANIFEST.json');rows=[]
    bp=OUT/'results/GPU_RESOURCE_BLOCKERS.json';resource_blockers=load(bp) if bp.exists() else {}
    for model in MODELS:
        for corpus in ['wiki','c4']:
            b=next(r for r in records if r['model']==model and r['corpus']==corpus and r['policy']=='four_over_six' and ('V31_ppl_primary' in r['source'] or 'V62_ppl_primary' in r['source']))
            panel=[b];ns=[0]
            for N in [8,16,32,64,128,256]:
                mp=next((m for m in maps if m['model']==model and m['N']==N),None)
                rr=next((r for r in records if mp and r['model']==model and r['corpus']==corpus and r['payload']==mp['payload_sha256'] and r['compatibility_key']==b['compatibility_key']),None)
                if rr:panel.append(rr);ns.append(N)
                else:
                    block=resource_blockers.get(model+'_granularity',{})
                    blocked=N in block.get('N',[]) and block.get('status')=='blocked_retry_limit'
                    rows.append(dict(model=model,draw='seed0',N=N,corpus=corpus,status='blocked_retry_limit' if blocked else ('pending_evaluation' if mp else 'blocked_exact_covariance'),backend_status='analytical_only',map_hash=mp['sha256'] if mp else None,blocker=block.get('required_external_resolution') if blocked else None))
            x,boot=paired_bootstrap(np.stack([r['cluster_loss'] for r in panel]),b['cluster_tokens'])
            for i,N in enumerate(ns):
                if not N:continue
                j=ns.index(8);gain=x[0]-x[j];gainbs=boot[:,0]-boot[:,j];glo,ghi=np.quantile(gainbs,[.025,.975])
                retention_ok=gain>0 and glo>0
                d=x[i]-x[0];lo,hi=np.quantile(boot[:,i]-boot[:,0],[.025,.975])
                row=dict(model=model,draw='seed0',N=N,corpus=corpus,status=result_status(panel[i]),backend_status='software_emulated',
                    map_hash=panel[i]['map_sha256'],source=panel[i]['source'],source_sha256=panel[i]['source_sha256'],
                    nll=float(x[i]),ppl=float(np.exp(x[i])),delta_nll_vs_baseline=float(d),relative_ppl=float(np.expm1(d)),ci_low=float(lo),ci_high=float(hi),
                    nll_gap_vs_n8=float(x[i]-x[j]),gain_retention=float((x[0]-x[i])/gain) if retention_ok else None,
                    retention_status='defined; no clipping' if retention_ok else 'NA: N8 gain nonpositive or CI crosses zero',n8_gain_ci_low=float(glo),n8_gain_ci_high=float(ghi),
                    cluster_count=len(b['cluster_tokens']),bootstrap_repeats=2000,bootstrap_seed=20260921)
                if retention_ok:
                    valid=gainbs>0;rat=(boot[valid,0]-boot[valid,i])/gainbs[valid]
                    row.update(retention_ci_low=float(np.quantile(rat,.025)),retention_ci_high=float(np.quantile(rat,.975)),nonpositive_bootstrap_denominators=int((~valid).sum()))
                rows.append(row)
            path=OUT/'results/arrays'/f'{model}_{corpus}_granularity.npz'
            np.savez_compressed(path,N=np.array(ns),cluster_ids=np.array(b['cluster_ids']),cluster_tokens=b['cluster_tokens'],cluster_nll_sum=np.stack([r['cluster_loss'] for r in panel]))
    csvout(OUT/'results/granularity_quality.csv',rows)
    log('T3-measured','python scripts/granularity_quality.py',start,[OUT/'results/granularity_quality.csv'])
    print('Measured cells',sum(r['status'] in ['validated_reuse','validated_new'] for r in rows),'/',len(rows))

if __name__=='__main__':main()
