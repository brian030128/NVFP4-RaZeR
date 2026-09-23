"""Explicit five-draw count/density summaries; no rank or quality inference."""
from common import *

def main():
    start=now();source=OUT/'results/map_counts.csv';rows=list(csv.DictReader(source.open()));out=[]
    for model,module in sorted({(r['model'],r['module']) for r in rows}):
        group=[r for r in rows if (r['model'],r['module'])==(model,module)]
        assert len(group)==5 and {r['draw'] for r in group}==set(DRAWS)
        universe={int(r['U']) for r in group};assert len(universe)==1
        U=universe.pop();assert U>0
        k=np.array([int(r['K']) for r in group]);density=np.array([float(r['density']) for r in group])
        assert np.allclose(density,k/U,atol=1e-15,rtol=0)
        result=dict(model=model,module=module,U=U,draws=5,all_empty=bool((k==0).all()),
            source_sha256=sha(source),interpretation='Across-draw descriptive variation; count and density share fixed U; not identity stability or independent evidence')
        for name,values in [('K',k),('density',density)]:
            result.update({name+'_mean':float(values.mean()),name+'_sample_sd':float(values.std(ddof=1)),
                name+'_min':float(values.min()),name+'_max':float(values.max()),name+'_range':float(np.ptp(values))})
        base=next(int(r['K']) for r in group if r['draw']=='seed0')
        result['density_ratio_denominator_zero']=base==0
        for r in group:
            ratio=r['density_ratio_vs_seed0']
            if base==0:assert ratio==''
            else:assert abs(float(ratio)-int(r['K'])/base)<1e-12
        out.append(result)
    path=OUT/'results/map_count_summary.csv';csvout(path,out)
    log('T1-density-summary','python scripts/map_count_summary.py',start,[path])
    print('Verified module/ALL five-draw summaries',len(out))

if __name__=='__main__':main()
