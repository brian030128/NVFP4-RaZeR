"""Full-window N8 (default) or N16 reproduction, not CPU reader acceptance."""
import argparse
from common import *
from quality import collect, result_status

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--N',type=int,choices=[8,16],default=8);args=ap.parse_args()
    N=args.N;start=now();records,excluded=collect();rows=[];gates={}
    for model in MODELS:
        comparisons=[]
        for corpus in ['wiki','c4']:
            old=[r for r in records if r['model']==model and r['corpus']==corpus and r['N']==N
                 and ('V31_ppl_primary' in r['source'] or 'V62_ppl_primary' in r['source'])]
            fresh=[r for r in records if r['model']==model and r['corpus']==corpus
                   and r['N']==N and r['policy']==f'n{N}_reproduction' and result_status(r)=='validated_new']
            for candidate in fresh:
                refs=[r for r in old if r['payload']==candidate['payload'] and r['compatibility_key']==candidate['compatibility_key']]
                assert len(refs)==1,(model,corpus,N,'ambiguous or missing reference')
                reference=refs[0]
                identical=(candidate['weight_sha256']==reference['weight_sha256']
                    and candidate['cluster_ids']==reference['cluster_ids']
                    and np.array_equal(candidate['window_tokens'],reference['window_tokens']))
                delta=candidate['window_nll']-reference['window_nll']
                passed=identical and bool((np.abs(delta)<=1e-10).all())
                row=dict(model=model,N=N,corpus=corpus,passed=passed,identities_match=identical,
                    windows=len(delta),max_abs_delta_nll=float(np.abs(delta).max()),absolute_tolerance=1e-10,
                    reference=reference['source'],reference_sha256=reference['source_sha256'],
                    source=candidate['source'],source_sha256=candidate['source_sha256'],
                    payload_sha256=candidate['payload'])
                rows.append(row);comparisons.append(row)
        valid={r['corpus'] for r in comparisons if r['passed']}
        gates[model]=dict(passed=valid=={'wiki','c4'},status='passed' if valid=={'wiki','c4'} else 'pending_or_failed',
            qualification=f'Full windows, matching installed N{N} weights, matched runtime/input identities; baseline verified by ingestion gate')
    path=OUT/f'results/N{N}_REPRODUCTION.json'
    jsonout(path,dict(models=gates,comparisons=rows,
        input_exclusions=excluded,checked_utc=start))
    log(f'N{N}-reproduction',f'python scripts/n8_reproduction_audit.py --N {N}',start,[path])
    print(gates)

if __name__=='__main__':main()
