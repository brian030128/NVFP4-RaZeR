"""Small independent point recomputation and descriptive five-draw tables.

No GPU, new bootstrap, policy selection, or population-risk inference. Run only
after the T2 ingestion writer is idle; source hashes are checked again at end.
"""
from common import *

POLICIES=['ce_natural','kl_natural','joint','ce_matched','kl_matched']

def main():
    start=now();path=OUT/'results/quality_results.csv';initial=sha(path)
    rows=list(csv.DictReader(path.open()));sources={str(path.relative_to(OUT)):initial}
    text=['# Objective comparison: observed calibration-draw variation','',
          'N16K64, k=3; delta NLL versus the paired FourOverSix baseline (negative is favorable).',
          'SD is the sample SD across available calibration draws, not a confidence interval.',
          'Worst means the largest observed delta NLL; regression means delta NLL > 0.',
          'Five draws and two corpora are not independent population tail-risk replications.',
          'Pointwise paired-cluster intervals are in `quality_results.csv` and `quality_contrasts.csv`;',
          'this table does not replace those intervals or imply simultaneous coverage.','']
    verified=0;table_rows=[]
    for model in MODELS:
        text += ['## '+model,'',
                 '| Corpus | Policy | Draws / 5 | Mean delta NLL | Sample SD | Minimum | Observed worst | Regressions |',
                 '|---|---|---:|---:|---:|---:|---:|---:|']
        for corpus in ['wiki','c4']:
            arrpath=OUT/'results/arrays'/f'{model}_{corpus}.npz'
            sources[str(arrpath.relative_to(OUT))]=sha(arrpath)
            with np.load(arrpath,allow_pickle=False) as arr:
                labels=list(arr['policies']);tokens=arr['cluster_tokens'];loss=arr['cluster_nll_sum']
                assert len(set(labels))==len(labels) and len(set(arr['cluster_ids']))==len(tokens)
                assert (tokens>0).all() and np.isfinite(loss).all()
                estimates=loss.sum(axis=1)/tokens.sum();baseline=estimates[labels.index('baseline')]
                for policy in POLICIES:
                    selected=[r for r in rows if r['model']==model and r['corpus']==corpus and r['policy']==policy
                              and r['status'] in ['validated_new','validated_reuse']]
                    assert len({r['draw'] for r in selected})==len(selected)<=5
                    values=[]
                    for r in selected:
                        delta=float(estimates[labels.index(r['draw']+'/'+policy)]-baseline)
                        assert abs(delta-float(r['delta_nll']))<1e-12
                        assert abs(np.expm1(delta)-float(r['relative_ppl']))<1e-12
                        assert int(r['valid_tokens'])==int(tokens.sum()) and int(r['doc_count'])==len(tokens)
                        values.append(delta);verified+=1
                    n=len(values)
                    if not n:
                        text.append(f'| {corpus} | {policy} | 0 | — | — | — | — | — |');continue
                    mean=float(np.mean(values));sd=float(np.std(values,ddof=1)) if n>1 else None
                    low=min(values);worst=max(values);reg=sum(v>0 for v in values)
                    sd_text=f'{sd:.6f}' if sd is not None else '—'
                    text.append(f'| {corpus} | {policy} | {n} | {mean:+.6f} | {sd_text} | {low:+.6f} | {worst:+.6f} | {reg}/{n} |')
                    table_rows.append(dict(model=model,corpus=corpus,policy=policy,draws=n,complete=n==5,
                        mean_delta_nll=mean,sample_sd=sd,minimum=low,observed_worst=worst,observed_regressions=reg))
        text.append('')
    assert sha(path)==initial,'Concurrent ingestion changed the input; rerun after it finishes'
    for rel,h in sources.items():assert sha(OUT/rel)==h,'Concurrent input change'
    output=OUT/'results/OBJECTIVE_TABLES.md';output.write_text('\n'.join(text)+'\n')
    audit=OUT/'results/OBJECTIVE_TABLE_AUDIT.json'
    jsonout(audit,dict(checked_utc=start,verified_cells=verified,required_cells=150,
        verified_from='Paired natural-cluster loss sums and token counts, independently recomputed points',
        sources=sources,table_sha256=sha(output),rows=table_rows,
        no_new_bootstrap=True,no_population_tail_risk_claim=True))
    log('T5-objective-table','python scripts/objective_tables.py',start,[output,audit])
    print('Independently verified point estimates',verified,'/150')

if __name__=='__main__':main()
