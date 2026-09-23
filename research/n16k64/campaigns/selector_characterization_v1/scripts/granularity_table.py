"""Array-checked measured quality table; never fill missing coarse cells."""
from common import *

def main():
    start=now();source=OUT/'results/granularity_quality.csv';rows=list(csv.DictReader(source.open()))
    assert len(rows)==36 and len({(r['model'],r['corpus'],r['N']) for r in rows})==36
    inputs={str(source.relative_to(OUT)):sha(source)};checks=[]
    text=['# Measured granularity curve: frozen seed0, joint k=3','',
        'Same candidates/scales; floating-point fake-quant evaluation. ΔNLL is '
        'policy minus FourOverSix; relative PPL is exp(ΔNLL)−1. Intervals are '
        '2,000 paired natural-cluster bootstrap pointwise 95% intervals, not '
        'simultaneous or multiplicity-adjusted. Missing is not zero.','']
    for model in MODELS:
        text += [f'## {model}','',
            '| Corpus | N | Status | PPL | ΔNLL [pointwise 95% CI] | Relative PPL (%) | N8 gain retained (%) |',
            '|---|---:|---|---:|---|---:|---:|']
        for corpus in ['wiki','c4']:
            path=OUT/'results/arrays'/f'{model}_{corpus}_granularity.npz'
            inputs[str(path.relative_to(OUT))]=sha(path)
            with np.load(path,allow_pickle=False) as a:
                ns=a['N'].tolist();nll=a['cluster_nll_sum'].sum(axis=1)/a['cluster_tokens'].sum()
                assert ns[0]==0 and len(a['cluster_ids'])==len(set(a['cluster_ids'].tolist()))
                for r in sorted([r for r in rows if r['model']==model and r['corpus']==corpus],key=lambda r:int(r['N'])):
                    N=int(r['N'])
                    if r['status'] not in ['validated_new','validated_reuse']:
                        assert N not in ns
                        text.append(f'| {corpus} | {N} | {r["status"]} | missing | missing | missing | missing |');continue
                    i=ns.index(N);delta=float(nll[i]-nll[0])
                    assert abs(float(nll[i])-float(r['nll']))<1e-12
                    assert abs(delta-float(r['delta_nll_vs_baseline']))<1e-12
                    assert abs(np.exp(nll[i])-float(r['ppl']))<1e-10
                    assert abs(np.expm1(delta)-float(r['relative_ppl']))<1e-12
                    retained='NA'
                    if r['gain_retention']:
                        expected=(nll[0]-nll[i])/(nll[0]-nll[ns.index(8)])
                        assert abs(expected-float(r['gain_retention']))<1e-10
                        retained=f'{100*expected:.2f}'
                    text.append(f'| {corpus} | {N} | {r["status"]} | {float(r["ppl"]):.6f} | {delta:+.6f} [{float(r["ci_low"]):+.6f}, {float(r["ci_high"]):+.6f}] | {100*np.expm1(delta):+.4f} | {retained} |')
                    checks.append(dict(model=model,corpus=corpus,N=N,delta_nll=delta,status=r['status']))
        text.append('')
    table=OUT/'results/GRANULARITY_TABLE.md';table.write_text('\n'.join(text)+'\n')
    audit=OUT/'results/GRANULARITY_TABLE_AUDIT.json'
    jsonout(audit,dict(checked_utc=start,script_sha256=sha(Path(__file__)),source_hashes=inputs,verified_points=checks,
        qualification='Independent point/unit/retention recomputation from paired arrays; CIs retained from frozen bootstrap script, not a second bootstrap.'))
    log('granularity-table','python scripts/granularity_table.py',start,[table,audit]);print('Array-checked granularity points',len(checks))

if __name__=='__main__':main()
