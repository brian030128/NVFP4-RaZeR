"""Bounded historical ratio audit; never merge this tensor-wide setup with T3."""
from common import *

def ratios(base,before,after,fine):
    return dict(total_fine_gain=base-fine,remaining_gap_before=before-fine,
        retention=(base-after)/(base-fine) if base>fine else None,
        incremental_recovery=(before-after)/(before-fine) if before>fine else None)

def main():
    start=now();root=REPO/'results/task_reorder/cluster_20260919';idx=load(root/'published_evidence/INDEX.json')
    measured=root/'published_evidence/measured_both_ppl/report.json'
    assert sha(measured)==idx[str(measured.relative_to(REPO))]['sha256']
    q=load(measured);qfine=REPO/q['published_token_windows_verified'];oldq=load(qfine)
    llama=REPO/'results/task_reorder/transfer_20260920/renewed_llama/joint192_ppl/report.json';l=load(llama)
    lfine=REPO/l['published_token_windows_verified'];oldl=load(lfine)
    supplied=root/'raw256_reference.json';raw=load(supplied);rows=[]
    for model,r,old,p,fp in [('qwen27b',q,oldq,measured,qfine),('llama8b',l,oldl,llama,lfine)]:
        for key in ['model','length','activation','wiki_use_cache','c4_use_cache','torch_version','transformers_version']:
            assert r[key]==old[key],(model,key)
        for corpus,key in [('wiki','wiki'),('c4','c4_paper')]:
            assert r['data'][key]['token_sha256']==old['data'][key]['token_sha256']
            assert r['data'][key]['revision']==old['data'][key]['revision']
            a=r['evaluation']['both' if model=='qwen27b' else 'refined'][corpus]
            b=old['evaluation']['four_over_six'][corpus];f=old['evaluation']['k3'][corpus]
            before=q['evaluation']['raw256'][corpus] if model=='qwen27b' else None
            for scale in ['PPL','NLL']:
                val=lambda x:float(x['ppl']) if scale=='PPL' else float(np.mean(x['nll'],dtype=np.float64))
                bv,av,fv=val(b),val(a),val(f)
                pre=val(before) if before else (raw['models']['llama8b']['raw256'][corpus] if scale=='PPL' else None)
                rr=ratios(bv,pre,av,fv) if pre is not None else dict(total_fine_gain=bv-fv,remaining_gap_before=None,retention=(bv-av)/(bv-fv) if bv>fv else None,incremental_recovery=None)
                rows.append(dict(model=model,corpus=corpus,scale=scale,baseline=bv,before=pre,after=av,fine=fv,**rr,
                    before_identity='same measured report, local 195-tile raw map' if before else 'supplied rounded 187-tile reference; raw measurement unavailable here',
                    evidence_status='historical descriptive ratios; not a new causal or paired significance claim' if before else 'retention recomputed; PPL incremental recovery approximate from supplied rounded reference; NLL incremental NA',
                    activation=r['activation'],window_hashes_equal=True,source=str(p.relative_to(REPO)),source_sha256=sha(p),fine_source=str(fp.relative_to(REPO)),fine_sha256=sha(fp),
                    before_reference_source=str((p if before else supplied).relative_to(REPO)),before_reference_sha256=sha(p if before else supplied)))
    csvout(OUT/'results/reorder_ratio_audit.csv',rows)
    ledger=list(csv.DictReader((OUT/'evidence_ledger.csv').open()));rel='results/reorder_ratio_audit.csv'
    ledger=[x for x in ledger if x['source']!=rel]+[dict(question='historical retention versus incremental recovery',source=rel,sha256=sha(OUT/rel),verification='ratios recomputed; model/config and window identities checked; supplied Llama before reference qualified',status='bounded historical context')]
    csvout(OUT/'evidence_ledger.csv',ledger)
    log('T4-reorder-context','python scripts/reorder_context.py',start,[OUT/rel])
    for r in rows:print(r['model'],r['corpus'],r['scale'],'retention',r['retention'],'incremental',r['incremental_recovery'])

if __name__=='__main__':main()
