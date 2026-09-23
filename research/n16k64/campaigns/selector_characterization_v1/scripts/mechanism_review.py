"""Bounded existing-data review; no new interventions or significance selection."""
from common import *
from scipy.stats import spearmanr

def sealed(p):
    run=p.parents[1];sp=run/'SHA256SUMS_run.txt'
    top=PRIMARY/'runs/V83_validate_artifacts_attempt7/artifact_validation/ARTIFACT_MANIFEST.sha256'
    def members(f):return {x.split(maxsplit=1)[1].strip().lstrip('*'):x.split()[0] for x in f.read_text().splitlines()}
    assert members(top)[sp.relative_to(PRIMARY).as_posix()]==sha(sp)
    assert members(sp)[p.relative_to(run).as_posix()]==sha(p)

def main():
    import torch
    torch.set_num_threads(1)
    start=now();print('TASK_STATUS',[(r['id'],r['status']) for r in load(OUT/'TASK_STATUS.json')['tasks']])
    inputs=load(OUT/'results/INPUTS.json');rows=[];groups=[];summaries=[];ledger=[]
    for model in MODELS:
        p=next((PRIMARY/'runs').glob(f'V43_fidelity_{model}_attempt1/fidelity/fidelity_report.json'));sealed(p)
        r=load(p);assert r['status']=='complete';v=r['results']['n16']
        # Preserve the failed restore-loss flag. The check also changes batch
        # size (8 vs 16); this flag alone cannot diagnose corruption or noise.
        inp=next(x for x in inputs if x['model']==model and x['draw']=='seed0')
        mom=torch.load(REPO/inp['moments'],weights_only=False,mmap=True)
        # Exact realized strata use the recorded thresholds, not a newly sampled quantile.
        upper=[]
        for n in mom['names']:
            mc,sc=moments(mom['n16'][n],'ce');mk,sk=moments(mom['n16'][n],'kl');upper.append(np.maximum(mc+3*sc,mk+3*sk))
        u=np.concatenate(upper);q=v['thresholds']
        sizes={'selected':int((u<0).sum()),'near_threshold':int(((u>=0)&(u<q['q10_positive_U3'])).sum()),
               'rejected':int((u>q['median_positive_U3']).sum()),'random':len(u)}
        assert sizes['selected']==q['selected_total'];assert len(v['singles'])==160
        for x in v['singles']:
            for objective in ['ce','kl']:
                a=np.array(x['actual_'+objective+'_per_seq']);mean=a.mean();se=a.std(ddof=1)/np.sqrt(len(a))
                assert len(a)==128 and abs(mean-x['actual_'+objective])<1e-12 and abs(se-x['actual_'+objective+'_se'])<1e-12
                pred=x['pred_'+objective];scorese=x['se_'+objective]
                rows.append(dict(model=model,N=16,stratum=x['stratum'],module=x['module'],tile=x['tile'],objective=objective,
                    baseline_loss_restored_exact=v['baseline_restored_exact'],predicted=pred,predicted_se=scorese,actual=mean,actual_se=se,pointwise_ci_low=mean-1.96*se,pointwise_ci_high=mean+1.96*se,
                    effect_over_se=mean/se if se else None,sign_agreement=bool(np.sign(pred)==np.sign(mean)),
                    pointwise_resolvable=bool(abs(mean)>1.96*se),se_unit='128 calibration sequences; not held-out documents',
                    valid_target_tokens=128*511,stratum_size=sizes[x['stratum']],within_stratum_inclusion_probability=40/sizes[x['stratum']],
                    predictor_data='same frozen seed0 calibration sequences as finite effect',reference='all FourOverSix; causal-per-token activation',
                    precision='BF16 forward, FP32 log_softmax, FP64 paired sequence summaries',source=str(p.relative_to(REPO)),source_sha256=sha(p)))
        for x in v['batched']:groups.append(dict(model=model,N=16,intervention='simultaneous selected prefix',**x,source=str(p.relative_to(REPO)),source_sha256=sha(p)))
        for objective in ['ce','kl']:
            rr=[x for x in rows if x['model']==model and x['objective']==objective]
            summaries.append(dict(model=model,objective=objective,n=len(rr),unique_tiles=len({(x['module'],x['tile']) for x in rr}),
                sample_sign_agreement=float(np.mean([x['sign_agreement'] for x in rr])),
                pointwise_resolvable_fraction=float(np.mean([x['pointwise_resolvable'] for x in rr])),
                sample_spearman=float(spearmanr([x['predicted'] for x in rr],[x['actual'] for x in rr]).statistic),
                qualification='Stratified/overlapping sample descriptive only; no population or significant-subset claim'))
        ledger.append(dict(question='individual finite effects',source=str(p.relative_to(REPO)),sha256=sha(p),verification='sealed raw records and paired SE independently recomputed',status='bounded evidence'))
    for rel in ['research/n16k64/campaigns/mechanism/MECHANISM_VERDICT.md','research/n16k64/campaigns/followup/FOLLOWUP_VERDICT.md',
                'research/n16k64/campaigns/boundary/BOUNDARY_CORRUPTION_VERDICT.md','results/task_reorder/cluster_20260919/TARGET_90_PERCENT.json',
                'research/n16k64/validation/CURRENT_EVIDENCE_AUDIT.json']:
        p=REPO/rel
        ledger.append(dict(question='historical aggregate/composition/breadth/reorder context',source=rel,sha256=sha(p),verification='source read and hashed; not re-executed',status='historical bounded context'))
    csvout(OUT/'results/effect_diagnostics.csv',rows);csvout(OUT/'results/group_interventions.csv',groups)
    jsonout(OUT/'results/INDIVIDUAL_SUMMARY.json',summaries);csvout(OUT/'evidence_ledger.csv',ledger)
    log('T4','python scripts/mechanism_review.py',start,[OUT/'results/effect_diagnostics.csv',OUT/'results/group_interventions.csv',OUT/'results/INDIVIDUAL_SUMMARY.json'])
    print(summaries)

if __name__=='__main__':
    start=now()
    try:main()
    except Exception as e:log('T4','python scripts/mechanism_review.py',start,[],status='failed',error=repr(e));raise
