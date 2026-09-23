"""Require exact historical sequence-score identity before accepting parents."""
from common import *
import torch

def require_monitoring(lr, pre, during, post):
    assert lr['status']=='complete' and not lr.get('invalid_reasons')
    assert pre and pre[-1]['passed'] and during and post['passed']
    assert all(row['passed'] for row in during)
    times=[lr['started_utc']]+[row['timestamp_utc'] for row in during]+[lr['finished_utc']]
    times=[datetime.fromisoformat(t.replace('Z','+00:00')) for t in times]
    assert all(0 <= (b-a).total_seconds() <= 60 for a,b in zip(times,times[1:])), 'calibration monitoring gap'

def main():
    start=now();torch.set_num_threads(1);runtime=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z'
    inputs=load(OUT/'results/INPUTS.json');accepted=[];excluded=[]
    for manifest in sorted((runtime/'runs').glob('*/parent_moments/manifest.json')):
        run=manifest.parent.parent;lr=load(run/'launch_record.json')
        if lr['status']!='complete':
            excluded.append(dict(run=str(run.relative_to(REPO)),reason='not complete valid run'));continue
        try:
            require_monitoring(lr,
                [json.loads(x) for x in (run/'preflight_candidates.jsonl').read_text().splitlines()],
                [json.loads(x) for x in (run/'gpu_monitor.jsonl').read_text().splitlines()],load(run/'postflight.json'))
            assert lr['protocol_sha256']==sha(OUT/'FROZEN_PROTOCOL.yaml')
            assert lr['actual_source_hashes_sha256']==sha(run/'source_hashes.json')
            adapter=load(run/'parent_moments/adapter.json')
            assert adapter['adapter_sha256']==sha(OUT/'scripts/stream_calibrate.py')
            assert adapter['original_source_sha256']==load(run/'source_hashes.json')['campaign/calibrate.py']
            repairs=[str(x) for x in lr['command'] if str(x).endswith(('stream_calibrate_historical.py','stream_calibrate_historical_v2.py','stream_calibrate_llama_historical.py'))]
            if repairs:
                assert len(repairs)==1
                repair_path=Path(repairs[0]);assert repair_path==OUT/'scripts'/repair_path.name
                repair=load(run/'parent_moments/historical_repair.json')
                assert repair['repair_adapter_sha256']==lr['calibration_repair_adapter_sha256']==sha(repair_path)
                assert repair['recovery_function_sha256']==lr['recovery_function_sha256']==sha(OUT/'scripts/calibration_identity_diagnostic.py')
                assert repair['observer_adapter_sha256']==sha(OUT/'scripts/stream_calibrate.py')
                assert repair['recovered_quant_sha256']=='210d182478d9bab9ce6c9b8e6d8ff831420ca83f31a2ecf170998c13ca641aca'
                llama_repair=repair_path.name=='stream_calibrate_llama_historical.py'
                assert repair['raw']=='sample' and repair['subset_moments'] is (not llama_repair)
                if llama_repair:
                    from calibration_llama_retry_run import require_authorization,PARENT_SHA
                    authorization=run/'additional_retry_authorization.json';grant=load(authorization)
                    require_authorization(grant)
                    assert run.name=='calibration_llama8b_attempt4_authorized'
                    assert sha(authorization)==lr['additional_retry_authorization_sha256']
                    assert lr['launcher_sha256']==sha(OUT/'scripts/calibration_llama_retry_run.py')
                    assert lr['calibration_repair_launcher_parent_sha256']==PARENT_SHA
                if repair_path.name=='stream_calibrate_historical_v2.py':
                    from calibration_repair_run_v2 import require_authorization,PARENT_SHA
                    authorization=run/'additional_retry_authorization.json';grant=load(authorization)
                    require_authorization(grant);assert grant['run_name']==run.name
                    assert sha(authorization)==lr['additional_retry_authorization_sha256']
                    assert lr['launcher_sha256']==sha(OUT/'scripts/calibration_repair_run_v2.py')
                    assert lr['calibration_repair_launcher_parent_sha256']==PARENT_SHA
        except (AssertionError,KeyError,OSError,ValueError) as exc:
            excluded.append(dict(run=str(run.relative_to(REPO)),reason='GPU/provenance admission failed',detail=str(exc)));continue
        r=load(run/'calibration/calibration_report.json');inp=next(x for x in inputs if x['model']==r['model'] and x['draw']=='seed0')
        old=load(REPO/inp['run']/'calibration/calibration_report.json');failures=[]
        for key in ['model','spec','draw','sequences','sequence_token_sha256','calibration_manifest_sha256','candidate_weight_sha256',
                    'module_manifest_sha256','tokenizer_manifest_sha256','weight_baseline','alternative','activation_quantizer',
                    'score_stream_sha256','score_stream_sha256_all']:
            if r.get(key)!=old.get(key):failures.append(key)
        if r['status']!='complete':failures.append('status')
        if failures:
            excluded.append(dict(run=str(run.relative_to(REPO)),reason='historical identity failed',fields=failures));continue
        base=torch.load(REPO/inp['moments'],weights_only=True,mmap=True,map_location='cpu')
        reg=run/'calibration/moments/moments_full.pt';assert sha(reg)==r['moment_files']['moments_full.pt']
        state=torch.load(reg,weights_only=True,mmap=True,map_location='cpu')
        assert state['names']==base['names'] and state['shapes']==base['shapes']
        for N in [8,16]:
            for n in base['names']:
                a,b=base[f'n{N}'][n],state[f'n{N}'][n]
                for k in a:
                    assert torch.equal(a[k],b[k]) if torch.is_tensor(a[k]) else a[k]==b[k],(run.name,N,n,k)
        mm=load(manifest);assert set(mm)=={'32','64','128','256'};files={}
        for N in [32,64,128,256]:
            p=Path(mm[str(N)]['path']);assert p.resolve().is_relative_to(run.resolve())
            assert sha(p)==mm[str(N)]['sha256'];d=torch.load(p,weights_only=True,mmap=True,map_location='cpu')
            assert set(d)==set(base['names'])
            for name in base['names']:
                ro,co=base['shapes'][name];assert d[name]['n']==128
                for obj in ['ce','kl']:
                    mu,se=moments(d[name],obj);oldmu,oldse=moments(base['n8'][name],obj)
                    expected=oldmu.reshape(ro//N,N//8,co//64).sum(1).ravel()
                    assert np.allclose(mu,expected,atol=1e-12,rtol=1e-10)
                    assert (se<=oldse.reshape(ro//N,N//8,co//64).sum(1).ravel()+1e-12).all()
                    allpass=(oldmu+3*oldse<0).reshape(ro//N,N//8,co//64).all(1).ravel()
                    assert ((mu+3*se<0)|~allpass).all()
            files[str(N)]=dict(path=str(p.relative_to(REPO)),sha256=sha(p))
        accepted.append(dict(model=r['model'],run=str(run.relative_to(REPO)),report_sha256=sha(run/'calibration/calibration_report.json'),
            historical_report_sha256=sha(REPO/inp['run']/'calibration/calibration_report.json'),
            score_stream_sha256_all=r['score_stream_sha256_all'],manifest_sha256=sha(manifest),files=files,
            verification='Exact per-module sequence-score digests AND N8/N16 moments; parent means, SE bounds and all-children-pass invariants'))
    jsonout(OUT/'results/PARENT_MOMENT_REUSE.json',accepted);jsonout(OUT/'results/PARENT_MOMENT_REJECTIONS.json',excluded)
    log('T3-parent-validation','python scripts/validate_parent_moments.py',start,[OUT/'results/PARENT_MOMENT_REUSE.json',OUT/'results/PARENT_MOMENT_REJECTIONS.json'])
    print('Validated regenerated parent sets',len(accepted),'rejected',len(excluded))

if __name__=='__main__':main()
