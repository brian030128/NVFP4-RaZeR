"""Record the actual three-item user grant without changing scientific gates."""
from common import *

def main():
    start=now();grant_path=OUT/'plans/additional_retry_authorization_20260923.json';grant=load(grant_path)
    assert grant['authorization_type']=='explicit_user_additional_retry'
    assert '授權三項各增加一次修復嘗試' in grant['user_evidence']
    bp=OUT/'results/GPU_RESOURCE_BLOCKERS.json';blocks=load(bp)
    old=OUT/'results/GPU_RESOURCE_BLOCKERS_before_20260923_grant.json'
    if not old.exists():jsonout(old,blocks)
    mapping={'llama8b_granularity':'calibration_llama8b_attempt4_authorized',
        'mistral7b_calibration':'calibration_mistral7b_attempt4_authorized',
        'mistral7b_granularity':'calibration_mistral7b_attempt4_authorized',
        'qwen4b_granularity':'granularity_qwen4b_attempt4_authorized'}
    for key,run in mapping.items():
        assert any(x['run_name']==run and x['additional_attempts']==1 for x in grant['scopes'])
        blocks[key].update(status='additional_attempt_authorized',authorized_run=run,
            authorization=str(grant_path.relative_to(OUT)),authorization_sha256=sha(grant_path),
            required_external_resolution='One additional attempt explicitly authorized on 2026-09-23; resource waits and original numerical identity gates remain. No fifth attempt authorized.')
    jsonout(bp,blocks)
    status=load(OUT/'TASK_STATUS.json')
    t3=next(t for t in status['tasks'] if t['id']=='T3')
    t3['blockers']=[b for b in t3.get('blockers',[]) if not b.startswith(tuple(k+': ' for k in mapping))]
    t3['additional_retry_authorization']=dict(path=str(grant_path.relative_to(OUT)),sha256=sha(grant_path),runs=sorted(set(mapping.values())))
    status['last_updated']=start;jsonout(OUT/'TASK_STATUS.json',status)
    result=OUT/'results/ADDITIONAL_RETRY_PLAN.json'
    jsonout(result,dict(recorded_utc=start,grant_sha256=sha(grant_path),previous_blockers_sha256=sha(old),
        plans=[dict(run='calibration_mistral7b_attempt4_authorized',entry='scripts/calibration_repair_run_v2.py',source_sha256=sha(OUT/'scripts/calibration_repair_run_v2.py'),repair='CPU-tested sibling import fix; original Mistral raw sample/subset allocation and hash-proven quant source'),
               dict(run='granularity_qwen4b_attempt4_authorized',entry='scripts/cadence_launch.py',source_sha256=sha(OUT/'scripts/cadence_launch.py'),plan='plans/qwen4b_granularity_anchored_v2.json',plan_sha256=sha(OUT/'plans/qwen4b_granularity_anchored_v2.json'),repair='Existing CPU-validated header repair; exclusive A6000 with cadence guard'),
               dict(run='calibration_llama8b_attempt4_authorized',entry='scripts/calibration_llama_retry_run.py',source_sha256=sha(OUT/'scripts/calibration_llama_retry_run.py'),adapter_sha256=sha(OUT/'scripts/stream_calibrate_llama_historical.py'),repair='Separate Llama historical raw sample, no subset moments; hash-proven quant restore; wait for available campaign slot')],
        exclusions='No fourth Llama secondary accuracy attempt; no acceptance or retry-limit relaxation beyond the three named attempts.'))
    log('additional-retry-authorization','python scripts/record_additional_retries.py',start,[grant_path,old,bp,result])
    print('Three scoped retries recorded; Llama secondary accuracy remains blocked.')

if __name__=='__main__':main()
