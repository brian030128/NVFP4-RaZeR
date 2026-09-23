"""Frozen full8 adapter over the unchanged, hash-pinned GPU policy launcher.

Evaluator payload/cache/labels, optional UUID restriction and prelaunch
inventory-error waiting differ; during-run/postflight checks are unchanged.
No GPU runs on import or --inspect-only. No optional benchmark/limit/ranking
arguments are accepted. Existing launchers/source files are never modified.
"""
import argparse
import ast
import hashlib
import importlib.metadata as md
import sys
from common import *
from gpu_run_wait import inventory_or_wait

BASE_SHA='35e40c567436c78bcab5b51116751c58f66e6c4b970349409c148565d3e2ffa4'
WAIT_SHA='7c3d65eecf72b83d0a8cf4416d846593ea8d7f8f1e4dfca4f8db17abe27ac8e4'
RECOVERED=Path('/share3/saves/JAAAAAA/mixfp4_n16k64_followup_20260917T090438Z/cache/hf/hub')

def transform(source,only_uuid=None):
    assert hashlib.sha256(source.encode()).hexdigest()==BASE_SHA,'Base launcher changed: re-audit before adapting'
    assert sha(OUT/'scripts/gpu_run_wait.py')==WAIT_SHA,'Inventory wait helper changed: re-audit'
    start=source.index('    if a.calibrate_stream:\n');end=source.index('    source_sha=',start)
    payload="""    assert a.plan and a.plan.exists()
    cmd += ['-m','campaign.evaluate_lmeval','--model',a.model,'--plan',str(a.plan.resolve()),
            '--suite','full8','--batch-size',str(ACCURACY_BATCH_SIZE),
            '--freeze',str(frozen),'--freeze-sha256',freezehash]
"""
    modified=source[:start]+payload+source[end:]
    replacements={
        "HF_DATASETS_CACHE=str(RUNTIME/'cache/datasets')":"HF_DATASETS_CACHE=str(RUNTIME/'cache/accuracy_datasets')",
        "rec=dict(status='running',":"rec=dict(launcher_base_sha256=BASE_SHA, inventory_wait_sha256=WAIT_SHA, adapter_generated_sha256=GENERATED_SHA, status='running',",
        "task='T3' if a.calibrate_stream else 'T2'":"task='T2-secondary'",
        "gp.smi_gpus()":"inventory_or_wait(gp,root)",
    }
    for old,new in replacements.items():
        assert modified.count(old)==1,old
        modified=modified.replace(old,new)
    if only_uuid is not None:
        from gpu_run_on_uuid import transformed
        transformed(only_uuid)  # Shared literal UUID/base-hash check; no GPU query.
        old="if g['uuid'] in active or g['name'] not in gp.ALLOWED_MODELS:continue"
        assert modified.count(old)==1
        modified=modified.replace(old,"if g['uuid'] != "+repr(only_uuid)+" or "+old[3:],1)
        marker="    code={str(f.relative_to(SOURCE)):sha(f)"
        modified=modified.replace(marker,"    rec.update(allowed_uuid="+repr(only_uuid)+")\n"+marker,1)
    ast.parse(modified)
    return modified

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,choices=MODELS)
    ap.add_argument('--name',required=True);ap.add_argument('--inspect-only',action='store_true');ap.add_argument('--only-uuid');a=ap.parse_args()
    start=now();protocol=load(OUT/'results/SECONDARY_ACCURACY_PLAN.json')
    item=next(r for r in protocol['plans'] if r['model']==a.model)
    plan=OUT/item['plan'];assert sha(plan)==item['plan_sha256']
    assert item['suite']=='full8' and md.version('lm_eval')==item['lm_eval_version']
    for policy in load(plan):
        if policy['kind']=='map':assert sha(policy['map_path'])==policy['map_sha256']
    cachefile=OUT/'results/ACCURACY_CACHE_AUDIT.json';assert sha(cachefile)==protocol['cache_audit']['sha256']
    cache=load(cachefile);assert cache['status']=='copied_and_byte_verified'
    for entry in cache['files']:assert sha(REPO/entry['path'])==entry['sha256']
    base=OUT/'scripts/gpu_run.py';source=base.read_text();modified=transform(source,a.only_uuid)
    generated=hashlib.sha256(modified.encode()).hexdigest()
    hub=PRIMARY/'cache/hf/hub' if a.model=='llama8b' else RECOVERED
    argv=[str(Path(__file__).resolve()),'--name',a.name,'--model',a.model,'--plan',str(plan),
          '--device-type','a6000','--hub-cache',str(hub)]
    if a.inspect_only:
        path=OUT/f'results/ACCURACY_ADAPTER_{a.model}.json'
        jsonout(path,dict(status='static_inspection_only',model=a.model,checked_utc=start,base_sha256=BASE_SHA,
            generated_sha256=generated,adapter_sha256=sha(__file__),inventory_wait_sha256=WAIT_SHA,plan_sha256=sha(plan),
            cache_audit_sha256=sha(cachefile),lm_eval_version=md.version('lm_eval'),batch_size=item['batch_size'],
            allowed_uuid=a.only_uuid,no_gpu_launched=True,qualification='Checks source transformation, fixed plan/cache and package version; not model/prompt/numerical correctness'))
        log('accuracy-adapter-inspect','python scripts/accuracy_run.py --inspect-only --model '+a.model,start,[path])
        print('Static accuracy adapter PASS',a.model,'No GPU launched');return
    assert sha(OUT/'FROZEN_PROTOCOL.yaml')=='dbf2a3cb353e168a9bbe104309a345781f51f7d04ca281c4415e8097463e2306'
    sys.argv=argv
    namespace=dict(__name__='__main__',__file__=str(Path(__file__).resolve()),
        BASE_SHA=BASE_SHA,WAIT_SHA=WAIT_SHA,inventory_or_wait=inventory_or_wait,
        GENERATED_SHA=generated,ACCURACY_BATCH_SIZE=item['batch_size'])
    exec(compile(modified,str(base)+'::accuracy-adapter','exec'),namespace)

if __name__=='__main__':main()
