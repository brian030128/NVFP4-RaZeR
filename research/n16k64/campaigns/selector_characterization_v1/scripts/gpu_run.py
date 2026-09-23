"""Exclusive local GPU launcher. Only this campaign's child may be signalled.

Checks both nvidia-smi and NVML plus PID ownership before, every 30 seconds,
at child phase boundaries (historical job_wrapper), and after the job.
No GPU is available => wait, not an experimental failure. One GPU per job;
the registry disallows more than three live physical leases in this campaign.
"""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from common import *

RUNTIME=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z'
SOURCE=PRIMARY/'source/NVFP4-RaZeR-main'
PYTHON=PRIMARY/'env/venv_main/bin/python'
sys.path.insert(0,str(SOURCE))
from campaign import gpu_preflight as gp

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--model',required=True,choices=MODELS)
    p.add_argument('--plan',type=Path);p.add_argument('--pilot',action='store_true');p.add_argument('--calibrate-stream',action='store_true')
    p.add_argument('--limit-sequences',type=int);p.add_argument('--teacher',choices=['none','instance'],default='none')
    p.add_argument('--hub-cache',type=Path,default=PRIMARY/'cache/hf/hub')
    p.add_argument('--device-type',choices=['a6000','ada','any'],default='any');a=p.parse_args()
    if not a.pilot:
        gate=OUT/'results/ANCHOR_GATE.json'
        assert gate.exists() and load(gate).get(a.model,{}).get('passed') is True, 'Numerical anchor required before new scientific GPU work'
    assert a.name.replace('_','').replace('-','').isalnum()
    root=RUNTIME/'runs'/a.name;root.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(CAMPAIGN_ROOT=str(PRIMARY),CAMPAIGN_RUN_DIR=str(root),CAMPAIGN_RUN_ID=a.name,
        CAMPAIGN_ENV_NAME='main',HF_HOME=str(PRIMARY/'cache/hf'),HF_HUB_CACHE=str(a.hub_cache.resolve()),
        HF_DATASETS_CACHE=str(RUNTIME/'cache/datasets'),PYTHONPATH=str(SOURCE),PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',
        TRITON_CACHE_DIR=str(RUNTIME/'cache/triton'),TORCH_HOME=str(RUNTIME/'cache/torch'),
        XDG_CACHE_HOME=str(RUNTIME/'cache/xdg'),TMPDIR=str(RUNTIME/'cache/tmp'))
    for name in ['triton','torch','xdg','tmp','datasets']:(RUNTIME/'cache'/name).mkdir(parents=True,exist_ok=True)
    leases=RUNTIME/'leases';leases.mkdir(exist_ok=True)
    lock=(leases/'registry.lock').open('a')
    uuid=None;lease=None
    while uuid is None:
        fcntl.flock(lock,fcntl.LOCK_EX)
        active=[]
        for lf in leases.glob('GPU-*.json'):
            d=load(lf)
            if Path('/proc',str(d['launcher_pid'])).exists():active.append(d['uuid'])
        if len(active)<3:
            for g in sorted(gp.smi_gpus(),key=lambda g:(g['name']!='NVIDIA RTX A6000',g['index'])):
                if a.device_type=='a6000' and g['name']!='NVIDIA RTX A6000':continue
                if a.device_type=='ada' and g['name']!='NVIDIA RTX 6000 Ada Generation':continue
                if g['uuid'] in active or g['name'] not in gp.ALLOWED_MODELS:continue
                e=dict(env,CUDA_VISIBLE_DEVICES=g['uuid'],CAMPAIGN_ALLOCATED_UUIDS=g['uuid'])
                check=gp.check('prelaunch',expected_uuids=[g['uuid']],allocation_id=a.name,env=e,own_pids=[])
                with (root/'preflight_candidates.jsonl').open('a') as f:f.write(json.dumps(check)+'\n')
                if check['passed']:
                    uuid=g['uuid'];lease=leases/(uuid+'.json');jsonout(lease,dict(uuid=uuid,launcher_pid=os.getpid(),run=a.name,start=now()));env=e;break
        fcntl.flock(lock,fcntl.LOCK_UN)
        if uuid is None:print('WAIT no exclusive GPU or all 3 campaign leases active',flush=True);time.sleep(30)
    env['CAMPAIGN_LEASE_ID']=a.name
    frozen=PRIMARY/'freeze/PROTOCOL_FREEZE.json';freezehash=sha(frozen)
    cmd=[str(PYTHON),'-m','campaign.job_wrapper','--gpu-run','--']
    if a.calibrate_stream:
        cmd += [str(OUT/'scripts/stream_calibrate.py'),'--model',a.model,'--draw','seed0','--raw','none',
                '--freeze',str(frozen),'--freeze-sha256',freezehash]
        if a.limit_sequences:cmd += ['--limit-sequences',str(a.limit_sequences)]
    else:
        if a.pilot:
            inputs=load(OUT/'results/INPUTS.json');r=next(r for r in inputs if r['model']==a.model and r['draw']=='seed0')
            mp=next(m for m in r['maps'] if m['rule']=='ce_kl');path=REPO/mp['path']
            plan=[dict(name='four_over_six',kind='four_over_six'),dict(name='n16_k3',kind='map',map_path=str(path),map_sha256=mp['sha256'],map_policy='n16_k3',type_block=[16,64])]
            a.plan=root/'pilot_plan.json';jsonout(a.plan,plan)
        assert a.plan and a.plan.exists()
        cmd += ['-m','campaign.evaluate_ppl','--model',a.model,'--plan',str(a.plan.resolve()),'--teacher',a.teacher,
                '--freeze',str(frozen),'--freeze-sha256',freezehash]
        if a.pilot:cmd += ['--limit-windows','2']
    source_sha=load(PRIMARY/'runs/V30_calib_llama8b_seed0_attempt1/launch_record.json')['source_manifest_sha256']
    rec=dict(status='running',started_utc=now(),command=cmd,uuid=uuid,gpu_model=g['name'],source_manifest_sha256=source_sha,hub_cache=env['HF_HUB_CACHE'],
             source_note='Historical source manifest retained for input map checks; actual current source tree is independently hashed below, not asserted equal to that historical run',
             executable_sha256=sha(PYTHON),launcher_sha256=sha(__file__),protocol_sha256=sha(OUT/'FROZEN_PROTOCOL.yaml'),gpu_hours=0)
    code={str(f.relative_to(SOURCE)):sha(f) for folder in ['campaign','quantize','models'] for f in (SOURCE/folder).rglob('*.py')}
    jsonout(root/'source_hashes.json',code);rec['actual_source_hashes_sha256']=sha(root/'source_hashes.json')
    jsonout(root/'launch_record.json',rec)
    t0=time.monotonic();invalid=None;child=None
    try:
        with (root/'stdout.log').open('w') as logf:
            child=subprocess.Popen(cmd,env=env,cwd=SOURCE,stdout=logf,stderr=subprocess.STDOUT,start_new_session=True)
            rec['pid']=child.pid;jsonout(root/'launch_record.json',rec)
            print('START',a.name,'gpu',g['index'],g['name'],'pid',child.pid,flush=True)
            while child.poll() is None:
                c=gp.check('during',expected_uuids=[uuid],allocation_id=a.name,env=env,own_pids=[child.pid])
                with (root/'gpu_monitor.jsonl').open('a') as f:f.write(json.dumps(c)+'\n')
                if not c['passed']:
                    invalid=c['reasons'];print('INVALID stop own child only',invalid,flush=True)
                    child.terminate()
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    break
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
            after=gp.check('postflight',expected_uuids=[uuid],allocation_id=a.name,env=env,own_pids=[])
            jsonout(root/'postflight.json',after)
            if not after['passed']:invalid=after['reasons']
        status=load(root/'job_status.json') if (root/'job_status.json').exists() else {}
        rec.update(status='invalid' if invalid else ('complete' if child.returncode==0 and status.get('status')=='complete' else 'failed'),
                   exit_code=child.returncode,finished_utc=now(),gpu_hours=(time.monotonic()-t0)/3600,invalid_reasons=invalid,
                   wall_seconds=time.monotonic()-t0,peak_memory=status.get('torch_peak_memory_bytes'))
    finally:
        if child is not None and child.poll() is None:child.terminate();child.wait()
        jsonout(root/'launch_record.json',rec)
        # Only our own exact lease; preserve its history in the run record.
        fcntl.flock(lock,fcntl.LOCK_EX)
        if lease and load(lease).get('launcher_pid')==os.getpid():lease.unlink()
        fcntl.flock(lock,fcntl.LOCK_UN)
        with (OUT/'RUN_LOG.jsonl').open('a') as f:f.write(json.dumps(dict(rec,run_id=a.name,task='T3' if a.calibrate_stream else 'T2',path=str(root.relative_to(REPO))))+'\n')
    print('END',a.name,rec['status'],rec['gpu_hours'],flush=True)
    return 0 if rec['status']=='complete' else 1

if __name__=='__main__':raise SystemExit(main())
