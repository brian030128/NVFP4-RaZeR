"""Resume frozen T2 plans after validated predecessors; no automatic retries.

This is a scheduling layer, not a selector. Every GPU child uses gpu_run.py's
exclusive leases, ownership checks and three-device cap. Existing launchers
are observed, never stopped. The attached controller emits terminal events;
it does not claim an unattended log will notify the agent.
"""
import argparse
import fcntl
import subprocess
import sys
import time
from common import *

RUNTIME=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z'
PYTHON=PRIMARY/'env/venv_main/bin/python'
RECOVERED=Path('/share3/saves/JAAAAAA/mixfp4_n16k64_followup_20260917T090438Z/cache/hf/hub')

def launchers():
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if not any(x.endswith('selector_characterization_v1/scripts/'+entry) for x in argv for entry in ['gpu_run.py','gpu_run_on_uuid.py','gpu_run_wait.py']):continue
            found.append(dict(pid=int(p.name),run=argv[argv.index('--name')+1],model=argv[argv.index('--model')+1]))
        except (OSError,ValueError,UnicodeError):continue
    return found

def choose(rows, live, blocked):
    # One in-flight plan per model. The ordering never uses observed NLL.
    busy={x['model'] for x in live}
    for model in MODELS:
        if model in busy:continue
        for draw in DRAWS:
            if (model,draw) in blocked:continue
            pending=[r for r in rows if r['model']==model and r['draw']==draw and r['status']=='pending_evaluation']
            if pending:return model,draw
    return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inspect-only',action='store_true');ap.add_argument('--only-uuid');args=ap.parse_args()
    if args.only_uuid:
        from gpu_run_on_uuid import transformed
        transformed(args.only_uuid)  # Validate the pinned adapter without querying GPUs.
    assert Path.cwd().resolve()==REPO.resolve(),'Run from repository root'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()=='db63419cc33b2bbbda2117aad636435a2956532d'
    assert sha(OUT/'FROZEN_PROTOCOL.yaml')=='dbf2a3cb353e168a9bbe104309a345781f51f7d04ca281c4415e8097463e2306'
    control=RUNTIME/'controller';control.mkdir(exist_ok=True)
    lock=(control/'t2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
    observed=None;children={};blocked=set()
    while True:
        task_status=load(OUT/'TASK_STATUS.json')
        assert task_status['source_head']=='db63419cc33b2bbbda2117aad636435a2956532d'
        terminals={str(p):sha(p) for p in (RUNTIME/'runs').glob('*/launch_record.json') if load(p).get('status')!='running'}
        if not args.inspect_only and terminals!=observed:
            # Ingestion validates full-window baselines, maps, runtime and raw
            # arrays. A successful process exit alone never admits outcomes.
            rc=subprocess.call([str(PYTHON),str(OUT/'scripts/quality.py')],cwd=REPO,env=env)
            if rc:raise RuntimeError('T2 ingestion failed; no successor launch authorized')
            observed=terminals
        for name,p in list(children.items()):
            if p.poll() is not None:
                print('CONTROLLER child terminal',name,p.returncode,flush=True);del children[name]
        live=launchers()
        rows=list(csv.DictReader((OUT/'job_plan.csv').open()))
        remaining=sum(r['status']=='pending_evaluation' for r in rows)
        for path in terminals:
            record=load(path);cmd=record.get('command',[])
            if '--plan' not in cmd:continue
            planname=Path(cmd[cmd.index('--plan')+1]).name
            for model in MODELS:
                for draw in DRAWS:
                    if planname==f'{model}_{draw}_anchored.json' and any(r['model']==model and r['draw']==draw and r['status']=='pending_evaluation' for r in rows):
                        blocked.add((model,draw))
        # Completed/failed prior attempts without admitted cells require a
        # human/agent diagnosis. Never repeatedly rerun identical failures.
        for model in MODELS:
            for draw in DRAWS:
                name=f'continued_t2_{model}_{draw}_attempt1';root=RUNTIME/'runs'/name
                if root.exists() and not any(x['run']==name for x in live):
                    if any(r['model']==model and r['draw']==draw and r['status']=='pending_evaluation' for r in rows):
                        blocked.add((model,draw))
        choice=choose(rows,live,blocked) if len(live)<3 else None
        state=dict(checked_utc=now(),remaining_cells=remaining,live=live,blocked=[list(x) for x in sorted(blocked)],
                   next_choice=choice,inspect_only=args.inspect_only,allowed_uuid=args.only_uuid,
                   policy='fixed model/draw order; no outcome-based selection; no automatic repair retries')
        if args.inspect_only:print(json.dumps(state));return
        jsonout(control/'t2_status.json',state)
        if remaining==0:
            print('CONTROLLER T2 all cells validated; T3/T5 not implied complete',flush=True);return
        if choice:
            model,draw=choice;name=f'continued_t2_{model}_{draw}_attempt1'
            plan=OUT/'plans'/f'{model}_{draw}_anchored.json';assert plan.exists()
            gate=load(OUT/'results/ANCHOR_GATE.json');assert gate[model]['passed']
            for policy in load(plan):
                if policy['kind']=='map':assert sha(Path(policy['map_path']))==policy['map_sha256']
            cmd=[str(PYTHON),str(OUT/'scripts/gpu_run.py'),'--name',name,'--model',model,
                 '--plan',str(plan),'--teacher','instance','--device-type','a6000']
            if args.only_uuid:
                cmd[1]=str(OUT/'scripts/gpu_run_on_uuid.py');cmd+=['--only-uuid',args.only_uuid]
            if model!='llama8b':cmd+=['--hub-cache',str(RECOVERED)]
            logfile=control/(name+'.log')
            with logfile.open('x') as f:
                child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
            children[name]=child
            event=dict(time=now(),run=name,pid=child.pid,command=cmd,plan_sha256=sha(plan),protocol_sha256=sha(OUT/'FROZEN_PROTOCOL.yaml'))
            with (control/'t2_events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            print('CONTROLLER submitted',name,child.pid,flush=True)
        elif not live:
            print('CONTROLLER unresolved remaining cells; diagnosis required',json.dumps(state),flush=True);return
        else:print('CONTROLLER waiting',remaining,'cells;',len(live),'live launchers',flush=True)
        time.sleep(30)

if __name__=='__main__':main()
