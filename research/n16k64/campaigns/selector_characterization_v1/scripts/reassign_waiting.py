"""Reassign only this campaign's verified, never-launched waiting process.

The registry lock closes the check/launch race. Never signal a GPU child or
foreign/unknown PID. Preserve the old directory and explicit reassignment
record; a resource wait is not an additional scientific attempt.
"""
import argparse
import fcntl
import signal
import subprocess
import time
from common import *
from gpu_run_wait import transformed

RUNTIME=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z'
WAITING_SCRIPTS={str(OUT/'scripts/gpu_run_wait.py'),str(OUT/'scripts/accuracy_run.py')}
ACCURACY_SHA='431a256b849b34a517f4e08b009ab58d3f0d94bd7f4db4f3a6204319617f4ed8'

def validate_snapshot(s, uid, run, ticks):
    assert s['uid']==uid,'Not our process'
    assert s['start_ticks']==ticks,'PID identity changed'
    assert s['script'] in WAITING_SCRIPTS,'Not this campaign waiting launcher'
    assert s['cwd']==str(REPO),'Unexpected working directory'
    assert s['run']==run,'Different run'
    assert not s['launch_record'] and not s['leases'] and not s['children'],'Already launched or has a child/lease'

def snapshot(pid, run):
    p=Path('/proc')/str(pid);argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    script=Path(argv[1]);script=script if script.is_absolute() else (p/'cwd').resolve()/script
    return dict(uid=p.stat().st_uid,start_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19],
        script=str(script.resolve()),cwd=str((p/'cwd').resolve()),run=argv[argv.index('--name')+1],argv=argv,
        launch_record=(RUNTIME/'runs'/run/'launch_record.json').exists(),
        leases=[str(f) for f in (RUNTIME/'leases').glob('GPU-*.json') if load(f).get('launcher_pid')==pid],
        children=(p/'task'/str(pid)/'children').read_text().split())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pid',type=int,required=True);ap.add_argument('--start-ticks',required=True)
    ap.add_argument('--run',required=True);ap.add_argument('--new-run',required=True);ap.add_argument('--to-uuid',required=True)
    ap.add_argument('--inspect-only',action='store_true');a=ap.parse_args();start=now()
    assert all(x.replace('_','').replace('-','').isalnum() for x in [a.run,a.new_run])
    assert a.run!=a.new_run
    transformed(a.to_uuid)  # Hash-pinned adapter and UUID validation, no GPU query.
    root=RUNTIME/'runs'/a.run;assert root.is_dir()
    dest=root/'waiting_reassignment.json';assert not dest.exists()
    assert not (RUNTIME/'runs'/a.new_run).exists(),'New run already exists'
    lock=(RUNTIME/'leases/registry.lock').open('a')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        s=snapshot(a.pid,a.run);validate_snapshot(s,os.getuid(),a.run,a.start_ticks)
        if s['script']==str(OUT/'scripts/accuracy_run.py'):
            assert sha(s['script'])==ACCURACY_SHA,'Accuracy adapter changed: re-audit before reassignment'
            from accuracy_run import transform
            transform((OUT/'scripts/gpu_run.py').read_text(),a.to_uuid)
        prehash=sha(root/'preflight_candidates.jsonl') if (root/'preflight_candidates.jsonl').exists() else None
        cmd=list(s['argv']);cmd[cmd.index('--name')+1]=a.new_run;cmd[cmd.index('--only-uuid')+1]=a.to_uuid
        if a.inspect_only:
            print(json.dumps(dict(status='safe_prelaunch_snapshot_only',pid=a.pid,snapshot=s,new_command=cmd)));return
        # Re-read identity/state immediately before signalling, still holding
        # the same lease lock used by all campaign launchers.
        validate_snapshot(snapshot(a.pid,a.run),os.getuid(),a.run,a.start_ticks)
        os.kill(a.pid,signal.SIGTERM)
        for _ in range(50):
            p=Path('/proc')/str(a.pid)
            if not p.exists():break
            try:stat=(p/'stat').read_text().rsplit(')',1)[1].split()
            except FileNotFoundError:break
            if stat[19]!=a.start_ticks or stat[0]=='Z':break
            time.sleep(.1)
        else:raise RuntimeError('Waiting launcher not terminated; no replacement started')
        assert not (root/'launch_record.json').exists()
        event=dict(timestamp_utc=now(),status='reassigned_before_any_gpu_launch',old_pid=a.pid,old_snapshot=s,
            new_run=a.new_run,new_command=cmd,gpu_hours=0,additional_scientific_attempt=False,
            preflight_history_sha256=prehash,helper_sha256=sha(__file__),
            reason='Avoid idle usable A6000 while a different fixed UUID stays foreign-occupied; retain all original waiting evidence')
        jsonout(dest,event)
        with (OUT/'RUN_LOG.jsonl').open('a') as f:f.write(json.dumps(dict(event,task='GPU-wait-reassignment',run_id=a.run))+'\n')
    print('REASSIGNED prelaunch wait only',a.run,'->',a.new_run,flush=True)
    child=subprocess.Popen(cmd,cwd=REPO,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'))
    print('ATTACHED replacement launcher PID',child.pid,flush=True)
    return child.wait()

if __name__=='__main__':raise SystemExit(main())
