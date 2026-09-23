"""Move only an identity-verified, never-launched cadence-guarded waiter."""
import argparse
import fcntl
import signal
import subprocess
import time
from common import *
from reassign_waiting import snapshot,RUNTIME
from gpu_run_wait import transformed

GUARD_SHA='d8b33ad150bfebdd2749a1fc661123c3745d9bd39ee1fb4ca9b94d626ec71ee5'

def validate(s,uid,run,ticks):
    assert s['uid']==uid and s['start_ticks']==ticks,'Process identity mismatch'
    assert s['script']==str(OUT/'scripts/cadence_launch.py'),'Not a guarded launcher'
    assert s['cwd']==str(REPO) and s['run']==run,'Wrong workspace/run'
    assert not s['launch_record'] and not s['leases'] and not s['children'],'Already launched/leased/has children'
    args=s['argv'];assert '--entry' in args and args[args.index('--entry')+1] in ['accuracy_run.py','gpu_run_wait.py']
    assert '--only-uuid' in args and '--inspect-only' not in args

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pid',required=True,type=int);ap.add_argument('--start-ticks',required=True)
    ap.add_argument('--run',required=True);ap.add_argument('--new-run',required=True);ap.add_argument('--to-uuid',required=True)
    ap.add_argument('--inspect-only',action='store_true');a=ap.parse_args()
    assert all(x.replace('_','').replace('-','').isalnum() for x in [a.run,a.new_run]) and a.run!=a.new_run
    assert sha(OUT/'scripts/cadence_launch.py')==GUARD_SHA
    transformed(a.to_uuid)  # Existing strict UUID/hash validation, no GPU query.
    root=RUNTIME/'runs'/a.run;dest=root/'guarded_waiting_reassignment.json'
    assert root.is_dir() and not dest.exists() and not (RUNTIME/'runs'/a.new_run).exists()
    with (RUNTIME/'leases/registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        s=snapshot(a.pid,a.run);validate(s,os.getuid(),a.run,a.start_ticks)
        cmd=list(s['argv']);cmd[cmd.index('--name')+1]=a.new_run;cmd[cmd.index('--only-uuid')+1]=a.to_uuid
        if a.inspect_only:
            print(dict(status='safe_prelaunch_snapshot_only',snapshot=s,new_command=cmd));return 0
        validate(snapshot(a.pid,a.run),os.getuid(),a.run,a.start_ticks)
        os.kill(a.pid,signal.SIGTERM)
        for _ in range(50):
            p=Path('/proc')/str(a.pid)
            if not p.exists():break
            try:st=(p/'stat').read_text().rsplit(')',1)[1].split()
            except FileNotFoundError:break
            if st[19]!=a.start_ticks or st[0]=='Z':break
            time.sleep(.1)
        else:raise RuntimeError('Original waiter remains; no replacement')
        assert not (root/'launch_record.json').exists()
        d=dict(checked_utc=now(),old_snapshot=s,new_command=cmd,additional_scientific_attempt=False,gpu_hours=0,
            reason='Availability-only reassignment; identical scientific arguments and cadence policy',
            helper_sha256=sha(Path(__file__)),guard_sha256=GUARD_SHA)
        jsonout(dest,d)
        with (OUT/'RUN_LOG.jsonl').open('a') as f:f.write(json.dumps(dict(d,task='guarded-wait-reassignment',run_id=a.run))+'\n')
    child=subprocess.Popen(cmd,cwd=REPO,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'))
    print('ATTACHED guarded replacement',child.pid,flush=True);return child.wait()

if __name__=='__main__':raise SystemExit(main())
