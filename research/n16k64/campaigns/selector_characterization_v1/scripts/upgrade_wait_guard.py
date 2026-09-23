"""Replace only a verified never-launched accuracy waiter with cadence wrapper."""
import argparse
import fcntl
import signal
import subprocess
import time
from common import *
from reassign_waiting import snapshot,validate_snapshot,RUNTIME,ACCURACY_SHA

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pid',required=True,type=int);ap.add_argument('--start-ticks',required=True)
    ap.add_argument('--run',required=True);ap.add_argument('--new-run',required=True);a=ap.parse_args()
    assert all(x.replace('_','').replace('-','').isalnum() for x in [a.run,a.new_run])
    root=RUNTIME/'runs'/a.run;assert root.is_dir() and not (RUNTIME/'runs'/a.new_run).exists()
    dest=root/'waiting_guard_upgrade.json';assert not dest.exists()
    with (RUNTIME/'leases/registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        s=snapshot(a.pid,a.run);validate_snapshot(s,os.getuid(),a.run,a.start_ticks)
        assert s['script']==str(OUT/'scripts/accuracy_run.py') and sha(s['script'])==ACCURACY_SHA
        rest=list(s['argv'][2:]);rest[rest.index('--name')+1]=a.new_run
        cmd=[s['argv'][0],str(OUT/'scripts/cadence_launch.py'),'--entry','accuracy_run.py']+rest
        validate_snapshot(snapshot(a.pid,a.run),os.getuid(),a.run,a.start_ticks)
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
            reason='Add fail-closed cadence policy before any GPU allocation; identical scientific arguments',
            helper_sha256=sha(Path(__file__)))
        jsonout(dest,d)
        with (OUT/'RUN_LOG.jsonl').open('a') as f:f.write(json.dumps(dict(d,task='waiting-guard-upgrade',run_id=a.run))+'\n')
    child=subprocess.Popen(cmd,cwd=REPO,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'))
    print('ATTACHED guarded waiter',child.pid,flush=True);return child.wait()

if __name__=='__main__':raise SystemExit(main())
