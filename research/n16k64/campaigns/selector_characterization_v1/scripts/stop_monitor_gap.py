"""Stop only an identity-verified campaign child after a persisted >60s gap."""
import argparse
import signal
from common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--pid',required=True,type=int);ap.add_argument('--start-ticks',required=True);a=ap.parse_args()
    assert '/' not in a.run and a.run not in ['.','..']
    root=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z/runs'/a.run
    lr=load(root/'launch_record.json');assert lr['status']=='running' and lr['pid']==a.pid
    proc=Path('/proc')/str(a.pid);assert proc.stat().st_uid==os.getuid()
    stat=(proc/'stat').read_text().split();assert stat[21]==a.start_ticks
    args=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    assert args==lr['command'] and 'campaign.job_wrapper' in args
    assert (proc/'cwd').resolve()==PRIMARY/'source/NVFP4-RaZeR-main'
    env=dict(x.split(b'=',1) for x in (proc/'environ').read_bytes().split(b'\0') if b'=' in x)
    assert env[b'CAMPAIGN_RUN_ID'].decode()==a.run and env[b'CAMPAIGN_RUN_DIR'].decode()==str(root)
    rows=[json.loads(x) for x in (root/'gpu_monitor.jsonl').read_text().splitlines()]
    ts=[datetime.fromisoformat(x['timestamp_utc'].replace('Z','+00:00')) for x in rows]
    gaps=[dict(before=b.isoformat(),after=c.isoformat(),seconds=(c-b).total_seconds()) for b,c in zip(ts,ts[1:]) if (c-b).total_seconds()>60]
    assert gaps,'No persisted disqualifying gap'
    record=dict(run=a.run,pid=a.pid,start_ticks=a.start_ticks,checked_utc=now(),reason='GPU ownership monitoring gap exceeds frozen 60-second maximum',gaps=gaps,signal='SIGTERM',scope='Exact own child only; launcher remains to perform postflight and terminal accounting',scientific_status='invalid',source_sha256=sha(Path(__file__)))
    dest=root/'external_invalidation.json';assert not dest.exists();jsonout(dest,record)
    # Recheck against PID reuse immediately before the only signal.
    assert proc.stat().st_uid==os.getuid() and (proc/'stat').read_text().split()[21]==a.start_ticks
    os.kill(a.pid,signal.SIGTERM)
    print(record)

if __name__=='__main__':main()
