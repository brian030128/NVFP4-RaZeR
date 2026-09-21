"""Host-side, read-only GPU occupancy sampler (evidence for external-contention blockers).
Every INTERVAL seconds appends one JSON line per sample to CR/logs/gpu_occupancy.jsonl: per GPU the model, used memory and the
owners of compute processes (nvidia-smi + /proc). It never signals or touches any process."""
import json
import os
import pwd
import subprocess
import sys
import time
from pathlib import Path

CR = Path(os.environ['CAMPAIGN_ROOT'])
OUT = CR / 'logs' / 'gpu_occupancy.jsonl'
INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def owner(pid):
    try:
        return pwd.getpwuid(os.stat(f'/proc/{pid}').st_uid).pw_name
    except Exception:
        return None


def sample():
    g = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, timeout=60)
    a = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, timeout=60)
    if g.returncode or a.returncode:
        return dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), error=(g.stderr or a.stderr)[-300:])
    gpus = {}
    for line in g.stdout.strip().splitlines():
        i, u, n, m, ut = [x.strip() for x in line.split(',')]
        gpus[u] = dict(index=int(i), name=n, memory_used_mib=int(m), util=int(ut), owners=[])
    for line in a.stdout.strip().splitlines():
        if not line.strip():
            continue
        u, p, m = [x.strip() for x in line.split(',')]
        if u in gpus:
            gpus[u]['owners'].append(dict(pid=int(p), user=owner(p), used_mib=int(m)))
    return dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), gpus=gpus)


while True:
    try:
        rec = sample()
    except Exception as exc:
        rec = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), error=repr(exc))
    with open(OUT, 'a') as f:
        f.write(json.dumps(rec, sort_keys=True) + '\n')
    time.sleep(INTERVAL)
