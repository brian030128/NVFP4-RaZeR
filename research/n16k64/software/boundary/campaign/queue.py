"""Unattended campaign job queue on top of the lease/Docker launcher.

queue/jobs/<job_id>.json : {job_id, matrix_id, protocol_id, gpus, gpu_model ('a6000'|'ada'|'any'), env, cpus, memory,
                            command: [...], depends_on: [job_id...], priority (lower first), max_invalid_retries}
Attempts are runs named <job_id>_attemptN. 'invalid' (co-tenancy) attempts are retried automatically;
'failed' attempts stop the job until a human/agent inspects it (the job file is then edited or re-added).
The daemon never launches when /home free space is below FREE_DISK_FLOOR_GIB.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from campaign import gpu_preflight as gp
from campaign import launcher as L

CR = L.CAMPAIGN_ROOT
QDIR = CR / 'queue'
FREE_DISK_FLOOR_GIB = 45
QUIET_SECONDS = L.QUIET_SECONDS  # per-GPU window: launcher.quiet_seconds (1800 s on GPUs with observed mid-run co-tenancy, else 300 s)
_last_busy = {}
PY = str(CR / 'env' / 'venv_main' / 'bin' / 'python')


def jobs():
    out = {}
    for p in sorted((QDIR / 'jobs').glob('*.json')):
        j = json.loads(p.read_text())
        out[j['job_id']] = j
    return out


def attempts(job_id):
    res = []
    for d in sorted((CR / 'runs').glob(f'{job_id}_attempt*')):
        lr = d / 'launch_record.json'
        st = json.loads(lr.read_text()).get('status') if lr.exists() else 'unknown'
        res.append((d.name, st))
    res.sort(key=lambda x: int(x[0].rsplit('attempt', 1)[1]))
    return res


def job_state(j, running):
    if j['job_id'] in running:
        return 'running'
    att = attempts(j['job_id'])
    if not att:
        return 'pending'
    last = att[-1][1]
    if last == 'complete':
        token = QDIR / 'rerun' / f'{j["job_id"]}.json'
        if token.exists() and json.loads(token.read_text()).get('after_attempt') == att[-1][0]:
            return 'pending'  # recorded request to refresh an analysis with newer inputs
        return 'complete'
    if last in ('starting', 'running', 'unknown'):
        return 'running_orphan'
    if last == 'invalid':
        n_invalid = sum(1 for _, s in att if s == 'invalid')
        return 'pending' if n_invalid <= j.get('max_invalid_retries', 3) else 'failed'
    if last in ('not_started', 'preflight_failed'):
        return 'pending'
    token = QDIR / 'retry' / f'{j["job_id"]}.json'
    if token.exists():
        t = json.loads(token.read_text())
        if t.get('after_attempt') == att[-1][0]:
            return 'pending'
    return 'failed'


def free_gpus():
    table = gp.smi_gpus()
    apps = gp.smi_compute_apps()
    busy = {a['gpu_uuid'] for a in apps}
    foreign = set()
    for a in apps:
        try:
            if gp.proc_owner(a['pid'])[0] != os.getuid():
                foreign.add(a['gpu_uuid'])
        except gp.QueryError:
            foreign.add(a['gpu_uuid'])
    free = {'a6000': [], 'ada': []}
    now = time.time()
    cot = L.cotenancy_gpus()
    for g in table:
        if g['uuid'] in foreign or (g['uuid'] not in busy and g['memory_used_mib'] >= L.FREE_MEMORY_MIB):
            _last_busy[g['uuid']] = now  # only other users' activity starts the quiet window
        if g['uuid'] in busy or g['memory_used_mib'] >= L.FREE_MEMORY_MIB:
            continue
        if _last_busy.get(g['uuid']) is not None and now - _last_busy[g['uuid']] < L.quiet_seconds(g['uuid'], cot):
            continue
        lock = CR / 'allocator' / 'locks' / f'{g["uuid"]}.lock'
        try:
            import fcntl
            fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
            except BlockingIOError:
                os.close(fd)
                continue
            os.close(fd)
        except OSError:
            continue
        for k, name in L.MODELS.items():
            if g['name'] == name:
                free[k].append(g['uuid'])
    return free


def launch(j, running):
    att = attempts(j['job_id'])
    n = (int(att[-1][0].rsplit('attempt', 1)[1]) + 1) if att else 1
    run_id = f'{j["job_id"]}_attempt{n}'
    model = j['gpu_model_resolved']
    cmd = [PY, '-m', 'campaign.launcher', '--run-id', run_id, '--matrix-id', j['matrix_id'], '--protocol-id', j['protocol_id'],
           '--gpus', str(j['gpus']), '--gpu-model', model, '--env', j.get('env', 'main'), '--cpus', str(j.get('cpus', 12)),
           '--memory', j.get('memory', '96g'), '--wait']
    for e in j.get('set_env', []):
        cmd += ['--set-env', e]
    cmd += ['--'] + j['command']
    log = open(QDIR / 'logs' / f'{run_id}.launcher.log', 'ab')
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(L.SOURCE), start_new_session=True)
    running[j['job_id']] = dict(pid=p.pid, run_id=run_id, started=time.time(), proc=p)
    L.append_registry(dict(event='queue_launch', job_id=j['job_id'], run_id=run_id, gpus=j['gpus'], gpu_model=model))
    print(f'{time.strftime("%H:%M:%S")} LAUNCH {run_id} gpus={j["gpus"]}x{model}', flush=True)


def loop(poll=30, once=False):
    (QDIR / 'jobs').mkdir(parents=True, exist_ok=True)
    (QDIR / 'logs').mkdir(parents=True, exist_ok=True)
    running = {}
    orphans = set()
    try:
        free_gpus()  # seed the quiet-window state from current foreign activity
    except gp.QueryError:
        pass
    while True:
        for jid in list(running):
            if running[jid]['proc'].poll() is not None:
                print(f'{time.strftime("%H:%M:%S")} EXIT {running[jid]["run_id"]} rc={running[jid]["proc"].returncode}', flush=True)
                del running[jid]
        js = jobs()
        states = {jid: job_state(j, running) for jid, j in js.items()}
        for jid, st in states.items():
            if st == 'running_orphan':
                orphans.add(jid)
            elif jid in orphans:
                orphans.discard(jid)
                att = attempts(jid)
                print(f'{time.strftime("%H:%M:%S")} EXIT {att[-1][0]} status={att[-1][1]} (adopted)', flush=True)
        free_disk = shutil.disk_usage(str(CR)).free / 2 ** 30
        status = dict(utc=gp._utc(), free_disk_gib=free_disk, running={k: v['run_id'] for k, v in running.items()},
                      counts={s: sum(1 for x in states.values() if x == s) for s in set(states.values())},
                      failed=[k for k, s in states.items() if s == 'failed'])
        (QDIR / 'status.json').write_text(json.dumps(status, indent=1) + '\n')
        if free_disk >= FREE_DISK_FLOOR_GIB and not (QDIR / 'PAUSE').exists():
            try:
                free = free_gpus()
            except gp.QueryError:
                free = {'a6000': [], 'ada': []}
            reserved = set()
            for jid, j in sorted(js.items(), key=lambda kv: (kv[1].get('priority', 50), kv[0])):
                if states[jid] != 'pending':
                    continue
                if j.get('hold') and not (QDIR / 'release' / f'{jid}').exists():
                    continue
                if any(states.get(d) != 'complete' for d in j.get('depends_on', [])):
                    continue
                prefs = ['a6000', 'ada'] if j.get('gpu_model', 'a6000') == 'any' else [j.get('gpu_model', 'a6000')]
                if j['gpus'] > 0 and all(m in reserved for m in prefs):
                    continue
                launched = False
                for m in prefs:
                    if m in reserved and j['gpus'] > 0:
                        continue
                    if len(free[m]) >= j['gpus']:
                        j['gpu_model_resolved'] = m
                        launch(j, running)
                        free[m] = free[m][j['gpus']:]
                        states[jid] = 'running'
                        launched = True
                        break
                if not launched and j.get('reserve') and j['gpus'] > 1:
                    # hold this GPU type for a multi-GPU job so single-GPU jobs cannot starve it
                    reserved.update(prefs)
        if once:
            return
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--poll', type=int, default=30)
    args = ap.parse_args()
    loop(args.poll, args.once)


if __name__ == '__main__':
    main()
