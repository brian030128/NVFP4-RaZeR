"""V01 real-hardware integration checks for the preflight and lease/Docker launcher.

Read-only against other users: foreign-PID cases only *query* GPUs that other users
occupy; nothing is signalled. The co-tenancy injection uses this user's own probe
process on this campaign's own lease.
"""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from campaign import gpu_preflight as gp
from campaign import launcher as L

CR = L.CAMPAIGN_ROOT
OUT = CR / 'reports' / 'V01'
PY = str(CR / 'env' / 'venv_main' / 'bin' / 'python')


def host_check(name, cvd, expected=None, **kw):
    rec = gp.check(name, env={'CUDA_VISIBLE_DEVICES': cvd}, expected_uuids=expected, allocation_id='v01-integration', **kw)
    path, digest = gp.write_record(rec, OUT / 'records' / f'{name}.json')
    return dict(case=name, passed=rec['passed'], reasons=rec['reasons'], record=path, sha256=digest,
                processes=[dict(pid=p['pid'], owner=p.get('owner'), gpu=p['gpu_uuid']) for p in rec['compute_processes']])


def pid_alive(pid):
    return Path(f'/proc/{pid}').exists()


def launch(run_id, gpus, model, extra_env=(), wait=False):
    cmd = [PY, '-m', 'campaign.launcher', '--run-id', run_id, '--matrix-id', 'V01', '--protocol-id', 'resource',
           '--gpus', str(gpus), '--gpu-model', model, '--env', 'main', '--cpus', '4', '--memory', '16g']
    for e in extra_env:
        cmd += ['--set-env', e]
    if wait:
        cmd.append('--wait')
    cmd += ['--', '-m', 'campaign.jobs.env_probe']
    return cmd


def summarize_run(run_id):
    rd = CR / 'runs' / run_id
    lr = json.loads((rd / 'launch_record.json').read_text())
    val = json.loads((rd / 'run_record_validation.json').read_text()) if (rd / 'run_record_validation.json').exists() else None
    return dict(run_id=run_id, status=lr.get('status'), exit_code=lr.get('exit_code'), leased=lr.get('leased_uuids'),
                invalid=lr.get('invalid_gpu_cotenancy'), sidecar=lr.get('sidecar'), record_valid=val,
                invalid_marker=(rd / 'INVALID_GPU_COTENANCY.json').exists())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'records').mkdir(exist_ok=True)
    table = gp.smi_gpus()
    apps = gp.smi_compute_apps()
    busy = {a['gpu_uuid'] for a in apps}
    a6 = [g for g in table if g['name'] == L.MODELS['a6000']]
    ada = [g for g in table if g['name'] == L.MODELS['ada']]
    results = dict(started_utc=gp._utc(), gpu_table=table, compute_apps=apps, cases=[])
    cases = results['cases']
    # Real negative cases (query-only).
    foreign = [a for a in apps if gp.proc_owner(a['pid'])[0] != os.getuid()]
    if foreign:
        u = foreign[0]['gpu_uuid']
        c = host_check('real_foreign_pid', u, [u])
        c['foreign_pid_still_alive_after_check'] = pid_alive(foreign[0]['pid'])
        c['expected_pass'] = False
        cases.append(c)
    cases.append(dict(host_check('real_four_visible', ','.join(g['uuid'] for g in a6[:4])), expected_pass=False))
    cases.append(dict(host_check('real_mixed_models', f'{a6[-1]["uuid"]},{ada[0]["uuid"]}'), expected_pass=False))
    cases.append(dict(host_check('real_duplicate_index', f'{a6[-1]["index"]},{a6[-1]["index"]}'), expected_pass=False))
    cases.append(dict(host_check('real_remapped_index_uuid', f'{a6[-1]["index"]},{a6[-1]["uuid"]}'), expected_pass=False))
    cases.append(dict(host_check('real_zero_visible', ''), expected_pass=False))

    def broken():
        return gp.smi_gpus(run=lambda cmd: gp._run(['/nonexistent/nvidia-smi'] + cmd[1:]))
    cases.append(dict(host_check('real_unavailable_query', a6[-1]['uuid'], gpu_query=broken), expected_pass=False))
    free_a6 = [g for g in a6 if g['uuid'] not in busy and g['memory_used_mib'] < L.FREE_MEMORY_MIB]
    free_ada = [g for g in ada if g['uuid'] not in busy and g['memory_used_mib'] < L.FREE_MEMORY_MIB]
    for label, pool in (('a6000', free_a6), ('ada', free_ada)):
        for n in (1, 2, 3):
            if len(pool) >= n:
                cases.append(dict(host_check(f'real_clean_{n}gpu_{label}', ','.join(g['uuid'] for g in pool[:n]),
                                             [g['uuid'] for g in pool[:n]]), expected_pass=True))
            else:
                cases.append(dict(case=f'real_clean_{n}gpu_{label}', passed=None, expected_pass=True,
                                  not_run_reason=f'only {len(pool)} free {label} GPU(s) at {gp._utc()}'))
    # Lease exclusivity with real flocks: hold every free A6000 lock, the allocator must refuse.
    held = []
    for g in free_a6:
        fd = os.open(CR / 'allocator' / 'locks' / f'{g["uuid"]}.lock', os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held.append(fd)
    lease, state = L.try_lease('a6000', 1)
    cases.append(dict(case='lease_refused_when_locked', passed=lease is None, expected_pass=True,
                      detail=f'{len(held)} free A6000 locks held by the test'))
    for fd in held:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    # Allocator refuses more homogeneous GPUs than are free.
    lease, state = L.try_lease('a6000', len(free_a6) + 1) if len(free_a6) < 3 else (None, None)
    cases.append(dict(case='lease_refused_insufficient_free', passed=lease is None, expected_pass=True,
                      detail=f'requested {len(free_a6)+1} with {len(free_a6)} free A6000'))
    if lease:
        lease.release()
    json.dump(results, open(OUT / 'v01_integration_partial.json', 'w'), indent=1, default=str)
    # Launched end-to-end runs.
    launched = []
    if free_ada:
        rid = 'V01_integration_clean1_ada_attempt1'
        p = subprocess.run(launch(rid, 1, 'ada'), capture_output=True, text=True)
        launched.append(dict(summarize_run(rid), launcher_rc=p.returncode, expected='complete'))
    if len(free_a6) >= 2:
        rid = 'V01_integration_clean2_a6000_attempt1'
        p = subprocess.run(launch(rid, 2, 'a6000'), capture_output=True, text=True)
        launched.append(dict(summarize_run(rid), launcher_rc=p.returncode, expected='complete'))
    # Co-tenancy injection on our own lease.
    rid = 'V01_integration_cotenancy_injection_attempt1'
    proc = subprocess.Popen(launch(rid, 1, 'a6000', extra_env=['PROBE_SLEEP=150']), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    rd = CR / 'runs' / rid
    uuid = None
    for _ in range(120):
        time.sleep(1)
        if (rd / 'container.cid').exists() and (rd / 'launch_record.json').exists():
            lr = json.loads((rd / 'launch_record.json').read_text())
            if lr.get('leased_uuids'):
                uuid = lr['leased_uuids'][0]
                break
    time.sleep(25)
    injector = subprocess.Popen([PY, '-c', 'import torch,time; x=torch.zeros(1,device="cuda"); time.sleep(240)'],
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid or 'none'))
    proc.wait(timeout=600)
    alive = injector.poll() is None
    launched.append(dict(summarize_run(rid), launcher_rc=proc.returncode, expected='invalid',
                         injector_pid=injector.pid, injector_alive_after_invalidation=alive))
    injector.terminate()
    injector.wait(timeout=60)
    results['launched'] = launched
    ok_cases = all(c['passed'] == c['expected_pass'] for c in cases if c.get('passed') is not None)
    ok_launch = all(r['status'] == r['expected'] and r['record_valid'] and r['record_valid']['valid'] for r in launched)
    ok_inject = any(r['expected'] == 'invalid' and r['invalid_marker'] and r['injector_alive_after_invalidation'] for r in launched)
    results.update(finished_utc=gp._utc(), all_cases_as_expected=ok_cases, launched_as_expected=ok_launch,
                   injection_detected_without_killing=ok_inject,
                   not_run=[c for c in cases if c.get('passed') is None])
    (OUT / 'v01_integration.json').write_text(json.dumps(results, indent=1, default=str) + '\n')
    print(json.dumps(dict(ok_cases=ok_cases, ok_launch=ok_launch, ok_inject=ok_inject,
                          not_run=[c['case'] for c in results['not_run']]), indent=1))
    sys.exit(0 if (ok_cases and ok_launch and ok_inject) else 1)


if __name__ == '__main__':
    main()
