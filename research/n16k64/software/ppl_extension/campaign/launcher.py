"""Campaign-local GPU lease allocator + Docker device isolation + co-tenancy sidecar.

The collection host had no site scheduler (owner decision D01 in provenance/01_owner_decisions.json).
This launcher is the documented substitute:

1. A run requests N (1-3) GPUs of ONE model (a6000 | ada).
2. Candidate GPUs must have zero compute processes and < 1 GiB used memory, and must not
   be leased by another campaign run (non-blocking flock on allocator/locks/<UUID>.lock).
3. A host-side preflight (gpu_preflight.check) must pass for exactly the leased UUIDs.
4. The job runs in a container started with `--gpus device=<UUIDs>`, so CUDA inside the
   container can only reach the leased devices. The in-container job wrapper runs its own
   preflights (before / phases / after) and cross-checks torch's device UUIDs.
5. A host sidecar samples the leased GPUs every 20 s. Any compute PID that is not inside
   this run's container cgroup, or a monitoring gap, invalidates the run and stops ONLY this
   run's container. Other users' processes are never signalled.
6. A host after-check runs once the container exits. Every attempt, including failures,
   is appended to registry/attempts.jsonl and never removed.
"""
import argparse
import calendar
import datetime as dt
import fcntl
import hashlib
import json
import os
import pwd
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid as uuidlib
from pathlib import Path

from campaign import gpu_preflight as gp

CAMPAIGN_ROOT = Path(os.environ.get('CAMPAIGN_ROOT', Path(__file__).resolve().parents[3]))
REPO_ROOT = Path(os.environ.get('CAMPAIGN_REPO_ROOT', CAMPAIGN_ROOT.parents[1]))
STORAGE_ROOT = Path(os.environ.get('CAMPAIGN_STORAGE_ROOT', CAMPAIGN_ROOT))
PARENT_ROOT = Path(os.environ.get('CAMPAIGN_PARENT_ROOT', '')) if os.environ.get('CAMPAIGN_PARENT_ROOT') else None
SOURCE = Path(os.environ.get('CAMPAIGN_SOURCE_ROOT', Path(__file__).resolve().parents[1])).resolve()
IMAGE = 'ubuntu@sha256:829f6df217bcbae2b371026e81711d1a787c61b2967ad09d015063663ebafbf7'
MODELS = {'a6000': 'NVIDIA RTX A6000', 'ada': 'NVIDIA RTX 6000 Ada Generation'}
NUMA_CPUS = {'a6000': '0-23,48-71', 'ada': '24-47,72-95'}
MONITOR_SECONDS = 20
MAX_MONITOR_FAILURES = 3
FREE_MEMORY_MIB = 1024
QUIET_SECONDS = 1800        # quiet window on GPUs where another user's process has appeared during one of our leased runs
SHORT_QUIET_SECONDS = 300   # quiet window on every other GPU
CAMPAIGN_GPU_CAP = 3
_UUID = re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_OWNER = re.compile(r"'owner': '([^']+)'")


def utc():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n')
    tmp.replace(path)


def append_registry(entry):
    reg = CAMPAIGN_ROOT / 'registry' / 'attempts.jsonl'
    reg.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(entry, logged_utc=utc()), sort_keys=True, default=str) + '\n'
    with open(reg, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


class Lease:
    def __init__(self, uuids, fds, lease_id):
        self.uuids, self.fds, self.lease_id = uuids, fds, lease_id

    def release(self):
        for fd in self.fds:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            except OSError:
                pass
        self.fds = []


def cotenancy_gpus(root=None):
    """GPU UUIDs on which a process owned by another user appeared while one of our runs held the lease
    (allocator/events records written by the sidecar). Our own synthetic injection tests do not count."""
    me = pwd.getpwuid(os.getuid()).pw_name
    out = set()
    for p in Path(root or CAMPAIGN_ROOT, 'allocator', 'events').glob('*.json'):
        try:
            reason = json.loads(p.read_text()).get('reason', '')
        except Exception:
            continue
        if any(o != me for o in _OWNER.findall(reason)):
            out.update(_UUID.findall(reason))
    return out


def quiet_seconds(uuid, cot):
    return QUIET_SECONDS if uuid in cot else SHORT_QUIET_SECONDS


def try_lease(model_key, count, prefer=None, exclude=(), last_busy=None):
    name = MODELS[model_key]
    table = gp.smi_gpus()
    apps = gp.smi_compute_apps()
    busy = {a['gpu_uuid'] for a in apps}
    now = time.time()
    if last_busy is not None:
        foreign = set()
        for a in apps:
            try:
                if gp.proc_owner(a['pid'])[0] != os.getuid():
                    foreign.add(a['gpu_uuid'])
            except gp.QueryError:
                foreign.add(a['gpu_uuid'])
        for g in table:
            if g['uuid'] in foreign:
                last_busy[g['uuid']] = now
    if count == 0:
        return None, dict(table=table, apps=apps)
    cot = cotenancy_gpus()
    quiet = lambda u: last_busy is None or last_busy.get(u) is None or now - last_busy[u] >= quiet_seconds(u, cot)
    cands = [g for g in table if g['name'] == name and g['uuid'] not in busy and quiet(g['uuid'])
             and g['memory_used_mib'] < FREE_MEMORY_MIB and g['uuid'] not in exclude]
    if prefer:
        order = {u: i for i, u in enumerate(prefer)}
        cands.sort(key=lambda g: (order.get(g['uuid'], 99), g['index']))
    locks = CAMPAIGN_ROOT / 'allocator' / 'locks'
    locks.mkdir(parents=True, exist_ok=True)
    # Serialize the count-and-acquire transaction across independent launchers.  Individual
    # GPU locks are held for each run's lifetime; probing all of them while this cap lock is
    # held gives an atomic campaign-wide maximum of three for v2 launchers.  Legacy waiters
    # are also visible because they hold the same individual lock files.
    cap_fd = os.open(CAMPAIGN_ROOT / 'allocator' / 'campaign_gpu_cap.lock', os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(cap_fd, fcntl.LOCK_EX)
    try:
        active = 0
        for lp in locks.glob('GPU-*.lock'):
            probe = os.open(lp, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(probe, fcntl.LOCK_UN)
            except BlockingIOError:
                active += 1
            finally:
                os.close(probe)
        if active + count > CAMPAIGN_GPU_CAP:
            return None, dict(table=table, apps=apps, candidates=[g['uuid'] for g in cands],
                              campaign_active_gpu_leases=active, campaign_gpu_cap=CAMPAIGN_GPU_CAP,
                              rejection='campaign aggregate GPU cap')
        got, fds = [], []
        for g in cands:
            fd = os.open(locks / f'{g["uuid"]}.lock', os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            got.append(g['uuid'])
            fds.append(fd)
            if len(got) == count:
                break
    finally:
        fcntl.flock(cap_fd, fcntl.LOCK_UN)
        os.close(cap_fd)
    if len(got) < count:
        for fd in fds:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        return None, dict(table=table, apps=apps, candidates=[g['uuid'] for g in cands])
    lease_id = f'lease-{dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")}-{uuidlib.uuid4().hex[:8]}'
    for fd in fds:
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(dict(lease_id=lease_id, pid=os.getpid(), utc=utc())).encode())
    return Lease(got, fds, lease_id), dict(table=table, apps=apps)


def source_manifest(root=SOURCE):
    """SHA-256 of every tracked source file (py/md/json/yaml/sh/txt/csv) under the campaign source tree."""
    entries = []
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or '.pytest_cache' in p.parts:
            continue
        if p.suffix not in ('.py', '.md', '.json', '.yaml', '.yml', '.sh', '.txt', '.csv', '.sbatch', '.in', '.toml'):
            continue
        rel = p.relative_to(root).as_posix()
        if rel.startswith('results/'):
            continue
        entries.append((rel, sha256_file(p)))
    blob = ''.join(f'{h}  {r}\n' for r, h in entries).encode()
    return sha256_bytes(blob), entries


def image_digest():
    out = subprocess.run(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}'], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def container_env(run_dir, run_id, lease, env_name, model_key, extra):
    env = {
        'CAMPAIGN_ROOT': str(CAMPAIGN_ROOT), 'CAMPAIGN_REPO_ROOT': str(REPO_ROOT),
        'CAMPAIGN_STORAGE_ROOT': str(STORAGE_ROOT),
        'CAMPAIGN_PARENT_ROOT': str(PARENT_ROOT) if PARENT_ROOT else '',
        'HF_HOME': str(STORAGE_ROOT / 'cache/hf'),
        'HF_HUB_CACHE': str(STORAGE_ROOT / 'cache/hf/hub'), 'HF_DATASETS_CACHE': str(STORAGE_ROOT / 'cache/hf/datasets'),
        'TRITON_CACHE_DIR': str(STORAGE_ROOT / 'cache/triton'), 'TORCH_HOME': str(STORAGE_ROOT / 'cache/torch'),
        'TORCHINDUCTOR_CACHE_DIR': str(STORAGE_ROOT / 'cache/torch/inductor'),
        'XDG_CACHE_HOME': str(STORAGE_ROOT / 'cache/xdg'), 'TMPDIR': str(STORAGE_ROOT / 'cache/tmp'),
        'HOME': str(STORAGE_ROOT / 'cache/home'), 'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'CAMPAIGN_RUN_ID': run_id, 'CAMPAIGN_RUN_DIR': str(run_dir), 'CAMPAIGN_LEASE_ID': lease.lease_id if lease else '',
        'CAMPAIGN_ALLOCATED_UUIDS': ','.join(lease.uuids) if lease else '', 'CAMPAIGN_CONTAINER_NAME': f'mixfp4_{run_id}',
        'CAMPAIGN_ENV_NAME': env_name, 'CAMPAIGN_GPU_MODEL': model_key or 'cpu',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
        'HF_HUB_DISABLE_PROGRESS_BARS': '1', 'TRANSFORMERS_VERBOSITY': 'warning', 'TQDM_DISABLE': '1',
    }
    env.update(extra or {})
    return env


def classify_sidecar_process(app, container_id, known_container_pids,
                             owner_query=None, proc_exists=None):
    """Classify one sidecar process while tolerating stale NVML exit rows.

    A PID is tolerated only when an earlier sample proved that exact PID was in
    this container and the process no longer exists. Unknown or still-live
    unresolvable PIDs remain foreign, preserving fail-closed behavior.
    """
    owner_query = owner_query or gp.proc_owner
    proc_exists = proc_exists or (lambda pid: Path(f'/proc/{pid}').exists())
    try:
        uid, owner, start, cgroup = owner_query(app['pid'])
    except gp.QueryError as exc:
        if app['pid'] in known_container_pids and not proc_exists(app['pid']):
            return dict(app, owner=None, uid=None, in_container=True,
                        exited_container_process_stale_nvml=True), None
        return None, dict(app, error=str(exc))
    mine = container_id in cgroup
    record = dict(app, owner=owner, uid=uid, in_container=mine)
    if mine:
        known_container_pids.add(app['pid'])
        return record, None
    return record, dict(app, owner=owner, uid=uid)


class Sidecar(threading.Thread):
    def __init__(self, run_dir, lease, container_id, stop_event):
        super().__init__(daemon=True)
        self.run_dir, self.lease, self.cid, self.stop_event = run_dir, lease, container_id, stop_event
        self.invalid_reasons, self.samples, self.failures = [], 0, 0
        self.peak_used_mib = {u: 0 for u in lease.uuids}
        self.known_container_pids = set()

    def run(self):
        log = open(self.run_dir / 'gpu_monitor.jsonl', 'a')
        while not self.stop_event.is_set():
            try:
                table = {g['uuid']: g for g in gp.smi_gpus()}
                apps = [a for a in gp.smi_compute_apps() if a['gpu_uuid'] in self.lease.uuids]
                self.failures = 0
            except gp.QueryError as exc:
                self.failures += 1
                log.write(json.dumps(dict(utc=utc(), error=str(exc))) + '\n'); log.flush()
                if self.failures >= MAX_MONITOR_FAILURES:
                    self._invalidate(f'monitoring gap: {self.failures} consecutive failed GPU queries')
                    return
                self.stop_event.wait(MONITOR_SECONDS)
                continue
            foreign = []
            procs = []
            for a in apps:
                record, outsider = classify_sidecar_process(
                    a, self.cid, self.known_container_pids)
                if record is not None:
                    procs.append(record)
                if outsider is not None:
                    foreign.append(outsider)
            for u in self.lease.uuids:
                self.peak_used_mib[u] = max(self.peak_used_mib[u], table[u]['memory_used_mib'])
            log.write(json.dumps(dict(utc=utc(), gpus={u: dict(used_mib=table[u]['memory_used_mib'],
                                 util=table[u]['utilization_gpu_pct']) for u in self.lease.uuids},
                                 processes=procs)) + '\n')
            log.flush()
            self.samples += 1
            if foreign:
                self._invalidate(f'foreign/non-job compute process on leased GPU: {foreign}')
                return
            self.stop_event.wait(MONITOR_SECONDS)

    def _invalidate(self, reason):
        self.invalid_reasons.append(reason)
        event = dict(utc=utc(), run_dir=str(self.run_dir), lease=self.lease.lease_id, reason=reason)
        atomic_json(self.run_dir / 'INVALID_GPU_COTENANCY.json', event)
        ev = CAMPAIGN_ROOT / 'allocator' / 'events' / f'{self.run_dir.name}_{int(time.time())}.json'
        atomic_json(ev, event)
        # Stop only this run's own container. Never signal the foreign process.
        subprocess.run(['docker', 'stop', '-t', '30', self.cid], capture_output=True)


def container_finished_epoch(inspect_stdout):
    """Container exit time (epoch seconds) from `docker inspect` State.FinishedAt, or None if unavailable."""
    try:
        st = json.loads(inspect_stdout)[0]['State']['FinishedAt'].rstrip('Z')
        base, _, frac = st.partition('.')
        return calendar.timegm(time.strptime(base, '%Y-%m-%dT%H:%M:%S')) + (float('0.' + frac) if frac else 0.0)
    except Exception:
        return None


def classify_after_processes(procs, finished_epoch, check_wall, uptime, clk=None, margin=1.0):
    """Split compute processes seen by the host after-check into (overlapping, post_exit).

    A process counts as post-exit (not co-tenancy with this run) only when its start time, reconstructed from
    /proc start_ticks recorded by the preflight, is provably later than the container's FinishedAt plus `margin` seconds.
    Unknown start time or unknown container exit time -> overlapping (fail closed)."""
    clk = clk or os.sysconf('SC_CLK_TCK')
    overlapping, post_exit = [], []
    for pr in procs:
        ticks = pr.get('start_ticks')
        if finished_epoch is None or ticks is None:
            overlapping.append(dict(pr, classification='overlapping (start time or container exit time unknown)'))
            continue
        started_wall = check_wall - (uptime - int(ticks) / clk)
        e = dict(pr, started_utc_estimate=time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(started_wall)) + f'.{int((started_wall % 1) * 1000):03d}Z',
                 container_finished_epoch=finished_epoch, seconds_after_container_exit=started_wall - finished_epoch)
        if started_wall > finished_epoch + margin:
            post_exit.append(dict(e, classification='post-exit (started after this run finished)'))
        else:
            overlapping.append(dict(e, classification='overlapping'))
    return overlapping, post_exit


def launch(args, command):
    run_id = args.run_id
    run_dir = CAMPAIGN_ROOT / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'preflight').mkdir()
    (CAMPAIGN_ROOT / 'cache' / 'home').mkdir(parents=True, exist_ok=True)
    manifest_sha, entries = source_manifest()
    (run_dir / 'source_manifest.txt').write_text(''.join(f'{h}  {r}\n' for r, h in entries))
    lock_file = CAMPAIGN_ROOT / 'env' / f'{args.env}.lock.txt'
    launch_record = dict(run_id=run_id, matrix_id=args.matrix_id, protocol_id=args.protocol_id,
                         command=command, env_name=args.env, env_lock_sha256=sha256_file(lock_file),
                         image=IMAGE, image_id=image_digest(), source_manifest_sha256=manifest_sha,
                         requested=dict(gpus=args.gpus, model=args.gpu_model), created_utc=utc(),
                         host=os.uname().nodename, scheduler_kind='campaign_local_lease+docker_device_cgroup',
                         owner_decision='provenance/01_owner_decisions.json#D01')
    atomic_json(run_dir / 'launch_record.json', launch_record)
    append_registry(dict(event='created', run_id=run_id, matrix_id=args.matrix_id, command=command))
    lease = None
    waited = 0
    last_busy = {}
    if args.gpus > 0:
        while True:
            try:
                # first attempt trusts the queue's quiet-window check; later waits require a fresh quiet window
                lease, state = try_lease(args.gpu_model, args.gpus, prefer=args.prefer.split(',') if args.prefer else None,
                                         last_busy=(last_busy if waited > 0 else None))
            except gp.QueryError as exc:
                lease, state = None, dict(error=str(exc))
            if lease is not None:
                break
            if not args.wait:
                launch_record.update(status='not_started', reason='no free homogeneous GPUs', state=state)
                atomic_json(run_dir / 'launch_record.json', launch_record)
                append_registry(dict(event='not_started', run_id=run_id, reason='no free GPUs'))
                return 3
            if waited == 0:
                try:
                    try_lease(args.gpu_model, 0, last_busy=last_busy)  # seed quiet-window state (count 0 leases nothing)
                except gp.QueryError:
                    pass
            time.sleep(60)
            waited += 60
        env_host = {'CUDA_VISIBLE_DEVICES': ','.join(lease.uuids)}
        before = gp.check('host_before', expected_uuids=lease.uuids, allocation_id=lease.lease_id, env=env_host,
                          scheduler_kind=launch_record['scheduler_kind'])
        bpath, bsha = gp.write_record(before, run_dir / 'preflight' / 'host_gpu_preflight_before.json')
        launch_record.update(lease_id=lease.lease_id, leased_uuids=lease.uuids, waited_seconds=waited,
                             host_preflight_before=dict(path=bpath, sha256=bsha, passed=before['passed'],
                                                        timestamp_utc=before['timestamp_utc']))
        atomic_json(run_dir / 'launch_record.json', launch_record)
        if not before['passed']:
            lease.release()
            launch_record.update(status='preflight_failed', reasons=before['reasons'])
            atomic_json(run_dir / 'launch_record.json', launch_record)
            append_registry(dict(event='preflight_failed', run_id=run_id, reasons=before['reasons']))
            return 4
    env = container_env(run_dir, run_id, lease, args.env, args.gpu_model if args.gpus else None,
                        dict(kv.split('=', 1) for kv in args.set_env))
    venv_py = CAMPAIGN_ROOT / 'env' / f'venv_{args.env}' / 'bin' / 'python'
    inner = [str(venv_py), '-m', 'campaign.job_wrapper', '--gpu-run' if args.gpus else '--cpu-run', '--'] + command
    cidfile = run_dir / 'container.cid'
    docker = ['docker', 'run', '-d', '--cidfile', str(cidfile), '--name', env['CAMPAIGN_CONTAINER_NAME'],
              '--user', f'{os.getuid()}:{os.getgid()}', '--pid=host', '--ipc=host', '--network', args.network,
              '-v', f'{REPO_ROOT}:{REPO_ROOT}:ro', '-v', f'{CAMPAIGN_ROOT}:{CAMPAIGN_ROOT}',
              '-v', f'{STORAGE_ROOT}:{STORAGE_ROOT}',
              '-v', '/etc/passwd:/etc/passwd:ro', '-v', '/etc/group:/etc/group:ro',
              '-w', str(SOURCE), '--cpus', str(args.cpus), '--memory', args.memory,
              '--cpuset-cpus', NUMA_CPUS[args.gpu_model] if args.gpus else '0-95',
              '--label', f'campaign.run_id={run_id}', '--entrypoint', str(venv_py)]
    if PARENT_ROOT:
        docker += ['-v', f'{PARENT_ROOT}:{PARENT_ROOT}:ro']
    if args.gpus:
        docker += ['--gpus', f'"device={",".join(lease.uuids)}"']
    for k, v in env.items():
        docker += ['-e', f'{k}={v}']
    docker += [IMAGE] + inner[1:]
    (run_dir / 'docker_command.sh').write_text(' '.join(shlex.quote(x) for x in docker) + '\n')
    start = time.time()
    launch_record.update(status='starting', started_utc=utc())
    atomic_json(run_dir / 'launch_record.json', launch_record)
    p = subprocess.run(docker, capture_output=True, text=True)
    if p.returncode != 0:
        if lease:
            lease.release()
        launch_record.update(status='docker_failed', docker_stderr=p.stderr[-4000:])
        atomic_json(run_dir / 'launch_record.json', launch_record)
        append_registry(dict(event='docker_failed', run_id=run_id, stderr=p.stderr[-500:]))
        return 5
    cid = cidfile.read_text().strip()
    append_registry(dict(event='started', run_id=run_id, container=cid, lease=getattr(lease, 'lease_id', None),
                         uuids=getattr(lease, 'uuids', None)))
    logf = open(run_dir / 'container.log', 'ab')
    logger = subprocess.Popen(['docker', 'logs', '-f', cid], stdout=logf, stderr=subprocess.STDOUT)
    stop = threading.Event()
    sidecar = Sidecar(run_dir, lease, cid, stop) if lease else None
    if sidecar:
        sidecar.start()
    w = subprocess.run(['docker', 'wait', cid], capture_output=True, text=True)
    stop.set()
    if sidecar:
        sidecar.join(timeout=MONITOR_SECONDS * 3)
    end = time.time()
    inspect = subprocess.run(['docker', 'inspect', cid], capture_output=True, text=True)
    (run_dir / 'docker_inspect.json').write_text(inspect.stdout)
    logger.wait(timeout=60)
    logf.close()
    exit_code = int(w.stdout.strip()) if w.stdout.strip().lstrip('-').isdigit() else None
    launch_record.update(exit_code=exit_code, finished_utc=utc(), wall_seconds=end - start,
                         gpu_hours=(args.gpus * (end - start) / 3600.0))
    if lease:
        # Host after-check: foreign processes that started before our container finished invalidate the run.
        after = gp.check('host_after', expected_uuids=lease.uuids, allocation_id=lease.lease_id,
                         env={'CUDA_VISIBLE_DEVICES': ','.join(lease.uuids)}, scheduler_kind=launch_record['scheduler_kind'])
        apath, asha = gp.write_record(after, run_dir / 'preflight' / 'host_gpu_preflight_after.json')
        late_foreign, post_exit = classify_after_processes(after.get('compute_processes', []), container_finished_epoch(inspect.stdout),
                                                           check_wall=time.time(), uptime=float(open('/proc/uptime').read().split()[0]))
        invalid = bool((sidecar and sidecar.invalid_reasons) or late_foreign)
        launch_record.update(host_preflight_after=dict(path=apath, sha256=asha, passed=after['passed'],
                                                       timestamp_utc=after['timestamp_utc']),
                             sidecar=dict(samples=sidecar.samples, invalid_reasons=sidecar.invalid_reasons,
                                          peak_used_mib=sidecar.peak_used_mib),
                             invalid_gpu_cotenancy=invalid, late_foreign_processes=late_foreign, post_exit_foreign_processes=post_exit,
                             host_after_post_exit_only=bool(post_exit) and not late_foreign and all(
                                 r.startswith(('foreign compute process', 'pre-existing compute process')) for r in after.get('reasons', [])))
        lease.release()
    subprocess.run(['docker', 'rm', cid], capture_output=True)
    job_status = {}
    if (run_dir / 'job_status.json').exists():
        job_status = json.loads((run_dir / 'job_status.json').read_text())
    status = 'invalid' if launch_record.get('invalid_gpu_cotenancy') else (
        'complete' if exit_code == 0 and job_status.get('status') == 'complete' else 'failed')
    launch_record['status'] = status
    atomic_json(run_dir / 'launch_record.json', launch_record)
    try:
        from campaign.records import assemble_run_record
        assemble_run_record(run_dir)
    except Exception as exc:
        launch_record['record_assembly_error'] = repr(exc)
        atomic_json(run_dir / 'launch_record.json', launch_record)
    append_registry(dict(event='finished', run_id=run_id, matrix_id=args.matrix_id, status=status,
                         exit_code=exit_code, wall_seconds=end - start,
                         invalid_gpu_cotenancy=launch_record.get('invalid_gpu_cotenancy', False)))
    return 0 if status == 'complete' else 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--matrix-id', required=True)
    ap.add_argument('--protocol-id', required=True)
    ap.add_argument('--gpus', type=int, required=True, choices=(0, 1, 2, 3))
    ap.add_argument('--gpu-model', choices=tuple(MODELS), default='a6000')
    ap.add_argument('--env', choices=('main', 'hist'), default='main')
    ap.add_argument('--prefer', default=None)
    ap.add_argument('--wait', action='store_true')
    ap.add_argument('--cpus', type=float, default=12)
    ap.add_argument('--memory', default='96g')
    ap.add_argument('--network', default='none')
    ap.add_argument('--set-env', action='append', default=[])
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not cmd:
        ap.error('missing job command after --')
    sys.exit(launch(args, cmd))


if __name__ == '__main__':
    main()
