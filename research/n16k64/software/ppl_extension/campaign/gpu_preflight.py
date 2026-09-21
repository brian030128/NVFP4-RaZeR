"""Fail-closed GPU allocation/exclusivity preflight (GPU_RESOURCE_POLICY.md).

Every GPU run calls this before importing/loading a model, at major phase
boundaries and after completion. The checker never terminates any process.

Inputs are obtained from `nvidia-smi` CSV queries (primary) and NVML via
`pynvml` (independent cross-check). Any query failure, malformed output,
disagreement between the two sources, unknown process owner, duplicate or
unresolvable device identifier, zero or more than three visible GPUs, mixed
GPU models, or a compute process not belonging to the current job makes the
check FAIL. There is no "warn and continue" path.
"""
import datetime as _dt
import hashlib
import json
import os
import pwd
import socket
import time
import subprocess
from pathlib import Path

ALLOWED_MODELS = ('NVIDIA RTX A6000', 'NVIDIA RTX 6000 Ada Generation')
MIN_GPUS, MAX_GPUS = 1, 3
SCHEMA_VERSION = 'gpu_preflight/v1'

PREFLIGHT_SCHEMA = {
    '$schema': 'https://json-schema.org/draft/2020-12/schema',
    'type': 'object',
    'required': ['schema', 'phase', 'passed', 'reasons', 'timestamp_utc', 'hostname', 'user', 'uid',
                 'allocation', 'visible', 'gpus', 'compute_processes', 'policy'],
    'properties': {
        'schema': {'const': SCHEMA_VERSION},
        'phase': {'type': 'string', 'minLength': 1},
        'passed': {'type': 'boolean'},
        'reasons': {'type': 'array', 'items': {'type': 'string'}},
        'timestamp_utc': {'type': 'string'},
        'hostname': {'type': 'string'},
        'user': {'type': 'string'},
        'uid': {'type': 'integer'},
        'allocation': {'type': 'object', 'required': ['scheduler_kind', 'allocation_id', 'expected_uuids']},
        'visible': {'type': 'object', 'required': ['cuda_visible_devices', 'resolved_uuids', 'count']},
        'gpus': {'type': 'array'},
        'compute_processes': {'type': 'array'},
        'policy': {'type': 'object'},
    },
    'allOf': [{
        'if': {'properties': {'passed': {'const': True}}},
        'then': {'properties': {'reasons': {'maxItems': 0},
                                'visible': {'properties': {'count': {'minimum': MIN_GPUS, 'maximum': MAX_GPUS}}}}},
    }],
}


class QueryError(RuntimeError):
    """Raised when the device/process state cannot be established with certainty."""


def _utc():
    return _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def _run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as exc:  # binary missing, timeout, ...
        raise QueryError(f'query failed: {cmd[0]}: {exc!r}') from exc
    if p.returncode != 0:
        raise QueryError(f'query failed: {" ".join(cmd)} rc={p.returncode} stderr={p.stderr.strip()[:300]}')
    return p.stdout


def smi_gpus(run=_run):
    out = run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,pci.bus_id',
               '--format=csv,noheader,nounits'])
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 7:
            raise QueryError(f'malformed nvidia-smi gpu row: {line!r}')
        idx, uuid, name, mtot, mused, util, bus = parts
        if not uuid.startswith('GPU-') or not idx.isdigit():
            raise QueryError(f'malformed nvidia-smi gpu row: {line!r}')
        try:
            gpus.append(dict(index=int(idx), uuid=uuid, name=name, memory_total_mib=int(mtot),
                             memory_used_mib=int(mused), utilization_gpu_pct=(None if util in ('[N/A]', 'N/A') else int(util)),
                             pci_bus_id=bus))
        except ValueError as exc:
            raise QueryError(f'malformed nvidia-smi gpu row: {line!r}') from exc
    if not gpus:
        raise QueryError('nvidia-smi reported no GPUs')
    return gpus


def smi_compute_apps(run=_run):
    out = run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory,process_name',
               '--format=csv,noheader,nounits'])
    apps = []
    for line in out.strip().splitlines():
        if not line.strip() or 'No running processes found' in line:
            continue
        parts = [p.strip() for p in line.split(',', 3)]
        if len(parts) != 4 or not parts[0].startswith('GPU-') or not parts[1].isdigit():
            raise QueryError(f'malformed nvidia-smi compute-app row: {line!r}')
        used = None if parts[2] in ('[N/A]', 'N/A') else int(parts[2])
        apps.append(dict(gpu_uuid=parts[0], pid=int(parts[1]), used_memory_mib=used, process_name=parts[3]))
    return apps


def nvml_state():
    """Independent NVML view: {uuid: set(pids)}. Raises QueryError on any failure."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception as exc:
        raise QueryError(f'NVML unavailable: {exc!r}') from exc
    try:
        state = {}
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(h)
            uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
            pids = {p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h)}
            state[uuid] = pids
        return state
    except Exception as exc:
        raise QueryError(f'NVML query failed: {exc!r}') from exc
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def proc_owner(pid, proc_root='/proc'):
    """(uid, username, start_ticks, cgroup_text). Raises QueryError if unresolvable."""
    base = Path(proc_root) / str(pid)
    try:
        uid = os.stat(base).st_uid
        stat = (base / 'stat').read_text()
        start = int(stat.rsplit(')', 1)[1].split()[19])
    except Exception as exc:
        raise QueryError(f'cannot resolve owner of PID {pid}: {exc!r}') from exc
    try:
        name = pwd.getpwuid(uid).pw_name
    except KeyError:
        name = f'uid:{uid}'
    try:
        cgroup = (base / 'cgroup').read_text()
    except Exception:
        cgroup = ''
    return uid, name, start, cgroup


def resolve_visible(cuda_visible_devices, gpus):
    """Map CUDA_VISIBLE_DEVICES onto physical UUIDs; returns (uuids, reasons)."""
    reasons = []
    by_index = {str(g['index']): g['uuid'] for g in gpus}
    uuids = [g['uuid'] for g in gpus]
    if cuda_visible_devices is None:
        return list(uuids), reasons
    tokens = [t.strip() for t in cuda_visible_devices.split(',')]
    if cuda_visible_devices.strip() == '' or tokens == ['']:
        return [], reasons
    kinds = {'uuid' if t.startswith('GPU-') else 'index' if t.isdigit() else 'other' for t in tokens}
    if 'other' in kinds:
        reasons.append(f'unresolvable CUDA_VISIBLE_DEVICES token(s): {cuda_visible_devices!r}')
        return [], reasons
    if len(kinds) > 1:
        reasons.append(f'mixed index/UUID CUDA_VISIBLE_DEVICES is ambiguous: {cuda_visible_devices!r}')
    resolved = []
    for t in tokens:
        if t.startswith('GPU-'):
            matches = [u for u in uuids if u == t or u.startswith(t)]
            if len(matches) != 1:
                reasons.append(f'CUDA_VISIBLE_DEVICES token {t!r} does not resolve to exactly one GPU')
                continue
            resolved.append(matches[0])
        else:
            if t not in by_index:
                reasons.append(f'CUDA_VISIBLE_DEVICES index {t!r} does not exist')
                continue
            resolved.append(by_index[t])
    if len(set(resolved)) != len(resolved):
        reasons.append(f'duplicate/remapped device identifiers in CUDA_VISIBLE_DEVICES: {cuda_visible_devices!r}')
    return resolved, reasons


def check(phase, *, expected_uuids=None, allocation_id=None, scheduler_kind='campaign_local_lease',
          own_cgroup=None, own_pids=(), strict_same_user=True, gpu_run=True, env=None,
          gpu_query=smi_gpus, app_query=smi_compute_apps, nvml_query=nvml_state, owner_query=proc_owner,
          uid=None, retries=3, retry_sleep=0.25):
    """Return a preflight record dict. Never raises for policy/query failures; `passed` carries the verdict."""
    env = os.environ if env is None else env
    uid = os.getuid() if uid is None else uid
    try:
        user = pwd.getpwuid(uid).pw_name
    except KeyError:
        user = f'uid:{uid}'
    record = dict(schema=SCHEMA_VERSION, phase=phase, passed=False, reasons=[], timestamp_utc=_utc(),
                  hostname=socket.gethostname(), user=user, uid=uid,
                  allocation=dict(scheduler_kind=scheduler_kind, allocation_id=allocation_id,
                                  expected_uuids=list(expected_uuids) if expected_uuids is not None else None,
                                  container=env.get('CAMPAIGN_CONTAINER_NAME')),
                  visible=dict(cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),
                               nvidia_visible_devices=env.get('NVIDIA_VISIBLE_DEVICES'),
                               resolved_uuids=[], count=0),
                  gpus=[], compute_processes=[], nvml_crosscheck=None,
                  policy=dict(min_gpus=MIN_GPUS, max_gpus=MAX_GPUS, allowed_models=list(ALLOWED_MODELS),
                              strict_same_user=strict_same_user, gpu_run=gpu_run),
                  environment=dict(slurm_job_id=env.get('SLURM_JOB_ID'), slurm_step_id=env.get('SLURM_STEP_ID'),
                                   lease_id=env.get('CAMPAIGN_LEASE_ID'), run_id=env.get('CAMPAIGN_RUN_ID')))
    reasons = record['reasons']
    if not gpu_run:
        record['passed'] = True
        record['note'] = 'CPU-only run: no GPU is required or inspected'
        return record
    # Query with a consistency retry: nvidia-smi and NVML must agree on UUIDs and PIDs.
    last_error = None
    for attempt in range(retries):
        if attempt:
            time.sleep(min(8.0, retry_sleep * (2 ** (attempt - 1))))  # capped exponential backoff; a phase-gate trip discards all in-flight work
        try:
            gpus = gpu_query()
            apps = app_query()
            nv = nvml_query() if nvml_query is not None else None
        except QueryError as exc:
            last_error = str(exc)
            continue
        if nv is not None:
            smi_uuids = {g['uuid'] for g in gpus}
            smi_pids = {}
            for a in apps:
                smi_pids.setdefault(a['gpu_uuid'], set()).add(a['pid'])
            agree = set(nv) == smi_uuids and all(nv.get(u, set()) == smi_pids.get(u, set()) for u in smi_uuids)
            record['nvml_crosscheck'] = dict(attempt=attempt, agree=agree)
            if not agree:
                last_error = 'nvidia-smi and NVML disagree on devices or compute PIDs'
                continue
        last_error = None
        break
    if last_error is not None:
        reasons.append(f'fail-closed: {last_error}')
        return record
    record['gpus'] = gpus
    visible, vis_reasons = resolve_visible(env.get('CUDA_VISIBLE_DEVICES'), gpus)
    reasons.extend(vis_reasons)
    record['visible'].update(resolved_uuids=visible, count=len(set(visible)))
    n = len(set(visible))
    if n < MIN_GPUS:
        reasons.append(f'{n} visible GPU(s); a GPU run needs {MIN_GPUS}-{MAX_GPUS}')
    if n > MAX_GPUS:
        reasons.append(f'{n} visible GPUs exceeds the per-run maximum of {MAX_GPUS}')
    if expected_uuids is not None and sorted(set(visible)) != sorted(set(expected_uuids)):
        reasons.append(f'visible GPUs {sorted(set(visible))} differ from the allocation {sorted(set(expected_uuids))}')
    by_uuid = {g['uuid']: g for g in gpus}
    names = {by_uuid[u]['name'] for u in visible if u in by_uuid}
    if len(names) > 1:
        reasons.append(f'mixed GPU models in one run: {sorted(names)}')
    bad_models = sorted(nm for nm in names if nm not in ALLOWED_MODELS)
    if bad_models:
        reasons.append(f'GPU model(s) not permitted by policy: {bad_models}')
    buses = [by_uuid[u]['pci_bus_id'] for u in visible if u in by_uuid]
    if len(set(buses)) != len(buses):
        reasons.append('duplicate PCI bus id among visible GPUs')
    procs = []
    own_pids = set(own_pids)
    for a in apps:
        if a['gpu_uuid'] not in set(visible):
            continue
        entry = dict(a)
        try:
            puid, pname, start, cgroup = owner_query(a['pid'])
            entry.update(owner_uid=puid, owner=pname, start_ticks=start)
            in_job = a['pid'] in own_pids or (own_cgroup is not None and own_cgroup != '' and own_cgroup in cgroup)
            entry['belongs_to_current_job'] = bool(in_job)
            if puid != uid:
                reasons.append(f'foreign compute process PID {a["pid"]} owned by {pname} on {a["gpu_uuid"]}')
            elif not in_job and strict_same_user:
                reasons.append(f'pre-existing compute process PID {a["pid"]} (same user, not this job) on {a["gpu_uuid"]}')
        except QueryError as exc:
            entry.update(owner_uid=None, owner=None, belongs_to_current_job=False)
            reasons.append(f'fail-closed: {exc}')
        procs.append(entry)
    record['compute_processes'] = procs
    record['passed'] = not reasons
    return record


def write_record(record, path):
    """Write immutably (O_EXCL, mode 0444) and return (path, sha256)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, indent=1, sort_keys=True) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    return str(path), hashlib.sha256(data).hexdigest()


def validate(record):
    import jsonschema
    jsonschema.validate(record, PREFLIGHT_SCHEMA)
    return True


def own_container_cgroup():
    """cgroup text of the current process, used to recognise this job's own GPU processes."""
    try:
        text = Path('/proc/self/cgroup').read_text().strip()
    except Exception:
        return None
    # cgroup v2: single line "0::/system.slice/docker-<id>.scope"; v1: several lines.
    for line in text.splitlines():
        path = line.split(':', 2)[-1]
        if 'docker' in path:
            return path
    return None


class PhaseGate:
    """In-job helper: `gate('load_model')` writes gpu_preflight_phase_<name>.json and aborts on failure."""

    def __init__(self, run_dir, gpu_run=True):
        self.run_dir = Path(run_dir)
        self.gpu_run = gpu_run
        self.records = []
        env = os.environ
        exp = env.get('CAMPAIGN_ALLOCATED_UUIDS')
        self.expected = exp.split(',') if exp else None
        self.allocation_id = env.get('CAMPAIGN_LEASE_ID')
        self.cgroup = own_container_cgroup()

    def __call__(self, phase, filename=None):
        rec = check(phase, expected_uuids=self.expected, allocation_id=self.allocation_id,
                    own_cgroup=self.cgroup, own_pids={os.getpid()}, gpu_run=self.gpu_run,
                    retries=40, retry_sleep=0.5)  # ~5 min envelope; the 20 s host sidecar remains active, and uncertainty still fails closed
        rec['torch_view'] = torch_view() if self.gpu_run else None
        if self.gpu_run and rec['passed'] and rec['torch_view'] is not None:
            tv = rec['torch_view']
            if tv.get('error'):
                rec['reasons'].append(f'fail-closed: torch device query failed: {tv["error"]}')
            else:
                if tv['device_count'] != rec['visible']['count']:
                    rec['reasons'].append(f'torch sees {tv["device_count"]} devices, NVML sees {rec["visible"]["count"]}')
                if tv.get('uuids') and sorted(tv['uuids']) != sorted(rec['visible']['resolved_uuids']):
                    rec['reasons'].append(f'torch UUIDs {tv["uuids"]} differ from resolved {rec["visible"]["resolved_uuids"]}')
            rec['passed'] = not rec['reasons']
        name = filename or f'gpu_preflight_phase_{phase}.json'
        path, digest = write_record(rec, self.run_dir / 'preflight' / name)
        entry = dict(path=path, sha256=digest, timestamp_utc=rec['timestamp_utc'], passed=rec['passed'], phase=phase)
        self.records.append(entry)
        if not rec['passed']:
            raise SystemExit(f'GPU preflight failed at phase {phase}: {rec["reasons"]}')
        return entry


def torch_view():
    """What CUDA inside this process actually sees (catches index remapping)."""
    try:
        import torch
        n = torch.cuda.device_count()
        uuids, names = [], []
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            names.append(p.name)
            u = getattr(p, 'uuid', None)
            if u is not None:
                s = str(u)
                uuids.append(s if s.startswith('GPU-') else 'GPU-' + s)
        return dict(device_count=n, names=names, uuids=uuids or None)
    except Exception as exc:
        return dict(error=repr(exc))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--expected-uuids', default=None)
    ap.add_argument('--allocation-id', default=None)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    exp = args.expected_uuids.split(',') if args.expected_uuids else None
    rec = check(args.phase, expected_uuids=exp, allocation_id=args.allocation_id, gpu_run=not args.cpu)
    path, digest = write_record(rec, args.out)
    print(json.dumps(dict(path=path, sha256=digest, passed=rec['passed'], reasons=rec['reasons'])))
    raise SystemExit(0 if rec['passed'] else 2)


if __name__ == '__main__':
    main()
