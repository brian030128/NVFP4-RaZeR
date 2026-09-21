"""Assemble and validate RESULT_SCHEMA.json run records for every launched attempt."""
import hashlib
import json
import os
from pathlib import Path

CAMPAIGN_ROOT = Path(os.environ.get('CAMPAIGN_ROOT', Path(__file__).resolve().parents[3]))
SCHEMA_PATH = CAMPAIGN_ROOT / 'handoff' / 'agent_handoff' / 'RESULT_SCHEMA.json'


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def run_manifest(run_dir, exclude=('run_record.json', 'run_record_validation.json', 'SHA256SUMS_run.txt')):
    lines = []
    for p in sorted(Path(run_dir).rglob('*')):
        if p.is_file() and p.name not in exclude and not p.name.endswith('.tmp'):
            lines.append(f'{sha256_file(p)}  {p.relative_to(run_dir).as_posix()}\n')
    path = Path(run_dir) / 'SHA256SUMS_run.txt'
    path.write_text(''.join(lines))
    return str(path), sha256_file(path)


def _null_source(launch):
    return dict(model_id=None, model_revision=None, tokenizer_revision=None, model_class=None,
                module_manifest_sha256=None, source_manifest_sha256=launch.get('source_manifest_sha256', ''))


def assemble_run_record(run_dir):
    run_dir = Path(run_dir)
    launch = json.loads((run_dir / 'launch_record.json').read_text())
    status = json.loads((run_dir / 'job_status.json').read_text()) if (run_dir / 'job_status.json').exists() else {}
    result = json.loads((run_dir / 'job_result.json').read_text()) if (run_dir / 'job_result.json').exists() else {}
    gpu_run = launch['requested']['gpus'] > 0
    preflights = []
    for key in ('host_preflight_before', 'host_preflight_after'):
        if launch.get(key):
            e = launch[key]
            preflights.append(dict(path=e['path'], sha256=e['sha256'], timestamp_utc=e['timestamp_utc'],
                                   passed=e['passed'], phase=key))
    for e in status.get('preflight_records', []):
        preflights.append(dict(path=e['path'], sha256=e['sha256'], timestamp_utc=e['timestamp_utc'],
                               passed=e['passed'], phase=e.get('phase')))
    names, uuids = [], list(launch.get('leased_uuids') or [])
    for e in preflights:
        try:
            rec = json.loads(Path(e['path']).read_text())
            by = {g['uuid']: g['name'] for g in rec.get('gpus', [])}
            names = [by[u] for u in uuids if u in by] or names
        except Exception:
            pass
    invalid = bool(launch.get('invalid_gpu_cotenancy')) or status.get('status') == 'invalid'
    # a host after-check that failed only because a foreign process started after the container had exited (launcher
    # classify_after_processes, fail-closed) is not co-tenancy with this run; the record and its evidence stay listed
    post_exit_only = bool(launch.get('host_after_post_exit_only'))
    foreign_absent = all(p['passed'] or (post_exit_only and p['phase'] == 'host_preflight_after') for p in preflights) and not invalid if gpu_run else True
    record_status = launch.get('status') or status.get('status') or 'failed'
    if record_status not in ('running', 'complete', 'failed', 'invalid'):
        record_status = 'failed'
    if invalid:
        record_status = 'invalid'
    sidecar = launch.get('sidecar') or {}
    torch_peak = status.get('torch_peak_memory_bytes') or {}
    nvml_peak = max([v for v in (sidecar.get('peak_used_mib') or {}).values()] or [0]) * 1024 * 1024
    torch_peak_max = max([int(v) for v in torch_peak.values()] or [0]) if isinstance(torch_peak, dict) else 0
    failures = list(result.get('failures', []))
    if status.get('status') in ('failed', 'invalid'):
        failures.append(dict(stage='job', error=status.get('error'), oom=status.get('oom', False),
                             after_preflight_failed=status.get('after_preflight_failed')))
    if launch.get('status') in ('preflight_failed', 'docker_failed', 'not_started'):
        failures.append(dict(stage='launch', status=launch.get('status'), reasons=launch.get('reasons')))
    record = dict(
        run_id=launch['run_id'], matrix_id=launch['matrix_id'], status=record_status,
        protocol_id=result.get('protocol_id', launch.get('protocol_id')),
        protocol_freeze_sha256=result.get('protocol_freeze_sha256', ''),
        source=result.get('source') or _null_source(launch),
        environment=result.get('environment') or dict(hostname=launch.get('host', ''), container_or_lock_sha256=launch.get('env_lock_sha256', ''),
                                                      driver=None, cuda=None, torch='', transformers='', datasets='',
                                                      lm_eval=None, attention_backend=None, activation_quantizer=None),
        gpu_allocation=dict(gpu_run=gpu_run, gpu_count=len(uuids), gpu_names=names, gpu_uuids=uuids,
                            same_model=len(set(names)) <= 1, scheduler_allocated=bool(launch.get('lease_id')) if gpu_run else False,
                            scheduler_kind=launch.get('scheduler_kind'), lease_id=launch.get('lease_id'),
                            foreign_compute_processes_absent=foreign_absent, invalid_gpu_cotenancy=invalid,
                            preflight_records=preflights),
        data=result.get('data') or dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=None,
                                        token_hashes={}, overlap_audit=None),
        policies=result.get('policies', []),
        results=result.get('results') or dict(raw_outputs=[], summary={}, uncertainty={},
                                              attempted_endpoints=[], missing_endpoints=['job produced no result']),
        artifacts=dict(logs=[str(run_dir / 'container.log')] + list(result.get('logs', [])), failures=failures,
                       compute_usage=dict(gpu_hours=launch.get('gpu_hours', 0.0),
                                          peak_gpu_memory_bytes=max(nvml_peak, torch_peak_max),
                                          torch_peak_memory_bytes=torch_peak, nvml_peak_used_mib=sidecar.get('peak_used_mib'),
                                          wall_seconds=launch.get('wall_seconds', status.get('wall_seconds', 0.0))),
                       sha256_manifest=''),
    )
    path, digest = run_manifest(run_dir)
    record['artifacts']['sha256_manifest'] = digest
    record['artifacts']['sha256_manifest_path'] = path
    (run_dir / 'run_record.json').write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + '\n')
    errors = validate_record(record)
    (run_dir / 'run_record_validation.json').write_text(json.dumps(dict(valid=not errors, errors=errors), indent=1) + '\n')
    return record, errors


def validate_record(record):
    import jsonschema
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    return [f'{"/".join(map(str, e.path))}: {e.message}' for e in validator.iter_errors(record)]
