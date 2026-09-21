"""Assemble and validate extension RESULT_SCHEMA.json records for every attempt."""
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
    if record_status == 'invalid':
        schema_status = 'invalid_cotenancy' if invalid else 'invalid_policy'
    elif record_status == 'failed':
        schema_status = 'oom' if any(f.get('oom') for f in failures) else 'failed'
    elif record_status == 'complete':
        schema_status = 'complete'
    else:
        schema_status = 'blocked'
    try:
        attempt = int(launch['run_id'].rsplit('_attempt', 1)[1])
    except Exception:
        attempt = 1
    source = result.get('source') or _null_source(launch)
    environment = result.get('environment') or {}
    data = result.get('data') or {}
    policies = result.get('policies') or []
    results = result.get('results') or dict(raw_outputs=[], summary={}, uncertainty={},
                                           attempted_endpoints=[], missing_endpoints=['job produced no result'])
    policy = policies[0] if len(policies) == 1 else {}
    protocol_sha = result.get('protocol_freeze_sha256') or launch.get('protocol_id') or 'prelock'
    pre_ok = all(p.get('passed') for p in preflights if p['phase'] in ('host_preflight_before', 'before'))
    post_ok = all(p.get('passed') or post_exit_only for p in preflights if p['phase'] in ('host_preflight_after', 'after'))
    during_ok = not invalid and (not gpu_run or int(sidecar.get('samples', 0)) >= 1)
    matrix_upper = str(launch.get('matrix_id', '')).upper()
    if 'ACCURACY' in matrix_upper:
        evaluation_kind = 'accuracy'
    elif 'GSM' in matrix_upper or 'GENERATION' in matrix_upper:
        evaluation_kind = 'generation'
    elif 'PORTABILITY' in matrix_upper:
        evaluation_kind = 'portability'
    elif 'PPL' in matrix_upper or 'ANCHOR' in matrix_upper:
        evaluation_kind = 'ppl'
    else:
        evaluation_kind = 'tile_effect'
    result_path = run_dir / 'job_result.json'
    result_sha = sha256_file(result_path) if result_path.exists() else sha256_file(run_dir / 'job_status.json')
    record = dict(
        schema_version='1.0', run_id=launch['run_id'], matrix_id=launch['matrix_id'], attempt=attempt,
        status=schema_status, invalid_reason=('; '.join(sidecar.get('invalid_reasons') or []) or status.get('error')),
        timestamps=dict(started_utc=launch.get('started_utc', status.get('started_utc', '')),
                        ended_utc=launch.get('finished_utc', status.get('finished_utc', ''))),
        provenance=dict(git_commit=launch.get('git_commit') or 'source-manifest-only',
                        dirty_diff_sha256=launch.get('source_manifest_sha256'), protocol_sha256=protocol_sha,
                        environment_identity=environment.get('container_or_lock_sha256') or launch.get('env_lock_sha256', ''),
                        parent_artifact_sha256='432d922a49751603740622a084a665983fb337295048112e68311cd8110e19ee'),
        gpu_policy=dict(device_models=names, device_uuids=uuids, campaign_concurrent_gpu_count=len(uuids),
                        preflight_passed=bool(pre_ok), during_checks_passed=bool(during_ok), postflight_passed=bool(post_ok),
                        foreign_compute_processes_absent=bool(foreign_absent),
                        event_log=str(run_dir / 'gpu_monitor.jsonl') if gpu_run else ''),
        model=dict(name=source.get('model_id') or 'campaign_inputs', revision=source.get('model_revision') or 'not_applicable',
                   weights_sha256=source.get('module_manifest_sha256') or 'not_applicable',
                   tokenizer_sha256=source.get('tokenizer_manifest_sha256') or source.get('tokenizer_revision') or 'not_applicable'),
        method=dict(name=policy.get('name') or ('multi_policy_plan' if policies else 'read_only_or_cpu_analysis'),
                    weight_format=str(policy.get('weight_format') or 'not_applicable'),
                    activation_policy=str(policy.get('activation_format') or 'not_applicable'),
                    quantized_scope=str(source.get('module_manifest_sha256') or 'not_applicable'),
                    tile_shape=policy.get('type_block'), k=policy.get('k'), selector=policy.get('selector'),
                    selected_tiles=policy.get('selected_tiles'),
                    selected_weights=(policy.get('selected_tiles') * policy.get('type_block', [0, 0])[0] * policy.get('type_block', [0, 0])[1]
                                      if policy.get('selected_tiles') is not None and policy.get('type_block') else None),
                    eligible_weights=None, effective_bits=None, high_precision_exceptions=[]),
        calibration=dict(draw_id=result.get('draw_id') or 'not_applicable',
                         manifest_sha256=data.get('calibration_manifest_sha256') or 'not_applicable',
                         math_sequences=int(result.get('math_sequences', 0)), code_sequences=int(result.get('code_sequences', 0)),
                         sequence_length=int(result.get('sequence_length', 1)), aggregation=result.get('aggregation')),
        evaluation=dict(kind=evaluation_kind, dataset=','.join(results.get('attempted_endpoints') or ['metadata_audit']),
                        manifest_sha256=data.get('evaluation_manifest_sha256') or 'not_applicable',
                        raw_ppl=None, mean_nll=None, dlogppl=None, ci95=None, p_raw=None, p_adjusted=None,
                        comparator_run_id=None),
        artifacts=dict(map_path=policy.get('map_path'), map_sha256=policy.get('map_sha256'),
                       per_example_or_window_path=None, result_sha256=result_sha,
                       logs=[str(run_dir / 'container.log')] + list(result.get('logs', [])), failures=failures,
                       compute_usage=dict(gpu_hours=launch.get('gpu_hours', 0.0),
                                          peak_gpu_memory_bytes=max(nvml_peak, torch_peak_max),
                                          torch_peak_memory_bytes=torch_peak, nvml_peak_used_mib=sidecar.get('peak_used_mib'),
                                          wall_seconds=launch.get('wall_seconds', status.get('wall_seconds', 0.0))),
                       sha256_manifest=''),
        protocol_id=result.get('protocol_id', launch.get('protocol_id')),
        source=source, environment=environment, data=data, policies=policies, results=results,
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
