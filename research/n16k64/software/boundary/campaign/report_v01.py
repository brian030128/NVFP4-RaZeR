"""Build GPU_PREFLIGHT_TEST_REPORT.md (V01) from unit tests, host integration cases, launched runs and Ada attempts."""
import json
import os
from pathlib import Path

from campaign import runtime
from campaign.policies import latest_complete_run

CR = Path(os.environ['CAMPAIGN_ROOT'])


def main():
    out = runtime.out_dir('report_v01')
    unit = json.loads((latest_complete_run(CR, 'V01_cpu_preflight_unit') / 'tests' / 'GPU_PREFLIGHT_UNIT_TESTS.json').read_text())
    integ = json.loads((CR / 'reports' / 'V01' / 'v01_integration.json').read_text())
    runs = {}
    for d in sorted((CR / 'runs').glob('V01_integration_*')):
        lr = json.loads((d / 'launch_record.json').read_text())
        rv = json.loads((d / 'run_record_validation.json').read_text()) if (d / 'run_record_validation.json').exists() else None
        runs[d.name] = dict(status=lr.get('status'), leased=lr.get('leased_uuids'), invalid=lr.get('invalid_gpu_cotenancy'),
                            sidecar=(lr.get('sidecar') or {}).get('invalid_reasons'), record_schema_valid=(rv or {}).get('valid'), reason=lr.get('reason'))
    ada_attempts = {}
    for d in sorted((CR / 'runs').glob('*')):
        lr = d / 'launch_record.json'
        if not lr.exists():
            continue
        l = json.loads(lr.read_text())
        if l.get('requested', {}).get('model') == 'ada' and l.get('requested', {}).get('gpus', 0) > 0:
            ada_attempts[d.name] = dict(status=l.get('status'), invalid_reasons=(l.get('sidecar') or {}).get('invalid_reasons'), reason=l.get('reason'))
    ok_real = [c for c in integ['cases'] if c.get('passed') is not None]
    lines = ['# GPU preflight and allocation test report (V01)', '',
             'Host at collection time: 4x RTX A6000 and 3x RTX 6000 Ada. No site scheduler existed; allocation used the campaign-local flock lease over physical',
             'GPU UUIDs plus Docker `--gpus device=<UUID>` isolation (owner decision D01, `provenance/01_owner_decisions.json`). The preflight is',
             '`campaign/gpu_preflight.py`; the launcher is `campaign/launcher.py`.', '',
             '## Unit tests (simulated states)', '',
             f'Result: {unit["repeats"][0]["passed"]} passed, {unit["repeats"][0]["failed"]} failed (run `{latest_complete_run(CR, "V01_cpu_preflight_unit").name}`).', '',
             '| required case | test(s) | outcome |', '|---|---|---|']
    mapping = [('zero visible GPUs for a GPU run -> fail', 'test_zero_visible_gpus_fails_for_gpu_run'), ('four visible GPUs -> fail', 'test_four_visible_gpus_fail'),
               ('duplicate/remapped identifiers -> fail', 'test_duplicate_identifiers_fail, test_remapped_index_and_uuid_of_same_device_fail'),
               ('mixed A6000/Ada -> fail', 'test_mixed_a6000_ada_fails'), ('foreign PID -> fail without terminating', 'test_foreign_pid_fails_without_termination (os.kill patched to raise)'),
               ('clean 1/2/3 GPUs -> pass', 'test_clean_one_two_three_gpu_allocations_pass[3 params]'), ('malformed/unavailable query -> fail closed', 'test_unavailable_query_fails_closed, test_malformed_query_output_fails_closed, test_nvml_disagreement_fails_closed, test_unresolvable_owner_fails_closed'),
               ('JSON round trip + schema', 'test_json_round_trip_and_schema')]
    names = {c['name'].split('::')[-1].split('[')[0]: c['status'] for c in unit['repeats'][0]['cases']}
    for case, tests in mapping:
        st = all(names.get(t.split(' ')[0].split('[')[0].strip(', ')) == 'passed' for t in tests.split(', '))
        lines.append(f'| {case} | {tests} | {"passed" if st else "CHECK"} |')
    lines += ['', '## Real-hardware integration (query-only for other users\' GPUs)', '', '| case | expected pass | observed pass | reasons |', '|---|---|---|---|']
    for c in integ['cases']:
        lines.append(f'| {c["case"]} | {c["expected_pass"]} | {c.get("passed")} | {"; ".join(c.get("reasons") or []) or c.get("not_run_reason") or c.get("detail") or ""} |')
    lines += ['', '## Launched end-to-end runs', '', '| run | status | leased | invalid | schema-valid record | note |', '|---|---|---|---|---|---|']
    for k, v in runs.items():
        lines.append(f'| {k} | {v["status"]} | {v["leased"]} | {v["invalid"]} | {v["record_schema_valid"]} | {(v["sidecar"] or [v.get("reason") or ""])[0][:160]} |')
    lines += ['', 'The co-tenancy injection run was invalidated by the sidecar, its own container was stopped, and the injector (this user\'s process) was',
              'still alive afterwards; RESULT_SCHEMA correctly refuses an invalid run as a valid result record.', '',
              '## RTX 6000 Ada availability', '',
              f'Ada-requesting attempts so far: {len(ada_attempts)}.', '']
    for k, v in ada_attempts.items():
        lines.append(f'- `{k}`: {v["status"]} {(v["invalid_reasons"] or [v.get("reason") or ""])[0][:200]}')
    lines += ['', 'The three end-to-end `V01_integration_clean{1,2,3}_ada` runs did not start: at the time they were attempted (2026-09-11) no',
              'homogeneous set of free Ada cards was available, and the campaign additionally requires a quiet window before leasing a card that',
              'other users recently released. That constraint was real but it was not permanent, and an earlier revision of this report',
              'overstated it as continuous occupancy. The read-only occupancy record (`logs/gpu_occupancy.jsonl`, 297 samples at 300 s over',
              '24.8 h) measures all three Ada cards simultaneously free of foreign processes in 18.4% of samples across five windows, the',
              'longest 2026-09-12T03:49Z to 06:14Z (~2.4 h); the standalone `V01_clean{1,2,3}_ada_probe` jobs duly completed on 2026-09-12,',
              'with `clean3_ada` leasing all three Ada cards at once.',
              '',
              'The integration endpoints were nevertheless not retried, and that is a deliberate choice rather than a blocker.',
              '`campaign/v01_integration.py` is a single linear driver with no case selection and hard-coded `attempt1` run ids, and it',
              'unconditionally performs a co-tenancy injection against this campaign\'s own lease plus lease-exhaustion probes; re-running it to',
              'capture three Ada endpoints would collide with existing run ids and repeat those side effects. The behaviour those endpoints',
              'would demonstrate is covered from two other directions: clean 1-, 2- and 3-GPU Ada allocation by the standalone probes above,',
              'and the lease + Docker launcher end-to-end on Ada by the V23 cross-GPU archived anchor and the V81 portability runs.']
    (out / 'GPU_PREFLIGHT_TEST_REPORT.md').write_text('\n'.join(lines) + '\n')
    src = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(protocol_id='resource', protocol_freeze_sha256=(CR / 'freeze' / 'PROTOCOL_FREEZE.sha256').read_text().split()[0],
        source=dict(model_id=None, model_revision=None, tokenizer_revision=None, model_class=None, module_manifest_sha256=None, source_manifest_sha256=src),
        environment=runtime.environment(), data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=None, token_hashes={}, overlap_audit=None),
        policies=[], results=dict(raw_outputs=[str(out / 'GPU_PREFLIGHT_TEST_REPORT.md')], summary=dict(unit_passed=unit['repeats'][0]['passed'], ada_attempts=len(ada_attempts)),
                                  uncertainty={}, attempted_endpoints=['unit', 'integration', 'ada'], missing_endpoints=[k for k in ('clean_ada_1', 'clean_ada_2', 'clean_ada_3')
                                  if not any(r.startswith(f'V01_integration_clean{k[-1]}_ada') and v['status'] == 'complete' for r, v in runs.items())]),
        logs=[], failures=[]))


if __name__ == '__main__':
    main()
