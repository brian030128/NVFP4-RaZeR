"""V83 artifact validation: schema, hashes, preflight records, maps, registry; campaign artifact manifest (CPU)."""
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from campaign import mapio as MIO
from campaign import records as REC
from campaign import runtime

CR = Path(os.environ['CAMPAIGN_ROOT'])
FSHA = (CR / 'freeze' / 'PROTOCOL_FREEZE.sha256').read_text().split()[0]


def sha(p):
    return runtime.sha256_file(p)


def main():
    out = runtime.out_dir('artifact_validation')
    me = runtime.run_dir.name
    registry = defaultdict(list)
    for line in (CR / 'registry' / 'attempts.jsonl').read_text().splitlines():
        e = json.loads(line)
        if e.get('run_id'):
            registry[e['run_id']].append(e['event'])
    runs, failures = [], []
    by_status = defaultdict(int)
    for d in sorted((CR / 'runs').iterdir()):
        if not d.is_dir() or d.name == me or not (d / 'launch_record.json').exists():
            continue
        lr = json.loads((d / 'launch_record.json').read_text())
        status = lr.get('status')
        by_status[status] += 1
        entry = dict(run=d.name, matrix_id=lr.get('matrix_id'), status=status, problems=[])
        if (d / 'run_record.json').exists():
            rec = json.loads((d / 'run_record.json').read_text())
            errs = REC.validate_record(rec)
            entry['schema_errors'] = errs
            if status == 'complete' and errs:
                entry['problems'].append('complete run with schema errors')
            if status == 'invalid' and not errs:
                entry['problems'].append('invalid run unexpectedly schema-valid')
            sums = d / 'SHA256SUMS_run.txt'
            if sums.exists():
                bad = []
                for line in sums.read_text().splitlines():
                    h, rel = line.split('  ', 1)
                    p = d / rel
                    if not p.exists():
                        bad.append(f'missing {rel}')
                    elif sha(p) != h:
                        bad.append(f'changed {rel}')
                entry['hash_problems'] = bad
                if bad:
                    entry['problems'].append(f'{len(bad)} file hash problems')
                if hashlib.sha256(sums.read_bytes()).hexdigest() != rec['artifacts']['sha256_manifest']:
                    entry['problems'].append('SHA256SUMS_run.txt digest differs from run record')
            for pf in rec['gpu_allocation']['preflight_records']:
                if not Path(pf['path']).exists() or sha(pf['path']) != pf['sha256']:
                    entry['problems'].append(f'preflight record hash mismatch {pf["path"]}')
            for pol in rec.get('policies', []):
                if pol.get('map_path') and pol.get('map_sha256'):
                    try:
                        MIO.read_map(pol['map_path'], expected_sha256=pol['map_sha256'])
                    except Exception as exc:
                        entry['problems'].append(f'map {pol["name"]}: {exc!r}')
            for p in rec['results'].get('raw_outputs', []):
                if isinstance(p, str) and p.startswith('/') and not Path(p).exists():
                    entry['problems'].append(f'raw output missing {p}')
        elif status not in ('not_started', 'preflight_failed', 'docker_failed'):
            entry['problems'].append('no run_record.json')
        ev = registry.get(d.name, [])
        if 'created' not in ev:
            entry['problems'].append('registry lacks created event')
        if status in ('complete', 'failed', 'invalid') and 'finished' not in ev:
            entry['problems'].append('registry lacks finished event')
        if status != 'complete':
            failures.append(dict(run=d.name, matrix_id=lr.get('matrix_id'), status=status, exit_code=lr.get('exit_code'),
                                 reason=(lr.get('reason') or lr.get('reasons') or (lr.get('sidecar') or {}).get('invalid_reasons') or
                                         (json.loads((d / 'job_status.json').read_text()).get('error') if (d / 'job_status.json').exists() else None))))
        runs.append(entry)
    manifest = []
    for top in ('freeze', 'provenance', 'reports', 'registry', 'plans', 'env'):
        for p in sorted((CR / top).rglob('*')):
            if p.is_file() and not p.name.endswith('.tmp'):
                manifest.append(f'{sha(p)}  {p.relative_to(CR).as_posix()}')
    for d in sorted((CR / 'runs').iterdir()):
        if d.is_dir() and (d / 'SHA256SUMS_run.txt').exists():
            manifest.append(f'{sha(d / "SHA256SUMS_run.txt")}  {(d / "SHA256SUMS_run.txt").relative_to(CR).as_posix()}')
    (out / 'ARTIFACT_MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    # a run is superseded when a later attempt of the same job completed; superseded attempts are kept as immutable
    # evidence and are never rewritten, so record-format defects fixed by a re-run remain visible on the old attempt only
    latest_complete = {}
    for r in runs:
        job, _, att = r['run'].rpartition('_attempt')
        if r['status'] == 'complete' and att.isdigit():
            latest_complete[job] = max(latest_complete.get(job, 0), int(att))
    for r in runs:
        job, _, att = r['run'].rpartition('_attempt')
        r['superseded_by_later_complete_attempt'] = bool(att.isdigit() and latest_complete.get(job, 0) > int(att))
    problems = [r for r in runs if r['problems']]
    current_problems = [r for r in problems if not r['superseded_by_later_complete_attempt']]
    superseded_problems = [r for r in problems if r['superseded_by_later_complete_attempt']]
    in_flight = [r for r in current_problems if r['status'] in ('starting', 'running')]
    blocking = [r for r in current_problems if r not in in_flight]
    runtime.atomic_json(out / 'FAILED_OR_SKIPPED_RUNS.json', dict(freeze_sha256=FSHA, attempts=failures))
    runtime.atomic_json(out / 'ARTIFACT_VALIDATION.json', dict(freeze_sha256=FSHA, runs=len(runs), by_status=dict(by_status), runs_with_problems=problems,
                                                              current_runs_with_problems=current_problems, superseded_runs_with_problems=superseded_problems,
                                                              in_flight_runs_with_problems=in_flight, blocking_runs_with_problems=blocking,
                                                              artifact_manifest_sha256=sha(out / 'ARTIFACT_MANIFEST.sha256'), entries=len(manifest)))
    lines = ['# Artifact validation (V83)', '', f'Freeze `{FSHA}`.', '', f'Runs inspected: {len(runs)}; by launch status: `{json.dumps(dict(by_status))}`.', '',
             f'Runs with problems: {len(problems)} total = {len(blocking)} current and settled, {len(in_flight)} still running (no record yet), '
             f'{len(superseded_problems)} superseded by a later complete attempt (kept unmodified as evidence).', '']
    for r in problems[:200]:
        lines.append(f'- `{r["run"]}` ({r["status"]}): ' + '; '.join(r['problems'][:5]))
    lines += ['', f'Artifact manifest: `artifact_validation/ARTIFACT_MANIFEST.sha256` ({len(manifest)} entries, sha256 `{sha(out / "ARTIFACT_MANIFEST.sha256")}`).',
              'Failed / invalid / not-started attempts: `artifact_validation/FAILED_OR_SKIPPED_RUNS.json` (never deleted from the registry).']
    (out / 'ARTIFACT_VALIDATION.md').write_text('\n'.join(lines) + '\n')
    src = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(protocol_id='artifact', protocol_freeze_sha256=FSHA,
        source=dict(model_id=None, model_revision=None, tokenizer_revision=None, model_class=None, module_manifest_sha256=None, source_manifest_sha256=src),
        environment=runtime.environment(), data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=None, token_hashes={}, overlap_audit=None),
        policies=[], results=dict(raw_outputs=[str(p) for p in sorted(out.iterdir())], summary=dict(by_status=dict(by_status), problems=len(problems)),
                                  uncertainty={}, attempted_endpoints=['all runs'], missing_endpoints=[]), logs=[], failures=[]))
    print(json.dumps(dict(runs=len(runs), by_status=dict(by_status), problems=len(problems))))


if __name__ == '__main__':
    main()
