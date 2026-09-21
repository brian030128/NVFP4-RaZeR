"""Read-only startup audit for the N16K64 PPL-improvement extension.

The audit deliberately never repairs an input.  It records exact hashes, parses every parent
map, verifies the parent artifact manifest, and distinguishes a rehydratable missing model cache
from a hash failure.  Run it again after cache rehydration; both attempts remain append-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import time
import zipfile
from pathlib import Path

from campaign import mapio
from campaign import runtime


EXPECTED_HANDOFF_SHA256 = '810483fcc621b4801081a8afbb0d12ebb9a0d317110bc91984b76b472d188e4a'
EXPECTED_PARENT_FREEZE_SHA256 = 'df78f1fbbd034cb8d2bc8e6f6a264e7e221e1b5f2b2ff99a3fe821369c5ac88f'
EXPECTED_PARENT_ARTIFACT_MANIFEST_SHA256 = '432d922a49751603740622a084a665983fb337295048112e68311cd8110e19ee'
EXPECTED_PARENT_RUNS = 338
EXPECTED_PARENT_ENTRIES = 78705
EXPECTED_ORIGINAL_HANDOFF_SHA256 = 'b4ba1a4b1af25dfa1dfc5e07429760af7ae0974918c5b43a8a12670bd48c2ce4'
EXPECTED_REFERENCE_HASHES = {
    'reviewer_core_zip': 'ecd04befcf9465f2f0bc218cfecb0cdf5c4a4e25901f6980211e6cef9cd65ea6',
    'EVIDENCE_BUNDLE_INDEX.json': '7da388d64b0ce3300ca31d1ffb693ad4c33ff59821564a10148b68b432c105f3',
    'BUNDLE_VERIFICATION_REPORT.json': '043a7dd43dd5f7d2942f697a9b75523633366b36fa48c12999a6af03b845f32a',
    'FINAL_SUBMISSION_RISK_AUDIT.md': '9fe489724b4a328bd867955829bb04d2086a1b2bab4c37b47bae7dada217602c',
    'N16_DECISION.md': 'f5f2a66b39032634d851370f25ba26197f92713c843c562ee59cdff55281867e',
    'STATISTICAL_VALIDITY_REPORT.md': 'b17d5b00bdcae06ecd67eb195d7b9a67316c6dee3b70a924cd06bc088cd71338',
    'REPRODUCTION_REPORT.md': 'b0f092f4481c152fe08403297baee8ae4dab5a85159073fe7d19ba6168fc1e9e',
}


def sha256_file(path: Path, progress_name: str | None = None) -> str:
    h = hashlib.sha256()
    read = 0
    next_report = 8 << 30
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b''):
            h.update(chunk)
            read += len(chunk)
            if progress_name and read >= next_report:
                print(f'HASH {progress_name} {read / 2**30:.1f} GiB', flush=True)
                next_report += 8 << 30
    return h.hexdigest()


def canonical_entry_hash(entry: dict) -> str:
    body = dict(entry)
    body.pop('entry_sha256', None)
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_amendments(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    errors = []
    prev = None
    for position, row in enumerate(rows, 1):
        if row.get('seq') != position:
            errors.append(f'row {position}: seq={row.get("seq")}')
        if row.get('prev_sha256') != prev:
            errors.append(f'row {position}: prev_sha256 mismatch')
        got = canonical_entry_hash(row)
        if got != row.get('entry_sha256'):
            errors.append(f'row {position}: entry_sha256 mismatch')
        prev = row.get('entry_sha256')
    return {'entries': len(rows), 'tail_sha256': prev, 'errors': errors, 'passed': not errors}


def handoff_audit(repo: Path, campaign_root: Path) -> dict:
    archive = repo / 'research_artifacts' / 'mixfp4_n16k64_ppl_improvement_agent_handoff.zip'
    extracted = campaign_root / 'handoff' / 'agent_handoff'
    outer = sha256_file(archive)
    checksum_sidecar = (archive.with_suffix(archive.suffix + '.sha256')).read_text().split()[0]
    expected = {}
    for line in (extracted / 'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split(None, 1)
        expected[name.strip().lstrip('*')] = digest
    internal_errors = []
    for name, digest in expected.items():
        path = extracted / name
        if not path.is_file() or sha256_file(path) != digest:
            internal_errors.append(name)
    with zipfile.ZipFile(archive) as zf:
        zip_names = sorted(i.filename for i in zf.infolist() if not i.is_dir())
        bad_zip = zf.testzip()
    extracted_names = sorted(p.relative_to(extracted).as_posix() for p in extracted.rglob('*') if p.is_file())
    readonly_errors = []
    for path in [extracted, *extracted.rglob('*')]:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o222:
            readonly_errors.append(path.relative_to(extracted.parent).as_posix())
    return {
        'archive': str(archive.resolve()), 'archive_size': archive.stat().st_size,
        'expected_sha256': EXPECTED_HANDOFF_SHA256, 'actual_sha256': outer,
        'sidecar_sha256': checksum_sidecar,
        'zip_crc_error': bad_zip, 'zip_files': len(zip_names),
        'internal_checksum_entries': len(expected), 'internal_errors': internal_errors,
        'zip_vs_extracted_equal': zip_names == extracted_names,
        'readonly_errors': readonly_errors,
        'passed': outer == EXPECTED_HANDOFF_SHA256 == checksum_sidecar and bad_zip is None
                  and not internal_errors and zip_names == extracted_names and not readonly_errors,
    }


def current_snapshot_path(hub: Path, record: dict) -> Path:
    return hub / ('models--' + record['repo'].replace('/', '--')) / 'snapshots' / record['revision']


def input_manifest_audit(parent: Path, hub: Path) -> dict:
    manifest_path = parent / 'provenance' / 'INPUT_FETCH_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    model_results = {}
    total_files = total_bytes = checked_files = checked_bytes = 0
    for key, record in manifest['models'].items():
        base = current_snapshot_path(hub, record)
        files = []
        for item in record['files']:
            total_files += 1
            total_bytes += int(item['size'])
            path = base / item['path']
            row = {'path': item['path'], 'expected_size': item['size'], 'expected_sha256': item['sha256']}
            if not path.is_file():
                row['status'] = 'missing_rehydrate_required'
            else:
                actual_size = path.stat().st_size
                actual_sha = sha256_file(path, f'{key}/{item["path"]}')
                checked_files += 1
                checked_bytes += actual_size
                row.update(actual_size=actual_size, actual_sha256=actual_sha,
                           status='verified' if actual_size == item['size'] and actual_sha == item['sha256'] else 'mismatch')
            files.append(row)
        statuses = {row['status'] for row in files}
        model_results[key] = {
            'repo': record['repo'], 'revision': record['revision'], 'resolved_revision': record.get('resolved_revision'),
            'snapshot': str(base), 'files': files, 'status': ('mismatch' if 'mismatch' in statuses else
                ('missing_rehydrate_required' if 'missing_rehydrate_required' in statuses else 'verified')),
        }
        print(f'INPUT MODEL {key} {model_results[key]["status"]}', flush=True)
    dataset_results = {}
    for key, record in manifest['datasets'].items():
        rows = []
        if record.get('files') is not None:
            base = hub / ('datasets--' + record['repo'].replace('/', '--')) / 'snapshots' / record['revision']
            items = record['files']
        else:
            base = Path(record['local_path']).parent
            items = [{'path': Path(record['local_path']).name, 'size': record['size'], 'sha256': record['sha256']}]
        for item in items:
            path = base / item['path']
            if not path.is_file():
                rows.append({'path': item['path'], 'status': 'missing', 'expected_sha256': item['sha256']})
                continue
            actual = sha256_file(path)
            rows.append({'path': item['path'], 'size': path.stat().st_size, 'expected_sha256': item['sha256'],
                         'actual_sha256': actual,
                         'status': 'verified' if actual == item['sha256'] and path.stat().st_size == item['size'] else 'mismatch'})
        dataset_results[key] = {'repo': record['repo'], 'revision': record['revision'], 'files': rows,
                                'status': 'verified' if rows and all(x['status'] == 'verified' for x in rows) else 'failed'}
        print(f'INPUT DATASET {key} {dataset_results[key]["status"]}', flush=True)
    mismatches = [k for k, v in model_results.items() if v['status'] == 'mismatch']
    missing = [k for k, v in model_results.items() if v['status'] == 'missing_rehydrate_required']
    dataset_failures = [k for k, v in dataset_results.items() if v['status'] != 'verified']
    return {
        'manifest_path': str(manifest_path), 'manifest_sha256': sha256_file(manifest_path),
        'models': model_results, 'datasets': dataset_results,
        'summary': {'model_files_expected': total_files, 'model_bytes_expected': total_bytes,
                    'model_files_checked': checked_files, 'model_bytes_checked': checked_bytes,
                    'missing_model_snapshots': missing, 'model_hash_mismatches': mismatches,
                    'dataset_failures': dataset_failures,
                    'all_required_inputs_readable': not missing and not mismatches and not dataset_failures},
    }


def bundle_source_hashes(repo: Path) -> dict:
    manifest = json.loads((repo / 'research_artifacts' / 'EVIDENCE_BUNDLE_MANIFEST.json').read_text())
    out = {}
    for entry in manifest['entries']:
        rel = entry['bundle_relative_path']
        if '/reports/authored/' in rel or rel.endswith('/registry/attempts.jsonl'):
            out[rel.split('/campaign/', 1)[-1]] = entry['source_sha256']
    return out


def verify_artifact_manifest(parent: Path, repo: Path) -> dict:
    path = parent / 'runs' / 'V83_validate_artifacts_attempt7' / 'artifact_validation' / 'ARTIFACT_MANIFEST.sha256'
    digest = sha256_file(path)
    expected_rows = []
    mismatches = []
    missing = []
    registry_expected = None
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        expected, rel = line.split('  ', 1)
        expected_rows.append((expected, rel))
        target = parent / rel
        if not target.is_file():
            missing.append(rel)
        else:
            actual = sha256_file(target)
            if actual != expected:
                mismatches.append({'path': rel, 'expected_sha256': expected, 'actual_sha256': actual})
        if rel == 'registry/attempts.jsonl':
            registry_expected = expected
        if line_number % 5000 == 0:
            print(f'PARENT MANIFEST {line_number}/{EXPECTED_PARENT_ENTRIES}', flush=True)
    source_hashes = bundle_source_hashes(repo)
    reconciled = []
    unreconciled = []
    for row in mismatches:
        source_expected = source_hashes.get(row['path'])
        item = dict(row, reviewer_bundle_source_sha256=source_expected,
                    reconciled=source_expected == row['actual_sha256'])
        (reconciled if item['reconciled'] else unreconciled).append(item)
    registry = parent / 'registry' / 'attempts.jsonl'
    lines = registry.read_bytes().splitlines(keepends=True)
    prefix_match_lines = None
    h = hashlib.sha256()
    for index, line in enumerate(lines, 1):
        h.update(line)
        if h.hexdigest() == registry_expected:
            prefix_match_lines = index
    suffix = [json.loads(line) for line in lines[prefix_match_lines or len(lines):]]
    return {
        'path': str(path), 'manifest_sha256': digest,
        'expected_manifest_sha256': EXPECTED_PARENT_ARTIFACT_MANIFEST_SHA256,
        'entries': len(expected_rows), 'missing': missing, 'mismatches': mismatches,
        'reconciled_post_manifest_changes': reconciled, 'unreconciled_changes': unreconciled,
        'registry_prefix_match_lines': prefix_match_lines, 'registry_total_lines': len(lines),
        'registry_appended_events': suffix,
        'passed': digest == EXPECTED_PARENT_ARTIFACT_MANIFEST_SHA256 and len(expected_rows) == EXPECTED_PARENT_ENTRIES
                  and not missing and not unreconciled and len(mismatches) == 6,
    }


def parent_state_audit(parent: Path, repo: Path) -> dict:
    freeze = parent / 'freeze' / 'PROTOCOL_FREEZE.json'
    freeze_sha = sha256_file(freeze)
    freeze_sidecar = (parent / 'freeze' / 'PROTOCOL_FREEZE.sha256').read_text().split()[0]
    amendments = verify_amendments(parent / 'provenance' / 'PROTOCOL_AMENDMENTS.jsonl')
    final = parent / 'runs' / 'V90_final_reports_attempt7' / 'final'
    final_index = json.loads((final / 'FINAL_DELIVERABLES_INDEX.json').read_text())
    validation = json.loads((parent / 'runs' / 'V83_validate_artifacts_attempt7' / 'artifact_validation' /
                             'ARTIFACT_VALIDATION.json').read_text())
    coverage = json.loads((final / 'MATRIX_COVERAGE.json').read_text())
    report_hashes = {name: sha256_file(final / name) for name in (
        'FINAL_SUBMISSION_RISK_AUDIT.md', 'N16_DECISION.md', 'STATISTICAL_VALIDITY_REPORT.md', 'REPRODUCTION_REPORT.md')}
    report_hash_match = {name: report_hashes[name] == EXPECTED_REFERENCE_HASHES[name] for name in report_hashes}
    authored_hashes = {p.name: sha256_file(p) for p in sorted((parent / 'reports' / 'authored').iterdir()) if p.is_file()}
    artifacts = verify_artifact_manifest(parent, repo)
    maps = []
    map_errors = []
    # Only maps under runs/ are scientific artifacts.  The parent cache currently also
    # contains four pytest scratch maps; retaining them is harmless, counting them is not.
    for path in sorted((parent / 'runs').rglob('*.mixfp4map')):
        try:
            header, _, digest = mapio.read_map(path)
            maps.append({'path': path.relative_to(parent).as_posix(), 'sha256': digest,
                         'policy': header['policy']['name'], 'type_block': header['type_block'],
                         'selected_tiles': header['totals']['selected_tiles']})
        except Exception as exc:
            map_errors.append({'path': path.relative_to(parent).as_posix(), 'error': repr(exc)})
    print(f'PARENT MAPS parsed={len(maps)} errors={len(map_errors)}', flush=True)
    by_state = coverage.get('summary') or coverage.get('coverage') or {}
    return {
        'canonical_path': str(parent.resolve()), 'freeze_sha256': freeze_sha, 'freeze_sidecar_sha256': freeze_sidecar,
        'amendment_chain': amendments, 'authoritative_final_attempt': 'V90_final_reports_attempt7',
        'final_index_sha256': sha256_file(final / 'FINAL_DELIVERABLES_INDEX.json'),
        'final_index': final_index, 'artifact_validation': validation,
        'matrix_coverage_sha256': sha256_file(final / 'MATRIX_COVERAGE.json'), 'matrix_coverage_summary': by_state,
        'final_report_hashes': report_hashes, 'final_report_reference_matches': report_hash_match,
        'authored_current_hashes': authored_hashes, 'map_count': len(maps), 'map_errors': map_errors,
        'map_resolution_counts': {
            'n8': sum(x['type_block'] == [8, 64] for x in maps),
            'n16': sum(x['type_block'] == [16, 64] for x in maps),
        },
        'artifact_manifest_verification': artifacts,
        'passed': freeze_sha == freeze_sidecar == EXPECTED_PARENT_FREEZE_SHA256 and amendments['passed']
                  and validation.get('runs') == EXPECTED_PARENT_RUNS and validation.get('entries') == EXPECTED_PARENT_ENTRIES
                  and validation.get('current_runs_with_problems') == [] and all(report_hash_match.values())
                  and len(maps) == 630 and not map_errors and artifacts['passed'],
    }


def reference_artifacts(repo: Path) -> dict:
    artifacts = repo / 'research_artifacts'
    paths = {
        'reviewer_core_zip': artifacts / 'mixfp4_n16k64_final_evidence_core.zip',
        'EVIDENCE_BUNDLE_INDEX.json': artifacts / 'EVIDENCE_BUNDLE_INDEX.json',
        'BUNDLE_VERIFICATION_REPORT.json': artifacts / 'BUNDLE_VERIFICATION_REPORT.json',
    }
    result = {}
    for name, expected in EXPECTED_REFERENCE_HASHES.items():
        if name not in paths:
            continue
        actual = sha256_file(paths[name])
        result[name] = {'path': str(paths[name]), 'expected_sha256': expected, 'actual_sha256': actual,
                        'passed': actual == expected}
    original = artifacts / 'mixfp4_n16k64_top_tier_agent_handoff.zip'
    original_sha = sha256_file(original)
    result['original_implementation_handoff'] = {'path': str(original), 'expected_sha256': EXPECTED_ORIGINAL_HANDOFF_SHA256,
                                                  'actual_sha256': original_sha,
                                                  'passed': original_sha == EXPECTED_ORIGINAL_HANDOFF_SHA256}
    return result


def runtime_identity(campaign_root: Path, parent: Path) -> dict:
    image = 'ubuntu@sha256:829f6df217bcbae2b371026e81711d1a787c61b2967ad09d015063663ebafbf7'
    # The campaign image deliberately does not contain the Docker client/socket.  The host
    # launcher already resolved and recorded the immutable image ID before starting us.
    launch = json.loads((runtime.run_dir / 'launch_record.json').read_text())
    source = campaign_root / 'source' / 'NVFP4-RaZeR-main' / 'campaign'
    parent_source = parent / 'source' / 'NVFP4-RaZeR-main' / 'campaign'
    compared = {}
    for name in ('evaluate_ppl.py', 'data.py', 'mapio.py', 'quant.py', 'tiles.py'):
        compared[name] = {'extension_sha256': sha256_file(source / name), 'parent_sha256': sha256_file(parent_source / name),
                          'identical': sha256_file(source / name) == sha256_file(parent_source / name)}
    return {
        'container_image': image, 'docker_image_id': launch.get('image_id'),
        'image_identity_source': 'host launch_record.json (resolved before container start)',
        'main_lock_sha256': sha256_file(campaign_root / 'env' / 'main.lock.txt'),
        'hist_lock_sha256': sha256_file(campaign_root / 'env' / 'hist.lock.txt'),
        'evaluator_and_semantics_files': compared,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-parent-manifest', action='store_true', help='required for a gate-quality audit')
    args = parser.parse_args()
    if not args.full_parent_manifest:
        raise SystemExit('--full-parent-manifest is required; partial audits are not gate evidence')
    campaign_root = Path(os.environ['CAMPAIGN_ROOT']).resolve()
    repo = Path(os.environ['CAMPAIGN_REPO_ROOT']).resolve()
    parent = Path(os.environ['CAMPAIGN_PARENT_ROOT']).resolve()
    storage = Path(os.environ['CAMPAIGN_STORAGE_ROOT']).resolve()
    out = runtime.out_dir('input_audit')
    started = time.time()
    report = {
        'schema_version': '1.0', 'status': 'running', 'read_only_sources': True,
        'campaign_root': str(campaign_root), 'storage_root': str(storage), 'parent_root': str(parent),
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    runtime.atomic_json(out / 'READ_ONLY_INPUT_AUDIT.json', report)
    report['handoff'] = handoff_audit(repo, campaign_root)
    report['reference_artifacts'] = reference_artifacts(repo)
    report['parent'] = parent_state_audit(parent, repo)
    report['inputs'] = input_manifest_audit(parent, Path(os.environ['HF_HUB_CACHE']))
    report['runtime_identity'] = runtime_identity(campaign_root, parent)
    report['carryover_findings'] = {
        'independent_accuracy_task_rng': 'fixed_in_extension_source_and_unit_tested',
        'finite_monte_carlo_plus_one_pvalues': 'fixed_in_extension_source_and_unit_tested',
        'reviewer_redacted_chain': 'unresolved_in_parent_bundle; repair_required_in_extension_bundle',
        'windows_safe_names': 'unresolved_in_parent_bundle; repair_required_in_extension_bundle',
        'conditional_on_map_wording': 'required_in_extension_reports',
    }
    refs_ok = all(item['passed'] for item in report['reference_artifacts'].values())
    report['g0_input_audit'] = {
        'handoff_passed': report['handoff']['passed'], 'parent_passed': report['parent']['passed'],
        'reference_artifacts_passed': refs_ok,
        'all_required_inputs_readable': report['inputs']['summary']['all_required_inputs_readable'],
        'passed': report['handoff']['passed'] and report['parent']['passed'] and refs_ok
                  and report['inputs']['summary']['all_required_inputs_readable'],
    }
    report.update(status='complete', wall_seconds=time.time() - started,
                  finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    path = out / 'READ_ONLY_INPUT_AUDIT.json'
    runtime.atomic_json(path, report)
    source_manifest = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    runtime.atomic_json(runtime.run_dir / 'job_result.json', {
        'protocol_id': 'ppl-extension-prelock-audit', 'protocol_freeze_sha256': '',
        'source': {'model_id': None, 'model_revision': None, 'tokenizer_revision': None, 'model_class': None,
                   'module_manifest_sha256': None, 'source_manifest_sha256': source_manifest},
        'environment': runtime.environment(),
        'data': {'calibration_manifest_sha256': None, 'evaluation_manifest_sha256': None,
                 'token_hashes': {}, 'overlap_audit': None},
        'policies': [],
        'results': {'raw_outputs': [str(path)], 'summary': report['g0_input_audit'], 'uncertainty': {},
                    'attempted_endpoints': ['P00_INPUT_AUDIT'],
                    'missing_endpoints': ([] if report['g0_input_audit']['passed'] else
                                          report['inputs']['summary']['missing_model_snapshots'])},
        'logs': [], 'failures': [],
    })
    print(json.dumps(report['g0_input_audit'], sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
