"""Run a CPU correctness test module inside the pinned container and emit the matrix deliverable JSON."""
import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

from campaign import runtime

DELIVERABLE = {'V10': 'QUANTIZER_CORRECTNESS.json', 'V11': 'TILE_LAYOUT_TESTS.json', 'V12': 'SCORE_AGGREGATION_TESTS.json',
               'V13': 'MAP_SERIALIZATION_TESTS.json', 'V14': 'SMOKE_REPORT_CPU_E2E.json', 'V01': 'GPU_PREFLIGHT_UNIT_TESTS.json',
               'REPO': 'ARCHIVED_REPOSITORY_TESTS.json'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrix-id', required=True)
    ap.add_argument('--tests', nargs='+', required=True)
    ap.add_argument('--findings-env', default=None)
    ap.add_argument('--repeat', type=int, default=1)
    args = ap.parse_args()
    out = runtime.out_dir('tests')
    runs = []
    for r in range(args.repeat):
        xml = out / f'junit_{r}.xml'
        env = dict(os.environ)
        findings = out / f'findings_{r}.json'
        if args.findings_env:
            env[args.findings_env] = str(findings)
        p = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', f'--junitxml={xml}'] + args.tests,
                           capture_output=True, text=True, env=env, cwd=str(runtime.run_dir.parents[1] / 'source' / 'NVFP4-RaZeR-main'))
        (out / f'pytest_{r}.log').write_text(p.stdout + p.stderr)
        cases = []
        root = ET.parse(xml).getroot()
        for tc in root.iter('testcase'):
            status = 'passed'
            detail = None
            for tag in ('failure', 'error', 'skipped'):
                el = tc.find(tag)
                if el is not None:
                    status, detail = tag, (el.get('message') or '')[:2000]
            cases.append(dict(name=f'{tc.get("classname")}::{tc.get("name")}', time=float(tc.get('time', 0)), status=status, detail=detail))
        runs.append(dict(repeat=r, returncode=p.returncode, passed=sum(c['status'] == 'passed' for c in cases),
                         failed=sum(c['status'] in ('failure', 'error') for c in cases), skipped=sum(c['status'] == 'skipped' for c in cases),
                         cases=cases, findings=(json.loads(findings.read_text()) if findings.exists() else None)))
    deliverable = dict(matrix_id=args.matrix_id, tests=args.tests, repeats=runs, all_passed=all(r['returncode'] == 0 for r in runs),
                       environment=runtime.environment())
    name = DELIVERABLE.get(args.matrix_id, f'{args.matrix_id}_TESTS.json')
    runtime.atomic_json(out / name, deliverable)
    src = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(
        protocol_id='correctness' if args.matrix_id != 'V01' else 'resource', protocol_freeze_sha256=os.environ.get('FREEZE_SHA256', ''),
        source=dict(model_id=None, model_revision=None, tokenizer_revision=None, model_class=None, module_manifest_sha256=None, source_manifest_sha256=src),
        environment=runtime.environment(), data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=None, token_hashes={}, overlap_audit=None),
        policies=[], results=dict(raw_outputs=[str(out / name)], summary={f'repeat{r["repeat"]}': dict(passed=r['passed'], failed=r['failed'], skipped=r['skipped']) for r in runs},
                                  uncertainty={}, attempted_endpoints=args.tests, missing_endpoints=[]),
        logs=[str(out / f'pytest_{r}.log') for r in range(args.repeat)], failures=[c for r in runs for c in r['cases'] if c['status'] in ('failure', 'error')]))
    print(json.dumps({k: v for k, v in deliverable.items() if k not in ('repeats', 'environment')}), flush=True)
    for r in runs:
        print(f'repeat {r["repeat"]}: passed={r["passed"]} failed={r["failed"]} skipped={r["skipped"]}', flush=True)
    if not deliverable['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
