"""V90/V91 final synthesis job: N16_DECISION.md, CONFIRMATORY_FREEZE.json, REPRODUCTION_REPORT.md, STATISTICAL_VALIDITY_REPORT.md,
FINAL_SUBMISSION_RISK_AUDIT.md, MATRIX_COVERAGE.json and FINAL_DELIVERABLES_INDEX.json, all in this run's final/ directory."""
import json
import os
from pathlib import Path

from campaign import assemble_deliverables as AD
from campaign import final_reports as FR
from campaign import final_reports_text as FT
from campaign import risk_audit as RA
from campaign import runtime


def main():
    out = runtime.out_dir('final')
    FR.main()
    FT.main()
    rows = RA.main(final_dir=out)
    me = runtime.run_dir

    def resolve(job, rel):
        if job == 'V90_final_reports':
            return me / rel
        if job == '@':
            r = AD.maybe('V90_assemble_deliverables')
            return (r / 'deliverables' / rel) if r else None
        if job is None:
            return AD.CR / rel
        r = AD.maybe(job)
        return (r / rel) if r else None

    cov = AD.matrix_coverage(resolve, AD.jobs_by_matrix(exclude=('V90_final_reports',)))
    runtime.atomic_json(out / 'MATRIX_COVERAGE.json', cov)
    index = {}
    for p in sorted(out.iterdir()):
        if p.is_file():
            index[p.name] = runtime.sha256_file(p)
    extra = {}
    for job, rel in (('V83_validate_artifacts', 'artifact_validation/ARTIFACT_MANIFEST.sha256'), ('V83_validate_artifacts', 'artifact_validation/FAILED_OR_SKIPPED_RUNS.json'),
                     ('V83_validate_artifacts', 'artifact_validation/ARTIFACT_VALIDATION.md')):
        p = resolve(job, rel)
        extra[rel] = dict(path=str(p.relative_to(AD.CR)), sha256=runtime.sha256_file(p)) if p and p.exists() else None
    for name in ('SMOKE_REPORT.json', 'ALIGNED_MAP_MANIFEST.json', 'CONFIRMATORY_MAP_MANIFEST.json', 'K_SENSITIVITY.json', 'OBJECTIVE_ABLATION.json', 'SELECTOR_CONTROLS.json',
                 'CALIBRATION_SEED_STABILITY.json', 'CALIBRATION_SIZE.json', 'CALIBRATION_DOMAIN.json', 'QWEN_SMOOTHING_DIAGNOSTIC.json', 'ADDITIONAL_BASELINES.json',
                 'COMPUTE_DISCLOSURE.json'):
        p = resolve('@', name)
        extra[name] = dict(path=str(p.relative_to(AD.CR)), sha256=runtime.sha256_file(p)) if p and p.exists() else None
    runtime.atomic_json(out / 'FINAL_DELIVERABLES_INDEX.json', dict(freeze_sha256=AD.FSHA, final=index, referenced=extra,
                                                                    amendments_sha256=runtime.sha256_file(AD.CR / 'provenance' / 'PROTOCOL_AMENDMENTS.jsonl')))
    src = json.loads((me / 'launch_record.json').read_text())['source_manifest_sha256']
    runtime.atomic_json(me / 'job_result.json', dict(protocol_id='synthesis', protocol_freeze_sha256=AD.FSHA,
        source=dict(model_id=None, model_revision=None, tokenizer_revision=None, model_class=None, module_manifest_sha256=None, source_manifest_sha256=src),
        environment=runtime.environment(), data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=None, token_hashes={}, overlap_audit=None),
        policies=[], results=dict(raw_outputs=[str(out / n) for n in index], summary=dict(coverage=cov['summary'], classifications={r['id']: r['classification'] for r in rows}),
                                  uncertainty={}, attempted_endpoints=list(index), missing_endpoints=[r['id'] for r in cov['rows'] if r['status'] != 'complete']),
        logs=[], failures=[]))
    print(json.dumps(dict(coverage=cov['summary'], classes={r['id']: r['classification'] for r in rows}), indent=1))


if __name__ == '__main__':
    main()
