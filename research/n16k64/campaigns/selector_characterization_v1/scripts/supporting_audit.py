"""Verify nested input seals and preserve bounded historical context evidence."""
from common import *
from mechanism_review import sealed

def main():
    start=now();checks=[];top=PRIMARY/'runs/V83_validate_artifacts_attempt7/artifact_validation/ARTIFACT_MANIFEST.sha256'
    def entries(p):return {r.split(maxsplit=1)[1].strip().lstrip('*'):r.split()[0] for r in p.read_text().splitlines()}
    outer=entries(top)
    for r in load(OUT/'results/INPUTS.json'):
        run=REPO/r['run'];s=run/'SHA256SUMS_run.txt';assert outer[s.relative_to(PRIMARY).as_posix()]==sha(s);members=entries(s)
        targets=[REPO/r['moments'],run/'calibration/calibration_report.json']+[REPO/m['path'] for m in r['maps']]
        for p in targets:
            assert members[p.relative_to(run).as_posix()]==sha(p),p
            checks.append(dict(path=str(p.relative_to(REPO)),sha256=sha(p),seal_sha256=sha(s),outer_seal_sha256=sha(top),passed=True))
    noise=[]
    for m in MODELS:
        p=PRIMARY/f'runs/V43_fidelity_noise_{m}_attempt1/fidelity_noise/fidelity_noise_report.json';sealed(p);r=load(p)
        assert r['status']=='complete'
        for objective in ['ce','kl']:
            b=[x[objective] for x in r['measurements'] if x['state']=='baseline']
            noise.append(dict(model=m,objective=objective,n_baseline_measurements=len(b),range=max(b)-min(b),source=str(p.relative_to(REPO)),sha256=sha(p)))
    # Record the real source drift, without claiming the currently available
    # historical source directory is the exact earlier executable snapshot.
    old=PRIMARY/'runs/V31_ppl_primary_llama8b_attempt1/source_manifest.txt'
    code=entries(old);drift=[];source=PRIMARY/'source/NVFP4-RaZeR-main'
    for name in ['campaign/quant.py','campaign/models.py','campaign/evaluate_ppl.py','campaign/policies.py','quantize/causal_four_over_six.py']:
        expected=code.get(name);actual=sha(source/name)
        drift.append(dict(path=name,original_sha256=expected,current_sha256=actual,equal=expected==actual))
    jsonout(OUT/'results/NESTED_SEAL_AUDIT.json',checks);jsonout(OUT/'results/NOISE_CONTROL_AUDIT.json',noise)
    jsonout(OUT/'results/SOURCE_DRIFT.json',dict(original_manifest=str(old.relative_to(REPO)),original_manifest_sha256=sha(old),files=drift,
        qualification='A filename or historical folder does not establish original executable identity; weight and data identities checked separately'))
    log('T0-seals','python scripts/supporting_audit.py',start,[OUT/'results/NESTED_SEAL_AUDIT.json',OUT/'results/NOISE_CONTROL_AUDIT.json',OUT/'results/SOURCE_DRIFT.json'])
    print('sealed files',len(checks),'noise',noise,'source drift',drift)

if __name__=='__main__':main()
