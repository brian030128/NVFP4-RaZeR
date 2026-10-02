"""Writes results/kernel_opt/registration_13.json (amendment 13, the 8x64 plan's P5), CPU only: the sha256 of every
file the run reads and, per build directory, every build the run gates or measures, in the format of check_provenance.py."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
KO = Path('/home/dev/n16k64_campaign/kernel_opt')
ART = Path('/home/dev/n16k64_campaign/paper/artifacts')
GPU = 'nvidia_rtx_pro_6000_blackwell_workstation_edition'
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
WBT0 = ('n8k64_wB_m16_t0', 'n8k64_wB_m32_t0', 'n8k64_wB_m64_t0', 'n8k64_wB_t0', 'n8k64_wB_n64_t0')
FILES = ['results/kernel_opt/PROTOCOL.md'] + [f'sm120/mixfp4_sm120/{m}.py' for m in (
    '__init__', 'artifact', 'configs', 'lib', 'linear', 'mapio', 'model', 'numerics', 'quant_act', 'select')] + [
    'sm120/build.py', 'sm120/kernel/scripts/patch_mixed_nvfp4_gemm.py', 'sm120/kernel/scripts/gen_mixed_mma_blob.py',
    f'sm120/configs/{GPU}.json', f'sm120/configs/{GPU}.ko.json', 'sm120/bench/common.py', 'sm120/bench/kernel.py',
    'experiments/paper/bench_gemm_isolated.py', 'sm120/tests/conftest.py', 'sm120/tests/test_select.py'] + [
    f'experiments/kernel_opt/{s}' for s in (
        'bench_ab_isolated.py', 'bench_w8p5_isolated.py', 'w8p5_report.py', 'check_provenance.py', 'check_sass.py',
        'run_w8p5.sh')] + [
    'results/kernel_opt/w8/p5/build_P5.sh', 'results/kernel_opt/w8/p5/make_registration_13.py'] + [
    str(ART / f'{m}_{u}' / 'artifact.json') for m in MODELS for u in ('fo6', 'tc_8x64')]
BUILDS = {
    str(KO / 'build_P3freq'): WBT0 + ('stock_wB_e64',),
    str(KO / 'build_7'): ('stock_wA_n16', 'stock_wA_n32', 'stock_wA_n64', 'stock_wA_e64'),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def entry(d):
    m = json.loads((d / 'manifest.json').read_text())
    assert sha(d / m['library']) == m['library_sha256'], d
    return {k: m.get(k) for k in ('library_sha256', 'sass_sha256', 'unpatched_sass_sha256', 'extra_defines', 'built_utc',
                                  'repo_head')}


def main():
    git = lambda *a: subprocess.run(['git', *a], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()  # noqa: E731
    files = {f: sha(f if f.startswith('/') else REPO / f) for f in FILES}
    builds = {root: {n: entry(Path(root) / n) for n in names} for root, names in BUILDS.items()}
    p5 = KO / 'build_P5'
    builds[str(p5)] = {d.name: entry(d) for d in sorted(p5.iterdir()) if (d / 'manifest.json').exists()}
    n_src = bad = 0
    for man in p5.glob('*/manifest.json'):
        for f, h in json.loads(man.read_text())['sources'].items():
            if not f.startswith('<generated>'):
                n_src += 1
                bad += sha(REPO / f) != h
    assert not bad, f'{bad} recorded source hashes differ from the files'
    reg = dict(protocol='results/kernel_opt/PROTOCOL.md',
               amendment="Amendment 13 (the 8x64 plan's P5: the adopted 8x64 path's no-dispatch ceiling at every width)",
               protocol_sha256=files['results/kernel_opt/PROTOCOL.md'],
               registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               parent_commit=git('rev-parse', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
               note=('before any registered GPU run of amendment 13 (the disclosed smoke test excepted); build_P5 (all 55 '
                     'configurations) was built CPU-only (no GPU visible) from the clean main checkout at 09f67e4; all '
                     f'{n_src} source hashes its manifests record equal the files at this registration (checked); the '
                     '8x64 path and stock_wB_e64 are build_P3freq\'s (amendment 12), stock_ko build_7\'s'),
               files=files, builds=builds)
    out = REPO / 'results' / 'kernel_opt' / 'registration_13.json'
    out.write_text(json.dumps(reg, indent=1) + '\n')
    print(f'{out}: {len(files)} files, {sum(len(v) for v in builds.values())} builds')


if __name__ == '__main__':
    main()
