"""Writes results/kernel_opt/registration_12.json (amendment 12, the 8x64 plan's P3 with P4), CPU only: the sha256 of
every file the run reads and, per build directory, every build the run gates, tunes or measures, in the format of
check_provenance.py."""
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
    f'sm120/configs/{GPU}.json', f'sm120/configs/{GPU}.ko.json',
    'sm120/bench/common.py', 'sm120/bench/kernel.py', 'sm120/bench/tune_tiles.py', 'sm120/eval/common.py',
    'experiments/paper/bench_gemm_isolated.py',
    'sm120/tests/conftest.py', 'sm120/tests/test_gemm.py', 'sm120/tests/test_select.py'] + [
    f'experiments/kernel_opt/{s}' for s in (
        'bench_ab_isolated.py', 'bench_w8p3_isolated.py', 'p3_tables.py', 'w8p3_report.py', 'check_provenance.py',
        'check_sass.py', 'check_same_sass.py', 'check_bitwise.py', 'check_model_logits.py', 'run_w8p3.sh')] + [
    'results/kernel_opt/w8/p3/build_P3.sh', 'results/kernel_opt/w8/p3/make_registration_12.py'] + [
    str(ART / f'{m}_tc_8x64.mixfp4map') for m in MODELS] + [
    str(ART / f'{m}_{u}' / 'artifact.json') for m in MODELS for u in ('fo6', 'tc_8x64')]
BUILDS = {
    str(KO / 'build_P2freq'): WBT0,                                       # amendment 11's adopted 8x64 path (G5 before)
    str(KO / 'build_7'): ('stock_wA_n16', 'stock_wA_n32', 'stock_wA_n64', 'stock_wA_e64'),   # stock_ko
    str(REPO / 'sm120' / 'build'): ('stock_wB', 'n8k64_wB'),             # M1/G5 reference; G4's reference
    str(KO / 'build_P3freq'): WBT0 + ('stock_wB_e64',),
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
    p3 = KO / 'build_P3'
    builds[str(p3)] = {d.name: entry(d) for d in sorted(p3.iterdir()) if (d / 'manifest.json').exists()}
    n_src = bad = 0
    for root in (p3, KO / 'build_P3freq'):
        for man in root.glob('*/manifest.json'):
            for f, h in json.loads(man.read_text())['sources'].items():
                if not f.startswith('<generated>'):
                    n_src += 1
                    bad += sha(REPO / f) != h
    assert not bad, f'{bad} recorded source hashes differ from the files'
    reg = dict(protocol='results/kernel_opt/PROTOCOL.md',
               amendment="Amendment 12 (the 8x64 plan's P3, with P4: the adopted 8x64 path's widths and scheduler rows; "
                         "stock_wB tuned alike)",
               protocol_sha256=files['results/kernel_opt/PROTOCOL.md'],
               registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               parent_commit=git('rev-parse', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
               note=('before any registered GPU run of amendment 12 (the disclosed smoke test excepted). build_P3 (all '
                     '51 configurations) and build_P3freq (the five 8x64 t0 builds with MIXFP4_DISPATCH_FREQ=1, and '
                     'stock_wB_e64) were built CPU-only (no GPU visible) from the clean main checkout at the P3 source '
                     f'commit 097d1e5; all {n_src} source hashes their manifests record equal the files at this '
                     'registration (checked). build_P2freq is amendment 11\'s adopted 8x64 path; build_7 and sm120/build '
                     'are amendment 10\'s.'),
               files=files, builds=builds)
    out = REPO / 'results' / 'kernel_opt' / 'registration_12.json'
    out.write_text(json.dumps(reg, indent=1) + '\n')
    print(f'{out}: {len(files)} files, {sum(len(v) for v in builds.values())} builds, {n_src} source hashes checked')


if __name__ == '__main__':
    main()
