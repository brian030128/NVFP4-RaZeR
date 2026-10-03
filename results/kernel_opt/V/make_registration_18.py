"""Writes results/kernel_opt/registration_18.json (amendment 18, V: the 256x64 path brought to the 16x64 state), CPU only:
the sha256 of every file the run reads and, per build directory, every build the run gates or measures, in the format of
check_provenance.py."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KO = Path('/home/dev/n16k64_campaign/kernel_opt')
ART = Path('/home/dev/n16k64_campaign/paper/artifacts')
GPU = 'nvidia_rtx_pro_6000_blackwell_workstation_edition'
MODELS = ('llama8b', 'mistral7b', 'phi4', 'qwen27b')
FILES = ['results/kernel_opt/PROTOCOL.md'] + [f'sm120/mixfp4_sm120/{m}.py' for m in (
    '__init__', 'artifact', 'configs', 'lib', 'linear', 'mapio', 'model', 'numerics', 'quant_act', 'select')] + [
    'sm120/build.py', 'sm120/kernel/scripts/patch_mixed_nvfp4_gemm.py', 'sm120/kernel/scripts/gen_mixed_mma_blob.py',
    'sm120/kernel/src/collective/sm120_blockscaled_mma_tma_mixed.hpp', 'sm120/kernel/src/mixed_nvfp4_gemm.cu',
    f'sm120/configs/{GPU}.json', f'sm120/configs/{GPU}.ko.json', 'sm120/bench/common.py', 'sm120/bench/kernel.py',
    'sm120/bench/tune_tiles.py', 'experiments/paper/bench_gemm_isolated.py', 'sm120/tests/conftest.py',
    'sm120/tests/test_gemm.py', 'sm120/tests/test_select.py', 'sm120/tests/test_g32.py'] + [
    f'experiments/kernel_opt/{s}' for s in (
        'bench_ab_isolated.py', 'bench_V_isolated.py', 'c2_freq.py', 'c2_V.py', 'V_tables.py', 'V_report.py',
        'check_provenance.py', 'check_sass.py', 'check_uniform_sass.py', 'check_same_sass.py', 'check_patcher_sites.py',
        'check_bitwise.py', 'check_model_logits.py', 'run_V.sh')] + [
    'results/kernel_opt/V/build_V.sh', 'results/kernel_opt/V/make_registration_18.py',
    'results/kernel_opt/V/EXPLORATION.md'] + [
    str(ART / f'{m}_{u}' / 'artifact.json') for m in MODELS for u in ('fo6', 'tc_256x64')] + [
    str(ART / f'{m}_tc_256x64.mixfp4map') for m in MODELS]
BUILDS = {
    str(KO / 'build_7'): ('stock_wA_n16', 'stock_wA_n32', 'stock_wA_n64', 'stock_wA_e64'),
    str(KO / 'build_A1'): ('n16k64_wA_g32', 'n16k64_wA_g32_n64', 'n16k64_wA_g32_n32', 'n16k64_wA_g32_n16'),
    str(KO / 'build_C3k'): ('n16k64_wA_nodisp_n16_t0', 'n16k64_wA_nodisp_n32_t0', 'n16k64_wA_nodisp_n64_t0',
                            'n16k64_wA_nodisp_e64_t0'),
    str(KO / 'build_U'): ('n16k64_wA_n16_t0', 'n16k64_wA_n32_t0', 'n16k64_wA_n64_t0', 'n16k64_wA_e64_t0', 'n8k64_wB_m16_t0',
                          'n8k64_wB_m32_t0', 'n8k64_wB_m64_t0', 'n8k64_wB_n64_t0', 'n8k64_wB_t0', 'stock_wB_e64'),
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
    heads, n_src, bad = set(), 0, 0
    for root in (KO / 'build_V', KO / 'build_Vall'):
        builds[str(root)] = {d.name: entry(d) for d in sorted(root.iterdir()) if (d / 'manifest.json').exists()}
        for man in root.glob('*/manifest.json'):
            m = json.loads(man.read_text())
            heads.add((m['repo_head'], m['repo_dirty']))
            for f, h in m['sources'].items():
                if not f.startswith('<generated>'):
                    n_src += 1
                    bad += sha(REPO / f) != h
    assert not bad, f'{bad} recorded source hashes differ from the files'
    assert len(heads) == 1 and not next(iter(heads))[1], heads
    assert len(builds[str(KO / 'build_V')]) == 22 and len(builds[str(KO / 'build_Vall')]) == 56
    reg = dict(protocol='results/kernel_opt/PROTOCOL.md',
               amendment=("Amendment 18 (V: the 256x64 path brought to the 16x64 state -- 'mixed256_ko', A''s g32 builds "
                          "with t0, #4's tile at width 128 and the uniform-branch dispatch, with its re-tuned widths and "
                          'scheduler rows)'),
               protocol_sha256=files['results/kernel_opt/PROTOCOL.md'],
               registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               parent_commit=git('rev-parse', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
               note=('before any registered GPU run of amendment 18 (the disclosed exploration and smoke tests excepted); '
                     f'build_V (22) and build_Vall (all 56 configurations) were built CPU-only (no GPU visible) from the '
                     f'clean main checkout at {next(iter(heads))[0][:7]}; all {n_src} source hashes their manifests '
                     "record equal the files at this registration (checked); today's 256x64 path is build_A1, stock_ko "
                     "build_7's, the ceiling's builds build_C3k's, amendment 17's adopted builds build_U's"),
               files=files, builds=builds)
    out = REPO / 'results' / 'kernel_opt' / 'registration_18.json'
    out.write_text(json.dumps(reg, indent=1) + '\n')
    print(f'{out}: {len(files)} files, {sum(len(v) for v in builds.values())} builds')


if __name__ == '__main__':
    main()
