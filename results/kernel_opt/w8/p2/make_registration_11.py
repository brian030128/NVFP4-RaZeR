"""Writes results/kernel_opt/registration_11.json (amendment 11, the 8x64 plan's P2), CPU only: the sha256 of every file
the run reads and, per build directory, every build the run gates or measures, in the format of check_provenance.py."""
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
WB = ('n8k64_wB_m16', 'n8k64_wB_m32', 'n8k64_wB_m64', 'n8k64_wB', 'n8k64_wB_n64')
WBT0 = tuple(n + '_t0' for n in WB)

FILES = ['results/kernel_opt/PROTOCOL.md'] + [f'sm120/mixfp4_sm120/{m}.py' for m in (
    '__init__', 'artifact', 'configs', 'lib', 'linear', 'mapio', 'model', 'numerics', 'quant_act', 'select')] + [
    'sm120/build.py', 'sm120/kernel/scripts/patch_mixed_nvfp4_gemm.py', 'sm120/kernel/scripts/gen_mixed_mma_blob.py',
    f'sm120/configs/{GPU}.json', f'sm120/configs/{GPU}.ko.json', 'results/kernel_opt/opt1b/table_opt1b.json',
    'sm120/bench/common.py', 'sm120/bench/kernel.py', 'sm120/eval/common.py', 'experiments/paper/bench_gemm_isolated.py',
    'sm120/tests/conftest.py', 'sm120/tests/test_gemm.py', 'sm120/tests/test_select.py'] + [
    f'experiments/kernel_opt/{s}' for s in (
        'bench_ab_isolated.py', 'bench_w8p2_isolated.py', 'c2_freq.py', 'c2_w8p2.py', 'w8p2_report.py',
        'check_provenance.py', 'check_sass.py', 'check_same_sass.py', 'check_patcher_sites.py', 'check_bitwise.py',
        'check_model_logits.py', 'run_w8p2.sh')] + [
    'results/kernel_opt/w8/p2/build_P2.sh', 'results/kernel_opt/w8/p2/make_registration_11.py',
    str(ART / 'llama8b_tc_8x64.mixfp4map')] + [
    str(ART / f'{m}_{u}' / 'artifact.json') for m in MODELS for u in ('fo6', 'tc_8x64')]
BUILDS = {
    str(KO / 'build'): WB,                       # optimization 1b's mixed_wB (amendment 10's G0)
    str(KO / 'build_freq'): WB,                  # #2's mixed_wB (amendment 10's G0)
    str(KO / 'build_7'): ('stock_wA_n16', 'stock_wA_n32', 'stock_wA_n64', 'stock_wA_e64'),   # stock_ko
    str(REPO / 'sm120' / 'build'): ('stock_wA', 'stock_wB', 'n8k64_wB'),   # C2w''' references; G4's reference
    str(KO / 'build_W'): ('n8k64_wB_nodisp', 'n8k64_wB_nodisp_t0'),        # the ceilings (C2w''')
    str(KO / 'build_P2freq'): WBT0,
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
    p2 = KO / 'build_P2'
    builds[str(p2)] = {d.name: entry(d) for d in sorted(p2.iterdir()) if (d / 'manifest.json').exists()}
    reg = dict(protocol='results/kernel_opt/PROTOCOL.md',
               amendment="Amendment 11 (the 8x64 plan's P2: t0 for the weights-on-B family; P1's dispatch choice)",
               protocol_sha256=files['results/kernel_opt/PROTOCOL.md'],
               registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               parent_commit=git('rev-parse', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
               note=('before any registered GPU run of amendment 11 (the disclosed smoke test excepted). build_P2 (all '
                     '50 configurations) and build_P2freq (the five wB t0 builds with MIXFP4_DISPATCH_FREQ=1) were built '
                     'CPU-only (no GPU visible) from the clean main checkout at the P2 source commit c5bb523; all 935 '
                     'source hashes their manifests record equal the files at this registration (checked), and those '
                     'registered here equal the registered sha256. The kernel-opt '
                     'build, build_freq, build_7, build_W and sm120/build entries are amendment 10\'s builds.'),
               files=files, builds=builds)
    out = REPO / 'results' / 'kernel_opt' / 'registration_11.json'
    out.write_text(json.dumps(reg, indent=1) + '\n')
    print(f'{out}: {len(files)} files, {sum(len(v) for v in builds.values())} builds')


if __name__ == '__main__':
    main()
