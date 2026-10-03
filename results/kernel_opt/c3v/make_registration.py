"""Writes results/kernel_opt/c3v/registration.json (amendment 19, C3v), CPU only: the sha256 of every file the sweep reads
and, per build directory, every build it times, in the format of check_provenance.py."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KO = Path('/home/dev/n16k64_campaign/kernel_opt')
ART = Path('/home/dev/n16k64_campaign/paper/artifacts')
GPU = 'nvidia_rtx_pro_6000_blackwell_workstation_edition'
FILES = ['results/kernel_opt/PROTOCOL.md', 'results/kernel_opt/c3v/PROTOCOL.md'] + [f'sm120/mixfp4_sm120/{m}.py' for m in (
    '__init__', 'artifact', 'configs', 'lib', 'linear', 'mapio', 'model', 'numerics', 'quant_act', 'select')] + [
    f'sm120/configs/{GPU}.json', f'sm120/configs/{GPU}.ko.json', 'sm120/bench/common.py',
    'experiments/paper/bench_gemm_isolated.py', 'results/kernel_opt/c3k/C3k_raw.json',
    'results/kernel_opt/c3w/C3w_raw.json', 'experiments/kernel_opt/c3k_fraction.py',
    'experiments/kernel_opt/c3w_fraction.py'] + [
    f'experiments/kernel_opt/{s}' for s in ('c3v_fraction.py', 'c3v_analyze.py', 'check_provenance.py', 'run_c3v.sh')] + [
    'results/kernel_opt/c3v/make_registration.py'] + [str(ART / f'llama8b_tc_{u}.mixfp4map') for u in ('256x64', '16x64', '8x64')]
BUILDS = {
    str(KO / 'build_V'): ('n16k64_wA_g32_n16_t0', 'n16k64_wA_g32_n32_t0', 'n16k64_wA_g32_n64_t0', 'n16k64_wA_g32_e64_t0',
                          'n16k64_wA_n16_t0', 'n16k64_wA_n32_t0', 'n16k64_wA_n64_t0', 'n16k64_wA_e64_t0', 'n8k64_wB_m16_t0',
                          'n8k64_wB_m32_t0', 'n8k64_wB_m64_t0', 'n8k64_wB_n64_t0', 'n8k64_wB_t0', 'stock_wB_e64'),
    str(KO / 'build_A1'): ('n16k64_wA_g32', 'n16k64_wA_g32_n64', 'n16k64_wA_g32_n32', 'n16k64_wA_g32_n16'),
    str(KO / 'build_7freq'): ('n16k64_wA_n16_t0', 'n16k64_wA_n32_t0', 'n16k64_wA_n64_t0', 'n16k64_wA_e64_t0'),
    str(KO / 'build_P3freq'): ('n8k64_wB_m16_t0', 'n8k64_wB_m32_t0', 'n8k64_wB_m64_t0', 'n8k64_wB_n64_t0', 'n8k64_wB_t0'),
    str(KO / 'build_C3k'): ('n16k64_wA_nodisp_n16_t0', 'n16k64_wA_nodisp_n32_t0', 'n16k64_wA_nodisp_n64_t0',
                            'n16k64_wA_nodisp_e64_t0'),
    str(KO / 'build_P5'): ('n8k64_wB_m16_nodisp_t0', 'n8k64_wB_m32_nodisp_t0', 'n8k64_wB_m64_nodisp_t0',
                           'n8k64_wB_n64_nodisp_t0', 'n8k64_wB_nodisp_t0'),
    str(KO / 'build_7'): ('stock_wA_n16', 'stock_wA_n32', 'stock_wA_n64', 'stock_wA_e64'),
    str(REPO / 'sm120' / 'build'): ('n16k64_wA', 'n16k64_wA_n16', 'n16k64_wA_n32', 'n16k64_wA_n64', 'n8k64_wB',
                                    'stock_wA_n16', 'stock_wA_n32', 'stock_wA_n64', 'stock_wA'),
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
    files ={f: sha(f if f.startswith('/') else REPO / f) for f in FILES}
    builds = {root: {n: entry(Path(root) / n) for n in names} for root, names in BUILDS.items()}
    # the build_V builds are the ones amendment 18 registered
    r18 = json.loads((REPO / 'results' / 'kernel_opt' / 'registration_18.json').read_text())['builds'][str(KO / 'build_V')]
    assert all(builds[str(KO / 'build_V')][n] == r18[n] for n in BUILDS[str(KO / 'build_V')])
    reg = dict(protocol='results/kernel_opt/c3v/PROTOCOL.md', amendment=('Amendment 19 (C3v, the E0M3-fraction sweep on the '
               'adopted paths of the three units: 256x64, and 16x64 / 8x64 again on build_V)'),
               protocol_sha256=files['results/kernel_opt/c3v/PROTOCOL.md'],
               registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               parent_commit=git('rev-parse', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
               note=('before any registered GPU run of the sweep (the disclosed smoke tests excepted); no new builds: the '
                     'builds listed here are registered in amendments 3, 7, 8, 12, 13 and 18 and the paper builds; the '
                     "build_V entries equal registration_18.json's"),
               files=files, builds=builds)
    out = REPO / 'results' / 'kernel_opt' / 'c3v' / 'registration.json'
    out.write_text(json.dumps(reg, indent=1) + '\n')
    print(f'{out}: {len(files)} files, {sum(len(v) for v in builds.values())} builds')


if __name__ == '__main__':
    main()
