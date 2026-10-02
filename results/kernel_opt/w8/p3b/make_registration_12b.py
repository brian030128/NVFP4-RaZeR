"""Writes results/kernel_opt/registration_12b.json (amendment 12b, the 8x64 plan's P3b), CPU only: the sha256 of every
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
    f'sm120/configs/{GPU}.json', f'sm120/configs/{GPU}.ko.json', 'sm120/bench/common.py', 'sm120/bench/kernel.py',
    'sm120/eval/common.py', 'experiments/paper/bench_gemm_isolated.py',
    f'results/kernel_opt/w8/p3/table/{GPU}.json', 'results/kernel_opt/w8/p3b/cells.json',
    f'results/kernel_opt/w8/p3b/table_p3b/{GPU}.json', f'results/kernel_opt/w8/p3b/table_a/{GPU}.json'] + [
    f'experiments/kernel_opt/{s}' for s in (
        'bench_ab_isolated.py', 'bench_w8p3b_isolated.py', 'w8p3_report.py', 'w8p3b_report.py', 'p3b_tables.py',
        'check_provenance.py', 'check_bitwise.py', 'check_model_logits.py', 'run_w8p3b.sh')] + [
    'results/kernel_opt/w8/p3b/make_registration_12b.py'] + [
    str(ART / f'{m}_{u}' / 'artifact.json') for m in MODELS for u in ('fo6', 'tc_8x64')]
BUILDS = {
    str(KO / 'build_P3freq'): WBT0 + ('stock_wB_e64',),
    str(KO / 'build_P2freq'): WBT0,
    str(KO / 'build_7'): ('stock_wA_n16', 'stock_wA_n32', 'stock_wA_n64', 'stock_wA_e64'),
    str(REPO / 'sm120' / 'build'): ('stock_wB', 'n8k64_wB'),
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
    reg = dict(protocol='results/kernel_opt/PROTOCOL.md',
               amendment="Amendment 12b (the 8x64 plan's P3b, option C: a fresh confirmation of a reduced 8x64 table)",
               protocol_sha256=files['results/kernel_opt/PROTOCOL.md'],
               registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               parent_commit=git('rev-parse', 'HEAD'), branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
               note=('before any registered GPU run of amendment 12b (the disclosed smoke test excepted); no build: the '
                     'libraries are amendment 12\'s build_P3freq (registered in registration_12.json); the cell list '
                     'and both tables were produced CPU-only by p3b_tables.py from amendment 12\'s registered M1 records'),
               files=files, builds=builds)
    out = REPO / 'results' / 'kernel_opt' / 'registration_12b.json'
    out.write_text(json.dumps(reg, indent=1) + '\n')
    print(f'{out}: {len(files)} files, {sum(len(v) for v in builds.values())} builds')


if __name__ == '__main__':
    main()
