#!/usr/bin/env python3
"""Kernel-opt measurements M2 (prefill) and M3 (decode): end-to-end latency before vs after, CUDA graphs.

    python experiments/kernel_opt/ab_e2e.py --what prefill,decode [--models ...] [--rounds 5]

Protocol: results/kernel_opt/PROTOCOL.md (M2, M3). The per-process scripts are the paper's, unchanged:
experiments/paper/bench_prefill.py (step 05) and experiments/paper_extra/bench_decode.py (Experiment D).
- Policies (artifacts from the paper run, PAPER_OUT/artifacts):
    ours-8x64      TM-OPT+TC 8x64 on n8k64_wB from sm120/build: "before", as in the paper
    ours-8x64-opt  the same artifact on 'auto_wB' (the width-selecting weights-on-B set) with SM120_BUILD_DIR set to
                   the kernel-opt build directory: "after"
    fo6            FourOverSix on 'auto_stock' (sm120/build), the deployment reference
    fo6-wB         FourOverSix on stock_wB (sm120/build), the paper's same-placement reference
- Prefill: every model, the paper's prompt shapes (1x128 ... 1x8192, 4x2048), 7 repetitions per process.
- Decode: llama8b, mistral7b, phi4 (Experiment D's models), D's settings, 64 generated tokens.
- Order: one process per (model, policy, round); round r (1-based) runs the policy list rotated by r - 1.
- Registered checks (a failure stops the run): prefill, every shape captured and the graph's logits equal eager's
  bitwise; decode, the graph's first 33 greedy tokens equal an eager StaticCache decode's in every setting; every
  process records the kernel builds it loaded, and "after" processes must have loaded every build from the
  kernel-opt build directory and "before" processes from sm120/build.
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'experiments' / 'paper'))
import paper_common as P  # noqa: E402

AFTER_ROOT = Path(os.environ.get('KERNEL_OPT_BUILD', '/home/dev/n16k64_campaign/kernel_opt/build'))
OUT = Path(os.environ.get('KERNEL_OPT_OUT', '/home/dev/n16k64_campaign/kernel_opt'))
POLICIES = {'ours-8x64': ('tc_8x64', 'n8k64_wB', False), 'ours-8x64-opt': ('tc_8x64', 'auto_wB', True),
            'fo6': ('fo6', 'auto_stock', False), 'fo6-wB': ('fo6', 'stock_wB', False)}
DECODE_MODELS = ('llama8b', 'mistral7b', 'phi4')
DECODE_SETTINGS = '1x512,4x512,16x512,1x2048,4x2048,16x2048'


def build_check(path, after):
    """The libraries a process loaded come from the expected build directory (by sha256 against the manifests)."""
    rec = json.loads(Path(path).read_text())
    inst = rec.get('install') or {}
    root = AFTER_ROOT if after else REPO / 'sm120' / 'build'
    shas = {}
    for d in root.iterdir():
        m = d / 'manifest.json'
        if m.exists():
            shas[json.loads(m.read_text())['library_sha256']] = d.name
    used = list((inst.get('kernel_set') or {}).get('kernels', {}).values()) or [inst.get('kernel_sha256')]
    missing = [s for s in used if s not in shas]
    return [f'library {s} is not a build of {root}' for s in missing]


def prefill_check(path):
    r = json.loads(Path(path).read_text())
    bad = [f"{s}: {g.get('error') or 'logits differ from eager (max |diff| %s)' % g.get('max_abs_diff_eager')}"
           for s, g in r['graph'].items() if g.get('error') or not g.get('logits_equal_eager')]
    bad += [f'{s}: not captured' for s in r['eager'] if s not in r['graph'] and 'error' not in r['eager'][s]]
    return bad


def decode_check(path):
    r = json.loads(Path(path).read_text())
    return [f"{s}: {e.get('error') or 'graph tokens differ from eager StaticCache decode (match %s)' % e.get('token_match')}"
            for s, e in r['decode'].items() if e.get('error') or not e.get('tokens_equal_static_eager')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--what', default='prefill,decode')
    ap.add_argument('--models', default=','.join(P.MODELS))
    ap.add_argument('--policies', default=','.join(POLICIES))
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--shapes', default=','.join(P.PREFILL_SHAPES))
    ap.add_argument('--settings', default=DECODE_SETTINGS)
    ap.add_argument('--reps', type=int, default=7)
    ap.add_argument('--out', type=Path, default=OUT / 'e2e')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'logs').mkdir(exist_ok=True)
    if not P.gpu_idle():
        P.die('the GPU is busy: latency needs an idle GPU')
    names = [p for p in POLICIES if p in args.policies.split(',')]
    for what in args.what.split(','):
        models = [m for m in args.models.split(',') if what == 'prefill' or m in DECODE_MODELS]
        for model in models:
            for r in range(1, args.rounds + 1):
                shift = (r - 1) % len(names)
                order = names[shift:] + names[:shift]
                P.log(args.out, f'{what} {model} round {r} order {",".join(order)}')
                for pol in order:
                    kind, kernel, after = POLICIES[pol]
                    out = args.out / what / model / pol / f'round{r}.json'
                    check = prefill_check if what == 'prefill' else decode_check
                    if P.complete(out):
                        bad = check(out) + build_check(out, after)
                        if bad:
                            P.die(f'registered check failed: {what} {model} {pol} round {r}: {bad}')
                        print(f'{what} {model} {pol} round {r}: done')
                        continue
                    art = P.artifact(P.OUT_DEFAULT, model, kind)
                    if not (art / 'artifact.json').exists():
                        P.die(f'{art} missing')
                    if what == 'prefill':
                        cmd = [P.PY, REPO / 'experiments' / 'paper' / 'bench_prefill.py', '--model', model, '--label', pol,
                               '--round', str(r), '--shapes', args.shapes, '--reps', str(args.reps), '--out', out,
                               '--artifact', art, '--kernel', kernel]
                    else:
                        cmd = [P.PY, REPO / 'experiments' / 'paper_extra' / 'bench_decode.py', '--model', model, '--label',
                               pol, '--round', str(r), '--settings', args.settings, '--out', out, '--artifact', art,
                               '--kernel', kernel]
                    extra = {'SM120_BUILD_DIR': str(AFTER_ROOT)} if after else {}
                    if not after:
                        os.environ.pop('SM120_BUILD_DIR', None)
                    rc = P.run(args.out, f'{what}_{model}_{pol}_r{r}', cmd, extra_env=extra)
                    if rc != 0 or not P.complete(out):
                        P.die(f'{what} failed: {model} {pol} round {r} (log: {args.out}/logs)')
                    bad = check(out) + build_check(out, after)
                    if bad:
                        P.log(args.out, f'CHECK FAILED {what} {model} {pol} round {r}: {bad}')
                        P.die(f'registered check failed: {what} {model} {pol} round {r}: {bad}')


if __name__ == '__main__':
    main()
