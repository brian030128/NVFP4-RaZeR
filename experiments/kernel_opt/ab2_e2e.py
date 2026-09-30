#!/usr/bin/env python3
"""Kernel-opt amendment 2b (#2's M2'): end-to-end CUDA-graph prefill, default vs MIXFP4_DISPATCH_FREQ.

    python experiments/kernel_opt/ab2_e2e.py [--models ...] [--rounds 5] [--out DIR]

Protocol: results/kernel_opt/PROTOCOL.md (amendment 2b). The per-process script is the paper's, unchanged:
experiments/paper/bench_prefill.py (step 05), as in M2 (ab_e2e.py, whose checks this imports).
- Policies (artifacts from the paper run, PAPER_OUT/artifacts); each names its kernel and its build directory:
    ours-16x64 / ours-256x64        TM-OPT+TC 16x64 / 256x64 on 'auto' (the 'mixed' set) from sm120/build: before
    ours-16x64-freq / -256x64-freq  the same on 'auto' from build_freq: after
    ours-8x64-opt                   TM-OPT+TC 8x64 on 'auto_wB' from optimization 1b's build directory: before
    ours-8x64-freq                  the same on 'auto_wB' from build_freq: after
    fo6                             FourOverSix on 'auto_stock' (sm120/build), the deployment reference
- Prefill: every model, the paper's prompt shapes (1x128 ... 1x8192, 4x2048), 7 repetitions per process.
- Order: one process per (model, policy, round); round r (1-based) runs the policy list rotated by r - 1.
- Registered checks (a failure stops the run): every shape captured and the graph's logits equal eager's bitwise; every
  library a process loaded is a build of its policy's build directory (by sha256 against the manifests).
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'experiments' / 'paper'))
sys.path.insert(0, str(REPO / 'experiments' / 'kernel_opt'))
import paper_common as P  # noqa: E402
from ab_e2e import prefill_check  # noqa: E402

KO = Path('/home/dev/n16k64_campaign/kernel_opt')
PAPER, FREQ, OPT1B = REPO / 'sm120' / 'build', KO / 'build_freq', KO / 'build'
POLICIES = {'ours-16x64': ('tc_16x64', 'auto', PAPER), 'ours-16x64-freq': ('tc_16x64', 'auto', FREQ),
            'ours-256x64': ('tc_256x64', 'auto', PAPER), 'ours-256x64-freq': ('tc_256x64', 'auto', FREQ),
            'ours-8x64-opt': ('tc_8x64', 'auto_wB', OPT1B), 'ours-8x64-freq': ('tc_8x64', 'auto_wB', FREQ),
            'fo6': ('fo6', 'auto_stock', PAPER)}


def build_check(path, root):
    """The libraries a process loaded are builds of `root` (by sha256 against its manifests)."""
    rec = json.loads(Path(path).read_text())
    inst = rec.get('install') or {}
    shas = {}
    for d in root.iterdir():
        m = d / 'manifest.json'
        if m.exists():
            shas[json.loads(m.read_text())['library_sha256']] = d.name
    used = list((inst.get('kernel_set') or {}).get('kernels', {}).values()) or [inst.get('kernel_sha256')]
    bad = [f'library {s} is not a build of {root}' for s in used if s not in shas]
    if root == FREQ:
        for s in used:
            if s in shas:
                man = json.loads((root / shas[s] / 'manifest.json').read_text())
                if man.get('extra_defines', {}).get('MIXFP4_DISPATCH_FREQ') != 1:
                    bad.append(f'{shas[s]} in {root} was not built with MIXFP4_DISPATCH_FREQ=1')
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default=','.join(P.MODELS))
    ap.add_argument('--policies', default=','.join(POLICIES))
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--shapes', default=','.join(P.PREFILL_SHAPES))
    ap.add_argument('--reps', type=int, default=7)
    ap.add_argument('--out', type=Path, default=KO / 'opt2' / 'e2e')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'logs').mkdir(exist_ok=True)
    if not P.gpu_idle():
        P.die('the GPU is busy: latency needs an idle GPU')
    names = [p for p in POLICIES if p in args.policies.split(',')]
    for model in args.models.split(','):
        for r in range(1, args.rounds + 1):
            shift = (r - 1) % len(names)
            order = names[shift:] + names[:shift]
            P.log(args.out, f'prefill {model} round {r} order {",".join(order)}')
            for pol in order:
                kind, kernel, root = POLICIES[pol]
                out = args.out / 'prefill' / model / pol / f'round{r}.json'
                if P.complete(out):
                    bad = prefill_check(out) + build_check(out, root)
                    if bad:
                        P.die(f'registered check failed: prefill {model} {pol} round {r}: {bad}')
                    print(f'prefill {model} {pol} round {r}: done')
                    continue
                art = P.artifact(P.OUT_DEFAULT, model, kind)
                if not (art / 'artifact.json').exists():
                    P.die(f'{art} missing')
                cmd = [P.PY, REPO / 'experiments' / 'paper' / 'bench_prefill.py', '--model', model, '--label', pol,
                       '--round', str(r), '--shapes', args.shapes, '--reps', str(args.reps), '--out', out,
                       '--artifact', art, '--kernel', kernel]
                os.environ.pop('SM120_BUILD_DIR', None)
                extra = {} if root == PAPER else {'SM120_BUILD_DIR': str(root)}
                rc = P.run(args.out, f'prefill_{model}_{pol}_r{r}', cmd, extra_env=extra)
                if rc != 0 or not P.complete(out):
                    P.die(f'prefill failed: {model} {pol} round {r} (log: {args.out}/logs)')
                bad = prefill_check(out) + build_check(out, root)
                if bad:
                    P.log(args.out, f'CHECK FAILED prefill {model} {pol} round {r}: {bad}')
                    P.die(f'registered check failed: prefill {model} {pol} round {r}: {bad}')
    P.log(args.out, 'DONE M2prime prefill')


if __name__ == '__main__':
    main()
