#!/usr/bin/env python3
"""Kernel-opt amendment 15 (the 8x64 plan's P7, the cumulative registered 8x64 run): end-to-end CUDA-graph prefill and
decode, the adopted 8x64 path against the paper's 8x64 kernel and the tuned stocks with the weights on A and on B.

    python experiments/kernel_opt/cum8_e2e.py --what prefill,decode [--models ...] [--rounds 5] [--out DIR]

Protocol: results/kernel_opt/PROTOCOL.md (amendment 15). It is amendment 9's run (cum_e2e.py) with 8x64 policies. The
per-process scripts are the paper's, unchanged: experiments/paper/bench_prefill.py (step 05) and
experiments/paper_extra/bench_decode.py (Experiment D), with ab_e2e.py's checks.
- Policies (artifacts from the paper run, PAPER_OUT/artifacts); each names its kernel, its build directory and what the
  install must report (a set's family, or a single build's configuration):
    ours-8x64-paper      TM-OPT+TC 8x64 on 'paper_wB' (the paper's n8k64_wB) from sm120/build
    ours-8x64-adopted    the same artifact on 'auto' from build_P3freq: 'mixed_wB_ko' (t0, #2's dispatch, the adopted
                         table's 8x64 widths; the SASS of build_P2freq, amendment 12's gate)
    fo6-ko               FourOverSix on 'auto_stock' from build_7: 'stock_ko' (#4 + the 4b widths + scheduler rows)
    fo6-wB-ko            FourOverSix on 'auto_stock_wB' from build_P3freq: 'stock_wB_ko' (stock_wB_e64 + its rows)
- Prefill: every model, the paper's prompt shapes (1x128 ... 1x8192, 4x2048), 7 repetitions per process.
- Decode: llama8b, mistral7b, phi4 (Experiment D's models; the harness does not support Qwen3.8-27B's hybrid cache),
  D's settings (batch 1, 4, 16 x prompt 512, 2048), 64 generated tokens.
- Order: one process per (model, policy, round); round r (1-based) runs the policy list rotated by r - 1.
- Registered checks (a failure stops the run):
  - prefill: every shape captured, and the graph's logits equal eager's bitwise;
  - decode: the graph's first 33 greedy tokens equal an eager StaticCache decode's in every setting;
  - every library a process loaded is a build of its policy's build directory (by sha256 against the manifests);
  - build_P3freq's 8x64 builds carry MIXFP4_DISPATCH_FREQ=1, and no other loaded library does;
  - the install reports the policy's set family on the adopted table, or the policy's single build.
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
from ab_e2e import decode_check, prefill_check  # noqa: E402

KO = Path('/home/dev/n16k64_campaign/kernel_opt')
PAPER, B7, BP3F = REPO / 'sm120' / 'build', KO / 'build_7', KO / 'build_P3freq'
# policy -> (artifact kind, kernel, build directory, set family or None, single build or None)
POLICIES = {'ours-8x64-paper': ('tc_8x64', 'paper_wB', PAPER, None, 'n8k64_wB'),
            'ours-8x64-adopted': ('tc_8x64', 'auto', BP3F, 'mixed_wB_ko', None),
            'fo6-ko': ('fo6', 'auto_stock', B7, 'stock_ko', None),
            'fo6-wB-ko': ('fo6', 'auto_stock_wB', BP3F, 'stock_wB_ko', None)}
DECODE_MODELS = ('llama8b', 'mistral7b', 'phi4')
DECODE_SETTINGS = '1x512,4x512,16x512,1x2048,4x2048,16x2048'


def build_check(path, root, family, single):
    """The libraries a process loaded are builds of `root` (by sha256 against its manifests), with #2's define exactly
    on build_P3freq's 8x64 builds, and the install is `family` on the adopted table or the single build `single`."""
    rec = json.loads(Path(path).read_text())
    inst = rec.get('install') or {}
    ks = inst.get('kernel_set') or {}
    man = {}
    for d in root.iterdir():
        m = d / 'manifest.json'
        if m.exists():
            mm = json.loads(m.read_text())
            man[mm['library_sha256']] = mm
    used = list((ks.get('kernels') or {}).values()) if family else [inst.get('kernel_sha256')]
    bad = [] if all(used) and used else ['no kernel recorded']
    for s in used:
        if s not in man:
            bad.append(f'library {s} is not a build of {root}')
            continue
        freq = (man[s].get('extra_defines') or {}).get('MIXFP4_DISPATCH_FREQ')
        want = 1 if (root == BP3F and man[s]['config'].startswith('n8k64_wB')) else None
        if freq != want:
            bad.append(f"{man[s]['config']} in {root}: MIXFP4_DISPATCH_FREQ={freq}")
        if single and man[s]['config'] != single:
            bad.append(f"installed {man[s]['config']}, expected {single}")
    if family:
        if ks.get('family') != family:
            bad.append(f"installed family {ks.get('family')}, expected {family} (routing: {inst.get('routing')})")
        if not str(ks.get('table', '')).endswith('.ko.json'):
            bad.append(f"table {ks.get('table')} for family {family}")
    elif ks:
        bad.append(f'a kernel set was installed, expected the single build {single}')
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--what', default='prefill,decode')
    ap.add_argument('--models', default=','.join(P.MODELS))
    ap.add_argument('--policies', default=','.join(POLICIES))
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--shapes', default=','.join(P.PREFILL_SHAPES))
    ap.add_argument('--settings', default=DECODE_SETTINGS)
    ap.add_argument('--reps', type=int, default=7)
    ap.add_argument('--out', type=Path, default=KO / 'w8p7' / 'e2e')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'logs').mkdir(exist_ok=True)
    if not P.gpu_idle():
        P.die('the GPU is busy: latency needs an idle GPU')
    names = [p for p in POLICIES if p in args.policies.split(',')]
    for what in args.what.split(','):
        models = [m for m in args.models.split(',') if what == 'prefill' or m in DECODE_MODELS]
        check = prefill_check if what == 'prefill' else decode_check
        for model in models:
            for r in range(1, args.rounds + 1):
                shift = (r - 1) % len(names)
                order = names[shift:] + names[:shift]
                P.log(args.out, f'{what} {model} round {r} order {",".join(order)}')
                for pol in order:
                    kind, kernel, root, family, single = POLICIES[pol]
                    out = args.out / what / model / pol / f'round{r}.json'
                    if P.complete(out):
                        bad = check(out) + build_check(out, root, family, single)
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
                    os.environ.pop('SM120_BUILD_DIR', None)
                    extra = {} if root == PAPER else {'SM120_BUILD_DIR': str(root)}
                    rc = P.run(args.out, f'{what}_{model}_{pol}_r{r}', cmd, extra_env=extra)
                    if rc != 0 or not P.complete(out):
                        P.die(f'{what} failed: {model} {pol} round {r} (log: {args.out}/logs)')
                    bad = check(out) + build_check(out, root, family, single)
                    if bad:
                        P.log(args.out, f'CHECK FAILED {what} {model} {pol} round {r}: {bad}')
                        P.die(f'registered check failed: {what} {model} {pol} round {r}: {bad}')
        P.log(args.out, f'DONE {what}')


if __name__ == '__main__':
    main()
