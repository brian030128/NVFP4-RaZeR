#!/usr/bin/env python3
"""Deviation 2 of results/paper/PROTOCOL.md: GEMM latency with isolated launches and cold weights, the primary GEMM table.

    PAPER_PYTHON experiments/paper/06b_gemm_isolated.py [--models ...] [--smoke]

Protocol: results/paper/PROTOCOL_GEMM_ISOLATED.md. Step 06 (bench_gemm.py: CUPTI, 20 back-to-back calls, L2-warm
weights of the densest module) stays as the alternative method; its records are untouched.
- **Step 1, the cold-cache verification** (check_l2_cold.py, Llama-3.1-8B), into <out>/gemm_isolated/l2_check.json. Its
  pass rule is registered: a failure stops here, before any model is timed.
- **Step 2:** one bench_gemm_isolated.py process per model, into <out>/gemm_isolated/<model>.json (with
  <model>.json.telemetry.csv). Every text projection, T = 128 ... 8192, the CONFIGS of bench_gemm_isolated.py.
- A finished step or model is skipped. Refuses to start on a busy GPU.
- **--smoke:** Llama-3.1-8B, q_proj only, T = 128 and 2048, 5 repetitions per round, the check with 10, into
  PAPER_SMOKE_OUT.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as P  # noqa: E402

TOKENS = '128,256,512,1024,2048,4096,8192'
KINDS = ('fo6', 'nvfp4', 'tc_8x64', 'tc_16x64', 'tc_256x64')


def main():
    ap = P.parser(__doc__)
    ap.add_argument('--tokens', default=TOKENS)
    ap.add_argument('--projections', default=None)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--check-reps', type=int, default=50)
    args = P.setup(ap.parse_args())
    if args.smoke:
        args.models, args.tokens, args.projections, args.reps, args.check_reps = ['llama8b'], '128,2048', 'q_proj', 5, 10
    if not P.gpu_idle():
        P.die('the GPU is busy: latency needs an idle GPU')
    gdir = args.out / 'gemm_isolated'
    gdir.mkdir(exist_ok=True)
    arts = P.OUT_DEFAULT                     # the paper run's artifacts (the smoke reuses them too)
    check = gdir / 'l2_check.json'
    if not (P.complete(check) and not args.force):
        cmd = [P.PY, Path(__file__).resolve().parent / 'check_l2_cold.py', '--fo6', P.artifact(arts, 'llama8b', 'fo6'),
               '--tc16', P.artifact(arts, 'llama8b', 'tc_16x64'), '--reps', str(args.check_reps), '--out', check]
        rc = P.run(args.out, '06b_l2_check', cmd)
        if rc != 0 or not P.complete(check):
            P.log(args.out, 'CHECK FAILED cold-cache verification (step 1 of deviation 2)')
            P.die(f'the cold-cache verification failed (registered check; log: {args.out}/logs/06b_l2_check.log)')
    for model in args.models:
        out = gdir / f'{model}.json'
        if P.complete(out) and not args.force:
            print(f'{model}: done')
            continue
        cmd = [P.PY, Path(__file__).resolve().parent / 'bench_gemm_isolated.py', '--model', model, '--tokens', args.tokens,
               '--reps', str(args.reps), '--rounds', str(args.rounds), '--out', out]
        for kind in KINDS:
            art = P.artifact(arts, model, kind)
            if not (art / 'artifact.json').exists():
                P.die(f'{art} missing: run step 02 first')
            cmd += ['--artifact', f'{kind}={art}']
        if args.projections:
            cmd += ['--projections', args.projections]
        rc = P.run(args.out, f'06b_gemm_{model}', cmd)
        if rc != 0 or not P.complete(out):
            P.log(args.out, f'CHECK FAILED or error: isolated GEMM latency {model}')
            P.die(f'isolated GEMM latency failed: {model} (log: {args.out}/logs/06b_gemm_{model}.log)')


if __name__ == '__main__':
    main()
