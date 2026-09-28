#!/usr/bin/env python3
"""Step 06 (Experiment 3.2): GEMM latency per text-Linear shape, and the activation quantizer (supplementary).

    PAPER_PYTHON experiments/paper/06_gemm_latency.py [--models ...] [--units ...] [--tokens ...] [--smoke]

One bench_gemm.py process per model (its docstring has the configurations), into <out>/gemm/<model>.json.
- **Tokens:** T = 128 ... 8192, the prefill shapes' batch x prompt; 4x2048 is T = 8192.
- **Kernel time:** CUPTI, the median of 20 calls.
- **Comparisons:**
  - TM-OPT+TC 16x64 and 256x64 (n16k64_wA through the tile table) against stock_wA;
  - TM-OPT+TC 8x64 (n8k64_wB) against stock_wB (same placement) and stock_wA;
  - FourOverSix and NVFP4 share the stock GEMM.
- **Per-forward sums:** step 07. A finished model is skipped. Refuses to start on a busy GPU.
- **--smoke:** Llama-3.1-8B, q_proj only, T = 128 and 2048, without the 256x64 artifact.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as P  # noqa: E402

TOKENS = '128,256,512,1024,2048,4096,8192'


def main():
    ap = P.parser(__doc__)
    ap.add_argument('--tokens', default=TOKENS)
    ap.add_argument('--projections', default=None)
    ap.add_argument('--iters', type=int, default=20)
    args = P.setup(ap.parse_args())
    if args.smoke:
        args.models, args.units, args.tokens, args.projections = ['llama8b'], ['8x64', '16x64'], '128,2048', 'q_proj'
    if not P.gpu_idle():
        P.die('the GPU is busy: latency needs an idle GPU')
    for model in args.models:
        out = args.out / 'gemm' / f'{model}.json'
        if P.complete(out) and not args.force:
            print(f'{model}: done')
            continue
        kinds = ['fo6', 'nvfp4'] + [f'tc_{u}' for u in args.units]
        cmd = [P.PY, Path(__file__).resolve().parent / 'bench_gemm.py', '--model', model, '--tokens', args.tokens,
               '--iters', str(args.iters), '--out', out]
        for kind in kinds:
            art = P.artifact(args.out, model, kind)
            if not (art / 'artifact.json').exists():
                P.die(f'{art} missing: run step 02 first')
            cmd += ['--artifact', f'{kind}={art}']
        if args.projections:
            cmd += ['--projections', args.projections]
        rc = P.run(args.out, f'06_gemm_{model}', cmd)
        if rc != 0 or not P.complete(out):
            P.die(f'GEMM latency failed: {model} (log: {args.out}/logs/06_gemm_{model}.log)')


if __name__ == '__main__':
    main()
