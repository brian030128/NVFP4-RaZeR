#!/usr/bin/env python3
"""Step 03 (Experiment 1): WikiText-2 and C4 perplexity, run_ppl_deploy.py, convention (c).

    PAPER_PYTHON experiments/paper/03_ppl.py [--models ...] [--units ...] [--policies bf16,nvfp4,fo6,ours-16x64] [--smoke]

Policies (paper_common.ACCURACY_POLICIES):
  bf16          the model as loaded
  nvfp4, fo6    NativeLinear (c) on the stock weights-on-A kernels ('auto_stock', width by the tile table)
  ours-8x64     the TM-OPT+TC 8x64 artifact on n8k64_wB (weights on B)
  ours-16x64    the TM-OPT+TC 16x64 artifact on the mixed weights-on-A set ('auto')
  ours-256x64   the TM-OPT+TC 256x64 artifact as 16x64 granules on 'auto'
Every quantized Linear runs natively (run_ppl_deploy.py checks coverage in every forward). The windows are the
released protocol's: the WikiText-2 test split in 2048-token windows and 256 C4 validation crops, one per forward.
One process per (model, policy), into <out>/ppl/<model>/<policy>/report.json (per-window NLL, for the paired
comparisons of step 07). A finished (model, policy) is skipped.
--smoke: Llama-3.1-8B, every policy but 256x64, the first 4 windows of each corpus.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as P  # noqa: E402


def policies(args):
    names = [p for p in P.ACCURACY_POLICIES if P.unit_of(p) is None or P.unit_of(p) in args.units]
    if args.policies:
        names = [p for p in names if p in args.policies.split(',')]
    return names


def main():
    args = P.setup(P.parser(__doc__).parse_args())
    if args.smoke:
        args.models, args.units = ['llama8b'], ['8x64', '16x64']
    for model in args.models:
        for pol in policies(args):
            out = args.out / 'ppl' / model / pol
            report = out / 'report.json'
            if P.complete(report) and not args.force:
                print(f'{model} {pol}: done')
                continue
            kind, kernel = P.ACCURACY_POLICIES[pol]
            if kind is None:
                spec = f'{pol}=bf16'
            else:
                art = P.artifact(args.out, model, kind)
                if not (art / 'artifact.json').exists():
                    P.die(f'{art} missing: run step 02 first')
                spec = f'{pol}=native:{art}:{kernel}'
            cmd = [P.PY, 'run_ppl_deploy.py', '--model', model, '--data-root', P.DATA[model], *P.DEVIATION.get(model, []),
                   '--evaluate', spec, '--out', out]
            if args.smoke:
                cmd += ['--limit-windows', '4']
            rc = P.run(args.out, f'03_ppl_{model}_{pol}', cmd, extra_env=P.ACCURACY_ENV)
            if rc != 0 or not P.complete(report):
                P.die(f'PPL failed: {model} {pol} (log: {args.out}/logs/03_ppl_{model}_{pol}.log)')
            ev = json.loads(report.read_text())['evaluations'][pol]['evaluation']
            print(f"{model} {pol}: WikiText-2 {ev['wiki']['ppl']:.4f}, C4 {ev['c4']['ppl']:.4f}", flush=True)


if __name__ == '__main__':
    main()
