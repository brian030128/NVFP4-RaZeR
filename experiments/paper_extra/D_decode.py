#!/usr/bin/env python3
"""Experiment D: decode latency with a CUDA graph over a StaticCache (results/paper_extra/D/PROTOCOL.md).

    PAPER_PYTHON experiments/paper_extra/D_decode.py [--models llama8b,mistral7b,phi4] [--policies ...] [--rounds 5]

- Policies (the paper run's artifacts, PAPER_OUT/artifacts):
    bf16                   the model as loaded
    nvfp4, fo6             NVFP4 / FourOverSix on the stock weights-on-A set ('auto_stock', RTX PRO 6000 tile table)
    ours-16x64             TM-OPT+TC 16x64 on the mixed weights-on-A set ('auto')
    ours-8x64              TM-OPT+TC 8x64 on n8k64_wB
    nvfp4-wB, fo6-wB       NVFP4 / FourOverSix on stock_wB, the 8x64 same-placement references
- Settings: batch {1, 4, 16} x prompt {512, 2048}, then 64 generated tokens (bench_decode.py).
- Protocol: one process per (model, policy, round) into <D_OUT>/decode/<model>/<policy>/round<r>.json; 5 rounds, the
  policy order shuffled per round (seed 20260929 + round, logged); the default caching allocator; an idle GPU.
- Registered check (stops the run): in every setting, the graph's first 33 greedy tokens equal an eager StaticCache
  decode's, and no setting failed.
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'paper'))
import paper_common as P  # noqa: E402

OUT = Path(os.environ.get('PAPER_EXTRA_OUT', '/home/dev/n16k64_campaign/paper_extra')) / 'D'
SEED = 20260929
SETTINGS = '1x512,4x512,16x512,1x2048,4x2048,16x2048'
POLICIES = {'bf16': (None, None), 'nvfp4': ('nvfp4', 'auto_stock'), 'fo6': ('fo6', 'auto_stock'),
            'ours-16x64': ('tc_16x64', 'auto'), 'ours-8x64': ('tc_8x64', 'n8k64_wB'),
            'nvfp4-wB': ('nvfp4', 'stock_wB'), 'fo6-wB': ('fo6', 'stock_wB')}


def check(path):
    r = json.loads(Path(path).read_text())
    return [f"{s}: {e.get('error') or 'graph tokens differ from eager StaticCache decode (match %s)' % e.get('token_match')}"
            for s, e in r['decode'].items() if e.get('error') or not e.get('tokens_equal_static_eager')]


def main():
    ap = P.parser(__doc__)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--settings', default=SETTINGS)
    args = ap.parse_args()
    if args.smoke:
        args.rounds, args.settings = 1, '1x512,4x2048'
    args.out = args.out or (P.SMOKE_OUT / 'extra_D' if args.smoke else OUT)
    args.models = args.models if args.models != ','.join(P.MODELS) else 'llama8b,mistral7b,phi4'
    args = P.setup(args)
    if not P.gpu_idle():
        P.die('the GPU is busy: latency needs an idle GPU')
    names = [p for p in POLICIES if not args.policies or p in args.policies.split(',')]
    for model in (['llama8b'] if args.smoke else args.models):
        for r in range(1, args.rounds + 1):
            order = list(names)
            random.Random(SEED + r).shuffle(order)
            P.log(args.out, f'D {model} round {r} order {",".join(order)} (seed {SEED + r})')
            for pol in order:
                out = args.out / 'decode' / model / pol / f'round{r}.json'
                if P.complete(out) and not args.force:
                    bad = check(out)
                    if bad:
                        P.die(f'registered check failed (decode graph tokens): {model} {pol} round {r}: {bad}')
                    print(f'{model} {pol} round {r}: done')
                    continue
                cmd = [P.PY, Path(__file__).resolve().parent / 'bench_decode.py', '--model', model, '--label', pol,
                       '--round', str(r), '--settings', args.settings, '--out', out]
                kind, kernel = POLICIES[pol]
                if kind is not None:
                    art = P.artifact(P.OUT_DEFAULT, model, kind)
                    if not (art / 'artifact.json').exists():
                        P.die(f'{art} missing (the paper run exports it)')
                    cmd += ['--artifact', art, '--kernel', kernel]
                rc = P.run(args.out, f'D_{model}_{pol}_r{r}', cmd)
                if rc != 0 or not P.complete(out):
                    P.die(f'decode latency failed: {model} {pol} round {r} (log: {args.out}/logs)')
                bad = check(out)
                if bad:
                    P.log(args.out, f'CHECK FAILED decode graph tokens: {model} {pol} round {r}: {bad}')
                    P.die(f'registered check failed (decode graph tokens): {model} {pol} round {r}: {bad}')


if __name__ == '__main__':
    main()
