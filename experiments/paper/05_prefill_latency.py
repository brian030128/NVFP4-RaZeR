#!/usr/bin/env python3
"""Step 05 (Experiment 3.1): end-to-end prefill latency on the SM120 kernels, eager and CUDA-graph captured.

    PAPER_PYTHON experiments/paper/05_prefill_latency.py [--models ...] [--units ...] [--policies ...] \
        [--rounds 5] [--shapes 1x128,...,4x2048] [--smoke]

Policies (paper_common.LATENCY_POLICIES; artifacts from step 02):
  bf16                  reference: the model as loaded
  nvfp4, fo6            NVFP4 / FourOverSix on the stock weights-on-A set ('auto_stock', RTX PRO 6000 tile table),
                        with their own activation quantizers (nvfp4_rows / four_over_six_rows)
  nvfp4-wB, fo6-wB      the same artifacts on stock_wB: the same-placement references of the 8x64 maps
  ours-<unit>           TM-OPT+TC as deployed: FourOverSix activations; 8x64 on n8k64_wB, 16x64 and 256x64 on 'auto'
  ours-<unit>-nvfp4act  LATENCY ONLY: the same TM-OPT+TC artifact with the activation quantizer switched to
                        nvfp4_rows (NM.install activation_quantizer=, recorded as an override). Not an evaluated
                        configuration: the maps were calibrated with FourOverSix activations.
Prompt shapes (batch x prompt): 1x128 ... 1x8192 and 4x2048 (--shapes).

**Registered check:** every shape is captured, and the graph's logits equal eager's bitwise; a failure stops the step.

**Protocol (Part R's):**
- one process per (model, policy, round), into <out>/latency/<model>/<policy>/round<r>.json;
- --rounds rounds, the policy order shuffled per round (seed 20260928 + round, recorded in commands.log);
- the reported value is the median over rounds of each process's median.
- A finished (model, policy, round) is skipped. Refuses to start on a busy GPU; runs with the default caching allocator.

**--smoke:** Llama-3.1-8B, every policy but 256x64, one round, 1x128 and 1x2048, 3 repetitions.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as P  # noqa: E402

SEED = 20260928


def graph_check(path):
    """The registered check: every shape was captured, and the graph's logits equal eager's bitwise. Returns the
    failures (empty when it passed)."""
    r = json.loads(Path(path).read_text())
    bad = [f"{s}: {g.get('error') or 'logits differ from eager (max |diff| %s)' % g.get('max_abs_diff_eager')}"
           for s, g in r['graph'].items() if g.get('error') or not g.get('logits_equal_eager')]
    bad += [f'{s}: not captured' for s in r['eager'] if s not in r['graph'] and 'error' not in r['eager'][s]]
    return bad


def policies(args):
    names = ['bf16'] + [p for p in P.LATENCY_POLICIES
                        if (P.unit_of(p) is None or P.unit_of(p) in args.units) and not (p.endswith('-wB') and '8x64' not in args.units)]
    if args.policies:
        names = [p for p in names if p in args.policies.split(',')]
    return names


def main():
    ap = P.parser(__doc__)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--shapes', default=','.join(P.PREFILL_SHAPES))
    ap.add_argument('--reps', type=int, default=7)
    ap.add_argument('--no-graph', action='store_true', help='eager only')
    args = P.setup(ap.parse_args())
    if args.smoke:
        args.models, args.units, args.rounds, args.shapes, args.reps = ['llama8b'], ['8x64', '16x64'], 1, '1x128,1x2048', 3
    if not P.gpu_idle():
        P.die('the GPU is busy: latency needs an idle GPU')
    for model in args.models:
        names = policies(args)
        for r in range(1, args.rounds + 1):
            order = list(names)
            random.Random(SEED + r).shuffle(order)
            P.log(args.out, f'05 {model} round {r} order {",".join(order)} (seed {SEED + r})')
            for pol in order:
                out = args.out / 'latency' / model / pol / f'round{r}.json'
                if P.complete(out) and not args.force:
                    bad = [] if args.no_graph else graph_check(out)
                    if bad:                              # a recorded failure stays a failure
                        P.die(f'registered check failed (CUDA graph = eager logits): {model} {pol} round {r}: {bad}')
                    print(f'{model} {pol} round {r}: done')
                    continue
                cmd = [P.PY, Path(__file__).resolve().parent / 'bench_prefill.py', '--model', model, '--label', pol,
                       '--round', str(r), '--shapes', args.shapes, '--reps', str(args.reps), '--out', out]
                if pol != 'bf16':
                    kind, kernel, override = P.LATENCY_POLICIES[pol]
                    art = P.artifact(args.out, model, kind)
                    if not (art / 'artifact.json').exists():
                        P.die(f'{art} missing: run step 02 first')
                    cmd += ['--artifact', art, '--kernel', kernel] + (['--act-override', override] if override else [])
                if args.no_graph:
                    cmd.append('--no-graph')
                rc = P.run(args.out, f'05_prefill_{model}_{pol}_r{r}', cmd)
                if rc != 0 or not P.complete(out):
                    P.die(f'prefill latency failed: {model} {pol} round {r} (log: {args.out}/logs)')
                bad = [] if args.no_graph else graph_check(out)
                if bad:
                    P.log(args.out, f'CHECK FAILED graph = eager: {model} {pol} round {r}: {bad}')
                    P.die(f'registered check failed (CUDA graph = eager logits): {model} {pol} round {r}: {bad}')


if __name__ == '__main__':
    main()
