#!/usr/bin/env python3
"""flipquant GEMM parity, a diagnostic (results/kernel_opt/flipquant_parity/NOTE.md): does the GEMM time of a cell
depend on where its buffers sit in device memory? One process of one side, the parity driver's Worker and its timing
(flipquant_parity_gemm.py), on fixed operands and a fixed kernel.

    python experiments/kernel_opt/flipquant_parity_placement.py --side rz --out JSON

Per trial, every cell is timed as in the parity run (3 rounds of --reps launches after --warmup, the median of all), then
the buffers are placed again before the next trial, alternately:
- 'weights': the rotation copies are cloned again after a pad allocation (the old copies freed afterwards), so the
  weights the GEMM streams sit at other addresses;
- 'activations': the input pool, the quantizer's outputs and the GEMM output are reallocated after a pad allocation.
The pad sizes are seeded. Trial 0 is the parity run's placement (copies made in setup).
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flipquant_parity_gemm as D  # noqa: E402

CELLS = [('256x64', 'q_proj', 1), ('256x64', 'k_proj', 1), ('256x64', 'v_proj', 1), ('256x64', 'k_proj', 16),
         ('256x64', 'gate_proj', 1), ('stock_wB_ko', 'gate_proj', 16), ('16x64', 'q_proj', 1), ('16x64', 'k_proj', 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--side', choices=('rz', 'fq'), default='rz')
    ap.add_argument('--model', default='llama8b')
    ap.add_argument('--fq-root', default='/home/dev/n16k64_campaign/fqport/wt')
    ap.add_argument('--trials', type=int, default=9)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    pols = sorted({c[0] for c in CELLS}, key=list(D.POLICIES).index)
    projs = sorted({c[1] for c in CELLS})
    wargs = argparse.Namespace(side=args.side, model=args.model, fq_root=args.fq_root,
                               fq_build=Path(args.fq_root) / 'kernels' / 'razer_sm120' / 'build_ko', flush_mib=512,
                               policies=pols, projections=projs)
    if args.side == 'fq':
        raise SystemExit('the fq side needs the module names of an rz setup; this diagnostic runs on rz')
    import torch
    sys.path.insert(0, str(D.RAZER / 'sm120' / 'bench'))
    import common as B
    B.require_idle()
    w = D.Worker(wargs)
    setup = w.setup({})
    rng = random.Random(args.seed)
    pads, res = [], dict(side=args.side, cells=[list(c) for c in CELLS], method=vars(args) | {'out': str(args.out)},
                          gpu=setup.get('gpu'), trials=[])
    for trial in range(args.trials):
        kind = 'setup' if trial == 0 else ('weights' if trial % 2 else 'activations')
        if kind == 'weights':
            pads.append(torch.empty(rng.randrange(1, 64) * 2 ** 20 + rng.randrange(0, 2 ** 20, 512), dtype=torch.uint8,
                                    device='cuda'))
            new = {k: [(p.clone(), s.clone()) for p, s in v] for k, v in w.copies.items()}
            w.copies = new
            del new
        elif kind == 'activations':
            w.pool_key = w.pool = w.x = None
            pads.append(torch.empty(rng.randrange(1, 4096) * 512, dtype=torch.uint8, device='cuda'))
        row = dict(trial=trial, placement=kind, cells={})
        for pol, proj, t in CELLS:
            vals = []
            for r in range(args.rounds):
                got = w.time(dict(policy=pol, proj=proj, t=t, reps=args.reps, warmup=args.warmup))
                vals += got['gemm_us']
            row['cells'][f'{pol}/{proj}/{t}'] = dict(median=statistics.median(vals), kernel=got['kernel'], width=got['width'],
                                                     schedule=got['schedule'])
        res['trials'].append(row)
        print(trial, kind, ' '.join(f"{k}={v['median']:.2f}" for k, v in row['cells'].items()), flush=True)
    res['summary'] = {}
    for key in res['trials'][0]['cells']:
        v = [t['cells'][key]['median'] for t in res['trials']]
        res['summary'][key] = dict(min=min(v), max=max(v), spread=(max(v) - min(v)) / statistics.median(v),
                                   by_placement={p: [t['cells'][key]['median'] for t in res['trials'] if t['placement'] == p]
                                                 for p in ('setup', 'weights', 'activations')})
        print(f"{key}: min {min(v):.2f} max {max(v):.2f} spread {res['summary'][key]['spread'] * 100:.1f} %")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
