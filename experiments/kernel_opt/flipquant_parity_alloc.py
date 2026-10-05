#!/usr/bin/env python3
"""flipquant GEMM parity, the allocation diagnostic (results/kernel_opt/flipquant_parity/NOTE.md): do the cell-level
differences of the parity run follow the device-memory placement of the timed buffers rather than the side?

    python experiments/kernel_opt/flipquant_parity_alloc.py --out JSON

The parity driver's two workers (flipquant_parity_gemm.py: fq = flipquant's real path, rz = the kernel-opt harness), on
the cells where the parity run differed and on control cells, timed in three phases with the driver's pairing (3 rounds;
the two sides of a policy back to back, the side order alternated):
  as_is      the parity run's state: fq's buffers come from the segments that transformers' allocator warmup and the
             model loads left (its install happens inside them), rz's from fresh segments;
  swapped    rz 'warm' (one cached 16 GiB segment, then its rotation copies and activation buffers allocated again, out
             of it) and fq 'rehome' (every timed buffer through host memory into fresh segments): the placements swap;
  both_fresh both 'rehome': both sides' timed buffers in fresh segments, allocated in the same order.
Each phase records the allocator segments that hold each side's buffers.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flipquant_parity_gemm as D  # noqa: E402

CELLS = [('256x64', 'q_proj', 1), ('256x64', 'k_proj', 1), ('256x64', 'v_proj', 1), ('256x64', 'gate_proj', 1),
         ('256x64', 'k_proj', 16), ('256x64', 'v_proj', 16), ('stock_wB_ko', 'gate_proj', 16),
         ('16x64', 'q_proj', 1), ('16x64', 'k_proj', 1), ('16x64', 'k_proj', 16)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='llama8b')
    ap.add_argument('--fq-root', default='/home/dev/n16k64_campaign/fqport/wt')
    ap.add_argument('--rz-build', default='/home/dev/n16k64_campaign/kernel_opt/build_V')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--warm-gib', type=float, default=16.0)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(D.RAZER / 'sm120' / 'bench'))
    import common as B
    B.require_idle()
    pols = [p for p in D.POLICIES if p in {c[0] for c in CELLS}]
    projs = sorted({c[1] for c in CELLS})
    pargs = argparse.Namespace(python=args.python, model=args.model, fq_root=str(Path(args.fq_root).resolve()),
                               fq_build=Path(args.fq_root).resolve() / 'kernels' / 'razer_sm120' / 'build_ko',
                               flush_mib=512, policies=pols, projections=projs)
    logs = Path(str(args.out) + '.logs')
    logs.mkdir(parents=True, exist_ok=True)
    base = {k: v for k, v in os.environ.items() if k not in ('SM120_BUILD_DIR', 'PYTHONPATH')}
    base.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
    res = dict(cells=[list(c) for c in CELLS], method=dict(reps=args.reps, warmup=args.warmup, rounds=args.rounds,
                                                          warm_gib=args.warm_gib),
               flipquant=D.git_state(pargs.fq_root), razer=D.git_state(D.RAZER), phases=[])
    peers = {}
    try:
        peers['rz'] = D.Peer('rz', pargs, dict(base, SM120_BUILD_DIR=args.rz_build), str(D.RAZER), logs / 'rz.log')
        rz = peers['rz']('setup')
        peers['fq'] = D.Peer('fq', pargs, base, pargs.fq_root, logs / 'fq.log')
        fq = peers['fq']('setup', modules=rz['modules'])
        assert all(rz['operands'][k]['packed_sha256'] == fq['operands'][k]['packed_sha256'] and
                   rz['operands'][k]['sf_sha256'] == fq['operands'][k]['sf_sha256'] for k in rz['operands'])
        res['modules'] = rz['modules']
        plan = [('as_is', {}), ('swapped', {'rz': ('warm', dict(gib=args.warm_gib)), 'fq': ('rehome', {})}),
                ('both_fresh', {'rz': ('rehome', {}), 'fq': ('rehome', {})})]
        for phase, ops in plan:
            seg = {}
            for side in ('rz', 'fq'):
                op, kw = ops.get(side, ('segments', {}))
                seg[side] = peers[side](op, **kw)
            row = dict(phase=phase, ops={s: ops.get(s, ('segments', {}))[0] for s in ('rz', 'fq')}, segments=seg, cells={})
            for pol, proj, t in CELLS:
                vals = {'fq': [], 'rz': []}
                rounds = {'fq': [], 'rz': []}
                for r in range(args.rounds):
                    order = ('fq', 'rz') if (r + list(D.POLICIES).index(pol)) % 2 == 0 else ('rz', 'fq')
                    for side in order:
                        got = peers[side]('time', policy=pol, proj=proj, t=t, reps=args.reps, warmup=args.warmup)
                        vals[side] += got['gemm_us']
                        rounds[side].append(statistics.median(got['gemm_us']))
                row['cells'][f'{pol}/{proj}/{t}'] = dict(
                    fq=statistics.median(vals['fq']), rz=statistics.median(vals['rz']),
                    ratio=statistics.median(vals['fq']) / statistics.median(vals['rz']),
                    round_ratios=[a / b for a, b in zip(rounds['fq'], rounds['rz'])], kernel=got['kernel'],
                    width=got['width'], schedule=got['schedule'])
            res['phases'].append(row)
            print(f"{phase}: copies' largest segment fq {seg['fq']['copies']['largest_gib']:.2f} GiB "
                  f"({seg['fq']['copies']['segments']} segments), rz {seg['rz']['copies']['largest_gib']:.2f} GiB "
                  f"({seg['rz']['copies']['segments']} segments)", flush=True)
            for key, c in row['cells'].items():
                print(f"   {key:28s} fq {c['fq']:7.2f}  rz {c['rz']:7.2f}  fq/rz {c['ratio']:.4f}  "
                      + ' '.join(f'{x:.3f}' for x in c['round_ratios']), flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(res, indent=1, default=str) + '\n')
    finally:
        for p in peers.values():
            p.close()
        res['flipquant_after'] = D.git_state(pargs.fq_root)
        res['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        args.out.write_text(json.dumps(res, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
