#!/usr/bin/env python3
"""Exploratory (disclosed, not registered), item 3, follow-up of quick_O.py: is the outlier a cold-weight effect at short K?

    python results/kernel_opt/O/exploration/quick_O2.py --out JSON

At a fixed tile count, K varies: 4096 x K x T=2048 (512 tiles, 2.72 passes of 188 CTAs) and 14336 x K x T=512 (448
tiles, 2.38 passes), K in {2048, 4096, 8192, 14336}. Each kernel of quick_O.py's sweep (stock_ko, stock_wB_ko, ceil16,
A16, ceil8, A8; default scheduler setting; f = 0) is timed cold (M1's condition) and warm (back-to-back on L2-resident
weights), 3 rotated rounds. If the excess over stock is a cold-weight latency exposed at the pass boundaries, it should
shrink as K grows (the weight stream spreads over a longer tile) and vanish warm.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quick_O as O  # noqa: E402
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

NAMES = ['stock_ko', 'stock_wB_ko', 'ceil16', 'A16', 'ceil8', 'A8']
CASES = [(4096, k, 2048) for k in (2048, 4096, 8192, 14336)] + [(14336, k, 512) for k in (2048, 4096, 8192)]


def warm_us(nm, op, n, k, t):
    q = O.act(nm, op)()
    fn = lambda: O.launcher(nm, op, n, k, t, (0, 1))(op['wp'], op['wsf'], q)  # noqa: E731
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(30):
            fn()
        torch.cuda.synchronize()
    d = [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]
    return statistics.median(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    res = []
    for n, k, t in CASES:
        o = {nm: O.ops(nm, n, k, t) for nm in NAMES}
        per = {m: {nm: [] for nm in NAMES} for m in ('cold', 'warm')}
        for r in range(3):
            for nm in O.rotate(NAMES, r, 3):
                per['cold'][nm].append(O.cold_us(O.launcher(nm, o[nm], n, k, t, (0, 1)), o[nm]['wp'], o[nm]['wsf'],
                                                 iters=30, act=O.act(nm, o[nm])))
                per['warm'][nm].append(warm_us(nm, o[nm], n, k, t))
        med = {m: {nm: statistics.median(v) for nm, v in d.items()} for m, d in per.items()}
        res.append(dict(n=n, k=k, t=t, tiles=O.tiles('stock_ko', n, k, t), us=med, rounds=per))
        for m in ('cold', 'warm'):
            ref = med[m]['stock_ko']
            print(f'{n}x{k} T={t} {m:4s} stock_ko {ref:8.2f} us | ' + ' '.join(
                f'{nm} {100 * (med[m][nm] / ref - 1):+6.2f} %' for nm in NAMES[1:]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(cases=res, kernels={nm: dict(config=c, root=r) for nm, (c, r) in O.K.items()}),
                                   indent=1) + '\n')


if __name__ == '__main__':
    main()
