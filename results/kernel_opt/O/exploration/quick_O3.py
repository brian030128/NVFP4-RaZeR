#!/usr/bin/env python3
"""Exploratory (disclosed, not registered), item 3, second follow-up: does the K = 4096 cold excess depend on where the
weights sit in memory?

    python results/kernel_opt/O/exploration/quick_O3.py --out JSON

quick_O2.py found the excess of the no-dispatch ceilings over stock only cold and only at K = 4096 (2 KB weight rows),
and earlier cold runs of the same cell disagreed by up to 5 points. Here the cold weight copies (M1's rotation, K x size
>= 4x L2) are carved from one buffer at a byte offset o (every copy at slot start + o; the scales 1 KB-aligned after the
weights), for o in OFFSETS, at 4096x4096 T=2048 and the control 4096x8192 T=2048. Each (offset, kernel): cold, 2 rotated
rounds x 20, activations quantized after the flush, default scheduler setting, f = 0. If the gap moves with o, the
outlier is an address-pattern effect of the cold weight stream, not a property of the kernels.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quick_O as O  # noqa: E402
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

NAMES = ['stock_ko', 'ceil16', 'A16', 'ceil8']
OFFSETS = [0, 1024, 2048, 4096, 8192, 16384, 65536, 196608]
CASES = [(4096, 4096, 2048), (4096, 8192, 2048)]
FLUSH = torch.ones(512 * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
L2 = torch.cuda.get_device_properties(0).L2_cache_size


def carve(op, offset):
    """K copies of (wp, wsf) in one uint8 buffer, each starting at its slot + offset."""
    wp, wsf = op['wp'], op['wsf']
    nb_w, nb_s = wp.numel() * wp.element_size(), wsf.numel() * wsf.element_size()
    gap = -(-nb_w // 1024) * 1024
    slot = -(-(offset + gap + nb_s) // 65536) * 65536
    n = math.ceil(4 * L2 / (nb_w + nb_s)) + 1
    buf = torch.empty(n * slot, dtype=torch.uint8, device='cuda')
    out = []
    for i in range(n):
        a = i * slot + offset
        w = buf[a:a + nb_w].view(wp.dtype).view(wp.shape)
        s = buf[a + gap:a + gap + nb_s].view(wsf.dtype).view(wsf.shape)
        w.copy_(wp)
        s.copy_(wsf)
        out.append((w, s))
    return buf, out


def cold(nm, op, n, k, t, copies, iters=20, warmup=3):
    launch = O.launcher(nm, op, n, k, t, (0, 1))
    act = O.act(nm, op)

    def one(i):
        FLUSH.sum()
        torch.cuda.synchronize()
        q = act()
        torch.cuda.synchronize()
        launch(*copies[i % len(copies)], q)
        torch.cuda.synchronize()
    for i in range(warmup):
        one(i)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            one(warmup + i)
    d = [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]
    assert len(d) == iters, len(d)
    return statistics.median(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    res = []
    for n, k, t in CASES:
        o = {nm: O.ops(nm, n, k, t) for nm in NAMES}
        # the carved copies compute exactly what the original operands do
        for nm in NAMES:
            q = O.act(nm, o[nm])()
            ref = O.launcher(nm, o[nm], n, k, t, (0, 1))(o[nm]['wp'], o[nm]['wsf'], q).clone()
            _, cp = carve(o[nm], 3072)
            got = O.launcher(nm, o[nm], n, k, t, (0, 1))(cp[1][0], cp[1][1], q)
            assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), (n, k, t, nm)
            del cp
        for off in OFFSETS:
            bufs, copies = {}, {}
            for nm in NAMES:
                bufs[nm], copies[nm] = carve(o[nm], off)
            per = {nm: [] for nm in NAMES}
            for r in range(2):
                for nm in O.rotate(NAMES, r, 2):
                    per[nm].append(cold(nm, o[nm], n, k, t, copies[nm]))
            med = {nm: statistics.median(v) for nm, v in per.items()}
            res.append(dict(n=n, k=k, t=t, offset=off, us=med, rounds=per))
            ref = med['stock_ko']
            print(f'{n}x{k} T={t} offset {off:6d} | stock_ko {ref:7.2f} us | ' + ' '.join(
                f'{nm} {100 * (med[nm] / ref - 1):+6.2f} %' for nm in NAMES[1:]), flush=True)
            del bufs, copies
            torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(rows=res, offsets=OFFSETS), indent=1) + '\n')


if __name__ == '__main__':
    main()
