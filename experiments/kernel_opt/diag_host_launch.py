#!/usr/bin/env python3
"""Kernel-opt diagnostic (not a registered measurement): host time to enqueue one GEMM, per build.

    python experiments/kernel_opt/diag_host_launch.py --after-root /home/dev/n16k64_campaign/kernel_opt/build --out JSON

Eager prefill at small T is host-bound, and the narrow builds' processes showed ~30 us more host time per launch than
the 128-wide ones. This times, per build, --calls enqueues of the same GEMM (the library's sm120_gemm through
Kernel.gemm_ptr, no synchronization between calls; the queue stays shallow because each GEMM is short) with
time.perf_counter, after 200 untimed calls. Recorded: the median and quartiles of the per-call host time, in blocks
of 50 calls.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import common as B  # noqa: E402
from kernel import operands  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402

BUILDS = [('n8k64_wB', None), ('n8k64_wB_m64', 'after'), ('n8k64_wB_m32', 'after'), ('n8k64_wB_m16', 'after'),
          ('n8k64_wB', 'after'), ('stock_wA', None), ('stock_wA_n64', None), ('stock_wA_n16', None), ('stock_wB', None)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--after-root', required=True)
    ap.add_argument('--n', type=int, default=4096)
    ap.add_argument('--k', type=int, default=4096)
    ap.add_argument('--t', type=int, default=128)
    ap.add_argument('--calls', type=int, default=2000)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    n, k, t = args.n, args.k, args.t
    op = operands(n, k, t, None, None, seed=1)
    res = dict(shape=[n, k, t], calls=args.calls, gpu=B.gpu_info(), note='diagnostic, not a registered measurement', builds={})
    stream = torch.cuda.current_stream().cuda_stream
    gs_x = op['gsx']
    for name, root in BUILDS:
        kern = Kernel.load(name, build_root=args.after_root if root == 'after' else None)
        y = torch.empty((t, n), dtype=torch.bfloat16, device='cuda')
        if kern.weight_operand == 0:
            call = lambda: kern.gemm_ptr(op['wp'].data_ptr(), op['wsf'].data_ptr(), op['xp'].data_ptr(),  # noqa: E731
                                         op['xsf'].data_ptr(), n, t, k, None, op['gsw'], gs_x.data_ptr(), 1.0, None, y, stream)
        else:
            call = lambda: kern.gemm_ptr(op['xp'].data_ptr(), op['xsf'].data_ptr(), op['wp'].data_ptr(),  # noqa: E731
                                         op['wsf'].data_ptr(), t, n, k, gs_x.data_ptr(), 1.0, None, op['gsw'], None, y, stream)
        for _ in range(200):
            call()
        torch.cuda.synchronize()
        blocks = []
        for _ in range(args.calls // 50):
            t0 = time.perf_counter()
            for _ in range(50):
                call()
            blocks.append((time.perf_counter() - t0) / 50 * 1e6)
            torch.cuda.synchronize()
        q = statistics.quantiles(blocks, n=4)
        res['builds'][f"{name}{'@after' if root else ''}"] = dict(host_us_median=statistics.median(blocks), q1=q[0], q3=q[2],
                                                                  library=str(kern.path))
        print(f"{name:14s} {'after' if root else 'before':6s} host us/call median {statistics.median(blocks):6.1f} "
              f"[{q[0]:.1f}, {q[2]:.1f}]", flush=True)
    B.write(args.out, res)


if __name__ == '__main__':
    main()
