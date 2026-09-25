#!/usr/bin/env python3
"""Nsight Compute profile of the GEMM kernels at chosen shapes (section 10: where the time goes).

For each (config, shape) one GEMM launch is profiled with ncu (caches flushed, clocks locked to
base by ncu's default) and a fixed metric set is extracted:
    duration, achieved TFLOP/s                          gpu__time_duration
    tensor-pipe issue: sm__inst_executed_pipe_tensor    (predicated / wasted MMA issue shows here)
    instructions issued                                 smsp__inst_executed.sum
    warp stall reasons (per issued instruction)         smsp__average_warp_latency_issue_stalled_*
    shared-memory load traffic                          l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
    DRAM / L2 throughput                                dram__throughput / lts__throughput (% of peak)

    python sm120/bench/profile_ncu.py --out sm120/results/bench/ncu_rtx5090.json
"""
import argparse
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as B  # noqa: E402

METRICS = [
    'gpu__time_duration.sum', 'sm__inst_executed_pipe_tensor.sum', 'smsp__inst_executed.sum',
    'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum', 'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'lts__throughput.avg.pct_of_peak_sustained_elapsed', 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'smsp__average_warp_latency_issue_stalled_math_pipe_throttle.ratio',
    'smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio',
    'smsp__average_warp_latency_issue_stalled_short_scoreboard.ratio',
    'smsp__average_warp_latency_issue_stalled_barrier.ratio',
    'smsp__average_warp_latency_issue_stalled_branch_resolving.ratio',
    'smsp__average_warp_latency_issue_stalled_no_instruction.ratio',
    'smsp__average_warp_latency_issue_stalled_wait.ratio',
    'smsp__average_warp_latency_issue_stalled_mio_throttle.ratio',
    'smsp__issue_active.avg.pct_of_peak_sustained_active',
]

RUNNER = r'''
import sys, torch
sys.path.insert(0, {sm120!r}); sys.path.insert(0, {bench!r})
from kernel import operands, selector_masks
from mixfp4_sm120.lib import Kernel
cfg, n, k, t, pattern = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
kern = Kernel.load(cfg)
mask = None
if pattern != 'e2m1' and kern.type_block is not None:
    g = torch.Generator().manual_seed(0)
    grid = (n // kern.type_block[0], k // kern.type_block[1])
    mask = torch.ones(grid, dtype=torch.bool) if pattern == 'all' else \
        torch.rand(grid, generator=g) < (0.5 if pattern == 'random50' else 0.0012)
op = operands(n, k, t, mask, kern.type_block if mask is not None else None, seed=1)
for _ in range(3):
    kern.gemm(op['wp'], op['wsf'], op['xp'], op['xsf'], n, t, k, scale_m_default=op['gsw'], scale_n=op['gsx'], check=False)
torch.cuda.synchronize()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='stock_wA:e2m1,n16k64_wA_nodisp:e2m1,n16k64_wA:e2m1,n16k64_wA:sparse,'
                                       'n16k64_wA:random50,n16k64_wA:all,n16k64_wA_8x1:random50,stock_wA_n16:e2m1,n16k64_wA_n16:random50')
    ap.add_argument('--shapes', default='4096x4096x4096,14336x4096x2048,4096x4096x1')
    ap.add_argument('--ncu', default=os.environ.get('NCU', '/usr/local/cuda-13.1/bin/ncu'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    B.require_idle()
    runner = Path(args.out).with_suffix('.runner.py')
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text(RUNNER.format(sm120=str(HERE.parent), bench=str(HERE)))
    res = dict(gpu=B.gpu_info(), metrics=METRICS, rows=[])
    for shape in args.shapes.split(','):
        n, k, t = (int(v) for v in shape.split('x'))
        for case in args.cases.split(','):
            cfg, pattern = case.split(':')
            cmd = [args.ncu, '-k', 'regex:device_kernel', '--launch-skip', '2', '--launch-count', '1', '--csv',
                   '--metrics', ','.join(METRICS), sys.executable, str(runner), cfg, str(n), str(k), str(t), pattern]
            out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
            lines = [ln for ln in out.splitlines() if ln.startswith('"')]
            vals = {}
            for row in csv.DictReader(io.StringIO('\n'.join(lines))):
                v = row.get('Metric Value', '').replace(',', '')
                try:
                    vals[row['Metric Name']] = float(v)
                except ValueError:
                    vals[row['Metric Name']] = v
            if 'gpu__time_duration.sum' in vals:
                unit_ns = 1.0  # ncu reports ns for gpu__time_duration.sum in CSV
                vals['tflops'] = 2.0 * n * k * t / (vals['gpu__time_duration.sum'] * unit_ns * 1e-9) / 1e12
            res['rows'].append(dict(config=cfg, pattern=pattern, out=n, inp=k, tokens=t, **vals))
            print(cfg, pattern, shape, {m.split('.')[0].split('__')[-1][:34]: vals.get(m) for m in METRICS[:4]}, flush=True)
            B.write(args.out, res)


if __name__ == '__main__':
    main()
