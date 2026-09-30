#!/usr/bin/env python3
"""Kernel-opt diagnostic (not a registered measurement): where the end-to-end prefill difference goes (#5).

    python experiments/kernel_opt/diag_e2e_split.py --model llama8b --shapes 1x2048,1x4096 \
        --after-root /home/dev/n16k64_campaign/kernel_opt/build --out JSON

The paper's consistency check found the 8x64 prefill overhead at T >= 1024 larger than the per-forward GEMM difference
predicts (by 1-2.5 pp). One hypothesis is power: a GEMM that draws more power lowers the clock of every kernel in the
forward under the 500 W cap. For each installation of one loaded model (policies below) and prompt shape, the forward is
captured in a CUDA graph and replayed --reps times back to back under torch.profiler, with an NVML sampler (SM clock,
power, the power-cap clock-event reason; every 5 ms). Per replay: the device time of the GEMMs, of the activation
quantizer, of every other kernel, the idle time between kernels, and the wall time. Policies:
  ours-8x64      TC 8x64 on n8k64_wB (sm120/build)       ours-8x64-opt  the same on the 'mixed_wB' set (--after-root)
  ours-16x64     TC 16x64 on 'auto' (sm120/build)        fo6            FourOverSix on 'auto_stock'
  fo6-wB         FourOverSix on stock_wB
The order is the list, then reversed (ABCDE EDCBA).
"""
import argparse
import importlib.util
import statistics
import sys
import threading
import time
from pathlib import Path

import pynvml
import torch
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import common as B  # noqa: E402
from mixfp4_sm120 import model as NM  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

ARTIFACTS = Path('/home/dev/n16k64_campaign/paper/artifacts')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def kind(name):
    n = name.lower()
    if 'quant_rows_kernel' in n:
        return 'quant'
    if 'device_kernel' in n and 'blockscaled' in n:
        return 'gemm'
    return 'other'


class Sampler:
    def __init__(self):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.rows, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            r = pynvml.nvmlDeviceGetCurrentClocksEventReasons(self.h)
            self.rows.append((pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM),
                              pynvml.nvmlDeviceGetPowerUsage(self.h) / 1e3, bool(r & pynvml.nvmlClocksEventReasonSwPowerCap)))
            time.sleep(0.005)

    def __enter__(self):
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join()

    def summary(self):
        sm = [r[0] for r in self.rows]
        pw = [r[1] for r in self.rows]
        return dict(samples=len(self.rows), sm_mhz_median=statistics.median(sm) if sm else None,
                    sm_mhz_min=min(sm) if sm else None, power_w_median=statistics.median(pw) if pw else None,
                    power_cap_share=sum(r[2] for r in self.rows) / len(self.rows) if self.rows else None)


@torch.no_grad()
def split(model, BP, ids, reps):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2):
            model(input_ids=ids, use_cache=True)
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with BP.eager_mask_decision(), torch.cuda.graph(g):
        model(input_ids=ids, use_cache=True)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    walls = []
    with Sampler() as smp, profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            walls.append(a.elapsed_time(b) * 1e3)
    ev = sorted((e for e in prof.events() if e.device_type.name == 'CUDA'), key=lambda e: e.time_range.start)
    per = len(ev) // reps
    parts = {k: [] for k in ('gemm', 'quant', 'other', 'idle')}
    for r in range(reps):
        seq = ev[r * per:(r + 1) * per]
        acc = dict(gemm=0.0, quant=0.0, other=0.0, idle=0.0)
        for i, e in enumerate(seq):
            acc[kind(e.name)] += e.device_time_total
            if i:
                acc['idle'] += max(0.0, e.time_range.start - seq[i - 1].time_range.end)
        for k in parts:
            parts[k].append(acc[k])
    del g
    return dict(wall_us=statistics.median(walls), **{f'{k}_us': statistics.median(v) for k, v in parts.items()},
                launches=per, telemetry=smp.summary())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='llama8b')
    ap.add_argument('--shapes', default='1x2048,1x4096')
    ap.add_argument('--after-root', required=True)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    C = load('sm120_eval_common', REPO / 'sm120' / 'eval' / 'common.py')
    BP = load('bench_prefill', REPO / 'experiments' / 'paper' / 'bench_prefill.py')
    model = C.load_model(args.model)
    pol = {'ours-8x64': ('tc_8x64', Kernel.load('n8k64_wB')),
           'ours-8x64-opt': ('tc_8x64', KernelSet('mixed_wB', build_root=args.after_root)),
           'ours-16x64': ('tc_16x64', KernelSet('mixed')), 'fo6': ('fo6', KernelSet('stock')),
           'fo6-wB': ('fo6', Kernel.load('stock_wB'))}
    order = list(pol) + list(pol)[::-1]
    res = dict(model=args.model, gpu=B.gpu_info(), note='diagnostic, not a registered measurement', order=order, shapes={})
    for spec in args.shapes.split(','):
        b, p = (int(v) for v in spec.split('x'))
        ids = torch.randint(100, 20000, (b, p), device='cuda', generator=torch.Generator('cuda').manual_seed(0))
        res['shapes'][spec] = {}
        for label in order:
            kd, kern = pol[label]
            NM.install(model, ARTIFACTS / f'{args.model}_{kd}', kernel=kern, loader=C.MODELS[args.model]['loader'])
            r = split(model, BP, ids, args.reps)
            res['shapes'][spec].setdefault(label, []).append(r)
            t = r['telemetry']
            print(f"{spec} {label:14s} wall {r['wall_us']:9.0f}  gemm {r['gemm_us']:9.0f}  quant {r['quant_us']:7.0f}  "
                  f"other {r['other_us']:9.0f}  idle {r['idle_us']:7.0f}  sm {t['sm_mhz_median']} MHz (min {t['sm_mhz_min']})  "
                  f"{t['power_w_median']:.0f} W  cap {100 * t['power_cap_share']:.0f} %", flush=True)
            B.write(args.out, res)


if __name__ == '__main__':
    main()
