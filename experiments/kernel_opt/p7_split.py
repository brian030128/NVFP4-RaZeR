#!/usr/bin/env python3
"""Kernel-opt amendment 15 (the 8x64 plan's P7): the in-graph split with the SM clock, D4's method on the adopted 8x64 path.

    python experiments/kernel_opt/p7_split.py --model llama8b --shapes 1x2048,1x4096 --passes 1 --out JSON
    python experiments/kernel_opt/p7_split.py --model phi4 --shapes 1x512 --passes 3 --out JSON

The method is D4's (diag_e2e_split.py; results/kernel_opt/dispatch/SUMMARY.md). For each installation of one loaded model
and each prompt shape, the forward is captured in a CUDA graph and replayed --reps times back to back under
torch.profiler, with an NVML sampler every 5 ms (SM clock, power, the power-cap clock-event reason). Per replay it
records the device time of the GEMMs, of the activation quantizer and of every other kernel, the idle time between
kernels, and the wall time. Each capture is one row.

Policies, loaded explicitly (cum8_e2e.py's):
  ours-8x64-paper    TC 8x64 on n8k64_wB (sm120/build)
  ours-8x64-adopted  TC 8x64 on 'mixed_wB_ko' (build_P3freq, the tracked adopted table)
  fo6-ko             FourOverSix on 'stock_ko' (build_7, the adopted table)
  fo6-wB-ko          FourOverSix on 'stock_wB_ko' (build_P3freq, the adopted table)
One pass is the list then reversed (ABCD DCBA). --passes repeats it, so each policy is captured 2 x passes times. That
is how the per-capture idle bimodality D4 found shows up.
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
KO = Path('/home/dev/n16k64_campaign/kernel_opt')
PAPER, B7, BP3F = REPO / 'sm120' / 'build', KO / 'build_7', KO / 'build_P3freq'


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
    ap.add_argument('--passes', type=int, default=1)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    C = load('sm120_eval_common', REPO / 'sm120' / 'eval' / 'common.py')
    BP = load('bench_prefill', REPO / 'experiments' / 'paper' / 'bench_prefill.py')
    model = C.load_model(args.model)
    pol = {'ours-8x64-paper': ('tc_8x64', Kernel.load('n8k64_wB', build_root=PAPER)),
           'ours-8x64-adopted': ('tc_8x64', KernelSet('mixed_wB_ko', build_root=BP3F)),
           'fo6-ko': ('fo6', KernelSet('stock_ko', build_root=B7)),
           'fo6-wB-ko': ('fo6', KernelSet('stock_wB_ko', build_root=BP3F))}
    for name, (_, k) in pol.items():
        if isinstance(k, KernelSet):
            assert str(k.table_source).endswith('.ko.json'), (name, k.table_source)
    assert all(k.manifest.get('extra_defines', {}).get('MIXFP4_DISPATCH_FREQ') == 1
               for k in pol['ours-8x64-adopted'][1].kernels.values())
    order = (list(pol) + list(pol)[::-1]) * args.passes
    res = dict(model=args.model, gpu=B.gpu_info(), protocol='results/kernel_opt/PROTOCOL.md, amendment 15 (the in-graph split)',
               order=order, reps=args.reps,
               kernels={n: (k.describe() if isinstance(k, KernelSet) else dict(kernel=k.cfg.name, sha256=k.sha256))
                        for n, (_, k) in pol.items()}, shapes={})
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
    res['status'] = 'complete'
    B.write(args.out, res)


if __name__ == '__main__':
    main()
