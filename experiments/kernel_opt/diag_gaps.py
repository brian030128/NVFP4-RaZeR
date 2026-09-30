#!/usr/bin/env python3
"""Kernel-opt diagnostic (not a registered measurement): where the time between kernels goes, and eager host time.

    python experiments/kernel_opt/diag_gaps.py --model llama8b --shapes 1x128,1x512 \
        --after-root /home/dev/n16k64_campaign/kernel_opt/build --out JSON

Installations of the model (loaded once): the TC 8x64 artifact on n8k64_wB (sm120/build, "before") and on the
'mixed_wB' set (--after-root, "after"); the FourOverSix artifact on 'auto_stock' ("fo6") and on stock_wB ("fo6-wB").
Per installation and prompt shape:
- CUDA graph (as diag_graph_kernels.py): --reps replays under torch.profiler. For every GEMM launch, the idle time
  before it (its start minus the previous kernel's end) and after it (the next kernel's start minus its end); medians
  per build name, and the replay's total idle time.
- Eager: --reps forwards with forward pre/post hooks on every NativeLinear timing the host time inside the module
  (time.perf_counter), and the whole forward's host time (no synchronization inside; synchronized before each).
"""
import argparse
import importlib.util
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

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


def is_gemm(name):
    n = name.lower()
    return 'device_kernel' in n and 'blockscaled' in n


def module_kernel(m, t):
    return (m.kernel_set.pick(m.out_features, m.in_features, t) if m.kernel_set is not None else m.kernel).cfg.name


@torch.no_grad()
def graph_gaps(model, BP, ids, reps):
    nat = NM.native_modules(model)
    order = []
    hooks = [m.register_forward_pre_hook(lambda mod, inp, name=n: order.append(name)) for n, m in nat.items()]
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2):
            model(input_ids=ids, use_cache=True)
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    order.clear()
    g = torch.cuda.CUDAGraph()
    with BP.eager_mask_decision(), torch.cuda.graph(g):
        model(input_ids=ids, use_cache=True)
    for h in hooks:
        h.remove()
    t = ids.numel()
    builds = [module_kernel(nat[n], t) for n in order]
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            g.replay()
            torch.cuda.synchronize()
    ev = sorted((e for e in prof.events() if e.device_type.name == 'CUDA'), key=lambda e: e.time_range.start)
    per_rep = len(ev) // reps
    before, after, idle = defaultdict(list), defaultdict(list), []
    for r in range(reps):
        seq = ev[r * per_rep:(r + 1) * per_rep]
        gi = 0
        tot = 0.0
        for i, e in enumerate(seq):
            if i:
                tot += max(0.0, e.time_range.start - seq[i - 1].time_range.end)
            if is_gemm(e.name):
                b = builds[gi % len(builds)]
                gi += 1
                if i:
                    before[b].append(e.time_range.start - seq[i - 1].time_range.end)
                if i + 1 < len(seq):
                    after[b].append(seq[i + 1].time_range.start - e.time_range.end)
        idle.append(tot)
    del g
    return dict(idle_us=statistics.median(idle), launches=per_rep,
                gap_before_us={b: statistics.median(v) for b, v in before.items()},
                gap_after_us={b: statistics.median(v) for b, v in after.items()},
                gemm_launches={b: len(v) // reps for b, v in before.items()})


@torch.no_grad()
def eager_host(model, ids, reps):
    nat = NM.native_modules(model)
    spent = defaultdict(float)
    start = {}

    def pre(mod, inp, name=None):
        start[name] = time.perf_counter()

    def post(mod, inp, out, name=None):
        spent[name.rsplit('.', 1)[-1]] += time.perf_counter() - start[name]
    hooks = []
    for n, m in nat.items():
        hooks.append(m.register_forward_pre_hook(lambda mod, inp, name=n: pre(mod, inp, name)))
        hooks.append(m.register_forward_hook(lambda mod, inp, out, name=n: post(mod, inp, out, name)))
    for _ in range(2):
        model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize()
    spent.clear()
    total = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(input_ids=ids, use_cache=True)
        total.append((time.perf_counter() - t0) * 1e3)
    torch.cuda.synchronize()
    for h in hooks:
        h.remove()
    return dict(forward_host_ms=statistics.median(total),
                native_host_ms_per_forward={p: v / reps * 1e3 for p, v in spent.items()},
                native_host_ms_total=sum(spent.values()) / reps * 1e3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='llama8b')
    ap.add_argument('--shapes', default='1x128,1x512')
    ap.add_argument('--after-root', required=True)
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    C = load('sm120_eval_common', REPO / 'sm120' / 'eval' / 'common.py')
    BP = load('bench_prefill', REPO / 'experiments' / 'paper' / 'bench_prefill.py')
    model = C.load_model(args.model)
    installs = dict(before=('tc_8x64', Kernel.load('n8k64_wB')),
                    after=('tc_8x64', KernelSet('mixed_wB', build_root=args.after_root)),
                    fo6=('fo6', KernelSet('stock')), **{'fo6-wB': ('fo6', Kernel.load('stock_wB'))})
    res = dict(model=args.model, gpu=B.gpu_info(), note='diagnostic, not a registered measurement', shapes={})
    for spec in args.shapes.split(','):
        b, p = (int(v) for v in spec.split('x'))
        ids = torch.randint(100, 20000, (b, p), device='cuda', generator=torch.Generator('cuda').manual_seed(0))
        res['shapes'][spec] = {}
        for label in ('before', 'after', 'fo6', 'fo6-wB', 'after', 'before'):
            kind, kern = installs[label]
            NM.install(model, ARTIFACTS / f'{args.model}_{kind}', kernel=kern, loader=C.MODELS[args.model]['loader'])
            r = dict(graph=graph_gaps(model, BP, ids, args.reps), eager=eager_host(model, ids, args.reps))
            res['shapes'][spec].setdefault(label, []).append(r)
            print(spec, label, f"graph idle {r['graph']['idle_us']:.0f} us; gaps before GEMM "
                  + ', '.join(f'{k} {v:.1f}' for k, v in r['graph']['gap_before_us'].items())
                  + f"; eager forward {r['eager']['forward_host_ms']:.2f} ms, in NativeLinear {r['eager']['native_host_ms_total']:.2f} ms",
                  flush=True)
            B.write(args.out, res)


if __name__ == '__main__':
    main()
