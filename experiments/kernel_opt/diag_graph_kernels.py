#!/usr/bin/env python3
"""Kernel-opt diagnostic (not a registered measurement): per-kernel device times inside a CUDA-graph prefill.

    python experiments/kernel_opt/diag_graph_kernels.py --model llama8b --shapes 1x256,1x512,1x1024 \
        --after-root /home/dev/n16k64_campaign/kernel_opt/build --out JSON

The model is loaded once. For each installation of its TC 8x64 artifact, n8k64_wB from sm120/build ("before") and
the 'mixed_wB' set from --after-root ("after"), and each prompt shape, the forward is captured in a CUDA graph
(bench_prefill.py's capture, with the eager mask decision) and replayed --reps times under torch.profiler. Every
kernel launch of a replay is attributed, in launch order, to the NativeLinear that issued it (the order of the
module calls recorded during capture) or to "other". Recorded: per projection and installation, the median over
replays of the summed GEMM device time, and each replay's total device time and wall time (CUDA events).
Purpose: to see whether a width the cold isolated tuning picked is slower inside a back-to-back forward.
"""
import argparse
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
sys.path.insert(0, str(REPO / 'experiments' / 'paper'))
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
    # the SM120 block-scaled GEMMs of the kernel libraries (not cuBLAS's BF16 GEMMs of the head, not attention)
    n = name.lower()
    return 'device_kernel' in n and 'blockscaled' in n


@torch.no_grad()
def profile_shape(model, BP, batch, prompt, reps):
    ids = torch.randint(100, 20000, (batch, prompt), device='cuda', generator=torch.Generator('cuda').manual_seed(0))
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
    calls = list(order)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    walls = []
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            walls.append(a.elapsed_time(b))
    ev = sorted((e for e in prof.events() if e.device_type.name == 'CUDA'), key=lambda e: e.time_range.start)
    gemms = [e for e in ev if is_gemm(e.name)]
    per_rep = len(gemms) // reps
    assert per_rep == len(calls) and len(gemms) == reps * per_rep, (len(gemms), reps, len(calls))
    by_proj = defaultdict(list)
    for r in range(reps):
        acc = defaultdict(float)
        for name, e in zip(calls, gemms[r * per_rep:(r + 1) * per_rep]):
            acc[name.rsplit('.', 1)[-1]] += e.device_time_total
        for p, v in acc.items():
            by_proj[p].append(v)
    total = sum(e.device_time_total for e in ev) / reps
    widths = defaultdict(int)
    for n in calls:
        m = nat[n]
        k = m.kernel_set.pick(m.out_features, m.in_features, batch * prompt) if m.kernel_set is not None else m.kernel
        widths[f"{n.rsplit('.', 1)[-1]}:{k.cfg.name}"] += 1
    del g
    return dict(gemm_us_by_proj={p: statistics.median(v) for p, v in by_proj.items()},
                gemm_us=statistics.median(sum(v[r] for v in by_proj.values()) for r in range(reps)),
                device_us_all_kernels=total, wall_ms=statistics.median(walls), wall_ms_all=walls,
                builds=dict(widths), launches_per_replay=len(ev) // reps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='llama8b')
    ap.add_argument('--shapes', default='1x256,1x512,1x1024')
    ap.add_argument('--after-root', required=True)
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    C = load('sm120_eval_common', REPO / 'sm120' / 'eval' / 'common.py')
    BP = load('bench_prefill', REPO / 'experiments' / 'paper' / 'bench_prefill.py')
    model = C.load_model(args.model)
    kernels = dict(before=Kernel.load('n8k64_wB'), after=KernelSet('mixed_wB', build_root=args.after_root))
    res = dict(model=args.model, gpu=B.gpu_info(), note='diagnostic, not a registered measurement', shapes={})
    order = ['before', 'after', 'after', 'before']          # ABBA per shape
    for spec in args.shapes.split(','):
        b, p = (int(v) for v in spec.split('x'))
        res['shapes'][spec] = {}
        for i, label in enumerate(order):
            NM.install(model, ARTIFACTS / f'{args.model}_tc_8x64', kernel=kernels[label], loader=C.MODELS[args.model]['loader'])
            r = profile_shape(model, BP, b, p, args.reps)
            res['shapes'][spec][f'{label}_{i}'] = r
            print(spec, label, f"wall {r['wall_ms']:.3f} ms, GEMM {r['gemm_us']:.0f} us", flush=True)
        B.write(args.out, res)


if __name__ == '__main__':
    main()
