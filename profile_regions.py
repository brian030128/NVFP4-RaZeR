"""Named profiling regions for run_multiround.py --profile, and a GPU-time breakdown by region.

region(name) is a torch.profiler.record_function range while profiling is on (ON[0]) and a no-op
context otherwise, so the default code path runs the same operations with no profiler in it.
breakdown_trace() reads the exported Kineto trace and counts every GPU kernel and memcpy exactly once:
kernel -> its launching runtime call (same correlation id) -> the innermost named region open on the
launching thread at launch time. This also covers launches from outside PyTorch (the native GEMM is
called through ctypes) and autograd's backward thread. Kernels outside any named region are
classified by kernel name (BF16 GEMM, attention, memcpy, other); phases are the top-level regions,
matched by launch time on any thread.
"""
import json
import contextlib
from collections import defaultdict

import torch

ON = [False]


def region(name):
    return torch.profiler.record_function(name) if ON[0] else contextlib.nullcontext()


def classify(kernel_name):
    k = kernel_name.lower()
    if 'memcpy' in k:
        return 'memcpy (outside named regions)'
    if any(s in k for s in ('flash', 'fmha', 'attention', 'attn', 'sdpa')):
        return 'attention'
    if any(s in k for s in ('gemm', 'nvjet', 'cutlass', 'xmma', 'cublas', 'sm80_', 'sm90_', 'sm100_', 'sm120_')):
        return 'GEMM (model Linear/lm_head, fwd+bwd)'
    return 'other kernels (norms, rotary, elementwise, ...)'


def breakdown(events, phases, regions):
    """{phase: {category: GPU milliseconds}} plus each phase's wall time (ms)."""
    spans = {}
    for e in events:
        if e.name in phases and e.name not in spans:
            spans[e.name] = (e.time_range.start, e.time_range.end)
    out = {p: defaultdict(float) for p in phases}
    for e in events:
        if not e.kernels:
            continue
        start = e.time_range.start
        phase = next((p for p, (s, f) in spans.items() if s <= start <= f), None)
        if phase is None:
            continue
        category, a = None, e
        while a is not None:
            if a.name in regions:
                category = a.name
                break
            a = a.cpu_parent
        for k in e.kernels:
            out[phase][category or classify(k.name)] += k.duration / 1000.0
    return {p: dict(gpu_ms=dict(sorted(v.items(), key=lambda kv: -kv[1])), gpu_ms_total=sum(v.values()),
                    wall_ms=(spans[p][1] - spans[p][0]) / 1000.0 if p in spans else None)
            for p, v in out.items()}


def breakdown_trace(trace_path, phases, regions):
    """({phase: {gpu_ms: {category: ms}, gpu_ms_total, wall_ms}}, kernels counted) from a chrome trace.

    Each GPU kernel/memcpy/memset is counted once, under the innermost named region open on its
    launching thread at launch time (a sweep over that thread's properly nested region intervals)."""
    events = json.loads(open(trace_path).read())['traceEvents']
    names = set(phases) | set(regions)
    launches, spans, device = {}, {}, []
    for e in events:
        cat, ph = e.get('cat', ''), e.get('ph')
        if ph != 'X':
            continue
        if cat in ('cuda_runtime', 'cuda_driver'):
            corr = e.get('args', {}).get('correlation')
            if corr is not None:
                launches[corr] = (e['tid'], e['ts'])
        elif cat in ('user_annotation', 'cpu_op') and e['name'] in names:
            spans.setdefault(e['name'], []).append((e['tid'], e['ts'], e['ts'] + e['dur']))
        elif cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            device.append(e)
    phase_spans = {p: [(s, f) for _, s, f in spans.get(p, [])] for p in phases}
    intervals = defaultdict(list)
    for name in regions:
        for tid, s, f in spans.get(name, []):
            intervals[tid].append((s, -f, name))           # outer before inner at equal start
    queries = defaultdict(list)
    for i, k in enumerate(device):
        tid, ts = launches.get(k.get('args', {}).get('correlation'), (None, k['ts']))
        queries[tid].append((ts, i))
    category = [None] * len(device)
    for tid, items in queries.items():
        items.sort()
        ivs = sorted(intervals.get(tid, []))
        stack, j = [], 0
        for ts, i in items:
            while j < len(ivs) and ivs[j][0] <= ts:
                s, negf, name = ivs[j]
                while stack and stack[-1][1] < s:
                    stack.pop()
                stack.append((s, -negf, name))
                j += 1
            while stack and stack[-1][1] < ts:
                stack.pop()
            category[i] = stack[-1][2] if stack else None
    out = {p: {} for p in phases}
    counted = 0
    for i, k in enumerate(device):
        tid, ts = launches.get(k.get('args', {}).get('correlation'), (None, k['ts']))
        phase = next((p for p, sp in phase_spans.items() if any(s <= ts <= f for s, f in sp)), None)
        if phase is None:
            continue
        c = category[i] or classify(k['name'])
        out[phase][c] = out[phase].get(c, 0.0) + k['dur'] / 1000.0
        counted += 1
    return {p: dict(gpu_ms=dict(sorted(v.items(), key=lambda kv: -kv[1])), gpu_ms_total=sum(v.values()),
                    wall_ms=sum(f - s for s, f in phase_spans[p]) / 1000.0) for p, v in out.items()}, counted
