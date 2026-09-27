"""Timing and environment helpers shared by the benchmarks.

Device time is measured with CUDA events around `iters` back-to-back launches (after warmup),
repeated `reps` times; the median and the min/max over repetitions are reported. Kernel-only
numbers additionally use the torch profiler (CUPTI) to read each GEMM kernel's own duration, which
excludes launch gaps. Every result file records the GPU, clocks, power limit, driver and software.
"""
import datetime
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import torch

SM120 = Path(__file__).resolve().parents[1]
for p in (str(SM120), str(SM120.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


def gpu_info():
    q = ('name,driver_version,power.limit,power.default_limit,clocks.max.sm,clocks.max.mem,'
         'clocks.applications.graphics,persistence_mode,temperature.gpu,utilization.gpu,memory.used')
    try:
        row = subprocess.run(['nvidia-smi', f'--query-gpu={q}', '--format=csv,noheader'], stdout=subprocess.PIPE,
                             text=True).stdout.strip().splitlines()[0]
        info = dict(zip(q.split(','), [c.strip() for c in row.split(',')]))
    except (OSError, IndexError):
        info = {}
    import triton
    info.update(torch=torch.__version__, torch_cuda=torch.version.cuda, triton=triton.__version__, host=platform.node(),
                device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability()),
                sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
                utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))
    return info


def require_idle(max_util=10, max_mem_mib=2000):
    """Refuse to benchmark on a busy GPU (another process makes every number wrong)."""
    out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                         stdout=subprocess.PIPE, text=True).stdout.strip().splitlines()[0]
    util, mem = (int(x) for x in out.split(','))
    procs = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], stdout=subprocess.PIPE,
                           text=True).stdout.split()
    import os
    others = [p for p in procs if p.strip() and int(p) != os.getpid()]
    if util > max_util or others:
        raise SystemExit(f'GPU is busy (util {util}%, {mem} MiB, other compute processes {others}); refusing to benchmark')


def time_fn(fn, iters=50, reps=7, warmup=10):
    """Median / min / max milliseconds per call over `reps` repetitions of `iters` calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    per = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        b.synchronize()
        per.append(a.elapsed_time(b) / iters)
    return dict(ms=statistics.median(per), min_ms=min(per), max_ms=max(per), reps=reps, iters=iters)


def time_host(fn, iters=200, warmup=20):
    """Host-side microseconds per call when the GPU is not the bottleneck (launch/overhead cost)."""
    import time
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) / iters * 1e6


def kernel_times(fn, iters=20, warmup=5, match=None):
    """Median device duration (us) of each CUDA kernel launched by `fn`, from CUPTI."""
    from torch.profiler import ProfilerActivity, profile
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    per = {}
    for e in prof.events():
        if e.device_type.name == 'CUDA' and (match is None or match(e.name)):
            per.setdefault(e.name, []).append(e.device_time_total if hasattr(e, 'device_time_total') else e.cuda_time_total)
    return {k[:90]: dict(us=statistics.median(v), n=len(v)) for k, v in per.items()}


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + '\n')


# Linear shapes (out, in) of the evaluated models; the token count is the GEMM's N.
MODEL_SHAPES = {
    'llama8b': {'q_proj': (4096, 4096), 'k_proj': (1024, 4096), 'v_proj': (1024, 4096), 'o_proj': (4096, 4096),
                'gate_proj': (14336, 4096), 'up_proj': (14336, 4096), 'down_proj': (4096, 14336)},
    'qwen4b': {'q_proj': (4096, 2560), 'k_proj': (1024, 2560), 'v_proj': (1024, 2560), 'o_proj': (2560, 4096),
               'gate_proj': (9728, 2560), 'up_proj': (9728, 2560), 'down_proj': (2560, 9728)},
    'mistral7b': {'q_proj': (4096, 4096), 'k_proj': (1024, 4096), 'v_proj': (1024, 4096), 'o_proj': (4096, 4096),
                  'gate_proj': (14336, 4096), 'up_proj': (14336, 4096), 'down_proj': (4096, 14336)},
}
