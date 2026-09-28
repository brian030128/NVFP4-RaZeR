#!/usr/bin/env python3
"""One prefill-latency process (step 05): one model, one policy, one round, every prompt shape.

    PAPER_PYTHON experiments/paper/bench_prefill.py --model llama8b --label fo6 --artifact ART --kernel auto_stock \
        --round 1 --shapes 1x128,1x2048 --out JSON [--act-override nvfp4_rows] [--no-graph]

- **Loading:** the Part R harness's (results/tm_opt/latency/bench_latency.py), from this repository's sm120.
  - The model comes from sm120/eval/common.load_model (BF16, SDPA, on the GPU).
  - NativeLinear is installed from the artifact on the given kernel: a configuration name, or 'auto' / 'auto_stock',
    the width-selecting sets with the RTX PRO 6000 tile table.
  - --act-override switches only the activation quantizer. It is used for the latency-only "Ours with NVFP4
    activations" policies, and the install record says so (activation_quantizer_override).
- **Eager:** sm120/bench/model.py prefill, the function Part R timed. It runs one forward over [batch, prompt] random
  tokens with the KV cache written and the logits returned. After 2 warm-ups, --reps forwards are timed with CUDA
  events; the median, min and max are recorded. The host's enqueue time of the same forward is also recorded (the
  median of 3 more forwards).
- **CUDA graph:** the same forward, captured once and replayed. Timing: 2 warm-up replays, then --reps replays timed
  with CUDA events. The graph's logits are compared with an eager forward on the same tokens, and must be equal
  bitwise (`logits_equal_eager`). A capture that fails is recorded with its error, and no later shape is captured in
  that process.
  - **transformers treats CUDA-graph capture as tracing** (`utils.import_utils.is_tracing`). Its masking code then
    builds an explicit causal mask, which takes SDPA off the flash kernel with is_causal=True. The captured forward
    would be a different computation: slower, and not equal to eager (the first smoke run showed both).
  - **So, during capture only, `transformers.masking_utils.is_tracing` ignores stream capture.** The mask decision
    is then the eager one. For these static, unpadded prefills that decision is exact, and it needs no host sync (there
    is no padding mask). Nothing else in transformers changes, and the bitwise logits check confirms the result.
- **Host-bound flag:** set when the graph is more than 5% faster than eager (eager_ms > 1.05 graph_ms), i.e. host
  overhead is a material part of the eager time. Without a graph, it is set when the host's enqueue time reaches 90%
  of the eager time (the GPU waits for the host).
- **Coverage:** every scoped Linear must be a NativeLinear and must have run.
"""
import argparse
import contextlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import common as B  # noqa: E402  (sm120/bench/common.py)
from mixfp4_sm120 import linear as NL  # noqa: E402
from mixfp4_sm120 import model as NM  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tokens(batch, prompt):
    # the same tokens as sm120/bench/model.py prefill (seed 0), so the graph's logits compare with eager ones
    return torch.randint(100, 20000, (batch, prompt), device='cuda', generator=torch.Generator('cuda').manual_seed(0))


@torch.no_grad()
def enqueue_ms(model, batch, prompt, reps):
    ids = tokens(batch, prompt)
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(input_ids=ids, use_cache=True)
        ts.append((time.perf_counter() - t0) * 1e3)
    torch.cuda.synchronize()
    return statistics.median(ts)


@contextlib.contextmanager
def eager_mask_decision():
    """During capture, transformers.masking_utils.is_tracing without its CUDA-stream-capture condition (see above)."""
    import transformers.masking_utils as MU
    from transformers.utils import import_utils as IU
    original = MU.is_tracing

    def is_tracing(tensor=None):
        t = IU.is_torchdynamo_compiling() or IU.is_jit_tracing()
        if tensor is not None:
            t = t or IU.is_torch_fx_proxy(tensor) or IU.is_fake_tensor(tensor) or IU.is_jax_jitting(tensor)
        return t
    MU.is_tracing = is_tracing
    try:
        yield
    finally:
        MU.is_tracing = original


@torch.no_grad()
def prefill_graph(model, batch, prompt, reps):
    ids = tokens(batch, prompt)
    ref = model(input_ids=ids, use_cache=True).logits.clone()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2):
            model(input_ids=ids, use_cache=True)
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with eager_mask_decision(), torch.cuda.graph(g):
        logits = model(input_ids=ids, use_cache=True).logits
    g.replay()
    torch.cuda.synchronize()
    equal = bool(torch.equal(logits, ref))
    diff = 0.0 if equal else float((logits.float() - ref.float()).abs().max())
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    del g, logits, ref
    NL.clear_quant_cache()             # it may hold the capture's scratch
    torch.cuda.empty_cache()
    return dict(ms=ts[len(ts) // 2], min_ms=ts[0], max_ms=ts[-1], tokens_per_s=batch * prompt / (ts[len(ts) // 2] / 1e3),
                logits_equal_eager=equal, max_abs_diff_eager=diff)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--artifact', default=None, help='omit for BF16')
    ap.add_argument('--kernel', default=None)
    ap.add_argument('--act-override', default=None, choices=('nvfp4_rows', 'four_over_six_rows'))
    ap.add_argument('--round', type=int, required=True)
    ap.add_argument('--shapes', required=True, help='comma-separated BATCHxPROMPT')
    ap.add_argument('--reps', type=int, default=7)
    ap.add_argument('--no-graph', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    torch.backends.cuda.matmul.allow_tf32 = False
    C = load('sm120_eval_common', REPO / 'sm120' / 'eval' / 'common.py')
    BM = load('sm120_bench_model', REPO / 'sm120' / 'bench' / 'model.py')
    res = dict(status='running', gpu=B.gpu_info(), model=args.model, label=args.label, artifact=args.artifact,
               kernel=args.kernel, act_override=args.act_override, round=args.round, reps=args.reps, pid=os.getpid(),
               allocator=os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default'), eager={}, graph={}, host_bound={},
               graph_capture_note='during capture, transformers.masking_utils.is_tracing ignores stream capture (the '
                                  'eager mask decision: flash SDPA with is_causal=True); logits compared bitwise with eager')
    t0 = time.time()
    model = C.load_model(args.model)
    res['load_seconds'] = time.time() - t0
    if args.artifact is not None:
        t0 = time.time()
        rep = NM.install(model, args.artifact, kernel=args.kernel, loader=C.MODELS[args.model]['loader'],
                         activation_quantizer=args.act_override)
        res['install_seconds'] = time.time() - t0
        res['install'] = rep.as_dict()
        assert not rep.fallback, rep.fallback
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    res['weights_gib'] = torch.cuda.memory_allocated() / 2 ** 30
    shapes = args.shapes.split(',')
    for spec in shapes:
        b, p = (int(v) for v in spec.split('x'))
        try:
            res['eager'][spec] = BM.prefill(model, b, p, reps=args.reps)
            res['eager'][spec]['enqueue_ms'] = enqueue_ms(model, b, p, min(3, args.reps))
        except torch.OutOfMemoryError as e:
            res['eager'][spec] = dict(error=f'OOM: {str(e)[:160]}')
            torch.cuda.empty_cache()
        print(args.label, spec, 'eager', json.dumps(res['eager'][spec]), flush=True)
    res['peak_eager_gib'] = torch.cuda.max_memory_allocated() / 2 ** 30
    capture_error = None
    for spec in shapes if not args.no_graph else ():
        if 'error' in res['eager'][spec]:
            continue
        if capture_error is not None:
            res['graph'][spec] = dict(error='not attempted: an earlier capture in this process failed')
            continue
        b, p = (int(v) for v in spec.split('x'))
        try:
            res['graph'][spec] = prefill_graph(model, b, p, args.reps)
        except Exception as e:  # noqa: BLE001  (recorded; the eager numbers stand)
            capture_error = f'{type(e).__name__}: {str(e)[:400]}'
            res['graph'][spec] = dict(error=capture_error)
        print(args.label, spec, 'graph', json.dumps(res['graph'][spec]), flush=True)
    for spec in shapes:
        e, g = res['eager'][spec], res['graph'].get(spec, {})
        if 'error' in e:
            continue
        if 'ms' in g:
            res['host_bound'][spec] = dict(flag=e['ms'] > 1.05 * g['ms'], rule='eager > 1.05 x graph',
                                           graph_saves=1 - g['ms'] / e['ms'])
        else:
            res['host_bound'][spec] = dict(flag=e['enqueue_ms'] >= 0.9 * e['ms'], rule='enqueue >= 0.9 x eager (no graph)',
                                           enqueue_over_eager=e['enqueue_ms'] / e['ms'])
    res['capture_error'] = capture_error
    if args.artifact is not None:
        cov = NM.coverage(model)
        res['coverage'] = {k: v for k, v in cov.items() if k != 'remaining_bf16_linears'}
        # every scoped Linear was replaced (a strict install) and ran
        assert cov['native_called'] == cov['native'] == res['install']['native_modules'], cov
        if res['install'].get('kernel_set') is not None:
            res['install']['kernel_set'] = next(m.kernel_set for m in NM.native_modules(model).values()).describe()
        # the quantization cache's reuse per projection (q/k/v and gate/up read one input): step 07's per-forward
        # quantizer sums use it
        reuse = {}
        for n, m in NM.native_modules(model).items():
            r = reuse.setdefault(n.rsplit('.', 1)[-1], dict(modules=0, calls=0, quant_reused=0))
            r['modules'] += 1
            r['calls'] += m.calls
            r['quant_reused'] += m.quant_reused
        res['quant_reuse'] = reuse
    res['peak_total_gib'] = torch.cuda.max_memory_allocated() / 2 ** 30
    res['gpu_end'] = B.gpu_info()
    res['status'] = 'complete'
    B.write(args.out, res)


if __name__ == '__main__':
    main()
