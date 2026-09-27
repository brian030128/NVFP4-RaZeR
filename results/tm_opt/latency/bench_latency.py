"""R2 (PROTOCOL_QR.md): prefill latency of one policy on the SM120 deployment kernels, with a profiler decomposition.

One process per (model, policy, round); the queue randomizes the policy order within each round. A policy is BF16
(the model as loaded) or an exported artifact on a kernel configuration: every scoped Linear becomes a NativeLinear
(the per-token activation quantizer and the FP4 GEMM with its fused epilogue, two launches from one ctypes call).

Timing: sm120/bench/model.py's prefill (one forward over [batch, prompt] random tokens with the KV cache written,
2 warm-ups, then --reps forwards timed with CUDA events; median, min, max).
Decomposition: --profile-iters more forwards under torch.profiler, each GPU kernel attributed to the innermost region
open on its launching thread at launch time (as profile_regions.breakdown_trace; the regions are forward hooks on
the modules), then classified:
  NativeLinear       quant_rows_kernel -> activation quantization; the CUTLASS kernel -> FP4 GEMM (per-token and
                     global scales, bias and the bf16 rounding are fused into its epilogue); anything else -> other
  BF16 nn.Linear     BF16 linear GEMM (the BF16 policy)
  lm_head            lm_head (BF16)
  *.self_attn        attention (SDPA, rotary, KV-cache write) outside its Linears
  *.linear_attn      Qwen3.5 linear attention (gated delta rule, conv1d) outside its Linears
  anything else      other (norms, MLP activation, embeddings, residual adds)
Decode (--decode, batch 1): sm120/bench/model.py's eager and CUDA-graph decode after a 512-token prompt.

python results/tm_opt/latency/bench_latency.py --model llama8b --label fo6@stock_wA --artifact DIR --kernel stock_wA \
    --round 1 --out JSON
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

SM120 = Path('/home/dev/n16k64_campaign/sm120_bench/sm120')
sys.path.insert(0, str(SM120))
sys.path.insert(0, str(SM120 / 'bench'))
import common as B  # noqa: E402  (sm120/bench/common.py)
from mixfp4_sm120 import model as NM  # noqa: E402
from mixfp4_sm120.linear import NativeLinear  # noqa: E402

PHASE = 'R2 prefill'
REGIONS = ('R2 native linear', 'R2 bf16 linear', 'R2 lm_head', 'R2 attention', 'R2 linear attention')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def region_hooks(model):
    """Forward hooks opening a record_function range per module class of interest; returns the handles."""
    head = model.get_output_embeddings()
    handles = []
    for n, m in model.named_modules():
        if isinstance(m, NativeLinear):
            name = 'R2 native linear'
        elif m is head:
            name = 'R2 lm_head'
        elif isinstance(m, torch.nn.Linear):
            name = 'R2 bf16 linear'
        elif n.endswith('.self_attn'):
            name = 'R2 attention'
        elif n.endswith('.linear_attn'):
            name = 'R2 linear attention'
        else:
            continue
        stack = []

        def pre(mod, inp, _name=name, _stack=stack):
            r = torch.profiler.record_function(_name)
            r.__enter__()
            _stack.append(r)

        def post(mod, inp, out, _stack=stack):
            _stack.pop().__exit__(None, None, None)

        handles.append(m.register_forward_pre_hook(pre))
        handles.append(m.register_forward_hook(post, always_call=True))
    return handles


def classify(region, name):
    n = name.lower()
    if region == 'R2 native linear':
        if 'quant_rows_kernel' in n:
            return 'activation quantization'
        if 'cutlass' in n or 'device_kernel' in n or 'gemm' in n:
            return 'FP4 GEMM (+ fused epilogue)'
        return 'native linear, other'
    return {'R2 bf16 linear': 'BF16 linear GEMM', 'R2 lm_head': 'lm_head (BF16)',
            'R2 attention': 'attention (SDPA, rotary, KV write)',
            'R2 linear attention': 'linear attention (gated delta rule, conv1d)'}.get(
        region, 'other (norms, MLP activation, embeddings, residual)')


def decompose(trace_path, iters):
    """Per-forward GPU milliseconds by class, the device span per forward, and the top kernels per class."""
    events = json.loads(Path(trace_path).read_text())['traceEvents']
    launches, spans, device = {}, defaultdict(list), []
    for e in events:
        cat = e.get('cat', '')
        if e.get('ph') != 'X':
            continue
        if cat in ('cuda_runtime', 'cuda_driver'):
            corr = e.get('args', {}).get('correlation')
            if corr is not None:
                launches[corr] = (e['tid'], e['ts'])
        elif cat in ('user_annotation', 'cpu_op') and (e['name'] in REGIONS or e['name'] == PHASE):
            spans[e['name']].append((e['tid'], e['ts'], e['ts'] + e['dur']))
        elif cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            device.append(e)
    phases = sorted((s, f) for _, s, f in spans[PHASE])
    assert len(phases) == iters, (len(phases), iters)
    intervals = defaultdict(list)
    for name in REGIONS:
        for tid, s, f in spans.get(name, []):
            intervals[tid].append((s, -f, name))
    queries = defaultdict(list)
    for i, k in enumerate(device):
        tid, ts = launches.get(k.get('args', {}).get('correlation'), (None, k['ts']))
        queries[tid].append((ts, i))
    region = [None] * len(device)
    launch_ts = [None] * len(device)
    for tid, items in queries.items():
        items.sort()
        ivs = sorted(intervals.get(tid, []))
        stack, j = [], 0
        for ts, i in items:
            launch_ts[i] = ts
            while j < len(ivs) and ivs[j][0] <= ts:
                s, negf, name = ivs[j]
                while stack and stack[-1][1] < s:
                    stack.pop()
                stack.append((s, -negf, name))
                j += 1
            while stack and stack[-1][1] < ts:
                stack.pop()
            region[i] = stack[-1][2] if stack else None
    ms, kernels = defaultdict(float), defaultdict(lambda: defaultdict(float))
    first, last, counted = [None] * iters, [None] * iters, 0
    for i, k in enumerate(device):
        p = next((j for j, (s, f) in enumerate(phases) if s <= launch_ts[i] <= f), None)
        if p is None:
            continue
        c = classify(region[i], k['name'])
        ms[c] += k['dur'] / 1000.0 / iters
        kernels[c][k['name'][:160]] += k['dur'] / 1000.0 / iters
        first[p] = k['ts'] if first[p] is None else min(first[p], k['ts'])
        last[p] = k['ts'] + k['dur'] if last[p] is None else max(last[p], k['ts'] + k['dur'])
        counted += 1
    span = [(f - s) / 1000.0 for s, f in zip(first, last)]
    total = sum(ms.values())
    return dict(gpu_ms=dict(sorted(ms.items(), key=lambda kv: -kv[1])), gpu_ms_total=total,
                device_span_ms=dict(median=statistics.median(span), all=span), gaps_ms=statistics.median(span) - total,
                kernels_per_forward=counted / iters,
                top_kernels={c: dict(sorted(v.items(), key=lambda kv: -kv[1])[:4]) for c, v in kernels.items()})


@torch.no_grad()
def profile_prefill(model, batch, prompt, iters, trace_path):
    ids = torch.randint(100, 20000, (batch, prompt), device='cuda', generator=torch.Generator('cuda').manual_seed(0))
    model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize()
    handles = region_hooks(model)
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                with torch.profiler.record_function(PHASE):
                    model(input_ids=ids, use_cache=True)
                torch.cuda.synchronize()
    finally:
        for h in handles:
            h.remove()
    prof.export_chrome_trace(str(trace_path))
    return decompose(trace_path, iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--artifact', default=None, help='omit for BF16')
    ap.add_argument('--kernel', default=None, help='a configuration name, or auto / auto_stock (the width-selecting sets)')
    ap.add_argument('--round', type=int, required=True)
    ap.add_argument('--prefill', default='1x512,1x2048,4x2048')
    ap.add_argument('--reps', type=int, default=7)
    ap.add_argument('--profile-iters', type=int, default=3)
    ap.add_argument('--decode', action='store_true')
    ap.add_argument('--keep-trace', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    B.require_idle()
    C = load('sm120_eval_common', SM120 / 'eval' / 'common.py')
    BM = load('sm120_bench_model', SM120 / 'bench' / 'model.py')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = dict(gpu=B.gpu_info(), model=args.model, label=args.label, artifact=args.artifact, kernel=args.kernel,
               round=args.round, pid=os.getpid(), prefill={}, decomposition={}, decode={})
    t0 = time.time()
    model = C.load_model(args.model)
    res['load_seconds'] = time.time() - t0
    if args.artifact is not None:
        t0 = time.time()
        rep = NM.install(model, args.artifact, kernel=args.kernel, loader=C.MODELS[args.model]['loader'])
        res['install_seconds'] = time.time() - t0
        res['install'] = rep.as_dict()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    res['weights_gib'] = torch.cuda.memory_allocated() / 2 ** 30
    for spec in args.prefill.split(','):
        b, p = (int(v) for v in spec.split('x'))
        try:
            res['prefill'][spec] = BM.prefill(model, b, p, reps=args.reps)
        except torch.OutOfMemoryError as e:
            res['prefill'][spec] = dict(error=f'OOM: {str(e)[:160]}')
            torch.cuda.empty_cache()
        print(args.label, spec, json.dumps(res['prefill'][spec]), flush=True)
    res['peak_prefill_gib'] = torch.cuda.max_memory_allocated() / 2 ** 30
    for spec in args.prefill.split(','):
        if 'error' in res['prefill'][spec]:
            continue
        b, p = (int(v) for v in spec.split('x'))
        trace = out.with_name(out.stem + f'.{spec}.trace.json')
        res['decomposition'][spec] = profile_prefill(model, b, p, args.profile_iters, trace)
        if not args.keep_trace:
            trace.unlink()
        print(args.label, spec, 'GPU ms', json.dumps({k: round(v, 3) for k, v in res['decomposition'][spec]['gpu_ms'].items()}), flush=True)
    if args.decode:
        try:
            res['decode']['eager_b1'], _ = BM.decode_eager(model, 1, 512, 64)
        except Exception as e:  # noqa: BLE001
            res['decode']['eager_b1'] = dict(error=f'{type(e).__name__}: {str(e)[:300]}')
        try:
            res['decode']['graph_b1'], graph_toks = BM.decode_graph(model, 1, 512, 64)
            _, ref = BM.decode_eager_tokens(model, 1, 512, graph_toks.shape[1])
            n = min(ref.shape[1], graph_toks.shape[1])
            same = ref[:, :n] == graph_toks[:, :n]
            res['decode']['graph_b1'].update(tokens_compared=n, token_match=float(same.double().mean()))
        except Exception as e:  # noqa: BLE001  (recorded, as sm120/bench/model.py does)
            res['decode']['graph_b1'] = dict(error=f'{type(e).__name__}: {str(e)[:300]}')
        print(args.label, 'decode', json.dumps(res['decode']), flush=True)
    if args.artifact is not None:
        res['coverage'] = {k: v for k, v in NM.coverage(model).items() if k != 'remaining_bf16_linears'}
        if res['install'].get('kernel_set') is not None:
            # the set is shared by every NativeLinear; after the run it holds the calls per CTA-tile width
            res['install']['kernel_set'] = next(m.kernel_set for m in NM.native_modules(model).values()).describe()
    res['peak_total_gib'] = torch.cuda.max_memory_allocated() / 2 ** 30
    res['gpu_end'] = B.gpu_info()
    B.write(out, res)


if __name__ == '__main__':
    main()
