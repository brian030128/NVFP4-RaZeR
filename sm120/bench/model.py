#!/usr/bin/env python3
"""C. Full-model benchmark: prefill latency, decode throughput and peak memory.

One process per (model, policy) so that peak memory is the policy's own. Policies:
    bf16                   the unmodified model
    native:<artifact>[:k]  every scoped Linear replaced by NativeLinear on kernel config k

Prefill: one forward over [batch, prompt] tokens with a KV cache being written (use_cache=True),
median of repeats, CUDA events. Decode: greedy generation of `--gen` tokens after a prompt, per
token, in two modes:
    eager   a Python loop over model forward with a DynamicCache (includes all host overhead);
    graph   the single-token forward captured in a CUDA graph over a StaticCache (host overhead
            removed); reported as unsupported with the reason if capture fails.
Peak memory: torch.cuda.max_memory_allocated after load and after each phase.

    python sm120/bench/model.py --model qwen4b --policy native:sm120/artifacts/qwen4b_n16_k3 \
        --out sm120/results/bench/model_qwen4b_native_n16.json
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as B  # noqa: E402  (bench/common.py; eval/common.py is loaded by path below)
from mixfp4_sm120 import model as NM  # noqa: E402


def _eval_common():
    import importlib.util
    spec = importlib.util.spec_from_file_location('eval_common', B.SM120 / 'eval' / 'common.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def prefill(model, batch, prompt, reps=5):
    ids = torch.randint(100, 20000, (batch, prompt), device='cuda', generator=torch.Generator('cuda').manual_seed(0))
    for _ in range(2):
        model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        model(input_ids=ids, use_cache=True)
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return dict(ms=ts[len(ts) // 2], min_ms=ts[0], max_ms=ts[-1], tokens_per_s=batch * prompt / (ts[len(ts) // 2] / 1e3))


@torch.no_grad()
def decode_eager(model, batch, prompt, gen):
    from transformers import DynamicCache
    ids = torch.randint(100, 20000, (batch, prompt), device='cuda', generator=torch.Generator('cuda').manual_seed(1))
    cache = DynamicCache()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1:].argmax(-1)
    for _ in range(3):  # warm the single-token path
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    toks = []
    t0 = time.perf_counter()
    for _ in range(gen):
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1:].argmax(-1)
        toks.append(nxt)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dict(ms_per_token=dt / gen * 1e3, tokens_per_s=batch * gen / dt), torch.cat(toks, 1)


def rewind(cache, s_pos, pos):
    """Set the next write position of a StaticCache and the graph's position input together.

    transformers 5.x StaticLayer writes at its own device counter `cumulative_length` (advanced in
    place by every update, including warm-up and capture calls), not at the cache_position it is
    given; RoPE and the mask use cache_position. Both must point at the same slot."""
    for layer in getattr(cache, 'layers', []):
        if hasattr(layer, 'cumulative_length') and torch.is_tensor(layer.cumulative_length):
            layer.cumulative_length.fill_(pos)
    s_pos.fill_(pos)


@torch.no_grad()
def decode_eager_tokens(model, batch, prompt, n):
    """Greedy tokens [first, ...] of an eager StaticCache decode (the reference for the graph run)."""
    from transformers import StaticCache
    L = prompt + n + 8
    try:
        cache = StaticCache(config=model.config, max_batch_size=batch, max_cache_len=L, device='cuda', dtype=torch.bfloat16)
    except TypeError:
        cache = StaticCache(config=model.config, max_cache_len=L)
    ids = torch.randint(100, 20000, (batch, prompt), device='cuda', generator=torch.Generator('cuda').manual_seed(1))
    out = model(input_ids=ids, past_key_values=cache, cache_position=torch.arange(prompt, device='cuda'), use_cache=True)
    toks = [out.logits[:, -1:].argmax(-1)]
    for i in range(n - 1):
        out = model(input_ids=toks[-1], past_key_values=cache, cache_position=torch.tensor([prompt + i], device='cuda'),
                    use_cache=True)
        toks.append(out.logits[:, -1:].argmax(-1))
    return None, torch.cat(toks, 1)


@torch.no_grad()
def decode_graph(model, batch, prompt, gen):
    from transformers import StaticCache
    L = prompt + gen + 8
    try:
        cache = StaticCache(config=model.config, max_batch_size=batch, max_cache_len=L, device='cuda', dtype=torch.bfloat16)
    except TypeError:
        cache = StaticCache(config=model.config, max_cache_len=L)
    ids = torch.randint(100, 20000, (batch, prompt), device='cuda', generator=torch.Generator('cuda').manual_seed(1))
    out = model(input_ids=ids, past_key_values=cache, cache_position=torch.arange(prompt, device='cuda'), use_cache=True)
    s_ids = out.logits[:, -1:].argmax(-1).clone()
    s_pos = torch.tensor([prompt], device='cuda')

    def step():
        o = model(input_ids=s_ids, past_key_values=cache, cache_position=s_pos, use_cache=True)
        return o.logits[:, -1:].argmax(-1)

    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        nxt = step()
    # replay: feed the argmax back and advance the position in place
    # correctness: replay from the prompt's first token and compare with an eager greedy decode
    first = out.logits[:, -1:].argmax(-1)
    s_ids.copy_(first)
    rewind(cache, s_pos, prompt)
    toks = []
    for i in range(min(gen, 32)):
        g.replay()
        toks.append(nxt.clone())
        s_ids.copy_(nxt)
        s_pos.add_(1)
    s_ids.copy_(first)
    rewind(cache, s_pos, prompt)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(gen):
        g.replay()
        s_ids.copy_(nxt)
        s_pos.add_(1)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dict(ms_per_token=dt / gen * 1e3, tokens_per_s=batch * gen / dt), torch.cat([first] + toks, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--policy', required=True, help='bf16 | native:<artifact>[:kernel]')
    ap.add_argument('--prefill', default='1x128,1x512,1x2048,4x512,4x2048,16x512')
    ap.add_argument('--decode', default='1,4,16')
    ap.add_argument('--decode-prompt', type=int, default=512)
    ap.add_argument('--gen', type=int, default=64)
    ap.add_argument('--out', required=True)
    ap.add_argument('--allow-busy', action='store_true')
    args = ap.parse_args()
    if not args.allow_busy:
        B.require_idle()
    C = _eval_common()
    res = dict(gpu=B.gpu_info(), model=args.model, policy=args.policy, prefill={}, decode_eager={}, decode_graph={})
    model = C.load_model(args.model)
    if args.policy != 'bf16':
        _, art, *k = args.policy.split(':')
        rep = NM.install(model, art, kernel=k[0] if k else 'auto', loader=C.MODELS[args.model]['loader'])
        res['install'] = rep.as_dict()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    res['weights_gib'] = torch.cuda.memory_allocated() / 2 ** 30
    for spec in args.prefill.split(','):
        b, p = (int(v) for v in spec.split('x'))
        try:
            res['prefill'][spec] = prefill(model, b, p)
        except torch.OutOfMemoryError as e:
            res['prefill'][spec] = dict(error=f'OOM: {str(e)[:120]}')
            torch.cuda.empty_cache()
        print(args.policy, 'prefill', spec, res['prefill'][spec], flush=True)
    res['peak_prefill_gib'] = torch.cuda.max_memory_allocated() / 2 ** 30
    for b in [int(v) for v in args.decode.split(',')]:
        res['decode_eager'][b], eager_toks = decode_eager(model, b, args.decode_prompt, args.gen)
        print(args.policy, 'decode eager', b, res['decode_eager'][b], flush=True)
        try:
            res['decode_graph'][b], graph_toks = decode_graph(model, b, args.decode_prompt, args.gen)
            # eager_toks start after the prompt's argmax + 3 warm-up tokens; compare from the start instead
            _, ref = decode_eager_tokens(model, b, args.decode_prompt, graph_toks.shape[1])
            n = min(ref.shape[1], graph_toks.shape[1])
            same = (ref[:, :n] == graph_toks[:, :n])
            prefix = int(same.long().cumprod(1).sum(1).min())
            res['decode_graph'][b].update(tokens_compared=n, token_match=float(same.double().mean()), min_identical_prefix=prefix)
        except Exception as e:  # noqa: BLE001
            res['decode_graph'][b] = dict(error=f'{type(e).__name__}: {str(e)[:300]}')
        print(args.policy, 'decode graph', b, res['decode_graph'][b], flush=True)
        B.write(args.out, res)
    res['peak_total_gib'] = torch.cuda.max_memory_allocated() / 2 ** 30
    if args.policy != 'bf16':
        res['coverage'] = {k: v for k, v in NM.coverage(model).items() if k != 'remaining_bf16_linears'}
    B.write(args.out, res)


if __name__ == '__main__':
    main()
