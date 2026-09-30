#!/usr/bin/env python3
"""One decode-latency process (Experiment D): one model, one policy, one round, every (batch, prompt) setting.

    PAPER_PYTHON experiments/paper_extra/bench_decode.py --model llama8b --label fo6 --artifact ART --kernel auto_stock \
        --round 1 --settings 1x512,4x512,16x512,1x2048,4x2048,16x2048 --gen 64 --out JSON

- Loading and install: as bench_prefill.py (sm120/eval/common.load_model; NM.install on the given kernel; the
  width-selecting sets use the RTX PRO 6000 tile table).
- Decode: sm120/bench/model.py decode_graph, Part R's harness.
  - A prompt of batch x prompt random tokens fills a StaticCache.
  - The single-token forward is captured in a CUDA graph.
  - After 32 replays for the token check, the cache is rewound and --gen replays are timed with the host clock (one
    synchronize at the end): ms per token and tokens per second (batch x gen / seconds).
- Registered check: the graph's greedy tokens equal an eager StaticCache greedy decode of the same prompt
  (decode_eager_tokens) on the first 33 tokens.
- Recorded, not a check: the same comparison against an eager DynamicCache decode (HF's default cache).
- Widths: for a width-selecting set, the CTA width the table gives each quantized Linear at T = batch (decode); stock_wB
  and n8k64_wB have a single build.
- Coverage: every scoped Linear is a NativeLinear and ran.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import common as B  # noqa: E402
from mixfp4_sm120 import model as NM  # noqa: E402
from mixfp4_sm120.select import key_order  # noqa: E402  (int widths, then string alternatives)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def dynamic_tokens(model, batch, prompt, n):
    """Greedy tokens [first, ...] of an eager DynamicCache decode of decode_graph's prompt (seed 1)."""
    from transformers import DynamicCache
    ids = torch.randint(100, 20000, (batch, prompt), device='cuda', generator=torch.Generator('cuda').manual_seed(1))
    cache = DynamicCache()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    toks = [out.logits[:, -1:].argmax(-1)]
    for _ in range(n - 1):
        out = model(input_ids=toks[-1], past_key_values=cache, use_cache=True)
        toks.append(out.logits[:, -1:].argmax(-1))
    return torch.cat(toks, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--artifact', default=None)
    ap.add_argument('--kernel', default=None)
    ap.add_argument('--round', type=int, required=True)
    ap.add_argument('--settings', required=True, help='comma-separated BATCHxPROMPT')
    ap.add_argument('--gen', type=int, default=64)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    torch.backends.cuda.matmul.allow_tf32 = False
    C = load('sm120_eval_common', REPO / 'sm120' / 'eval' / 'common.py')
    BM = load('sm120_bench_model', REPO / 'sm120' / 'bench' / 'model.py')
    res = dict(status='running', gpu=B.gpu_info(), model=args.model, label=args.label, artifact=args.artifact,
               kernel=args.kernel, round=args.round, gen=args.gen, pid=os.getpid(),
               allocator=os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default'), decode={})
    t0 = time.time()
    model = C.load_model(args.model)
    res['load_seconds'] = time.time() - t0
    if args.artifact is not None:
        rep = NM.install(model, args.artifact, kernel=args.kernel, loader=C.MODELS[args.model]['loader'])
        res['install'] = rep.as_dict()
        assert not rep.fallback, rep.fallback
    torch.cuda.synchronize()
    res['weights_gib'] = torch.cuda.memory_allocated() / 2 ** 30
    nat = NM.native_modules(model)
    for spec in args.settings.split(','):
        b, p = (int(v) for v in spec.split('x'))
        torch.cuda.reset_peak_memory_stats()
        entry = res['decode'][spec] = {}
        try:
            stats, graph_toks = BM.decode_graph(model, b, p, args.gen)
            entry.update(stats)
            _, ref = BM.decode_eager_tokens(model, b, p, graph_toks.shape[1])
            n = min(ref.shape[1], graph_toks.shape[1])
            same = ref[:, :n] == graph_toks[:, :n]
            entry.update(tokens_compared=n, token_match=float(same.double().mean()),
                         tokens_equal_static_eager=bool(same.all()))
            dyn = dynamic_tokens(model, b, p, n)
            entry['token_match_dynamic_eager'] = float((dyn[:, :n] == graph_toks[:, :n]).double().mean())
        except Exception as e:  # noqa: BLE001  (recorded; the orchestrator stops on it)
            entry['error'] = f'{type(e).__name__}: {str(e)[:400]}'
        entry['peak_gib'] = torch.cuda.max_memory_allocated() / 2 ** 30
        if nat:
            ks = next(iter(nat.values())).kernel_set
            entry['decode_widths'] = (sorted({ks.width(m.out_features, m.in_features, b) for m in nat.values()}, key=key_order)
                                      if ks is not None else next(iter(nat.values())).kernel.cfg.name)
        torch.cuda.empty_cache()
        print(args.label, spec, json.dumps(entry), flush=True)
    if nat:
        cov = NM.coverage(model)
        res['coverage'] = {k: v for k, v in cov.items() if k != 'remaining_bf16_linears'}
        assert cov['native_called'] == cov['native'] == res['install']['native_modules'], cov
        if res['install'].get('kernel_set') is not None:
            res['install']['kernel_set'] = next(iter(nat.values())).kernel_set.describe()
    res['gpu_end'] = B.gpu_info()
    res['status'] = 'complete'
    B.write(args.out, res)


if __name__ == '__main__':
    main()
