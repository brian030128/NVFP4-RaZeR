"""Controlled test: how much of the N16K64-vs-N8K64 prefill gap is the harness's output handling?

For N16K64 (weights-on-A build) and N8K64 (weights-on-B build), each policy is installed once and
the epilogue is toggled between the default three-step path (FP32 per-token scale, bf16 cast,
contiguous copy) and rq.SINGLE_PASS_EPILOGUE (one elementwise pass writing contiguous bf16), with
forwards alternating between the two modes so drift cancels. The two modes must give bit-identical
logits. Llama-3.1-8B, batch 1, T=2048, SDPA, real weights and the evaluated maps.
"""
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq  # noqa: E402
from bench_latency_llama import ROOT, categorize, load_masks  # noqa: E402
from campaign import models as MOD  # noqa: E402

REPS, WARMUP, NPROF = 12, 3, 3


def timed(model, ids):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    model(input_ids=ids, use_cache=False)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b)


@torch.no_grad()
def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    rq.VALIDATE = False
    model, _ = MOD.load_model('llama8b', attn='sdpa', device_map='cuda')
    modules = MOD.scope(model, 'llama8b')
    shapes = {n: tuple(m.weight.shape) for n, m in modules.items()}
    masks = {p: load_masks(p, shapes)[0] for p in ('n16_k3', 'n8_k3')}
    ids = torch.randint(0, model.config.vocab_size, (1, 2048), device='cuda')
    inst = rq.RealInstaller(modules, tokens=2048, check_calls=0)
    out = {}
    for label, cfg, mpol in (('N16K64 (wt_as_A)', 'wt_as_A', 'n16_k3'), ('N8K64 (b8x64)', 'b8x64', 'n8_k3')):
        inst.install('map', rq.Kernel(cfg), masks[mpol], None)
        logits = {}
        for mode in (False, True):
            rq.SINGLE_PASS_EPILOGUE = mode
            logits[mode] = model(input_ids=ids, use_cache=False).logits
        identical = torch.equal(logits[False], logits[True])
        del logits
        times = {False: [], True: []}
        for _ in range(WARMUP):
            for mode in (False, True):
                rq.SINGLE_PASS_EPILOGUE = mode
                timed(model, ids)
        for _ in range(REPS):
            for mode in (False, True):
                rq.SINGLE_PASS_EPILOGUE = mode
                times[mode].append(timed(model, ids))
        res = dict(logits_bit_identical=identical)
        for mode in (False, True):
            rq.SINGLE_PASS_EPILOGUE = mode
            cats = []
            for _ in range(NPROF):
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    model(input_ids=ids, use_cache=False)
                    torch.cuda.synchronize()
                cats.append(categorize(prof))
            cat = {k: statistics.mean(c[k] for c in cats) for k in cats[0]}
            cat['kernel_total'] = sum(cat[k] for k in ('fp4_gemm', 'bf16_gemm', 'attention', 'other'))
            wall = statistics.median(times[mode])
            cat['idle'] = wall - cat['kernel_total']
            res['single_pass' if mode else 'three_pass'] = dict(median_ms=wall, times_ms=times[mode], **cat)
        paired = [b - a for a, b in zip(times[False], times[True])]
        res['single_minus_three_ms'] = dict(median=statistics.median(paired), mean=statistics.mean(paired),
                                            sd=statistics.stdev(paired))
        out[label] = res
        rq.SINGLE_PASS_EPILOGUE = False
        print(label, json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 3) if isinstance(vv, float) else '...')
                                                                          for kk, vv in v.items()}) for k, v in res.items()}), flush=True)
    inst.remove()
    Path(ROOT / 'results/latency_epilogue_llama8b.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
