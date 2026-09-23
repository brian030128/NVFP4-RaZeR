"""Prefill latency with the fused Triton activation quantizer (fused_quant.py) on the native kernels.

Llama-3.1-8B, batch 1, T=2048, SDPA, real weights and the evaluated maps; single-pass epilogue
everywhere (rq.SINGLE_PASS_EPILOGUE). Policies are packed once and switched per forward, and each
quantized policy runs with the PyTorch reference quantizer and with the fused kernel, interleaved:
  BF16                       unquantized reference
  NVFP4 (stock)              stock CUTLASS NVFP4 GEMM, nvfp4_rows activations
  4Over6 (stock)             stock CUTLASS NVFP4 GEMM, four_over_six_rows activations
  N8K64 (b8x64)              weights on B, 8x64 granule
  N16K64 (wt_as_A_colD)      weights on A, 16x64 granule, column-major D (no transpose)
The fused path must give logits bit-identical to the reference path for every policy.
"""
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq  # noqa: E402
from bench_latency_llama import ROOT, kernel_rows, load_masks  # noqa: E402
from campaign import models as MOD  # noqa: E402

REPS, WARMUP, NPROF = 10, 2, 2
POLICIES = [('NVFP4 (stock)', 'nvfp4', 'stock', None),
            ('4Over6 (stock)', 'four_over_six', 'stock', None),
            ('N8K64 (b8x64)', 'map', 'b8x64', 'n8_k3'),
            ('N16K64 (wt_as_A_colD)', 'map', 'wt_as_A_colD', 'n16_k3')]


def categorize(prof):
    cat = dict(fp4_gemm=0.0, act_quant=0.0, bf16_gemm=0.0, attention=0.0, other=0.0)
    for key, count, t in kernel_rows(prof):
        low = key.lower()
        if 'cutlass13device_kernel' in low:
            cat['fp4_gemm'] += t
        elif '_quant_rows_kernel' in low:
            cat['act_quant'] += t
        elif any(s in low for s in ('flash', 'fmha', 'attention')):
            cat['attention'] += t
        elif any(s in low for s in ('gemm', 'xmma', 'cutlass', 'cublas', 'nvjet')):
            cat['bf16_gemm'] += t
        else:
            cat['other'] += t
    return {k: v / 1e3 for k, v in cat.items()}


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
    rq.SINGLE_PASS_EPILOGUE = True
    model, _ = MOD.load_model('llama8b', attn='sdpa', device_map='cuda')
    modules = MOD.scope(model, 'llama8b')
    shapes = {n: tuple(m.weight.shape) for n, m in modules.items()}
    masks = {p: load_masks(p, shapes)[0] for p in ('n16_k3', 'n8_k3')}
    ids = torch.randint(0, model.config.vocab_size, (1, 2048), device='cuda')
    pristine = {n: m.forward for n, m in modules.items()}
    inst = rq.RealInstaller(modules, tokens=2048, check_calls=0)
    forwards = {'BF16': pristine}
    for label, kind, cfg, mpol in POLICIES:
        inst.install(kind, rq.Kernel(cfg), None if mpol is None else masks[mpol], None)
        forwards[label] = {n: m.forward for n, m in modules.items()}

    def activate(label):
        for n, m in modules.items():
            m.forward = forwards[label][n]

    identical = {}
    for label, *_ in POLICIES:
        activate(label)
        rq.FUSED_ACT_QUANT = False
        ref = model(input_ids=ids, use_cache=False).logits
        rq.FUSED_ACT_QUANT = True
        fus = model(input_ids=ids, use_cache=False).logits
        identical[label] = torch.equal(ref, fus)
        del ref, fus
    print('fused vs reference logits bit-identical:', identical, flush=True)

    configs = [('BF16', False)] + [(p[0], f) for p in POLICIES for f in (False, True)]
    times = {c: [] for c in configs}
    for r in range(WARMUP + REPS):
        for label, fused in configs:
            activate(label)
            rq.FUSED_ACT_QUANT = fused
            t = timed(model, ids)
            if r >= WARMUP:
                times[(label, fused)].append(t)
    out = dict(logits_bit_identical=identical, reps=REPS, results={})
    for label, fused in configs:
        activate(label)
        rq.FUSED_ACT_QUANT = fused
        cats = []
        for _ in range(NPROF):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                model(input_ids=ids, use_cache=False)
                torch.cuda.synchronize()
            cats.append(categorize(prof))
        cat = {k: statistics.mean(c[k] for c in cats) for k in cats[0]}
        cat['kernel_total'] = sum(cats[0].keys() and cat[k] for k in ('fp4_gemm', 'act_quant', 'bf16_gemm', 'attention', 'other'))
        wall = statistics.median(times[(label, fused)])
        cat['idle'] = wall - cat['kernel_total']
        key = f"{label} [{'fused quant' if fused else 'PyTorch quant'}]" if label != 'BF16' else 'BF16'
        out['results'][key] = dict(median_ms=wall, min_ms=min(times[(label, fused)]), times_ms=times[(label, fused)], **cat)
        print(f"{key:44s} wall {wall:7.2f} ms | fp4_gemm {cat['fp4_gemm']:5.2f} act_quant {cat['act_quant']:5.2f} "
              f"bf16 {cat['bf16_gemm']:5.2f} attn {cat['attention']:4.2f} other {cat['other']:6.2f} idle {cat['idle']:5.2f}", flush=True)
    rq.FUSED_ACT_QUANT = False
    rq.SINGLE_PASS_EPILOGUE = False
    activate('BF16')
    Path(ROOT / 'results/latency_fused_llama8b.json').write_text(json.dumps(out, indent=1))
    print('wrote', ROOT / 'results/latency_fused_llama8b.json')


if __name__ == '__main__':
    main()
