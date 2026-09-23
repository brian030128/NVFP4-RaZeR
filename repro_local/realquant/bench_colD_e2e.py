"""Prefill latency with the column-major-D weights-on-A build: does it remove N16K64's transpose cost?

Llama-3.1-8B, batch 1, T=2048, SDPA, real weights and the evaluated maps. Five policies are packed
once and switched per forward (module forwards swapped), so every policy and epilogue mode is
measured interleaved in the same loop and GPU drift cancels:
  NVFP4 (stock)                  stock CUTLASS NVFP4, nvfp4_rows activations
  4Over6 (stock)                 stock CUTLASS NVFP4, four_over_six_rows activations
  N8K64 (b8x64)                  weights on B, 8x64 granule
  N16K64 (wt_as_A)               weights on A, 16x64 granule, D = [out, tokens] (needs a transpose)
  N16K64 (wt_as_A_colD)          same kernel, D stored column-major = [tokens, out] (no transpose)
each with the three-step epilogue (default) and rq.SINGLE_PASS_EPILOGUE. Both N16 variants must give
bit-identical logits.
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

REPS, WARMUP, NPROF = 10, 2, 2
POLICIES = [('NVFP4 (stock)', 'nvfp4', 'stock', None),
            ('4Over6 (stock)', 'four_over_six', 'stock', None),
            ('N8K64 (b8x64)', 'map', 'b8x64', 'n8_k3'),
            ('N16K64 (wt_as_A, transpose)', 'map', 'wt_as_A', 'n16_k3'),
            ('N16K64 (wt_as_A_colD)', 'map', 'wt_as_A_colD', 'n16_k3')]


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
    forwards = {}
    for label, kind, cfg, mpol in POLICIES:
        inst.install(kind, rq.Kernel(cfg), None if mpol is None else masks[mpol], None)
        forwards[label] = {n: m.forward for n, m in modules.items()}

    def activate(label):
        for n, m in modules.items():
            m.forward = forwards[label][n]

    # the two N16 variants must agree bit for bit, in both epilogue modes
    same = {}
    for mode in (False, True):
        rq.SINGLE_PASS_EPILOGUE = mode
        activate('N16K64 (wt_as_A, transpose)')
        a = model(input_ids=ids, use_cache=False).logits
        activate('N16K64 (wt_as_A_colD)')
        b = model(input_ids=ids, use_cache=False).logits
        same['single_pass' if mode else 'three_pass'] = torch.equal(a, b)
        del a, b
    print('N16 transpose vs colD logits bit-identical:', same, flush=True)

    times = {(p[0], m): [] for p in POLICIES for m in (False, True)}
    for r in range(WARMUP + REPS):
        for label, *_ in POLICIES:
            activate(label)
            for mode in (False, True):
                rq.SINGLE_PASS_EPILOGUE = mode
                t = timed(model, ids)
                if r >= WARMUP:
                    times[(label, mode)].append(t)
    out = dict(logits_bit_identical_n16=same, reps=REPS, results={})
    for label, *_ in POLICIES:
        activate(label)
        res = {}
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
            wall = statistics.median(times[(label, mode)])
            cat['idle'] = wall - cat['kernel_total']
            res['single_pass' if mode else 'three_pass'] = dict(median_ms=wall, times_ms=times[(label, mode)], **cat)
            print(f"{label:30s} {'single' if mode else 'three '}-pass wall {wall:7.2f} ms | fp4_gemm {cat['fp4_gemm']:5.2f} "
                  f"other {cat['other']:6.2f} attn {cat['attention']:4.2f} bf16 {cat['bf16_gemm']:4.2f} idle {cat['idle']:5.2f}",
                  flush=True)
        out['results'][label] = res
    rq.SINGLE_PASS_EPILOGUE = False
    inst.remove()
    for label, *_ in POLICIES:
        activate(label)
    for m in modules.values():   # restore the pristine forwards
        m.forward = inst.saved[[n for n, mm in modules.items() if mm is m][0]]
    Path(ROOT / 'results/latency_colD_llama8b.json').write_text(json.dumps(out, indent=1))
    print('wrote', ROOT / 'results/latency_colD_llama8b.json')


if __name__ == '__main__':
    main()
