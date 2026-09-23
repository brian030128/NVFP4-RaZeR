"""Llama-3.1-8B latency on the native SM120 kernels: NVFP4, 4Over6, N16K64 and N8K64.

All measurements use the real Llama-3.1-8B weights and the two evaluated maps
(repro_local/results/calib_llama8b_attempt1/maps), on an otherwise idle GPU.

A. GEMM kernel time (CUPTI device durations via torch.profiler, so launch gaps and host overhead are
   excluded): the 224 projection GEMMs of one prefill forward, per projection type, for
     bf16            cuBLAS BF16 on the unquantized weights (reference)
     stock           stock CUTLASS SM120 NVFP4 GEMM (no format dispatch), weights on B / on A
     *_nodisp        the mixed builds with the dispatch compiled out (same tile/warp arrangement)
     wt_as_A, b8x64  the mixed builds with every format flag clear (4Over6 weights)
     *_map           the mixed builds running the evaluated N16 / N8 maps
   NVFP4 and 4Over6 weights are both plain E2M1 data, so their GEMM is the same kernel.
B. Activation quantization per projection input (the harness's PyTorch implementation):
   nvfp4_rows (one block-scale candidate) vs four_over_six_rows (two candidates + MSE choice),
   plus nibble encoding/packing and scale placement.
C. End-to-end prefill forward (batch 1, T=2048, logits included): median wall time over repeated
   forwards, and one profiled forward split into FP4 GEMM, activation quantization, encode,
   scale placement, epilogue, attention and everything else.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq  # noqa: E402
from campaign import mapio as MIO  # noqa: E402
from campaign import models as MOD  # noqa: E402
from campaign import quant as Q  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAPS = ROOT / 'results/calib_llama8b_attempt1/maps'
PROJ = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def device_us(prof):
    """Total kernel time (us) recorded in a profiler window."""
    total = 0.0
    for e in prof.key_averages():
        total += getattr(e, 'self_device_time_total', None) or getattr(e, 'self_cuda_time_total', 0) or 0
    return total


def kernel_rows(prof):
    rows = []
    for e in prof.key_averages():
        t = getattr(e, 'self_device_time_total', None) or getattr(e, 'self_cuda_time_total', 0) or 0
        if t > 0:
            rows.append((e.key, e.count, t))
    return rows


def load_masks(policy, shapes):
    path = MAPS / f'llama8b_seed0_{policy}.mixfp4map'
    header, masks, digest = MIO.read_map(path)
    manifest = json.loads((ROOT / 'results/calib_llama8b_attempt1/calibration/map_manifest.json').read_text())
    want = {m['policy']: m['sha256'] for m in manifest}[policy]
    assert digest == want, f'{policy}: map digest {digest} != manifest {want}'
    assert [m['name'] for m in header['modules']] == list(shapes)
    return masks, header['totals']['selected_tiles']


class Packed:
    """Per-module packed weights (+ row-major scale bytes) for one weight format."""

    def __init__(self, modules, kind, masks=None, type_block=None):
        self.items = {}
        for n, m in modules.items():
            pw, sb = rq.pack_weight(m.weight.detach(), kind, None if masks is None else masks[n], type_block)
            self.items[n] = (pw, sb)


def gemm_closure(kern, pw, sb, act, t):
    """One projection GEMM on packed operands (act = packed activation, scale bytes for (t, k))."""
    packed, sbytes = act
    n, k = pw.n, pw.k
    if kern.weight_operand == 0:
        wsf = kern.place_scales(sb, 0, n, t, k)
        asf = kern.place_scales(sbytes, 1, n, t, k)
        return lambda: kern.gemm(pw.packed, wsf, packed, asf, n, t, k, pw.gs)
    wsf = kern.place_scales(sb, 1, t, n, k)
    asf = kern.place_scales(sbytes, 0, t, n, k)
    return lambda: kern.gemm(packed, asf, pw.packed, wsf, t, n, k, pw.gs)


@torch.no_grad()
def part_a(modules, packs, kernels, tokens, reps):
    out = {}
    names = list(modules)
    by_proj = {p: [n for n in names if n.endswith(p)] for p in PROJ}
    variants = [('bf16', None, None), ('stock_wB', 'stock', 'fo6'), ('stock_wA', 'stock_wA', 'fo6'),
                ('wt_as_A_nodisp', 'wt_as_A_nodisp', 'fo6'), ('wt_as_A', 'wt_as_A', 'fo6'),
                ('wt_as_A_n16map', 'wt_as_A', 'n16'),
                ('b8x64_nodisp', 'b8x64_nodisp', 'fo6'), ('b8x64', 'b8x64', 'fo6'),
                ('b8x64_n8map', 'b8x64', 'n8')]
    for t in tokens:
        acts = {}
        xs = {}
        for k in sorted({modules[n].weight.shape[1] for n in names}):
            x = torch.randn(t, k, device='cuda').bfloat16()
            code, scale, gs = rq.act_four_over_six_rows(x)
            acts[k] = (rq.pack_nibbles(rq.e2m1_nibbles(code)), rq.scale_bytes(scale))
            xs[k] = x
        res = {}
        for label, kname, pname in variants:
            per = {}
            for p in PROJ:
                calls = []
                for n in by_proj[p]:
                    w = modules[n].weight
                    if label == 'bf16':
                        x = xs[w.shape[1]]
                        calls.append(lambda x=x, w=w: F.linear(x, w))
                    else:
                        pw, sb = packs[pname].items[n]
                        calls.append(gemm_closure(kernels[kname], pw, sb, acts[pw.k], t))
                for c in calls:   # warm-up (and CUTLASS/cuBLAS first-call costs)
                    c()
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(reps):
                        for c in calls:
                            c()
                    torch.cuda.synchronize()
                per[p] = device_us(prof) / reps / 1e3      # ms per forward for this projection type
            per['total'] = sum(per[p] for p in PROJ)
            res[label] = per
            print(f'A T={t} {label:15s} ' + ' '.join(f'{p[:-5]}={per[p]:.3f}' for p in PROJ)
                  + f' | total {per["total"]:.3f} ms', flush=True)
        out[t] = res
    return out


@torch.no_grad()
def part_b(tokens, reps):
    out = {}
    for k in (4096, 14336):
        x = torch.randn(tokens, k, device='cuda').bfloat16()
        code, scale, gs = rq.act_four_over_six_rows(x)
        sb = rq.scale_bytes(scale)
        kern = rq.Kernel('stock')
        fns = {'nvfp4_rows': lambda: rq.act_nvfp4_rows(x),
               'four_over_six_rows': lambda: rq.act_four_over_six_rows(x),
               'encode_pack': lambda: (rq.pack_nibbles(rq.e2m1_nibbles(code)), rq.scale_bytes(scale)),
               'place_scales': lambda: kern.place_scales(sb, 0, tokens, 4096, k),
               'fake_quantize_rows': lambda: Q.ACTIVATION['four_over_six_rows'](x),
               'fake_nvfp4_rows': lambda: Q.ACTIVATION['nvfp4_rows'](x)}
        res = {}
        for name, fn in fns.items():
            fn()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(reps):
                    fn()
                torch.cuda.synchronize()
            res[name] = device_us(prof) / reps / 1e3
        out[k] = res
        print(f'B K={k} ' + ' '.join(f'{n}={v:.3f}ms' for n, v in res.items()), flush=True)
    return out


def forward_timer(model, ids, warmup, reps):
    for _ in range(warmup):
        model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        model(input_ids=ids, use_cache=False)
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return times


def categorize(prof):
    """Kernel time (ms) per category from a CUDA-only profile (kernel rows only: no double count)."""
    cat = dict(fp4_gemm=0.0, bf16_gemm=0.0, attention=0.0, other=0.0)
    for key, count, t in kernel_rows(prof):
        low = key.lower()
        if 'cutlass13device_kernel' in low:                 # the mangled mixfp4/stock CUTLASS GEMMs
            cat['fp4_gemm'] += t
        elif any(s in low for s in ('flash', 'fmha', 'attention')):
            cat['attention'] += t
        elif any(s in low for s in ('gemm', 'xmma', 'cutlass', 'cublas', 'nvjet')):
            cat['bf16_gemm'] += t
        else:
            cat['other'] += t
    return {k: v / 1e3 for k, v in cat.items()}


def breakdown(model, ids, nprof=3):
    """Kernel time per category (mean of `nprof` CUDA-only profiled forwards), plus the device time
    under each RealLinear stage range (one CPU+CUDA profiled forward)."""
    cats = []
    for _ in range(nprof):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            model(input_ids=ids, use_cache=False)
            torch.cuda.synchronize()
        cats.append(categorize(prof))
    out = {k: statistics.mean(c[k] for c in cats) for k in cats[0]}
    out['kernel_total'] = sum(out[k] for k in ('fp4_gemm', 'bf16_gemm', 'attention', 'other'))
    rq.PROFILE_RANGES = True
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            model(input_ids=ids, use_cache=False)
            torch.cuda.synchronize()
    finally:
        rq.PROFILE_RANGES = False
    ranges = {}
    for e in prof.key_averages():
        if e.key.startswith('rq/') and e.key != 'rq/gemm':
            ranges[e.key] = (getattr(e, 'device_time_total', None) or getattr(e, 'cuda_time_total', 0) or 0) / 1e3
    out.update(ranges)
    # the rq/ stages (activation quantization, encode, placement, epilogue) are elementwise kernels
    # inside `other`; what remains is norms, RoPE, SiLU*mul, residual adds, embedding, softmax etc.
    out['other_model_ops'] = out['other'] - sum(ranges.values())
    return out


@torch.no_grad()
def part_c(model, modules, kernels, masks, tokens, warmup, reps):
    ids = torch.randint(0, model.config.vocab_size, (1, tokens), device='cuda')
    inst = rq.RealInstaller(modules, tokens=tokens, check_calls=0)
    policies = [('bf16', None, None, None),
                ('nvfp4 (stock kernel)', 'nvfp4', 'stock', None),
                ('4over6 (stock kernel)', 'four_over_six', 'stock', None),
                ('4over6 (wt_as_A kernel, flags clear)', 'four_over_six', 'wt_as_A', None),
                ('N16K64 (wt_as_A kernel)', 'map', 'wt_as_A', 'n16_k3'),
                ('4over6 (b8x64 kernel, flags clear)', 'four_over_six', 'b8x64', None),
                ('N8K64 (b8x64 kernel)', 'map', 'b8x64', 'n8_k3')]
    out = {}
    for label, kind, kname, mpol in policies:
        if kind is None:
            inst.remove()
        else:
            inst.install(kind, kernels[kname], None if mpol is None else masks[mpol], None)
        times = forward_timer(model, ids, warmup, reps)
        bd = breakdown(model, ids)
        out[label] = dict(median_ms=statistics.median(times), min_ms=min(times), times_ms=times, breakdown=bd)
        print(f'C {label:40s} median {out[label]["median_ms"]:.2f} ms  min {out[label]["min_ms"]:.2f}  '
              + ' '.join(f'{k}={v:.2f}' for k, v in bd.items()), flush=True)
    inst.remove()
    # fake-quant 4Over6 for reference (dequantized BF16 weights + quantize_rows pre-hooks)
    saved = {n: m.weight.detach().clone() for n, m in modules.items()}
    for n, m in modules.items():
        m.weight.copy_(Q.four_over_six(m.weight.detach()))
    act = Q.ActivationQuant(modules, 'four_over_six_rows', ste=False)
    times = forward_timer(model, ids, warmup, reps)
    bd = breakdown(model, ids)
    out['4over6 (fake quant, BF16 GEMM)'] = dict(median_ms=statistics.median(times), min_ms=min(times), times_ms=times,
                                                 breakdown=bd)
    print(f'C fake 4over6 median {statistics.median(times):.2f} ms  '
          + ' '.join(f'{k}={v:.2f}' for k, v in bd.items()), flush=True)
    act.remove()
    for n, m in modules.items():
        m.weight.copy_(saved[n])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', default='512,2048,8192')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--e2e-tokens', type=int, default=2048)
    ap.add_argument('--e2e-reps', type=int, default=10)
    ap.add_argument('--out', default=str(ROOT / 'results/latency_llama8b.json'))
    ap.add_argument('--parts', default='ABC', help='subset of A (GEMM), B (activation quant), C (forward)')
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    rq.VALIDATE = False           # correctness was established by the accuracy runs and unit gates
    model, _ = MOD.load_model('llama8b', attn='sdpa', device_map='cuda')
    modules = MOD.scope(model, 'llama8b')
    shapes = {n: tuple(m.weight.shape) for n, m in modules.items()}
    masks = {}
    tiles = {}
    for p in ('n16_k3', 'n8_k3'):
        masks[p], tiles[p] = load_masks(p, shapes)
    kernels = {c: rq.Kernel(c) for c in ('stock', 'wt_as_A_nodisp', 'b8x64_nodisp', 'wt_as_A', 'b8x64')}
    kernels['stock_wA'] = rq.Kernel('stock', weight_operand=0)
    env = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               libs={c: str(k.path) for c, k in kernels.items()}, e0m3_tiles=tiles)
    out = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    out['env'] = env
    if 'A' in args.parts:
        t0 = time.time()
        packs = {'fo6': Packed(modules, 'four_over_six'),
                 'n16': Packed(modules, 'map', masks['n16_k3'], rq.TYPE_BLOCK['wt_as_A']),
                 'n8': Packed(modules, 'map', masks['n8_k3'], rq.TYPE_BLOCK['b8x64'])}
        print(f'packed 3 weight sets in {time.time() - t0:.0f}s; E0M3 tiles n16={tiles["n16_k3"]} '
              f'n8={tiles["n8_k3"]}', flush=True)
        out['gemm'] = part_a(modules, packs, kernels, [int(t) for t in args.tokens.split(',')], args.reps)
        del packs
        torch.cuda.empty_cache()
    if 'B' in args.parts:
        out['act_quant'] = part_b(args.e2e_tokens, 20)
    if 'C' in args.parts:
        out['forward'] = part_c(model, modules, kernels, masks, args.e2e_tokens, 3, args.e2e_reps)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
