"""#3 (PROTOCOL_ITEMS.md): GEMM-only time of the SM120 deployment kernel for E0M3-dense maps.

Adapted from sm120/bench/kernel.py (origin/SM120-kernel): packed operands, CUPTI kernel time (median; the largest
CUTLASS kernel of the call) and CUDA-event time per call. For every distinct Linear shape of the model (one per
projection), every T and every unit:
    unit 8x64    mixed kernel n8k64_wB (weights on B, 8x64 granule)   vs  stock NVFP4 kernel stock_wB
    unit 16x64   mixed kernel n16k64_wA (weights on A, 16x64 granule) vs  stock NVFP4 kernel stock_wA
with the mixed kernel's tag patterns: all E2M1 (FourOverSix), MR-OPT, TM-OPT, all E0M3. The MR-OPT and TM-OPT patterns
are the map of the module of that projection with the most E0M3 tiles in each map (its worst case). Weights and
activations are random (the GEMM's speed depends on the tags, not on the values). Idle GPU required.

python results/tm_opt/gemm/bench_gemm.py --model llama8b --out JSON [--tokens 128,512,2048,8192]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

SM120 = Path('/home/dev/n16k64_campaign/sm120_bench/sm120')
sys.path.insert(0, str(SM120))
sys.path.insert(0, str(SM120 / 'bench'))
import common as B  # noqa: E402  (sm120/bench/common.py)
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel, sf_buffer_size, sf_offset_formula  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_maps import source  # noqa: E402

PAIRS = {'8x64': ('n8k64_wB', 'stock_wB'), '16x64': ('n16k64_wA', 'stock_wA')}


def place(sb, k):
    rows = sb.shape[0]
    buf = torch.zeros(sf_buffer_size(rows, k), dtype=torch.uint8, device=sb.device)
    r = torch.arange(rows, device=sb.device)[:, None]
    kb = torch.arange(k // 16, device=sb.device)[None, :]
    buf[sf_offset_formula(r, kb, k).reshape(-1)] = sb.reshape(-1)
    return buf


def worst_modules(model, method, unit):
    """{projection: (mask, module, E0M3 tiles, tiles)}: per projection, the module with the most E0M3 tiles."""
    masks = torch.load(source(model, method, unit), map_location='cpu', weights_only=True)
    best = {}
    for name, m in masks.items():
        proj = name.rsplit('.', 1)[-1]
        s = int(m.sum())
        if proj not in best or s > best[proj][2]:
            best[proj] = (m, name, s, m.numel())
    return best


def weight(n, k, mask, tb, seed):
    g = torch.Generator(device='cpu').manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    wn, wsb, gsw = N.quantize_weight(w, 'four_over_six' if mask is None else 'map', mask, tb)
    return dict(wp=N.pack_nibbles(wn), wsf=place(wsb, k), gsw=float(gsw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tokens', default='128,512,2048,8192')
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    B.require_idle()
    kernels = {c: Kernel.load(c) for pair in PAIRS.values() for c in pair}
    res = dict(gpu=B.gpu_info(), model=args.model, kernels={c: dict(sha256=k.sha256, desc=k.desc) for c, k in kernels.items()},
               patterns={}, rows=[])
    tokens = [int(t) for t in args.tokens.split(',')]
    for unit, (mixed, stock) in PAIRS.items():
        rows = int(unit.split('x')[0])
        worst = {m: worst_modules(args.model, m, unit) for m in ('mropt', 'tmopt')}
        res['patterns'][unit] = {m: {p: dict(module=v[1], e0m3_tiles=v[2], tiles=v[3]) for p, v in w.items()} for m, w in worst.items()}
        for proj, (mask0, name0, _, _) in worst['tmopt'].items():
            n, k = mask0.shape[0] * rows, mask0.shape[1] * 64
            pats = [('e2m1', None), ('mropt', worst['mropt'][proj][0]), ('tmopt', mask0),
                    ('all_e0m3', torch.ones(mask0.shape, dtype=torch.bool))]
            ops = {p: weight(n, k, m, (rows, 64), seed=n + k) for p, m in pats}
            for t in tokens:
                g = torch.Generator(device='cpu').manual_seed(t)
                x = torch.randn(t, k, generator=g).cuda().bfloat16()
                xn, xsb, gsx = N.quantize_act(x, 'four_over_six_rows')
                xp, xsf = N.pack_nibbles(xn), place(xsb, k)
                for config, plist in ((stock, ['e2m1']), (mixed, [p for p, _ in pats])):
                    kern = kernels[config]
                    for p in plist:
                        op = ops[p]
                        if kern.weight_operand == 0:
                            fn = lambda: kern.gemm(op['wp'], op['wsf'], xp, xsf, n, t, k,  # noqa: E731
                                                   scale_m_default=op['gsw'], scale_n=gsx, check=False)
                        else:
                            fn = lambda: kern.gemm(xp, xsf, op['wp'], op['wsf'], t, n, k,  # noqa: E731
                                                   scale_m=gsx, scale_n_default=op['gsw'], check=False)
                        kt = B.kernel_times(fn, iters=args.iters, match=lambda s: 'cutlass' in s or 'Gemm' in s or 'device_kernel' in s)
                        us = max(v['us'] for v in kt.values()) if kt else float('nan')
                        ev = B.time_fn(fn, iters=args.iters)
                        e0 = 0 if p == 'e2m1' else int(dict(pats)[p].sum())
                        res['rows'].append(dict(unit=unit, proj=proj, out=n, inp=k, tokens=t, config=config, pattern=p,
                                                e0m3_tiles=e0, tiles=mask0.numel(), kernel_us=us, event_ms=ev['ms'],
                                                tflops=2.0 * n * k * t / us / 1e6))
                print(f'{args.model} {unit} {proj} ({n}x{k}) T={t} done', flush=True)
                B.write(args.out, res)
    B.write(args.out, res)


if __name__ == '__main__':
    main()
