#!/usr/bin/env python3
"""A. Kernel-only GEMM benchmark on real model Linear shapes.

For every (model projection shape, token count T) and every built configuration, times the GEMM
kernel alone (CUPTI device duration of the CUTLASS kernel, median) with operands already packed:

    stock_wA / stock_wB   stock CUTLASS SM120 NVFP4 mainloop, same epilogue (the baseline)
    n16k64_wA_nodisp      the mixed kernel's tile/arrangement with the format dispatch compiled out
    n16k64_wA             deployment kernel (weights on A, 16x64 granule)
    n16k64_wA_8x1         same granule, 8x1 warp arrangement
    n8k64_wB              weights on B, 8x64 granule
    bf16                  torch F.linear (cuBLAS), for scale

and, for the mixed kernels, several tag patterns: all E2M1 (tags clear), the real selector map of
the most-E0M3 module of that projection, 50% random tiles, all E0M3.

    python sm120/bench/kernel.py --models llama8b,qwen4b --out sm120/results/bench/kernel_rtx5090.json
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as B  # noqa: E402

from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel, sf_buffer_size, sf_offset_formula  # noqa: E402

MAPS = {('llama8b', 16): 'llama8b_seed0_n16_k3', ('llama8b', 8): 'llama8b_seed0_n8_k3',
        ('qwen4b', 16): 'qwen4b_seed0_n16_k3', ('qwen4b', 8): 'qwen4b_seed0_n8_k3',
        ('mistral7b', 16): 'mistral7b_seed0_n16_k3', ('mistral7b', 8): 'mistral7b_seed0_n8_k3'}


def place(sb, k):
    rows = sb.shape[0]
    buf = torch.zeros(sf_buffer_size(rows, k), dtype=torch.uint8, device=sb.device)
    r = torch.arange(rows, device=sb.device)[:, None]
    kb = torch.arange(k // 16, device=sb.device)[None, :]
    buf[sf_offset_formula(r, kb, k).reshape(-1)] = sb.reshape(-1)
    return buf


def selector_masks(model, rows):
    """{proj: (mask of the module with the most E0M3 tiles, its name, count)} from the frozen map."""
    name = MAPS.get((model, rows))
    if name is None:
        return {}
    header, masks, _ = mapio.read_map(B.SM120 / 'maps' / f'{name}.mixfp4map')
    best = {}
    for m in header['modules']:
        proj = m['name'].rsplit('.', 1)[-1]
        if m['selected'] > best.get(proj, (None, None, -1))[2]:
            best[proj] = (masks[m['name']], m['name'], m['selected'])
    return best


def operands(n, k, t, mask, tb, seed):
    g = torch.Generator(device='cpu').manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    kind = 'map' if mask is not None else 'four_over_six'
    wn, wsb, gsw = N.quantize_weight(w, kind, mask, tb)
    xn, xsb, gsx = N.quantize_act(x, 'four_over_six_rows')
    return dict(w=w, x=x, wp=N.pack_nibbles(wn), wsf=place(wsb, k), gsw=float(gsw),
                xp=N.pack_nibbles(xn), xsf=place(xsb, k), gsx=gsx)


def patterns(cfg_rows, n, k, sel, proj, seed):
    grid = (n // cfg_rows, k // 64)
    g = torch.Generator(device='cpu').manual_seed(seed)
    out = [('e2m1', None)]
    if proj in sel and tuple(sel[proj][0].shape) == grid:
        out.append((f'selector({sel[proj][2]} tiles)', sel[proj][0]))
    out.append(('random50', torch.rand(grid, generator=g) < 0.5))
    out.append(('all_e0m3', torch.ones(grid, dtype=torch.bool)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='llama8b,qwen4b')
    ap.add_argument('--tokens', default='1,16,128,512,2048,8192')
    ap.add_argument('--configs', default='stock_wA,n16k64_wA_nodisp,n16k64_wA,n16k64_wA_8x1,stock_wB,n8k64_wB')
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--out', required=True)
    ap.add_argument('--allow-busy', action='store_true')
    args = ap.parse_args()
    if not args.allow_busy:
        B.require_idle()
    kernels = {}
    for c in args.configs.split(','):
        try:
            kernels[c] = Kernel.load(c)
        except Exception as e:  # noqa: BLE001
            print(f'skip {c}: {e}', flush=True)
    res = dict(gpu=B.gpu_info(), kernels={c: dict(sha256=k.sha256, desc=k.desc) for c, k in kernels.items()}, rows=[])
    toks = [int(t) for t in args.tokens.split(',')]
    for model in args.models.split(','):
        sel16, sel8 = selector_masks(model, 16), selector_masks(model, 8)
        for proj, (n, k) in B.MODEL_SHAPES[model].items():
            for t in toks:
                flops = 2.0 * n * k * t
                base = operands(n, k, t, None, None, seed=n + k + t)
                ms = B.time_fn(lambda: F.linear(base['x'], base['w']), iters=args.iters)
                res['rows'].append(dict(model=model, proj=proj, out=n, inp=k, tokens=t, config='bf16_cublas', pattern='-',
                                        event_ms=ms['ms'], tflops=flops / ms['ms'] / 1e9))
                for cname, kern in kernels.items():
                    rows = kern.type_block[0] if kern.type_block else None
                    pats = [('e2m1', None)] if rows is None else patterns(rows, n, k, sel16 if rows == 16 else sel8, proj, seed=t)
                    for pname, mask in pats:
                        op = base if mask is None else operands(n, k, t, mask, kern.type_block, seed=n + k + t)
                        if kern.weight_operand == 0:
                            fn = lambda: kern.gemm(op['wp'], op['wsf'], op['xp'], op['xsf'], n, t, k,  # noqa: E731
                                                   scale_m_default=op['gsw'], scale_n=op['gsx'], check=False)
                        else:
                            fn = lambda: kern.gemm(op['xp'], op['xsf'], op['wp'], op['wsf'], t, n, k,  # noqa: E731
                                                   scale_m=op['gsx'], scale_n_default=op['gsw'], check=False)
                        kt = B.kernel_times(fn, iters=args.iters, match=lambda s: 'cutlass' in s or 'Gemm' in s or 'device_kernel' in s)
                        us = max(v['us'] for v in kt.values()) if kt else float('nan')
                        ev = B.time_fn(fn, iters=args.iters)
                        res['rows'].append(dict(model=model, proj=proj, out=n, inp=k, tokens=t, config=cname, pattern=pname,
                                                kernel_us=us, event_ms=ev['ms'], tflops=flops / us / 1e6))
                print(f'{model} {proj} T={t} done', flush=True)
                B.write(args.out, res)
    B.write(args.out, res)


if __name__ == '__main__':
    main()
