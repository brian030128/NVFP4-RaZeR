#!/usr/bin/env python3
"""Exploratory (disclosed, not registered): the 256x64 (A') build variants at fixed widths, against the no-dispatch
ceiling and stock, on all-E2M1, the real map and all-E0M3 tags (C2's question, by M1's cold condition).

    python results/kernel_opt/V/exploration/quick_V2.py > results/kernel_opt/V/exploration/quick_V2.out

Cells: width 128 at 4096^3, 4096x4096 T=1024, 14336x4096 T=2048, 4096x14336 T=512; width 16 at T=16 on the three
shapes. Kernels at each width (widths 128: today's A' build is the non-e64 n16k64_wA_g32):
  today   n16k64_wA_g32 / _n16 from build_A1          e / eF / eU / eFU   n16k64_wA_g32_e64 / _n16 from build_V*
  e_t0, eU_t0, eFU_t0   the t0 builds from build_VD0 / build_VU / build_VFU
  ceil    n16k64_wA_nodisp_e64_t0 / _n16_t0 (build_C3k)   stock   stock_wA_e64 / _n16 (build_7)
Maps: the typical module (lower median of the E0M3 count) of the projection in Llama-3.1-8B's TC 256x64 map (16x64
granules, uniform over 256x64 tiles); all-E2M1; all-E0M3. Every variant's output is checked bitwise against today's on
each map. Cold: weights rotated past the L2 after a 512 MiB flush, activations quantized after the flush, CUPTI, 3
rotated rounds x 30, the default scheduler setting.
"""
import math
import statistics
import sys

sys.path.insert(0, '/home/dev/n16k64_campaign/kernel_opt/wtV/sm120')
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import mixfp4_sm120  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import place_scales as place  # noqa: E402
assert '/wtV/' in mixfp4_sm120.__file__

KO = '/home/dev/n16k64_campaign/kernel_opt'
ART = '/home/dev/n16k64_campaign/paper/artifacts'
FLUSH = torch.ones(512 * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
L2 = torch.cuda.get_device_properties(0).L2_cache_size
PROJ = {(4096, 4096): 'o_proj', (14336, 4096): 'gate_proj', (4096, 14336): 'down_proj'}
MAP = mapio.read_map(f'{ART}/llama8b_tc_256x64.mixfp4map')
NAMES = ['today', 'e', 'eF', 'eU', 'eFU', 'e_t0', 'eU_t0', 'eFU_t0', 'ceil', 'stock']
ROOT = dict(today='build_A1', e='build_VD0', eF='build_VF', eU='build_VU', eFU='build_VFU', e_t0='build_VD0',
            eU_t0='build_VU', eFU_t0='build_VFU', ceil='build_C3k', stock='build_7')


def cfg(name, width):
    if name == 'stock':
        return 'stock_wA_e64' if width == 128 else f'stock_wA_n{width}'
    if name == 'ceil':
        return 'n16k64_wA_nodisp_e64_t0' if width == 128 else f'n16k64_wA_nodisp_n{width}_t0'
    base = ('n16k64_wA_g32' if name == 'today' else 'n16k64_wA_g32_e64') if width == 128 else f'n16k64_wA_g32_n{width}'
    return base + ('_t0' if name.endswith('_t0') else '')


def typical(proj):
    header, masks, _ = MAP
    mods = [m for m in header['modules'] if m['name'].rsplit('.', 1)[-1] == proj]
    order = sorted(range(len(mods)), key=lambda i: (mods[i]['selected'], i))
    m = mods[order[(len(mods) - 1) // 2]]
    return m['name'], masks[m['name']]


def operands(n, k, t, mask, seed):
    g = torch.Generator('cpu').manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    kind = 'map' if mask is not None else 'four_over_six'
    wn, wsb, gsw = N.quantize_weight(w, kind, mask, (16, 64) if mask is not None else None)
    return dict(wp=N.pack_nibbles(wn), wsf=place(wsb, k), gsw=float(gsw), x=x)


def call(kern, wp, wsf, op, xq, n, k, t):
    xp, xsf, gs = xq
    return kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs, check=False)


def cold(kern, op, n, k, t, iters=30, warmup=3):
    copies = [(op['wp'].clone(), op['wsf'].clone())
              for _ in range(math.ceil(4 * L2 / (op['wp'].numel() + op['wsf'].numel())) + 1)]

    def one(i):
        FLUSH.sum()
        torch.cuda.synchronize()
        xq = kern.quant_rows(op['x'], 'four_over_six_rows')
        torch.cuda.synchronize()
        wp, wsf = copies[i % len(copies)]
        call(kern, wp, wsf, op, xq, n, k, t)
        torch.cuda.synchronize()
    for i in range(warmup):
        one(i)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            one(warmup + i)
    d = [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]
    assert len(d) == iters, len(d)
    return statistics.median(d)


CELLS = [(128, 4096, 4096, 4096), (128, 4096, 4096, 1024), (128, 14336, 4096, 2048), (128, 4096, 14336, 512),
         (16, 4096, 4096, 16), (16, 14336, 4096, 16), (16, 4096, 14336, 16)]
print("cold us; % vs today (A', default dispatch); maps: the typical module of the projection, all-E2M1, all-E0M3",
      flush=True)
for width, n, k, t in CELLS:
    ks = {nm: Kernel.load(cfg(nm, width), build_root=f'{KO}/{ROOT[nm]}') for nm in NAMES}
    mod, real = typical(PROJ[(n, k)])
    grid = (n // 16, k // 64)
    fo6 = operands(n, k, t, None, n + k + t)
    for label in ('e2m1', 'real', 'e0m3'):
        mask = {'e2m1': torch.zeros(grid, dtype=torch.bool), 'e0m3': torch.ones(grid, dtype=torch.bool), 'real': real}[label]
        op = operands(n, k, t, mask, n + k + t)
        q = ks['today'].quant_rows(op['x'], 'four_over_six_rows')
        ref = call(ks['today'], op['wp'], op['wsf'], op, q, n, k, t).clone()
        for nm in NAMES[1:-2]:
            y = call(ks[nm], op['wp'], op['wsf'], op, ks[nm].quant_rows(op['x'], 'four_over_six_rows'), n, k, t)
            assert torch.equal(y.view(torch.int16), ref.view(torch.int16)), (width, n, k, t, label, nm)
        per = {nm: [] for nm in NAMES}
        for r in range(3):
            order = NAMES[r * 3:] + NAMES[:r * 3]
            for nm in order:
                per[nm].append(cold(ks[nm], fo6 if nm in ('ceil', 'stock') else op, n, k, t))
        med = {nm: statistics.median(v) for nm, v in per.items()}
        ref_us = med['today']
        print(f'w{width:3d} {n}x{k} T={t:5d} {label:4s} today {ref_us:7.2f} us | '
              + ' '.join(f'{nm} {100 * (med[nm] / ref_us - 1):+6.2f}%' for nm in NAMES[1:]) + f' | real module {mod}',
              flush=True)
