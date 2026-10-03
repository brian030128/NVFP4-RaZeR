#!/usr/bin/env python3
"""Exploratory (disclosed, not registered), item 2: why the 16x64 width-64 mixed build is relatively slow. Per shape and
T around the width crossover, every width's build of four kinds, cold (M1's condition: weights rotated past the L2 + a
512 MiB flush, activations quantized after the flush), CUPTI device time, 3 rotated rounds x 30, the default
scheduler setting (0, 1) everywhere:
  stock    stock_wA_n32 / _n64 / _e64 (build_7): stock_ko's builds
  nodisp   n16k64_wA_nodisp_n32_t0 / _n64_t0 / _e64_t0 (build_C3k): the same tiles, dispatch compiled out (E2M1 only)
  freq     n16k64_wA_n32_t0 / _n64_t0 / _e64_t0 (build_7freq): today's adopted 16x64 path (#2's dispatch)
  uni      the same from build_U1 (+ MIXFP4_UNIFORM_DISPATCH=1)
Weights: FourOverSix for stock and nodisp; for freq and uni the typical module's tags of Llama-3.1-8B's TC 16x64 map for
the projection, and all-E2M1 tags (the dispatch's floor: every k_tile takes arm 0). freq and uni outputs are checked
bitwise against each other on each map.
"""
import math
import statistics
import sys

sys.path.insert(0, '/home/dev/NVFP4-RaZeR/sm120')
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import place_scales as place  # noqa: E402

KO = '/home/dev/n16k64_campaign/kernel_opt'
ART = '/home/dev/n16k64_campaign/paper/artifacts'
FLUSH = torch.ones(512 * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
L2 = torch.cuda.get_device_properties(0).L2_cache_size
PROJ = {(4096, 4096): 'o_proj', (14336, 4096): 'gate_proj', (4096, 14336): 'down_proj'}
MAP = mapio.read_map(f'{ART}/llama8b_tc_16x64.mixfp4map')


def typical(proj):
    header, masks, _ = MAP
    mods = [m for m in header['modules'] if m['name'].rsplit('.', 1)[-1] == proj]
    if not mods:
        return None, None
    order = sorted(range(len(mods)), key=lambda i: (mods[i]['selected'], i))
    m = mods[order[(len(mods) - 1) // 2]]
    return m['name'], masks[m['name']]


def name(kind, w):
    suf = 'e64' if w == 128 else f'n{w}'
    return {'stock': f'stock_wA_{suf}', 'nodisp': f'n16k64_wA_nodisp_{suf}_t0', 'freq': f'n16k64_wA_{suf}_t0',
            'uni': f'n16k64_wA_{suf}_t0'}[kind]


ROOT = dict(stock='build_7', nodisp='build_C3k', freq='build_7freq', uni='build_U1')


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


SHAPES = [(4096, 4096), (4096, 14336), (14336, 4096)]
TOKENS = [64, 128, 256, 512]
WIDTHS = [32, 64, 128]
KINDS = ['stock', 'nodisp', 'freq', 'uni']
print('cold us, median of 3 rounds x 30 (round range in brackets); maps: fo6 for stock/nodisp; real typical module and '
      'all-E2M1 tags for freq/uni', flush=True)
for n, k in SHAPES:
    proj = PROJ[(n, k)]
    mod, real = typical(proj)
    tiles = (n // 16, k // 64)
    for t in TOKENS:
        fo6 = operands(n, k, t, None, n + k + t)
        ops = dict(real=operands(n, k, t, real, n + k + t), e2m1=operands(n, k, t, torch.zeros(tiles, dtype=torch.bool), n + k + t))
        res = {}
        for w in WIDTHS:
            ks = {kd: Kernel.load(name(kd, w), build_root=f'{KO}/{ROOT[kd]}') for kd in KINDS}
            for m in ('real', 'e2m1'):
                a = call(ks['freq'], ops[m]['wp'], ops[m]['wsf'], ops[m], ks['freq'].quant_rows(ops[m]['x'], 'four_over_six_rows'), n, k, t).clone()
                b = call(ks['uni'], ops[m]['wp'], ops[m]['wsf'], ops[m], ks['uni'].quant_rows(ops[m]['x'], 'four_over_six_rows'), n, k, t)
                assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (n, k, t, w, m)
            cfgs = [('stock', fo6), ('nodisp', fo6), ('freq', ops['real']), ('uni', ops['real']), ('freq_e2m1', ops['e2m1']),
                    ('uni_e2m1', ops['e2m1'])]
            vals = {c: [] for c, _ in cfgs}
            for r in range(3):
                order = cfgs[r * 2:] + cfgs[:r * 2]
                for c, op in order:
                    vals[c].append(cold(ks[c.split('_')[0]], op, n, k, t))
            res[w] = {c: (statistics.median(v), min(v), max(v)) for c, v in vals.items()}
        line = f'{n}x{k} T={t:4d} ({proj} {mod})'
        print(line, flush=True)
        for c in ('stock', 'nodisp', 'freq', 'uni', 'freq_e2m1', 'uni_e2m1'):
            cells = ' | '.join(f'w{w}: {res[w][c][0]:7.2f} [{res[w][c][1]:.2f}-{res[w][c][2]:.2f}]' for w in WIDTHS)
            best = min(WIDTHS, key=lambda w: res[w][c][0])
            print(f'   {c:10s} {cells} | best w{best}', flush=True)
