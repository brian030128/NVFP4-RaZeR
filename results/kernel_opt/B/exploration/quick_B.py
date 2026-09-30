"""Exploratory (before registration of kernel-opt B, disclosed, not a result): per-warp run tables vs the per-k_tile
dispatch. Builds: n16k64_wA (sm120/build, default), n16k64_wA from build_freq (#2), n16k64_wA_rt with its run table
(build_B), n16k64_wA_nodisp and stock_wA (sm120/build). Every output is checked bitwise against n16k64_wA first.
Part 1: 4096^3, width 128, warm isolated CUPTI (3 rotated rounds x 15): all-E2M1, real (Llama layer-0 o_proj 16x64),
        all-E0M3, p3 / p5 (every (warp, k_tile) pattern 3 / 5), and the densest 16x64 module of Llama's o_proj shape.
Part 2: Llama-3.1-8B layer-0 q / down / gate at T = 1, 16 (width 16) and 64 (the table's width), real tags,
        warm (isolated) and cold weights with the activations quantized after the flush (M1's condition), 3 x 30.
"""
import statistics
import sys
import time

sys.path.insert(0, '/home/dev/n16k64_campaign/kernel_opt/wtB/sm120')
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import mixfp4_sm120  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120 import runs as RT  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import place_scales as place  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402
assert '/wtB/' in mixfp4_sm120.__file__, mixfp4_sm120.__file__

PAPER, FREQ, BB = '/home/dev/NVFP4-RaZeR/sm120/build', '/home/dev/n16k64_campaign/kernel_opt/build_freq', '/home/dev/n16k64_campaign/kernel_opt/build_B'
NAMES = {16: '_n16', 32: '_n32', 64: '_n64', 128: ''}
header, masks, _ = mapio.read_map('/home/dev/n16k64_campaign/paper/artifacts/llama8b_tc_16x64.mixfp4map')


def iso(fn, reps=15):
    out = []
    for _ in range(reps):
        time.sleep(0.05)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        out += [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]
    return statistics.median(out)


def run(fns, rounds=3, measure=iso):
    res = {kk: [] for kk in fns}
    names = list(fns)
    for r in range(rounds):
        sh = r * len(names) // rounds
        for kk in names[sh:] + names[:sh]:
            res[kk].append(measure(kk))
    return {kk: statistics.median(v) for kk, v in res.items()}


def operands(n, k, t, mask, seed):
    g = torch.Generator('cpu').manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    xn, xsb, gsx = N.quantize_act(x, 'four_over_six_rows')
    wn, wsb, gsw = N.quantize_weight(w, 'map', mask, (16, 64))
    return dict(wp=N.pack_nibbles(wn), wsf=place(wsb, k), wsb=wsb, gsw=float(gsw), xp=N.pack_nibbles(xn),
                xsf=place(xsb, k), gsx=gsx, x=x)


# ---------------------------------------------------------------------------------------------- part 1
n = k = t = 4096
grid = (n // 16, k // 64)
blk = torch.arange(grid[0])[:, None] % 8
kbl = torch.arange(grid[1])[None, :] % 2
dense = max((m for m in header['modules'] if m['name'].endswith('o_proj')), key=lambda m: m['selected'])
tags = dict(e2m1=torch.zeros(grid, dtype=torch.bool), real=masks['model.layers.0.self_attn.o_proj'],
            densest=masks[dense['name']], e0m3=torch.ones(grid, dtype=torch.bool),
            p3=(kbl == 0).expand(grid).clone(), p5=(blk < 4).expand(grid).clone())
K = dict(default=Kernel.load('n16k64_wA', build_root=PAPER), freq=Kernel.load('n16k64_wA', build_root=FREQ),
         rt=Kernel.load('n16k64_wA_rt', build_root=BB))
nodisp, stock = Kernel.load('n16k64_wA_nodisp', build_root=PAPER), Kernel.load('stock_wA', build_root=PAPER)
fns, runs_info = {}, {}
for tg, m in tags.items():
    op = operands(n, k, t, m, 4096)
    table = RT.build(op['wsb'], n, k, K['rt'].runs_geometry).cuda()
    runs_info[tg] = (table.shape[1] - 1, float(table[:, 0].float().mean()))
    for kk, kern in K.items():
        rr = table if kk == 'rt' else None
        fns[(kk, tg)] = (lambda kern=kern, op=op, rr=rr: kern.gemm(op['wp'], op['wsf'], op['xp'], op['xsf'], n, t, k,
                                                                   scale_m_default=op['gsw'], scale_n=op['gsx'], check=False, runs=rr))
    ref = fns[('default', tg)]()
    for kk in ('freq', 'rt'):
        y = fns[(kk, tg)]()
        torch.cuda.synchronize()
        assert torch.equal(ref.view(torch.int16), y.view(torch.int16)), (kk, tg)
op0 = operands(n, k, t, tags['e2m1'], 4096)
fns[('nodisp', '-')] = lambda: nodisp.gemm(op0['wp'], op0['wsf'], op0['xp'], op0['xsf'], n, t, k, scale_m_default=op0['gsw'], scale_n=op0['gsx'], check=False)
fns[('stock', '-')] = lambda: stock.gemm(op0['wp'], op0['wsf'], op0['xp'], op0['xsf'], n, t, k, scale_m_default=op0['gsw'], scale_n=op0['gsx'], check=False)
print('bitwise: freq and rt (with its table) equal n16k64_wA on every tag set; runs per warp (max, mean):', runs_info, flush=True)


def warm(key):
    fns[key]()
    torch.cuda.synchronize()
    return iso(fns[key])


med = run(fns, measure=warm)
print(f"4096^3 width 128 (warm): stock_wA {med[('stock', '-')]:.1f} us, nodisp {med[('nodisp', '-')]:.1f} us")
for tg in tags:
    d = med[('default', tg)]
    print(f"  {tg:8s} default {d:6.1f} us | freq {100 * (med[('freq', tg)] / d - 1):+6.2f} % | rt {100 * (med[('rt', tg)] / d - 1):+6.2f} % "
          f"| rt vs nodisp {100 * (med[('rt', tg)] / med[('nodisp', '-')] - 1):+6.2f} % | rt vs stock {100 * (med[('rt', tg)] / med[('stock', '-')] - 1):+6.2f} %",
          flush=True)

# ---------------------------------------------------------------------------------------------- part 2
import math  # noqa: E402
FLUSH = torch.ones(512 * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
L2 = torch.cuda.get_device_properties(0).L2_cache_size


def cold_warm_act(kern, runs, op, n, k, t, iters=30, warmup=3):
    """M1's condition: weights cold (rotation over copies + 512 MiB flush), activations quantized after the flush."""
    copies = [(op['wp'].clone(), op['wsf'].clone()) for _ in range(math.ceil(4 * L2 / (op['wp'].numel() + op['wsf'].numel())) + 1)]

    def one(i):
        FLUSH.sum()
        torch.cuda.synchronize()
        xp, xsf, gs = kern.quant_rows(op['x'], 'four_over_six_rows')
        torch.cuda.synchronize()
        wp, wsf = copies[i % len(copies)]
        kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs, check=False, runs=runs)
        torch.cuda.synchronize()
    for i in range(warmup):
        one(i)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            one(warmup + i)
    d = [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]
    assert len(d) == iters, len(d)
    return statistics.median(d)


D = KernelSet('mixed', build_root=PAPER)
for layer, (n, k) in (('model.layers.0.self_attn.q_proj', (4096, 4096)), ('model.layers.0.mlp.down_proj', (4096, 14336)),
                      ('model.layers.0.mlp.gate_proj', (14336, 4096))):
    for t in (1, 16, 64):
        wd = D.width(n, k, t)
        op = operands(n, k, t, masks[layer], n + k + t)
        kerns = dict(default=Kernel.load('n16k64_wA' + NAMES[wd], build_root=PAPER),
                     freq=Kernel.load('n16k64_wA' + NAMES[wd], build_root=FREQ),
                     rt=Kernel.load('n16k64_wA_rt' + NAMES[wd], build_root=BB))
        table = RT.build(op['wsb'], n, k, kerns['rt'].runs_geometry).cuda()

        def launcher(kk, kern):
            rr = table if kk == 'rt' else None
            return lambda wp, wsf: kern.gemm(wp, wsf, op['xp'], op['xsf'], n, t, k, scale_m_default=op['gsw'],
                                             scale_n=op['gsx'], check=False, runs=rr)
        ls = {kk: launcher(kk, kern) for kk, kern in kerns.items()}
        ref = ls['default'](op['wp'], op['wsf'])
        for kk in ('freq', 'rt'):
            y = ls[kk](op['wp'], op['wsf'])
            torch.cuda.synchronize()
            assert torch.equal(ref.view(torch.int16), y.view(torch.int16)), (layer, t, kk)
        wm = run(ls, measure=lambda kk: (ls[kk](op['wp'], op['wsf']), torch.cuda.synchronize(),
                                         iso(lambda: ls[kk](op['wp'], op['wsf'])))[2])
        cm = run(ls, measure=lambda kk: cold_warm_act(kerns[kk], table if kk == 'rt' else None, op, n, k, t))
        print(f"{layer.split('.')[-1]:10s} {n}x{k} T={t:3d} width {wd:3d}: warm default {wm['default']:6.1f} us, freq "
              f"{100 * (wm['freq'] / wm['default'] - 1):+6.2f} %, rt {100 * (wm['rt'] / wm['default'] - 1):+6.2f} % | cold default "
              f"{cm['default']:6.1f} us, freq {100 * (cm['freq'] / cm['default'] - 1):+6.2f} %, rt {100 * (cm['rt'] / cm['default'] - 1):+6.2f} % "
              f"| runs/warp max {table.shape[1] - 1}", flush=True)
