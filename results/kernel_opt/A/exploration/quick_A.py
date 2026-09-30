"""Exploratory (before registration of kernel-opt A, disclosed, not a result): arm-subset dispatch variants at 4096^3.

Isolated CUPTI, 3 rotated rounds x 15 single calls. Builds: stock_wA, n16k64_wA_nodisp, n16k64_wA (sm120/build), freq
(build_freq), n16k64_wA_as2 / as6 / as8 (build_A2, worktree wtA2). Tags (16x64): all-E2M1, the real map (Llama
layer-0 o_proj), all-E0M3, and two fallback-forcing maps: 'p3' (both granules of every warp E0M3 on k_block 0 only:
pattern 3 on every (warp, k_tile)) and 'p5' (granule 0 E0M3 on both k_blocks: pattern 5). Every variant's output is
checked bitwise against n16k64_wA's first.
"""
import statistics
import sys
import time

sys.path.insert(0, '/home/dev/n16k64_campaign/kernel_opt/wtA2/sm120')
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import mixfp4_sm120  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import place_scales as place  # noqa: E402
assert '/wtA2/' in mixfp4_sm120.__file__, mixfp4_sm120.__file__

PAPER, FREQ, A2 = '/home/dev/NVFP4-RaZeR/sm120/build', '/home/dev/n16k64_campaign/kernel_opt/build_freq', '/home/dev/n16k64_campaign/kernel_opt/build_A2'
n = k = t = 4096
g = torch.Generator('cpu').manual_seed(4096)
w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
x = torch.randn(t, k, generator=g).cuda().bfloat16()
xn, xsb, gsx = N.quantize_act(x, 'four_over_six_rows')
xp, xsf = N.pack_nibbles(xn), place(xsb, k)
_, masks, _ = mapio.read_map('/home/dev/n16k64_campaign/paper/artifacts/llama8b_tc_16x64.mixfp4map')
grid = (n // 16, k // 64)
blk = torch.arange(grid[0])[:, None] % 8          # 16-row block within its 128-row panel: warp w -> blocks w, w + 4
kbl = torch.arange(grid[1])[None, :] % 2          # k_block within the k_tile
tags = dict(e2m1=torch.zeros(grid, dtype=torch.bool), real=masks['model.layers.0.self_attn.o_proj'],
            e0m3=torch.ones(grid, dtype=torch.bool),
            p3=(kbl == 0).expand(grid).clone(),                       # both granules, k_block 0
            p5=(blk < 4).expand(grid).clone())                        # granule 0 (the warp's first m-atom), both k_blocks
W = {}
for tg, m in tags.items():
    wn, wsb, gsw = N.quantize_weight(w, 'map', m, (16, 64))
    W[tg] = (N.pack_nibbles(wn), place(wsb, k), float(gsw))
wn, wsb, gsw = N.quantize_weight(w, 'four_over_six', None, None)
W['fo6'] = (N.pack_nibbles(wn), place(wsb, k), float(gsw))
wn, wsb, gsw = N.quantize_weight(w, 'nvfp4', None, None)
W['nvfp4'] = (N.pack_nibbles(wn), place(wsb, k), float(gsw))
K = dict(stock=Kernel.load('stock_wA', build_root=PAPER), nodisp=Kernel.load('n16k64_wA_nodisp', build_root=PAPER),
         default=Kernel.load('n16k64_wA', build_root=PAPER), freq=Kernel.load('n16k64_wA', build_root=FREQ),
         as2=Kernel.load('n16k64_wA_as2', build_root=A2), as6=Kernel.load('n16k64_wA_as6', build_root=A2),
         as8=Kernel.load('n16k64_wA_as8', build_root=A2))


def call(kk, wt):
    wp, wsf, gs = W[wt]
    return lambda: K[kk].gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=gs, scale_n=gsx, check=False)


for tg in tags:
    ref = call('default', tg)()
    for kk in ('freq', 'as2', 'as6', 'as8'):
        y = call(kk, tg)()
        torch.cuda.synchronize()
        assert torch.equal(ref.view(torch.int16), y.view(torch.int16)), f'{kk} {tg} differs from n16k64_wA'
print('bitwise: every variant equals n16k64_wA on every tag set', flush=True)
cfgs = [('stock', 'nvfp4'), ('nodisp', 'fo6')] + [(kk, tg) for tg in tags for kk in ('default', 'freq', 'as2', 'as6', 'as8')]


def iso(fn, reps=15):
    out = []
    for _ in range(reps):
        time.sleep(0.05)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        out += [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]
    return statistics.median(out)


res = {c: [] for c in cfgs}
for r in range(3):
    sh = r * len(cfgs) // 3
    for c in cfgs[sh:] + cfgs[:sh]:
        fn = call(*c)
        fn()
        torch.cuda.synchronize()
        res[c].append(iso(fn))
med = {c: statistics.median(v) for c, v in res.items()}
print(f"stock_wA {med[('stock', 'nvfp4')]:.1f} us, nodisp {med[('nodisp', 'fo6')]:.1f} us")
print('tags   ' + ' '.join(f'{kk:>16s}' for kk in ('default', 'freq', 'as2', 'as6', 'as8')))
for tg in tags:
    d = med[('default', tg)]
    print(f'{tg:6s} ' + ' '.join(f"{med[(kk, tg)]:7.1f} {100 * (med[(kk, tg)] / d - 1):+6.2f}%  " for kk in ('default', 'freq', 'as2', 'as6', 'as8')))
nd = med[('nodisp', 'fo6')]
print('vs nodisp, all-E2M1: ' + ' '.join(f"{kk} {100 * (med[(kk, 'e2m1')] / nd - 1):+.2f}%" for kk in ('default', 'freq', 'as2', 'as6', 'as8')))
