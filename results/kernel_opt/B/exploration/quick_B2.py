"""Exploratory (disclosed): the per-run cost of run tables. 4096x4096, warm isolated CUPTI, 3 rounds x 15; default n16k64_wA
vs n16k64_wA_rt with its table, at width 128 (T = 4096) and width 16 (T = 16). Maps (16x64 granules): 'e2m1'; 'altK':
granule 0 of EVERY warp alternates E2M1 / E0M3 on k_block 0 every K k_tiles (all warps change pattern at the same
k_tiles: 32/K runs per warp); 'stagK': the same but warp w's boundaries shifted by w (the warps' boundaries differ)."""
import statistics, sys, time
sys.path.insert(0, '/home/dev/n16k64_campaign/kernel_opt/wtB/sm120')
import torch
from torch.profiler import ProfilerActivity, profile
from mixfp4_sm120 import numerics as N, runs as RT
from mixfp4_sm120.lib import Kernel
from mixfp4_sm120.linear import place_scales as place
PAPER, BB = '/home/dev/NVFP4-RaZeR/sm120/build', '/home/dev/n16k64_campaign/kernel_opt/build_B'
def iso(fn, reps=15):
    out = []
    for _ in range(reps):
        time.sleep(0.03)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(); torch.cuda.synchronize()
        out += [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]
    return statistics.median(out)
n = k = 4096
grid = (n // 16, k // 64)
blk = torch.arange(grid[0])[:, None] % 8
kt = (torch.arange(grid[1])[None, :] // 2)
kb0 = (torch.arange(grid[1])[None, :] % 2) == 0
maps = {'e2m1': torch.zeros(grid, dtype=torch.bool)}
for K in (1, 2, 4):
    maps[f'alt{K}'] = ((blk < 4) & kb0 & ((kt // K) % 2 == 1)).expand(grid).clone()
    maps[f'stag{K}'] = ((blk < 4) & kb0 & (((kt + blk % 4) // K) % 2 == 1)).expand(grid).clone()
g = torch.Generator('cpu').manual_seed(4096)
w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
for t, suffix in ((4096, ''), (16, '_n16')):
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    xn, xsb, gsx = N.quantize_act(x, 'four_over_six_rows')
    xp, xsf = N.pack_nibbles(xn), place(xsb, k)
    D, R = Kernel.load('n16k64_wA' + suffix, build_root=PAPER), Kernel.load('n16k64_wA_rt' + suffix, build_root=BB)
    for name, m in maps.items():
        wn, wsb, gsw = N.quantize_weight(w, 'map', m, (16, 64))
        wp, wsf = N.pack_nibbles(wn), place(wsb, k)
        table = RT.build(wsb, n, k, R.runs_geometry).cuda()
        fd = lambda: D.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=float(gsw), scale_n=gsx, check=False)
        fr = lambda: R.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=float(gsw), scale_n=gsx, check=False, runs=table)
        a, b = fd(), fr(); torch.cuda.synchronize()
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), name
        res = {'d': [], 'r': []}
        for r in range(3):
            for kk, fn in ((('d', fd), ('r', fr)) if r % 2 == 0 else (('r', fr), ('d', fd))):
                fn(); torch.cuda.synchronize(); res[kk].append(iso(fn))
        d, rr = statistics.median(res['d']), statistics.median(res['r'])
        print(f"width {128 if not suffix else 16:3d} {name:6s} runs/warp {int(table[:, 0].max()):2d}: default {d:7.2f} us, rt {rr:7.2f} us ({100 * (rr / d - 1):+6.2f} %)", flush=True)
