"""Exploratory (disclosed, before registration of kernel-opt B'): the per-warp all-E2M1 bits vs today's dispatch.
Builds from worktree wtZ (B') and the amendment 7 build directories:
  t0      n16k64_wA_e64_t0 / _n16_t0 from build_7: the adopted 16x64 path, default (pipelined) dispatch
  freq    the same from build_7freq: #2's pattern-0-first dispatch
  zb1     n16k64_wA_e64_t0_zb / _n16_t0_zb from build_Z (MIXFP4_ZEROBIT=1: flat loop, funnel shift)
  zb2     the same from build_Z2 (MIXFP4_ZEROBIT=2: a loop over 32-k_tile windows, one shift per k_tile)
  stock   stock_wA_e64 / stock_wA_n16 from build_7 (FourOverSix weights), the reference
Maps: all-E2M1, the Llama-3.1-8B o_proj 16x64 map (real), all-E0M3. Every mixed output is checked bitwise against t0's
(zb with its bits) first. Timing: M1's condition (cold weights by rotation + 512 MiB flush, activations quantized
after the flush), CUPTI, 3 rotated rounds x 30; and back-to-back warm calls, 3 rounds x 50.
"""
import math
import statistics
import sys

sys.path.insert(0, '/home/dev/n16k64_campaign/kernel_opt/wtZ/sm120')
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import mixfp4_sm120  # noqa: E402
from mixfp4_sm120 import mapio  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120 import zbits as ZB  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import place_scales as place  # noqa: E402
assert '/wtZ/' in mixfp4_sm120.__file__

KO = '/home/dev/n16k64_campaign/kernel_opt'
FLUSH = torch.ones(512 * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
L2 = torch.cuda.get_device_properties(0).L2_cache_size
_, m16, _ = mapio.read_map('/home/dev/n16k64_campaign/paper/artifacts/llama8b_tc_16x64.mixfp4map')


def kernels(width):
    sfx = '' if width == 128 else f'_n{width}'
    mixed = 'n16k64_wA_e64_t0' if width == 128 else f'n16k64_wA{sfx}_t0'
    return dict(t0=Kernel.load(mixed, build_root=f'{KO}/build_7'), freq=Kernel.load(mixed, build_root=f'{KO}/build_7freq'),
                zb1=Kernel.load(mixed + '_zb', build_root=f'{KO}/build_Z'), zb2=Kernel.load(mixed + '_zb', build_root=f'{KO}/build_Z2'),
                stock=Kernel.load('stock_wA_e64' if width == 128 else f'stock_wA{sfx}', build_root=f'{KO}/build_7'))


def operands(n, k, t, mask, seed):
    g = torch.Generator('cpu').manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    kind = 'map' if mask is not None else 'four_over_six'
    wn, wsb, gsw = N.quantize_weight(w, kind, mask, (16, 64) if mask is not None else None)
    return dict(wp=N.pack_nibbles(wn), wsf=place(wsb, k), wsb=wsb, gsw=float(gsw), x=x)


def call(kern, op, n, k, t, wp, wsf, zb):
    xp, xsf, gs = kern.quant_rows(op['x'], 'four_over_six_rows')
    return kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs, check=False, zbits=zb)


def times(prof, n_):
    d = [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]
    assert len(d) == n_, len(d)
    return statistics.median(d)


def cold(kern, op, n, k, t, zb, iters=30, warmup=3):
    copies = [(op['wp'].clone(), op['wsf'].clone()) for _ in range(math.ceil(4 * L2 / (op['wp'].numel() + op['wsf'].numel())) + 1)]

    def one(i):
        FLUSH.sum()
        torch.cuda.synchronize()
        xp, xsf, gs = kern.quant_rows(op['x'], 'four_over_six_rows')
        torch.cuda.synchronize()
        wp, wsf = copies[i % len(copies)]
        kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs, check=False, zbits=zb)
        torch.cuda.synchronize()
    for i in range(warmup):
        one(i)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            one(warmup + i)
    return times(prof, iters)


def b2b(kern, op, n, k, t, zb, iters=50, warmup=5):
    xp, xsf, gs = kern.quant_rows(op['x'], 'four_over_six_rows')
    f = lambda: kern.gemm(op['wp'], op['wsf'], xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs, check=False, zbits=zb)  # noqa: E731
    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            f()
        torch.cuda.synchronize()
    return times(prof, iters)


CELLS = [(128, 4096, 4096, 4096), (128, 4096, 4096, 1024), (128, 14336, 4096, 2048), (128, 4096, 14336, 512),
         (16, 4096, 4096, 16), (16, 14336, 4096, 16), (16, 4096, 14336, 16)]
real = m16['model.layers.0.self_attn.o_proj']
print('us; % vs t0 (the adopted default dispatch)', flush=True)
for width, n, k, t in CELLS:
    ks = kernels(width)
    for label in ('e2m1', 'real', 'e0m3'):
        grid = (n // 16, k // 64)
        mask = {'e2m1': torch.zeros(grid, dtype=torch.bool), 'e0m3': torch.ones(grid, dtype=torch.bool),
                'real': real if tuple(real.shape) == grid else (torch.rand(grid, generator=torch.Generator().manual_seed(1)) < 0.12)}[label]
        op = operands(n, k, t, mask, n + k + t)
        so = operands(n, k, t, None, n + k + t)
        zb = ZB.build(op['wsb'], n, k, ks['zb1'].zbits_geometry).cuda()
        ref = call(ks['t0'], op, n, k, t, op['wp'], op['wsf'], None).clone()
        for c in ('freq', 'zb1', 'zb2'):
            y = call(ks[c], op, n, k, t, op['wp'], op['wsf'], zb if c.startswith('zb') else None)
            assert torch.equal(y.view(torch.int16), ref.view(torch.int16)), (width, n, k, t, label, c)
        names = ['t0', 'freq', 'zb1', 'zb2', 'stock']
        res = {m: {c: [] for c in names} for m in ('cold', 'b2b')}
        for r in range(3):
            order = names[r:] + names[:r]
            for c in order:
                o = so if c == 'stock' else op
                res['cold'][c].append(cold(ks[c], o, n, k, t, zb if c.startswith('zb') else None))
            for c in order:
                o = so if c == 'stock' else op
                res['b2b'][c].append(b2b(ks[c], o, n, k, t, zb if c.startswith('zb') else None))
        for mode in ('cold', 'b2b'):
            med = {c: statistics.median(v) for c, v in res[mode].items()}
            cells = ' '.join(f"{c} {100 * (med[c] / med['t0'] - 1):+5.2f}%" for c in names[1:])
            print(f"w{width:3d} {n}x{k} T={t:5d} {label:4s} {mode:4s}: t0 {med['t0']:8.2f} | {cells} | zero bits "
                  f"{100 * (1 - float((ZB.warp_patterns(ZB.block_flags(op['wsb'], n, k), n, k) != 0).float().mean())):.0f}%", flush=True)
