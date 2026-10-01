"""Exploratory (disclosed, before any registration): why n16k64_wA_nodisp is slower than stock_wA. Same weights (FourOverSix,
E2M1 only) on stock_wA and four nodisp builds, all from the current sources:
  nodisp        build_4 (as registered for #4; the blob's site-0 prmt tags, m-major MMA order)
  nodisp_t0     build_N: the site-0 prmt tags dropped
  nodisp_on     build_N: CUTLASS's MMA order (n outer, m serpentine)
  nodisp_t0on   build_N: both
Each build's output is checked bitwise against stock_wA's first. Timing: (1) M1's condition -- weights cold by rotation
+ 512 MiB flush, activations quantized after the flush, CUPTI, 3 rotated rounds x 30; (2) back-to-back warm launches
(C2's b2b column), 3 rotated rounds x 50.
"""
import math
import statistics
import sys

sys.path.insert(0, '/home/dev/n16k64_campaign/kernel_opt/wtN/sm120')
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import mixfp4_sm120  # noqa: E402
from mixfp4_sm120 import numerics as N  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import place_scales as place  # noqa: E402
assert '/wtN/' in mixfp4_sm120.__file__

KO = '/home/dev/n16k64_campaign/kernel_opt'
BUILDS = dict(stock=('stock_wA', f'{KO}/build_4'), nodisp=('n16k64_wA_nodisp', f'{KO}/build_4'),
              nodisp_t0=('n16k64_wA_nodisp_t0', f'{KO}/build_N'), nodisp_on=('n16k64_wA_nodisp_on', f'{KO}/build_N'),
              nodisp_t0on=('n16k64_wA_nodisp_t0on', f'{KO}/build_N'))
FLUSH = torch.ones(512 * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
L2 = torch.cuda.get_device_properties(0).L2_cache_size


def operands(n, k, t, seed):
    g = torch.Generator('cpu').manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(t, k, generator=g).cuda().bfloat16()
    wn, wsb, gsw = N.quantize_weight(w, 'four_over_six', None, None)
    return dict(wp=N.pack_nibbles(wn), wsf=place(wsb, k), gsw=float(gsw), x=x)


def device_times(prof, n):
    d = [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]
    assert len(d) == n, len(d)
    return d


def cold(kern, op, n, k, t, iters=30, warmup=3):
    copies = [(op['wp'].clone(), op['wsf'].clone()) for _ in range(math.ceil(4 * L2 / (op['wp'].numel() + op['wsf'].numel())) + 1)]

    def one(i):
        FLUSH.sum()
        torch.cuda.synchronize()
        xp, xsf, gs = kern.quant_rows(op['x'], 'four_over_six_rows')
        torch.cuda.synchronize()
        wp, wsf = copies[i % len(copies)]
        kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs, check=False)
        torch.cuda.synchronize()
    for i in range(warmup):
        one(i)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            one(warmup + i)
    return statistics.median(device_times(prof, iters))


def b2b(kern, op, n, k, t, iters=50, warmup=5):
    xp, xsf, gs = kern.quant_rows(op['x'], 'four_over_six_rows')
    for _ in range(warmup):
        kern.gemm(op['wp'], op['wsf'], xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs, check=False)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            kern.gemm(op['wp'], op['wsf'], xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs, check=False)
        torch.cuda.synchronize()
    return statistics.median(device_times(prof, iters))


def out(kern, op, n, k, t):
    xp, xsf, gs = kern.quant_rows(op['x'], 'four_over_six_rows')
    return kern.gemm(op['wp'], op['wsf'], xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs, check=False).clone()


ks = {c: Kernel.load(name, build_root=root) for c, (name, root) in BUILDS.items()}
assert all(str(ks[c].path).startswith(root) for c, (_, root) in BUILDS.items())
names = list(ks)
CELLS = [(4096, 4096, 4096), (4096, 4096, 1024), (4096, 4096, 256), (14336, 4096, 2048), (4096, 14336, 512), (4096, 14336, 64),
         (4096, 4096, 16)]
print('times in us; % vs stock; all builds are 128-wide (the nodisp builds exist at width 128 only)', flush=True)
for n, k, t in CELLS:
    op = operands(n, k, t, n + k + t)
    ref = out(ks['stock'], op, n, k, t)
    for c in names[1:]:
        assert torch.equal(out(ks[c], op, n, k, t).view(torch.int16), ref.view(torch.int16)), (n, k, t, c)
    res = {mode: {c: [] for c in names} for mode in ('cold', 'b2b')}
    for r in range(3):
        order = names[r:] + names[:r]
        for c in order:
            res['cold'][c].append(cold(ks[c], op, n, k, t))
        for c in order:
            res['b2b'][c].append(b2b(ks[c], op, n, k, t))
    for mode in ('cold', 'b2b'):
        med = {c: statistics.median(v) for c, v in res[mode].items()}
        cells = ' '.join(f"{c} {100 * (med[c] / med['stock'] - 1):+5.2f}%" for c in names[1:])
        print(f'{n}x{k} T={t:5d} {mode:4s}: stock {med["stock"]:8.2f} | {cells} | rounds stock {[round(v, 2) for v in res[mode]["stock"]]}'
              f' t0on {[round(v, 2) for v in res[mode]["nodisp_t0on"]]}', flush=True)
