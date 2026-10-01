"""Exploratory (disclosed, before registration of kernel-opt t0): the real 16x64 dispatch kernel with vs without the
site-0 tags, default and #2's dispatch, on a real Llama-3.1-8B map (o_proj) and on an all-E2M1 map. Timing as quick_N
(cold weights, warm activations, 3 rotated rounds x 30). Outputs checked bitwise first."""
import sys
sys.path.insert(0, '/home/dev/n16k64_campaign/kernel_opt/wtN/sm120')
import math, statistics, torch
from torch.profiler import ProfilerActivity, profile
from mixfp4_sm120 import mapio, numerics as N
from mixfp4_sm120.lib import Kernel
from mixfp4_sm120.linear import place_scales as place
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


KO = '/home/dev/n16k64_campaign/kernel_opt'
ks = dict(default=Kernel.load('n16k64_wA', build_root='/home/dev/NVFP4-RaZeR/sm120/build'), t0=Kernel.load('n16k64_wA_t0', build_root=f'{KO}/build_T'),
          freq=Kernel.load('n16k64_wA', build_root=f'{KO}/build_freq'), freqt0=Kernel.load('n16k64_wA_t0', build_root=f'{KO}/build_Tfreq'))
_, m16, _ = mapio.read_map('/home/dev/n16k64_campaign/paper/artifacts/llama8b_tc_16x64.mixfp4map')
real = m16['model.layers.0.self_attn.o_proj']
names = list(ks)
for label, mask in (('real', real), ('all-E2M1', torch.zeros_like(real))):
    for n, k, t in ((4096, 4096, 4096), (4096, 4096, 1024), (4096, 4096, 256)):
        g = torch.Generator('cpu').manual_seed(n + k + t)
        w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
        x = torch.randn(t, k, generator=g).cuda().bfloat16()
        wn, wsb, gsw = N.quantize_weight(w, 'map', mask, (16, 64))
        op = dict(wp=N.pack_nibbles(wn), wsf=place(wsb, k), gsw=float(gsw), x=x)
        ref = out(ks['default'], op, n, k, t)
        for c in names[1:]:
            assert torch.equal(out(ks[c], op, n, k, t).view(torch.int16), ref.view(torch.int16)), (label, c)
        res = {c: [] for c in names}
        for r in range(3):
            for c in names[r:] + names[:r]:
                res[c].append(cold(ks[c], op, n, k, t))
        med = {c: statistics.median(v) for c, v in res.items()}
        print(f"{label:8s} {n}x{k} T={t:5d}: t0 vs default {100*(med['t0']/med['default']-1):+5.2f}%  freqt0 vs freq {100*(med['freqt0']/med['freq']-1):+5.2f}%  "
              f"(default {med['default']:.2f}, t0 {med['t0']:.2f}, freq {med['freq']:.2f}, freqt0 {med['freqt0']:.2f} us)", flush=True)
