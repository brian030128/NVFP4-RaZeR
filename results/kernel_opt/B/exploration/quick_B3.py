"""Exploratory (disclosed): run tables on 256x64 maps (A'). Llama-3.1-8B layer-0 q / down / gate with the real 256x64 map,
T = 1, 16, 64, 256, 1024 at the table's width: n16k64_wA_g32 (build_A1) vs n16k64_wA_g32_rt with its table (build_B), warm
isolated and cold (weights rotated + flushed, activations quantized after the flush), 3 rounds."""
import math, statistics, sys, time
sys.path.insert(0, '/home/dev/n16k64_campaign/kernel_opt/wtB/sm120')
import torch
from torch.profiler import ProfilerActivity, profile
from mixfp4_sm120 import mapio, numerics as N, runs as RT
from mixfp4_sm120.lib import Kernel
from mixfp4_sm120.linear import place_scales as place
from mixfp4_sm120.select import KernelSet
A1, BB = '/home/dev/n16k64_campaign/kernel_opt/build_A1', '/home/dev/n16k64_campaign/kernel_opt/build_B'
NAMES = {16: '_n16', 32: '_n32', 64: '_n64', 128: ''}
_, masks, _ = mapio.read_map('/home/dev/n16k64_campaign/paper/artifacts/llama8b_tc_256x64.mixfp4map')
FLUSH = torch.ones(512 * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
L2 = torch.cuda.get_device_properties(0).L2_cache_size
def prof_gemm(fn_list):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for f in fn_list: f()
    return [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]
def warm(fn, reps=15):
    out = []
    for _ in range(reps):
        time.sleep(0.02)
        out += prof_gemm([lambda: (fn(), torch.cuda.synchronize())])
    return statistics.median(out)
def cold(kern, runs, op, n, k, t, iters=30):
    copies = [(op['wp'].clone(), op['wsf'].clone()) for _ in range(math.ceil(4 * L2 / (op['wp'].numel() + op['wsf'].numel())) + 1)]
    def one(i):
        FLUSH.sum(); torch.cuda.synchronize()
        xp, xsf, gs = kern.quant_rows(op['x'], 'four_over_six_rows'); torch.cuda.synchronize()
        wp, wsf = copies[i % len(copies)]
        kern.gemm(wp, wsf, xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gs, check=False, runs=runs); torch.cuda.synchronize()
    for i in range(3): one(i)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters): one(3 + i)
    d = [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'device_kernel' in e.name.lower()]
    return statistics.median(d)
D = KernelSet('mixed256', build_root=A1)
for layer, (n, k) in (('model.layers.0.self_attn.q_proj', (4096, 4096)), ('model.layers.0.mlp.down_proj', (4096, 14336)),
                      ('model.layers.0.mlp.gate_proj', (14336, 4096))):
    for t in (1, 16, 64, 256, 1024):
        wd = D.width(n, k, t)
        g = torch.Generator('cpu').manual_seed(n + k + t)
        w = (torch.randn(n, k, generator=g) * 0.02).cuda().bfloat16()
        x = torch.randn(t, k, generator=g).cuda().bfloat16()
        xn, xsb, gsx = N.quantize_act(x, 'four_over_six_rows')
        wn, wsb, gsw = N.quantize_weight(w, 'map', masks[layer], (16, 64))
        op = dict(wp=N.pack_nibbles(wn), wsf=place(wsb, k), gsw=float(gsw), x=x)
        xp, xsf = N.pack_nibbles(xn), place(xsb, k)
        G, R = Kernel.load('n16k64_wA_g32' + NAMES[wd], build_root=A1), Kernel.load('n16k64_wA_g32_rt' + NAMES[wd], build_root=BB)
        table = RT.build(wsb, n, k, R.runs_geometry).cuda()
        fg = lambda: G.gemm(op['wp'], op['wsf'], xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gsx, check=False)
        fr = lambda: R.gemm(op['wp'], op['wsf'], xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gsx, check=False, runs=table)
        a, b = fg(), fr(); torch.cuda.synchronize()
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (layer, t)
        wm = {'g': [], 'r': []}; cm = {'g': [], 'r': []}
        for r in range(3):
            order = (('g', G, None, fg), ('r', R, table, fr)) if r % 2 == 0 else (('r', R, table, fr), ('g', G, None, fg))
            for kk, kern, rt, fn in order:
                fn(); torch.cuda.synchronize()
                wm[kk].append(warm(fn)); cm[kk].append(cold(kern, rt, op, n, k, t))
        wg, wr, cg, cr = (statistics.median(v) for v in (wm['g'], wm['r'], cm['g'], cm['r']))
        print(f"{layer.split('.')[-1]:9s} {n}x{k} T={t:5d} w{wd:3d} runs/warp max {int(table[:, 0].max()):2d} mean {float(table[:, 0].float().mean()):4.1f}: "
              f"warm g32 {wg:7.2f} rt {100 * (wr / wg - 1):+6.2f} % | cold g32 {cg:7.2f} rt {100 * (cr / cg - 1):+6.2f} %", flush=True)
