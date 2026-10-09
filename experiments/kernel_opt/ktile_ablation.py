#!/usr/bin/env python3
"""Kernel-opt amendment 20 (the K-tile-dispatch ablation, the paper's Figure 2(a)): what one format dispatch per K-tile
buys over choosing the format per MMA, on the RTX PRO 6000, at M = N = K = 4096.

    python experiments/kernel_opt/ktile_ablation.py check  --out results/kernel_opt/ktile_ablation/check.json
    python experiments/kernel_opt/ktile_ablation.py census --out results/kernel_opt/ktile_ablation/census.json
    python experiments/kernel_opt/ktile_ablation.py time   --out results/kernel_opt/ktile_ablation/time.json
    python experiments/kernel_opt/ktile_ablation.py report --dir results/kernel_opt/ktile_ablation

Operands (as C2U): weights N(0, 0.02) seed 4096, 4096 tokens N(0, 1) quantized FourOverSix per token; n = k = t = 4096.
Kernels (GEMM only):
  stock_ko   build_V's 'stock_ko' at this shape: the adopted table's width (stock_wA_e64) and scheduler row; NVFP4
             weights, weights on A
  fp8        cuBLAS FP8 through torch._scaled_mm: e4m3 operands with per-tensor FP32 scales, bf16 output (x W^T)
  permma16   n16k64_wA_e64_t0_permma (build_KT): the 16x64 per-MMA branch -- mixed_ko's width-128 tile, warp arrangement,
             placement (weights on A) and epilogue, with each MMA's format chosen by its own C++ if/else
  ktile16    build_V's 'mixed_ko' at this shape (n16k64_wA_e64_t0: one dispatch per K-tile), its table scheduler row
  permma8    n8k64_wB_t0_permma (build_KT): the 8x64 per-MMA branch, mixed_wB_ko's width-128 tile (weights on B)
  ktile8     build_V's 'mixed_wB_ko' at this shape (n8k64_wB_t0)
The per-MMA builds run with their per-K-tile counterpart's scheduler setting.
Tags: 'real' = model.layers.0.self_attn.o_proj of the Llama-3.1-8B TC map of the unit (C2U's); 'e2m1' = every tile E2M1.
check:  the per-MMA output equals the per-K-tile build's bitwise on the same operands, for the real, all-E2M1, all-E0M3
        and random 30 % tags (no timing).
census: C2's static SASS census of each build's GEMM function (OMMAs, predicated OMMAs, BRX, WARPSYNC, OMMAs per path
        through the steady k-loop iteration) and the tensor-pipe estimate at 4096^3 (OMMAs per iteration x 32 k-tiles x
        8 MMA warps x 1024 output tiles). Static counts, not measured counters (ncu is unavailable on this machine).
time:   isolated -- M1's cold weights: per call a 512 MiB read-flush and the next weight copy of a rotation over >= 4x L2;
        3 warm-up + 30 timed calls per configuration per round; and b2b -- 20 back-to-back calls per block over the
        same rotating copies (C2U's b2b). CUPTI kernel times; 3 rounds, round r starting the list at r * len / 3;
        the value per round is the median of its calls, the reported value the median of the 3 rounds. No clock
        locking and no ncu (not available). The registered checks run first (check), and no other compute process may
        be on the GPU.
"""
import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
sys.path.insert(0, str(REPO / 'experiments' / 'kernel_opt'))
sys.path.insert(0, str(REPO / 'experiments' / 'paper'))
sys.path.insert(0, str(REPO / 'experiments' / 'paper_extra'))

KO = Path('/home/dev/n16k64_campaign/kernel_opt')
BUILD_V, BUILD_KT, BUILD_KT_NODEF, BUILD_KT_REF = KO / 'build_V', KO / 'build_KT', KO / 'build_KT_nodef', KO / 'build_KT_ref'
CUDA_HOME = Path('/home/dev/.conda/envs/mixfp4-cuda131')
N_ = K_ = T_ = 4096
FLOP = 2 * N_ * K_ * T_
UNITS = {16: dict(permma='n16k64_wA_e64_t0_permma', ktile='n16k64_wA_e64_t0', family='mixed_ko', rows=16, on_b=False),
         8: dict(permma='n8k64_wB_t0_permma', ktile='n8k64_wB_t0', family='mixed_wB_ko', rows=8, on_b=True)}
TAGS = ('real', 'e2m1')


def operands():
    import torch
    from c2_freq import LAYER, MAPS
    from kernel import place
    from mixfp4_sm120 import mapio
    from mixfp4_sm120 import numerics as N
    g = torch.Generator('cpu').manual_seed(4096)
    w = (torch.randn(N_, K_, generator=g) * 0.02).cuda().bfloat16()
    x = torch.randn(T_, K_, generator=g).cuda().bfloat16()
    xn, xsb, gsx = N.quantize_act(x, 'four_over_six_rows')
    xq = (N.pack_nibbles(xn), place(xsb, K_), gsx)
    real = {u: mapio.read_map(MAPS[u])[1][LAYER] for u in UNITS}

    def wt(kind, mask=None, rows=None):
        wn, wsb, gsw = N.quantize_weight(w, kind, mask, (rows, 64) if mask is not None else None)
        return N.pack_nibbles(wn), place(wsb, K_), float(gsw)

    def tags(u, tag):
        rows = UNITS[u]['rows']
        grid = (N_ // rows, K_ // 64)
        if tag == 'real':
            mask = real[u]
        elif tag == 'e2m1':
            mask = torch.zeros(grid, dtype=torch.bool)
        elif tag == 'e0m3':
            mask = torch.ones(grid, dtype=torch.bool)
        else:                                          # 'rand30'
            mask = torch.rand(grid, generator=torch.Generator('cpu').manual_seed(20261009)) < 0.3
        assert tuple(mask.shape) == grid, (u, tag, tuple(mask.shape))
        return wt('map', mask, rows), float(mask.float().mean())
    return w, x, xq, wt, tags


def kernels():
    from mixfp4_sm120 import select as S
    from mixfp4_sm120.lib import Kernel
    from mixfp4_sm120.select import KernelSet
    table = S.TABLE_DIR / f'{S.gpu_slug()}.ko.json'
    sets = {'stock_ko': KernelSet('stock_ko', build_root=BUILD_V, table=table)}
    for u, d in UNITS.items():
        sets[d['family']] = KernelSet(d['family'], build_root=BUILD_V, table=table)
    K = {'stock_ko': sets['stock_ko'].pick(N_, K_, T_)}
    sched = {'stock_ko': tuple(sets['stock_ko'].schedule(N_, K_, T_))}
    for u, d in UNITS.items():
        ks = sets[d['family']]
        K[f'ktile{u}'] = ks.pick(N_, K_, T_)
        assert K[f'ktile{u}'].cfg.name == d['ktile'], (u, K[f'ktile{u}'].cfg.name)
        sched[f'ktile{u}'] = sched[f'permma{u}'] = tuple(ks.schedule(N_, K_, T_))
        K[f'permma{u}'] = Kernel.load(d['permma'], build_root=BUILD_KT)
    return K, sched, sets, table


def call(kern, wts, xq, on_b, sched):
    wp, wsf, gsw = wts
    xp, xsf, gsx = xq
    if not on_b:
        return lambda wp=wp, wsf=wsf: kern.gemm(wp, wsf, xp, xsf, N_, T_, K_, scale_m_default=gsw, scale_n=gsx,
                                                check=False, schedule=sched)
    return lambda wp=wp, wsf=wsf: kern.gemm(xp, xsf, wp, wsf, T_, N_, K_, scale_m=gsx, scale_n_default=gsw, check=False,
                                            schedule=sched)


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + '\n')


# ------------------------------------------------------------------------------------------------------------- check
def check(args):
    import torch
    import common as B
    w, x, xq, wt, tags = operands()
    K, sched, sets, table = kernels()
    res = dict(protocol='results/kernel_opt/PROTOCOL.md, amendment 20 (K-tile ablation): check', gpu=B.gpu_info(),
               kernels={k: dict(config=v.cfg.name, sha256=v.sha256, root=str(v.path.parent.parent),
                                defines=v.manifest.get('extra_defines'), schedule=list(sched[k])) for k, v in K.items()},
               table=str(table), checks=[])
    ok_all = True
    for u, d in UNITS.items():
        for tag in ('real', 'e2m1', 'e0m3', 'rand30'):
            wts, frac = tags(u, tag)
            ya = call(K[f'ktile{u}'], wts, xq, d['on_b'], sched[f'ktile{u}'])()
            yb = call(K[f'permma{u}'], wts, xq, d['on_b'], sched[f'permma{u}'])()
            torch.cuda.synchronize()
            ok = bool(torch.equal(ya.view(torch.int16), yb.view(torch.int16)))
            res['checks'].append(dict(unit=f'{u}x64', tags=tag, e0m3_fraction=frac, equal=ok,
                                      finite=bool(torch.isfinite(ya.float()).all()), ref=d['ktile'], new=d['permma']))
            print(f'{u}x64 {tag:6s} e0m3={frac:.4f} permma == ktile: {ok}', flush=True)
            ok_all &= ok
    # the FP8 reference: the cuBLAS kernel it launches, and its error against the BF16 product
    xf, wf, sx, sw = fp8_operands(w, x)
    yf = torch._scaled_mm(xf, wf.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)
    ref = (x.float() @ w.float().t())
    res['fp8'] = dict(rel_err=float((yf.float() - ref).norm() / ref.norm()), kernels=fp8_kernel_names(xf, wf, sx, sw))
    res['status'] = 'pass' if ok_all else 'FAIL'
    write(args.out, res)
    print('check', res['status'], 'fp8', res['fp8'], flush=True)
    if not ok_all:
        raise SystemExit('a per-MMA build differs from its per-K-tile build')


def fp8_operands(w, x):
    import torch
    sx = (x.float().abs().amax() / 448.0).reshape(())
    sw = (w.float().abs().amax() / 448.0).reshape(())
    xf = (x.float() / sx).to(torch.float8_e4m3fn)
    wf = (w.float() / sw).to(torch.float8_e4m3fn)
    return xf, wf, sx.cuda().float(), sw.cuda().float()


def fp8_kernel_names(xf, wf, sx, sw):
    import torch
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch._scaled_mm(xf, wf.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
    return sorted({e.name for e in prof.events() if e.device_type.name == 'CUDA'})


# ------------------------------------------------------------------------------------------------------------ census
def census(args):
    from C2_sass import blocks, gemm_function, loop_paths, parse
    from collections import Counter
    cuobjdump = str(CUDA_HOME / 'bin' / 'cuobjdump')
    libs = {'stock_ko': (BUILD_V, 'stock_wA_e64'), 'ktile16': (BUILD_V, 'n16k64_wA_e64_t0'),
            'permma16': (BUILD_KT, 'n16k64_wA_e64_t0_permma'), 'ktile8': (BUILD_V, 'n8k64_wB_t0'),
            'permma8': (BUILD_KT, 'n8k64_wB_t0_permma')}
    res = dict(caveat='static SASS census, not measured counters (ncu is unavailable on this machine)', kernels={},
               estimate='OMMAs per steady k-loop iteration x 32 k-tiles x 8 MMA warps x 1024 output tiles (4096^3, '
                        '128 x 128 CTA tiles, k-tile 128); an issued OMMA, predicated or not, takes a tensor-pipe slot')
    for key, (root, cfg) in libs.items():
        man = json.loads((root / cfg / 'manifest.json').read_text())
        lib = root / cfg / man['library']
        sass = subprocess.run([cuobjdump, '--dump-sass', str(lib)], capture_output=True, text=True, check=True).stdout
        text = gemm_function(sass)
        insns = parse(text)
        ops = Counter(i['base'] for i in insns)
        omma = [i for i in insns if i['base'] == 'OMMA']
        fmts = Counter((re.search(r'OMMA\.SF\.\d+\.F32\.(E\dM\d)\.(E\dM\d)', i['op'] + i['args']) or
                        re.search(r'(.)(.)', '??')).groups() for i in omma)
        loops = loop_paths(blocks(insns))
        # the steady k-loop: the innermost loop holding OMMAs (no inner back edge), as C2's "every path issues the same
        # OMMAs per iteration" reading
        inner = [lp for lp in loops if lp['inner_back_edges'] == 0 and lp['omma_per_iteration_max'] is not None]
        steady = min(inner, key=lambda lp: lp['blocks']) if inner else None
        per_iter = (steady['omma_per_iteration_min'], steady['omma_per_iteration_max']) if steady else (None, None)
        warps = 8
        est = per_iter[1] * 32 * warps * 1024 if per_iter[1] is not None else None
        res['kernels'][key] = dict(
            config=cfg, root=str(root), library_sha256=man['library_sha256'], sass_sha256=man['sass_sha256'],
            instructions=len(insns), omma=len(omma), omma_predicated=sum(1 for i in omma if i['pred']),
            omma_formats={f'{a}x{b}': n for (a, b), n in fmts.items()}, brx=ops.get('BRX', 0), bra=ops.get('BRA', 0),
            warpsync=ops.get('WARPSYNC', 0), bssy=ops.get('BSSY', 0), prmt=ops.get('PRMT', 0), loops=loops,
            omma_per_steady_iteration=dict(min=per_iter[0], max=per_iter[1]),
            steady_loop=dict(head=steady['head'], latch=steady['latch'], blocks=steady['blocks'],
                             omma_in_body=steady['omma_in_body']) if steady else None,
            tensor_pipe_estimate_4096=est, patch_census=man['patch'].get('sites'),
            resource_usage=man.get('resource_usage'))
        k = res['kernels'][key]
        print(key, cfg, {x: k[x] for x in ('instructions', 'omma', 'omma_predicated', 'brx', 'warpsync',
                                           'omma_per_steady_iteration', 'tensor_pipe_estimate_4096')}, flush=True)
    # gate: the new hooks are inactive unless asked for -- the deployed builds rebuilt from these sources have
    # build_V's SASS; the per-MMA builds do not depend on their counterparts' (dispatch-only) defines
    gates = {}
    for cfg in ('n16k64_wA_e64_t0', 'n8k64_wB_t0', 'stock_wA_e64'):
        a = json.loads((BUILD_V / cfg / 'manifest.json').read_text())
        b = json.loads((BUILD_KT_REF / cfg / 'manifest.json').read_text())
        gates[f'build_KT_ref/{cfg} == build_V'] = dict(
            sass=a['sass_sha256'] == b['sass_sha256'], unpatched_sass=a['unpatched_sass_sha256'] == b['unpatched_sass_sha256'],
            defines=(a.get('extra_defines'), b.get('extra_defines')))
    for cfg in ('n16k64_wA_e64_t0_permma', 'n8k64_wB_t0_permma'):
        a = json.loads((BUILD_KT / cfg / 'manifest.json').read_text())
        b = json.loads((BUILD_KT_NODEF / cfg / 'manifest.json').read_text())
        gates[f'build_KT_nodef/{cfg} == build_KT'] = dict(sass=a['sass_sha256'] == b['sass_sha256'],
                                                          defines=(a.get('extra_defines'), b.get('extra_defines')))
    res['gates'] = gates
    print(json.dumps(gates, indent=1), flush=True)
    write(args.out, res)
    if not all(v['sass'] for v in gates.values()):
        raise SystemExit('a SASS-equality gate failed')


# -------------------------------------------------------------------------------------------------------------- time
def timing(args):
    import torch
    from torch.profiler import ProfilerActivity, profile
    import common as B
    import bench_gemm_isolated as G
    B.require_idle()
    tel = G.Telemetry()
    if tel.others():
        raise SystemExit(f'another compute process is on the GPU: {tel.others()}')
    w, x, xq, wt, tags = operands()
    K, sched, sets, table = kernels()
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    flush = torch.ones(args.flush_mib * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
    if args.flush_mib * 2 ** 20 < 4 * l2:
        raise SystemExit('the flush is below 4x the L2')

    def rotation(wts):
        wp, wsf, gsw = wts
        n = math.ceil(4 * l2 / (wp.numel() * wp.element_size() + wsf.numel() * wsf.element_size())) + 1
        return [(wp.clone(), wsf.clone(), gsw) for _ in range(n)]

    # configurations: name -> (kernel key, weights, weights on B, unit, tags)
    cfgs = {'stock_ko': ('stock_ko', wt('nvfp4'), False, None, 'nvfp4'), 'fp8': ('fp8', None, False, None, 'fp8')}
    for u, d in UNITS.items():
        for tag in TAGS:
            wts, frac = tags(u, tag)
            for kind in ('permma', 'ktile'):
                cfgs[f'{kind}{u}_{tag}'] = (f'{kind}{u}', wts, d['on_b'], u, tag)
    fns, copies, ys = {}, {}, {}
    xf, wf, sx, sw = fp8_operands(w, x)
    for name, (kk, wts, on_b, u, tag) in cfgs.items():
        if kk == 'fp8':
            n = math.ceil(4 * l2 / (wf.numel() * wf.element_size())) + 1
            copies[name] = [wf.clone() for _ in range(n)]
            fns[name] = (lambda c, xf=xf, sx=sx, sw=sw: torch._scaled_mm(xf, c.t(), scale_a=sx, scale_b=sw,
                                                                           out_dtype=torch.bfloat16))
        else:
            copies[name] = rotation(wts)
            f = call(K[kk], wts, xq, on_b, sched[kk])
            fns[name] = (lambda c, f=f: f(c[0], c[1]))
    # registered check on the timed operands: per-MMA == per-K-tile bitwise
    res = dict(protocol='results/kernel_opt/PROTOCOL.md, amendment 20 (K-tile ablation): time', gpu=B.gpu_info(),
               l2_bytes=l2, flush_mib=args.flush_mib, rounds=args.rounds, reps=args.reps, warmup=args.warmup, b2b=args.b2b,
               power_limit_w=tel.power_limit_w(),
               kernels={k: dict(config=v.cfg.name, sha256=v.sha256, root=str(v.path.parent.parent),
                                defines=v.manifest.get('extra_defines'), schedule=list(sched[k])) for k, v in K.items()},
               fp8=dict(api='torch._scaled_mm(e4m3 x [T, K], e4m3 W^T, per-tensor fp32 scales, bf16 out)', torch=torch.__version__,
                        kernels=fp8_kernel_names(xf, wf, sx, sw)),
               rotation_copies={k: len(v) for k, v in copies.items()}, checks={}, modes={})
    for u in UNITS:
        for tag in TAGS:
            a, b = fns[f'ktile{u}_{tag}'](copies[f'ktile{u}_{tag}'][0]), fns[f'permma{u}_{tag}'](copies[f'permma{u}_{tag}'][0])
            torch.cuda.synchronize()
            ok = bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
            res['checks'][f'permma{u}_{tag} == ktile{u}_{tag}'] = ok
            if not ok:
                write(args.out, res)
                raise SystemExit(f'permma{u}_{tag} differs from ktile{u}_{tag}')
    names = list(cfgs)
    counter = {nm: 0 for nm in names}

    def nxt(nm):
        c = copies[nm][counter[nm] % len(copies[nm])]
        counter[nm] += 1
        return c

    def kernel_us(prof, nm):
        # every configuration: the GEMM kernel only (bench_gemm_isolated.classify: a name with 'cutlass', 'device_kernel'
        # or 'gemm' -- the cuBLAS FP8 kernel is sm89_xmma_gemm_...); not the flush's reduce kernel or its memset
        # (deviation 1: the first run also counted the memset for fp8 and stopped at the count check)
        return [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and G.classify(e.name) == 'gemm']

    for mode in ('isolated', 'b2b'):
        per = {nm: [] for nm in names}
        for r in range(args.rounds):
            shift = (r * len(names) // args.rounds) % len(names)
            for pos, nm in enumerate(names[shift:] + names[:shift]):
                if tel.others():
                    res['checks']['other_processes'] = tel.others()
                    write(args.out, res)
                    raise SystemExit(f'another compute process on the GPU: {tel.others()}')
                fn = fns[nm]
                before = tel.snap()
                if mode == 'isolated':
                    for _ in range(args.warmup):
                        flush.sum()
                        torch.cuda.synchronize()
                        fn(nxt(nm))
                        torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        for _ in range(args.reps):
                            flush.sum()
                            torch.cuda.synchronize()
                            fn(nxt(nm))
                            torch.cuda.synchronize()
                    want = args.reps
                else:
                    fn(nxt(nm))
                    torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        for _ in range(args.b2b):
                            fn(nxt(nm))
                        torch.cuda.synchronize()
                    want = args.b2b
                us = kernel_us(prof, nm)
                if len(us) != want:
                    res['checks']['counts'] = f'{mode} {nm} round {r}: {len(us)} kernels profiled, expected {want}'
                    write(args.out, res)
                    raise SystemExit(res['checks']['counts'])
                per[nm].append(dict(round=r, position=pos, us=statistics.median(us), calls=us,
                                    telemetry=G.Telemetry.block(before, tel.snap())))
                print(f'{mode} r{r} {nm:16s} {per[nm][-1]["us"]:8.2f} us', flush=True)
        res['modes'][mode] = {nm: dict(us=statistics.median(b['us'] for b in per[nm]),
                                       pooled_us=statistics.median(v for b in per[nm] for v in b['calls']),
                                       rounds=per[nm]) for nm in names}
        write(args.out, res)
    res['gpu_end'] = B.gpu_info()
    res['status'] = 'complete'
    write(args.out, res)


# ------------------------------------------------------------------------------------------------------------ report
def report(args):
    d = Path(args.dir)
    t = json.loads((d / 'time.json').read_text())
    c = json.loads((d / 'census.json').read_text())
    ck = json.loads((d / 'check.json').read_text())
    tf = lambda us: FLOP / (us * 1e-6) / 1e12  # noqa: E731
    out = dict(source=dict(time='time.json', census='census.json', check='check.json'), modes={})
    L = ['# K-tile dispatch ablation at 4096³ on the RTX PRO 6000 (kernel-opt amendment 20; the paper\'s Figure 2(a))', '',
         'What one format dispatch per K-tile buys over choosing the format per MMA. GEMM only, M = N = K = 4096 '
         '(weights N(0, 0.02), 4096 tokens N(0, 1) with per-token FourOverSix, as C2U). Kernel time by CUPTI; median of 3 '
         'rotated rounds (each the median of its calls). No clock locking and no ncu (neither is available).', '']
    rows = [('stock_ko', 'stock NVFP4 (stock_ko)', None), ('fp8', 'FP8 (cuBLAS, torch._scaled_mm)', None)]
    for u in UNITS:
        for tag in TAGS:
            rows += [(f'permma{u}_{tag}', f'{u}x64 per-MMA branch, {tag} tags', u), (f'ktile{u}_{tag}', f'{u}x64 per-K-tile (build_V), {tag} tags', u)]
    for mode in ('isolated', 'b2b'):
        m = t['modes'][mode]
        s = m['stock_ko']['us']
        L += [f'## {mode}', '', '| kernel | µs | TFLOP/s | vs stock_ko | per round (µs) |', '|---|---:|---:|---:|---|']
        out['modes'][mode] = {}
        for key, label, _ in rows:
            v = m[key]
            out['modes'][mode][key] = dict(label=label, us=v['us'], tflops=tf(v['us']), vs_stock=100 * (v['us'] / s - 1),
                                           rounds_us=[b['us'] for b in v['rounds']])
            L.append(f'| {label} | {v["us"]:.2f} | {tf(v["us"]):.1f} | {100 * (v["us"] / s - 1):+.1f} % | '
                     + ' / '.join(f'{b["us"]:.2f}' for b in v['rounds']) + ' |')
        L.append('')
    L += ['## Static SASS census (C2 style; static counts, not measured counters)', '',
          '| kernel | build | instructions | OMMA | predicated OMMA | BRX | WARPSYNC | OMMA per steady k-iteration | '
          'tensor-pipe estimate at 4096³ |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, k in c['kernels'].items():
        it = k['omma_per_steady_iteration']
        L.append(f'| {key} | {k["config"]} | {k["instructions"]} | {k["omma"]} | {k["omma_predicated"]} | {k["brx"]} | '
                 f'{k["warpsync"]} | {it["min"]}–{it["max"]} | {k["tensor_pipe_estimate_4096"]:,} |')
    L += ['', 'Checks: ' + json.dumps(dict(bitwise=ck['status'], timed_operands=t['checks'], sass_gates={g: v['sass'] for g, v in c['gates'].items()})), '']
    out['census'] = {k: dict(omma=v['omma'], predicated=v['omma_predicated'], per_iteration=v['omma_per_steady_iteration'],
                             estimate=v['tensor_pipe_estimate_4096']) for k, v in c['kernels'].items()}
    (d / 'REPORT_tables.md').write_text('\n'.join(L) + '\n')
    (d / 'ktile_ablation.json').write_text(json.dumps(out, indent=1) + '\n')
    # LaTeX-ready numbers (isolated mode, real tags)
    m = out['modes']['isolated']
    tex = ['% K-tile dispatch ablation, 4096^3, RTX PRO 6000, isolated launches (CUPTI, cold weights), real tags',
           f'\\newcommand{{\\ktStockTF}}{{{m["stock_ko"]["tflops"]:.0f}}}',
           f'\\newcommand{{\\ktFPeightTF}}{{{m["fp8"]["tflops"]:.0f}}}']
    for u, w in ((16, 'Sixteen'), (8, 'Eight')):
        for kind, nm in (('permma', 'PerMMA'), ('ktile', 'PerKtile')):
            v = m[f'{kind}{u}_real']
            tex += [f'\\newcommand{{\\kt{nm}{w}TF}}{{{v["tflops"]:.0f}}}',
                    f'\\newcommand{{\\kt{nm}{w}Ovh}}{{{v["vs_stock"]:+.1f}}}']
    (d / 'numbers.tex').write_text('\n'.join(tex) + '\n')
    plot(out, d / 'tflops.png')
    print('\n'.join(L))


def plot(out, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = out['modes']['isolated']
    groups = [('16x64', 16), ('8x64', 8)]
    labels = ['stock NVFP4', 'FP8 (cuBLAS)', 'per-MMA branch', 'per-K-tile']
    colors = ['#7f7f7f', '#bcbd22', '#d62728', '#1f77b4']
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    width = 0.2
    for gi, (gname, u) in enumerate(groups):
        vals = [m['stock_ko']['tflops'], m['fp8']['tflops'], m[f'permma{u}_real']['tflops'], m[f'ktile{u}_real']['tflops']]
        for bi, v in enumerate(vals):
            xpos = gi + (bi - 1.5) * width
            ax.bar(xpos, v, width, color=colors[bi], label=labels[bi] if gi == 0 else None)
            ax.text(xpos, v + 15, f'{v:.0f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f'{g} format unit' for g, _ in groups])
    ax.set_ylabel('TFLOP/s (4096³, isolated, CUPTI)')
    ax.set_title('RTX PRO 6000: per-MMA branch vs per-K-tile dispatch (real tags)', fontsize=9)
    ax.legend(fontsize=7, ncol=4, loc='upper center', frameon=False)
    ax.set_ylim(0, max(m[k]['tflops'] for k in m) * 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=('check', 'census', 'time', 'report'))
    ap.add_argument('--out', type=Path)
    ap.add_argument('--dir', type=Path, default=REPO / 'results' / 'kernel_opt' / 'ktile_ablation')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--b2b', type=int, default=20)
    ap.add_argument('--flush-mib', type=int, default=512)
    args = ap.parse_args()
    {'check': check, 'census': census, 'time': timing, 'report': report}[args.what](args)


if __name__ == '__main__':
    main()
