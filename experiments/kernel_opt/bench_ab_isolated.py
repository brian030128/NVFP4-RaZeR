#!/usr/bin/env python3
"""Kernel-opt measurement M1: GEMM time before vs after, by the registered deviation-2 method.

    python experiments/kernel_opt/bench_ab_isolated.py --model llama8b --after-root /home/dev/n16k64_campaign/kernel_opt/build \
        --artifact fo6=ART --artifact tc_8x64=ART --out JSON

Protocol: results/kernel_opt/PROTOCOL.md (M1). The repetition, cold weights (rotation + a 512 MiB read-flush), CUPTI
kernel times, rotated order over --rounds rounds, tags (typical = lower-median module, worst = densest), telemetry and
registered checks are those of experiments/paper/bench_gemm_isolated.py (results/paper/PROTOCOL_GEMM_ISOLATED.md), whose
helpers this script imports. The differences:
- configurations (CONFIGS): the 8x64 maps on the current n8k64_wB (sm120/build, "before") and on the width-selecting
  weights-on-B set 'auto_wB' loaded from --after-root ("after"), with FourOverSix on the stock sets as references;
- tokens: decode-sized counts too (TOKENS);
- each row records the build the call ran (and, for a set, the width).
"""
import argparse
import dataclasses
import json
import math
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'experiments' / 'paper'))
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import bench_gemm_isolated as G  # noqa: E402  (the deviation-2 script; helpers only)
import common as B  # noqa: E402
from mixfp4_sm120 import artifact as A  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import NativeLinear  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

# configuration -> (artifact kind, tag variant, kernel, activation quantizer)
CONFIGS = OrderedDict(
    stock_wA=('fo6', 'first', 'stock', 'four_over_six_rows'),
    stock_wB=('fo6', 'first', 'stock_wB', 'four_over_six_rows'),
    n8k64_wB_typical=('tc_8x64', 'typical', 'n8k64_wB', 'four_over_six_rows'),
    n8k64_wB_worst=('tc_8x64', 'worst', 'n8k64_wB', 'four_over_six_rows'),
    auto_wB_typical=('tc_8x64', 'typical', 'auto_wB', 'four_over_six_rows'),
    auto_wB_worst=('tc_8x64', 'worst', 'auto_wB', 'four_over_six_rows'))
TOKENS = (1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', action='append', required=True, metavar='KIND=DIR')
    ap.add_argument('--after-root', required=True, help='build directory of the kernel-opt builds')
    ap.add_argument('--tokens', default=','.join(map(str, TOKENS)))
    ap.add_argument('--projections', default=None)
    ap.add_argument('--configs', default=None)
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--flush-mib', type=int, default=512)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    csv_path = Path(str(args.out) + '.telemetry.csv')
    tel = G.Telemetry(csv_path)
    try:
        run(args, tel)
    finally:
        tel.close()
    res = json.loads(args.out.read_text())
    res['sampler'] = G.Telemetry.summarize(csv_path)
    B.write(args.out, res)


def run(args, tel):
    torch.backends.cuda.matmul.allow_tf32 = False
    arts = dict(spec.split('=', 1) for spec in args.artifact)
    kernels = dict(stock=KernelSet('stock'), stock_wB=Kernel.load('stock_wB'), n8k64_wB=Kernel.load('n8k64_wB'),
                   auto_wB=KernelSet('mixed_wB', build_root=args.after_root))
    before_root = REPO / 'sm120' / 'build'
    for name in ('stock_wB', 'n8k64_wB'):
        assert Path(kernels[name].path).parent.parent == before_root, kernels[name].path
    assert all(Path(k.path).parent.parent == before_root for k in kernels['stock'].kernels.values())
    assert all(Path(k.path).parent.parent == Path(args.after_root) for k in kernels['auto_wB'].kernels.values())
    cfgs = [c for c, (kind, *_) in CONFIGS.items() if kind in arts and (not args.configs or c in args.configs.split(','))]
    res = dict(status='running', gpu=B.gpu_info(), power_limit_w=tel.power_limit_w(),
               l2_bytes=torch.cuda.get_device_properties(0).L2_cache_size, model=args.model, artifacts=arts,
               after_root=args.after_root,
               protocol=dict(method='isolated launches, cold weights (results/paper/PROTOCOL_GEMM_ISOLATED.md), '
                                    'kernel-opt M1 (results/kernel_opt/PROTOCOL.md)',
                             reps=args.reps, warmup=args.warmup, rounds=args.rounds, flush_mib=args.flush_mib,
                             order='rotated: round r starts at position r * len / rounds',
                             value='median over all rounds of the CUPTI device time of isolated launches'),
               kernels={k: (v.sha256 if isinstance(v, Kernel) else v.describe()) for k, v in kernels.items()},
               configs={c: dict(artifact=CONFIGS[c][0], tags=CONFIGS[c][1], kernel=CONFIGS[c][2],
                                activation_quantizer=CONFIGS[c][3]) for c in cfgs},
               projections={}, rows=[], blocks=[], checks=dict(bitwise=[], counts_ok=True, other_processes=[]))
    if args.flush_mib * 2 ** 20 < 4 * res['l2_bytes']:
        raise SystemExit(f'--flush-mib {args.flush_mib} is below 4x the L2 ({res["l2_bytes"]} bytes)')
    tags, pw = {}, {}
    for kind, path in arts.items():
        meta, weights = A.load(path, device='cpu')
        tags[kind] = G.tag_modules(meta, weights)
        for proj, t in tags[kind].items():
            p = res['projections'].setdefault(proj, dict(shape=t['shape'], modules=t['modules'], tags={}))
            assert p['shape'] == t['shape'] and p['modules'] == t['modules'], (kind, proj)
            p['tags'][kind] = dict(type_block=meta['type_block'], tiles_per_module=t['tiles_per_module'],
                                   artifact_weights_sha256=meta['weights_sha256'],
                                   **{v: dict(module=t[v][0], e0m3_tiles=t[v][1],
                                              e0m3_share=None if not t['tiles_per_module'] else t[v][1] / t['tiles_per_module'])
                                      for v in ('first', 'typical', 'worst')})
            for v in ('first', 'typical', 'worst'):
                pw[(kind, v, proj)] = weights[t[v][0]]
        del weights
    projs = list(res['projections'])
    if args.projections:
        projs = [p for p in projs if p in args.projections.split(',')]
    tokens = [int(t) for t in args.tokens.split(',')]
    flush = torch.ones(args.flush_mib * 2 ** 20 // 4, dtype=torch.float32, device='cuda')
    stream = torch.cuda.current_stream()
    for proj in projs:
        n, k = res['projections'][proj]['shape']
        lins, copies, counter = {}, {}, {}
        for c in cfgs:
            kind, variant, kname, act = CONFIGS[c]
            w = pw[(kind, variant, proj)]
            w = dataclasses.replace(w, packed=w.packed.cuda(), scales=w.scales.cuda(), bias=None if w.bias is None else w.bias.cuda())
            lins[c] = NativeLinear(w, kernels[kname], act, name=f'{proj}@{c}')
            lins[c].share_input = False
            size = lins[c].packed.numel() + lins[c].sf.numel()
            copies[c] = [(lins[c].packed.clone(), lins[c].sf.clone()) for _ in range(math.ceil(4 * res['l2_bytes'] / size) + 1)]
            counter[c] = 0
        res['projections'][proj]['rotation_copies'] = {c: len(copies[c]) for c in cfgs}
        for t in tokens:
            g = torch.Generator('cpu').manual_seed(n + k + t)
            pool = [torch.randn(t, k, generator=g).to('cuda', torch.bfloat16) for _ in range(2)]
            x = torch.empty_like(pool[0])
            per = {c: [] for c in cfgs}
            fns, ys = {}, {}
            for c in cfgs:
                lin = lins[c]
                kern = lin.kernel_set.pick(n, k, t) if lin.kernel_set is not None else lin.kernel
                y = torch.empty((t, n), dtype=torch.bfloat16, device='cuda')
                bptr = None if lin.bias_bf16 is None else lin.bias_bf16.data_ptr()

                def quant(kern=kern, act=lin.act_kind):
                    return kern.quant_rows(x, act)

                def gemm(q, kern=kern, lin=lin, y=y, bptr=bptr, c=c):
                    xp, xsf, gs = q
                    wp, wsf = copies[c][counter[c] % len(copies[c])]
                    counter[c] += 1
                    if lin.weights_on_a:
                        kern.gemm_ptr(wp.data_ptr(), wsf.data_ptr(), xp.data_ptr(), xsf.data_ptr(), n, t, k,
                                      None, lin.global_scale, gs.data_ptr(), 1.0, bptr, y, stream.cuda_stream)
                    else:
                        kern.gemm_ptr(xp.data_ptr(), xsf.data_ptr(), wp.data_ptr(), wsf.data_ptr(), t, n, k,
                                      gs.data_ptr(), 1.0, None, lin.global_scale, bptr, y, stream.cuda_stream)
                    return y
                fns[c] = (quant, gemm, kern)
                # registered check (deviation 2): the isolated path equals NativeLinear's fused forward bitwise
                x.copy_(pool[0])
                ref = lin(x)
                got = gemm(quant()).clone()
                ok = bool(torch.equal(ref.view(torch.int16), got.view(torch.int16)))
                res['checks']['bitwise'].append(dict(proj=proj, tokens=t, config=c, equal=ok))
                if not ok:
                    res['status'] = 'failed'
                    B.write(args.out, res)
                    raise G.CheckFailed(f'{proj} T={t} {c}: the isolated GEMM differs from NativeLinear')
                ys[c] = y
            # kernel-opt: before and after must give the same output bits on the same tags (G4 on the timed operands)
            for v in ('typical', 'worst'):
                a, b = f'n8k64_wB_{v}', f'auto_wB_{v}'
                if a in ys and b in ys:
                    ok = bool(torch.equal(ys[a].view(torch.int16), ys[b].view(torch.int16)))
                    res['checks']['bitwise'].append(dict(proj=proj, tokens=t, config=f'{b} == {a}', equal=ok))
                    if not ok:
                        res['status'] = 'failed'
                        B.write(args.out, res)
                        raise G.CheckFailed(f'{proj} T={t}: {b} differs from {a}')

            def rep(c, i, timed):
                quant, gemm, _ = fns[c]
                flush.sum()
                torch.cuda.synchronize()
                x.copy_(pool[i % 2])
                torch.cuda.synchronize()
                q = quant()
                torch.cuda.synchronize()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                gemm(q)
                b.record()
                torch.cuda.synchronize()
                return a.elapsed_time(b) * 1e3 if timed else None

            block_before = tel.snap()
            for r in range(args.rounds):
                shift = (r * len(cfgs) // args.rounds) % len(cfgs)
                for pos, c in enumerate(cfgs[shift:] + cfgs[:shift]):
                    others = tel.others()
                    if others:
                        res['checks']['other_processes'].append(dict(proj=proj, tokens=t, config=c, round=r, pids=others))
                        res['status'] = 'failed'
                        B.write(args.out, res)
                        raise G.CheckFailed(f'another compute process on the GPU: {others}')
                    for i in range(args.warmup):
                        rep(c, i, False)
                    before = tel.snap()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        ev = [rep(c, i, True) for i in range(args.reps)]
                    after = tel.snap()
                    kt = {'gemm': [], 'quant': []}
                    for e in prof.events():
                        if e.device_type.name == 'CUDA':
                            cls = G.classify(e.name)
                            if cls in kt:
                                kt[cls].append(e.device_time_total if hasattr(e, 'device_time_total') else e.cuda_time_total)
                    if len(kt['gemm']) != args.reps or len(kt['quant']) != args.reps:
                        res['checks']['counts_ok'] = False
                        res['status'] = 'failed'
                        B.write(args.out, res)
                        raise G.CheckFailed(f'{proj} T={t} {c} round {r}: {len(kt["gemm"])} GEMM / {len(kt["quant"])} '
                                            f'quantizer launches profiled, expected {args.reps}')
                    per[c].append(dict(round=r, position=pos, gemm_us=kt['gemm'], quant_us=kt['quant'], event_us=ev,
                                       telemetry=G.Telemetry.block(before, after)))
            res['blocks'].append(dict(proj=proj, tokens=t, **G.Telemetry.block(block_before, tel.snap())))
            for c in cfgs:
                kind, variant, kname, act = CONFIGS[c]
                kern = fns[c][2]
                g_all = [v for blk in per[c] for v in blk['gemm_us']]
                q_all = [v for blk in per[c] for v in blk['quant_us']]
                e_all = [v for blk in per[c] for v in blk['event_us']]
                res['rows'].append(dict(
                    proj=proj, out=n, inp=k, tokens=t, config=c, kind=kind, tags=variant, kernel=kern.cfg.name,
                    kernel_sha256=kern.sha256, act=act,
                    width=lins[c].kernel_set.width(n, k, t) if lins[c].kernel_set is not None else None,
                    gemm_us=statistics.median(g_all), gemm=G.stats(g_all), quant_us=statistics.median(q_all),
                    quant=G.stats(q_all), event_us=statistics.median(e_all), event=G.stats(e_all),
                    rounds=[dict(round=b['round'], position=b['position'], gemm_us=statistics.median(b['gemm_us']),
                                 quant_us=statistics.median(b['quant_us']), event_us=statistics.median(b['event_us']),
                                 telemetry=b['telemetry']) for b in per[c]]))
            print(f"{args.model} {proj} T={t} " + ' '.join(f"{r['config']}={r['gemm_us']:.1f}" for r in res['rows']
                                                           if r['proj'] == proj and r['tokens'] == t), flush=True)
            del pool, x, ys, fns
            B.write(args.out, res)
        del lins, copies
    res['kernel_sets_after'] = {k: v.describe() for k, v in kernels.items() if isinstance(v, KernelSet)}
    res['gpu_end'] = B.gpu_info()
    res['status'] = 'complete'
    B.write(args.out, res)


if __name__ == '__main__':
    main()
