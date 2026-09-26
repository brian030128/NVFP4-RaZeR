"""#3 timing diagnostic (PROTOCOL_ITEMS.md, deviation 1; not a replacement of the registered numbers).

The registered GEMM benchmark (bench_gemm.py) shows, on this GPU, mixed-kernel CUPTI durations above the per-call
event time at T = 8192 on the large MLP shapes, and overheads far above both the RTX 5090 reference and what full-model
prefill shows. This separates three timing methods on the same operands, in alternating shuffled order over rounds:
    isolated     one call between two events, synchronized before and after (no overlap with other kernels)
    chained      events around 20 back-to-back calls, per call (the registered event time; consecutive GEMMs can overlap)
    cupti        the registered CUPTI kernel duration (median; for chained launches it can include time spent waiting
                 at a programmatic-dependent-launch barrier)
and records the SM clock, power and temperature around each round. Idle GPU required.

python results/tm_opt/gemm/diagnose_timing.py --model llama8b --out JSON
"""
import argparse
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_gemm import B, PAIRS, N, Kernel, place, weight, worst_modules  # noqa: E402

SHAPES = ('q_proj', 'gate_proj', 'down_proj')


def telemetry():
    out = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active',
                          '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True).stdout.strip()
    sm, mem, power, temp, reasons = [v.strip() for v in out.split(',')]
    return dict(sm_mhz=int(sm), mem_mhz=int(mem), power_w=float(power), temp_c=int(temp), throttle=reasons)


def isolated(fn, n=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record(); fn(); b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='llama8b')
    ap.add_argument('--tokens', default='2048,8192')
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    B.require_idle()
    kernels = {c: Kernel.load(c) for pair in PAIRS.values() for c in pair}
    res = dict(gpu=B.gpu_info(), model=args.model, rounds=[], summary={})
    cases = []
    for unit, (mixed, stock) in PAIRS.items():
        rows = int(unit.split('x')[0])
        worst = worst_modules(args.model, 'tmopt', unit)
        for proj in SHAPES:
            mask = worst[proj][0]
            n, k = mask.shape[0] * rows, mask.shape[1] * 64
            ops = {'e2m1': weight(n, k, None, (rows, 64), seed=n + k), 'tmopt': weight(n, k, mask, (rows, 64), seed=n + k)}
            for t in (int(v) for v in args.tokens.split(',')):
                g = torch.Generator(device='cpu').manual_seed(t)
                x = torch.randn(t, k, generator=g).cuda().bfloat16()
                xn, xsb, gsx = N.quantize_act(x, 'four_over_six_rows')
                act = (N.pack_nibbles(xn), place(xsb, k), gsx)
                for config, pattern in ((stock, 'e2m1'), (mixed, 'e2m1'), (mixed, 'tmopt')):
                    cases.append((unit, proj, n, k, t, config, pattern, ops[pattern], act))
    rng = random.Random(0)
    for r in range(args.rounds):
        order = list(range(len(cases)))
        rng.shuffle(order)
        for i in order:
            unit, proj, n, k, t, config, pattern, op, (xp, xsf, gsx) = cases[i]
            kern = kernels[config]
            if kern.weight_operand == 0:
                fn = lambda: kern.gemm(op['wp'], op['wsf'], xp, xsf, n, t, k, scale_m_default=op['gsw'], scale_n=gsx, check=False)  # noqa: E731
            else:
                fn = lambda: kern.gemm(xp, xsf, op['wp'], op['wsf'], t, n, k, scale_m=gsx, scale_n_default=op['gsw'], check=False)  # noqa: E731
            before = telemetry()
            iso = isolated(fn)
            chained = B.time_fn(fn, iters=20)['ms'] * 1e3
            kt = B.kernel_times(fn, iters=20, match=lambda s: 'cutlass' in s or 'Gemm' in s or 'device_kernel' in s)
            cupti = max(v['us'] for v in kt.values()) if kt else float('nan')
            res['rounds'].append(dict(round=r, unit=unit, proj=proj, tokens=t, config=config, pattern=pattern,
                                      isolated_us=iso, chained_us=chained, cupti_us=cupti, telemetry_before=before,
                                      telemetry_after=telemetry()))
        print(f'round {r} done', flush=True)
        B.write(args.out, res)
    for unit, (mixed, stock) in PAIRS.items():
        for proj in SHAPES:
            for t in (int(v) for v in args.tokens.split(',')):
                sel = [x for x in res['rounds'] if x['unit'] == unit and x['proj'] == proj and x['tokens'] == t]
                med = {(c, p): {m: statistics.median(x[m] for x in sel if (x['config'], x['pattern']) == (c, p))
                                for m in ('isolated_us', 'chained_us', 'cupti_us')} for c, p in ((stock, 'e2m1'), (mixed, 'e2m1'), (mixed, 'tmopt'))}
                base = med[(stock, 'e2m1')]
                res['summary'][f'{unit} {proj} T={t}'] = {
                    f'{c} {p}': {m: dict(us=v, vs_stock_pct=100 * (v / base[m] - 1)) for m, v in d.items()} for (c, p), d in med.items()}
    res['telemetry_range'] = {key: [min(x['telemetry_before'][key] for x in res['rounds']), max(x['telemetry_before'][key] for x in res['rounds'])]
                              for key in ('sm_mhz', 'power_w', 'temp_c')}
    B.write(args.out, res)
    print(json.dumps(res['summary'], indent=1))
    print(json.dumps(res['telemetry_range']))


if __name__ == '__main__':
    main()
