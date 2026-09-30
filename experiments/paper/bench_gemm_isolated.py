#!/usr/bin/env python3
"""GEMM and activation-quantizer kernel times per text-Linear shape: isolated launches, cold weights (deviation 2).

    PAPER_PYTHON experiments/paper/bench_gemm_isolated.py --model llama8b --artifact fo6=ART --artifact nvfp4=ART \
        --artifact tc_8x64=ART --artifact tc_16x64=ART --artifact tc_256x64=ART --out JSON

Protocol: results/paper/PROTOCOL_GEMM_ISOLATED.md (results/paper/PROTOCOL.md, deviation 2). Step 06's CUPTI
measurement of 20 back-to-back calls on L2-resident weights of the densest module (bench_gemm.py) is kept as the
alternative method.

**One repetition.** Every timed kernel is launched on an idle GPU (synchronized before; nothing queued):
  1. flush: a reduction reads a --flush-mib buffer (default 512 MiB, 4x the 128 MiB L2); synchronize. The read evicts
     the weights from L2 and leaves clean lines, so the GEMM's misses cause no write-backs;
  2. a fresh activation: a seeded BF16 input [T, in] is copied from a pool of two into the input buffer; synchronize;
  3. TIMED: the activation quantizer (Kernel.quant_rows, the kernel sm120_linear launches); synchronize;
  4. TIMED: the GEMM with its fused epilogue (Kernel.gemm_ptr, the kernel sm120_linear launches), reading its weights
     from DRAM and the just-quantized activations from L2; synchronize.
**Kernel time:** the CUPTI device duration of each timed launch (torch.profiler over one block of --reps repetitions
of one configuration). An event pair around each GEMM launch is recorded too; it includes the host enqueue gap.
**Order:** at each (projection, T), every configuration is measured in each of --rounds rounds (default 3), round r
starting the configuration list at position r * len / rounds (deviation 1's rotation), --reps repetitions (default 30)
per configuration per round, after --warmup untimed repetitions.
**Configurations:** see CONFIGS. The FlipQuant (ours) maps are timed with two modules' tags per projection:
  typical  the module whose E0M3 tile share is the lower median over the projection's modules (ties: layer order);
  worst    the densest module (ties: the first), as step 06 did.
FourOverSix and NVFP4 have no tags: the projection's first module.
**Checks (registered; a failure stops the run):**
  - before timing, each configuration's output at each (projection, T) equals NativeLinear's fused forward bitwise;
  - every profiled block holds exactly --reps GEMM and --reps quantizer launches;
  - no other compute process on the GPU at any block (NVML).
**Telemetry:** NVML before and after every block: SM clock, power, energy, the cumulative power-cap violation time and
the clock-event reasons. A 100 ms nvidia-smi sampler runs over the whole process (<out>.telemetry.csv).
"""
import argparse
import dataclasses
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import pynvml
import torch
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'sm120'))
sys.path.insert(0, str(REPO / 'sm120' / 'bench'))
import common as B  # noqa: E402  (sm120/bench/common.py)
from mixfp4_sm120 import artifact as A  # noqa: E402
from mixfp4_sm120.lib import Kernel  # noqa: E402
from mixfp4_sm120.linear import NativeLinear  # noqa: E402
from mixfp4_sm120.select import KernelSet  # noqa: E402

# configuration -> (artifact kind, tag variant, kernel, activation quantizer)
CONFIGS = OrderedDict(
    stock_wA=('fo6', 'first', 'stock', 'four_over_six_rows'),
    stock_wA_nvfp4=('nvfp4', 'first', 'stock', 'nvfp4_rows'),
    stock_wB=('fo6', 'first', 'stock_wB', 'four_over_six_rows'),
    mixed_16x64_typical=('tc_16x64', 'typical', 'mixed', 'four_over_six_rows'),
    mixed_16x64_worst=('tc_16x64', 'worst', 'mixed', 'four_over_six_rows'),
    mixed_256x64_typical=('tc_256x64', 'typical', 'mixed', 'four_over_six_rows'),
    mixed_256x64_worst=('tc_256x64', 'worst', 'mixed', 'four_over_six_rows'),
    n8k64_wB_typical=('tc_8x64', 'typical', 'n8k64_wB', 'four_over_six_rows'),
    n8k64_wB_worst=('tc_8x64', 'worst', 'n8k64_wB', 'four_over_six_rows'))
L2_BYTES = 128 * 2 ** 20


class CheckFailed(SystemExit):
    pass


def classify(name):
    n = name.lower()
    if 'quant_rows_kernel' in n:
        return 'quant'
    if 'cutlass' in n or 'device_kernel' in n or 'gemm' in n:
        return 'gemm'
    return 'other'


def tag_modules(meta, weights):
    """{projection: dict(shape, modules, tiles_per_module, first, typical, worst)}; each choice is (name, E0M3 tiles)."""
    by = OrderedDict()
    for m in meta['modules']:
        by.setdefault(m['name'].rsplit('.', 1)[-1], []).append(m)
    tb = meta['type_block']
    out = OrderedDict()
    for proj, mods in by.items():
        shape = tuple(weights[mods[0]['name']].shape)
        assert all(tuple(weights[m['name']].shape) == shape for m in mods), proj
        order = sorted(range(len(mods)), key=lambda i: (mods[i]['e0m3_tiles'], i))
        typical = mods[order[(len(mods) - 1) // 2]]
        worst = max(mods, key=lambda m: m['e0m3_tiles'])          # the first maximum
        out[proj] = dict(shape=list(shape), modules=len(mods),
                         tiles_per_module=None if not tb else -(-shape[0] // tb[0]) * (shape[1] // tb[1]),
                         first=(mods[0]['name'], mods[0]['e0m3_tiles']), typical=(typical['name'], typical['e0m3_tiles']),
                         worst=(worst['name'], worst['e0m3_tiles']),
                         shares=sorted(m['e0m3_tiles'] for m in mods))
    return out


class Telemetry:
    """NVML snapshots, and a background nvidia-smi sampler."""

    def __init__(self, csv_path=None):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.pid = None
        self.proc = None
        if csv_path is not None:
            q = ('timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active,'
                 'clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown')
            self.csv = open(csv_path, 'w')
            self.proc = subprocess.Popen(['nvidia-smi', f'--query-gpu={q}', '--format=csv,nounits', '-lms', '100'],
                                         stdout=self.csv, stderr=subprocess.DEVNULL)

    def snap(self):
        reasons = (pynvml.nvmlDeviceGetCurrentClocksEventReasons(self.h) if hasattr(pynvml, 'nvmlDeviceGetCurrentClocksEventReasons')
                   else pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.h))
        return dict(t=time.time(), sm_mhz=pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM),
                    mem_mhz=pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_MEM),
                    power_w=pynvml.nvmlDeviceGetPowerUsage(self.h) / 1e3,
                    temp_c=pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU),
                    energy_mj=pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h),
                    power_cap_ns=pynvml.nvmlDeviceGetViolationStatus(self.h, pynvml.NVML_PERF_POLICY_POWER).violationTime,
                    reasons=hex(reasons))

    def others(self):
        import os
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.h)
        return [p.pid for p in procs if p.pid != os.getpid()]

    def power_limit_w(self):
        return pynvml.nvmlDeviceGetEnforcedPowerLimit(self.h) / 1e3

    @staticmethod
    def block(a, b):
        dt = b['t'] - a['t']
        return dict(seconds=dt, sm_mhz=[a['sm_mhz'], b['sm_mhz']], power_w=[a['power_w'], b['power_w']],
                    mean_power_w=(b['energy_mj'] - a['energy_mj']) / 1e3 / dt if dt > 0 else None,
                    power_cap_ms=(b['power_cap_ns'] - a['power_cap_ns']) / 1e6, temp_c=[a['temp_c'], b['temp_c']],
                    reasons_after=b['reasons'])

    def close(self):
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait()
            self.csv.close()


def stats(v):
    q = statistics.quantiles(v, n=4) if len(v) >= 4 else [min(v), statistics.median(v), max(v)]
    return dict(median=statistics.median(v), q1=q[0], q3=q[2], min=min(v), max=max(v), n=len(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--artifact', action='append', required=True, metavar='KIND=DIR')
    ap.add_argument('--tokens', default='128,256,512,1024,2048,4096,8192')
    ap.add_argument('--projections', default=None, help='comma-separated subset (default: every text projection)')
    ap.add_argument('--configs', default=None, help='comma-separated subset of CONFIGS (default: all with artifacts)')
    ap.add_argument('--reps', type=int, default=30, help='timed repetitions per configuration per round')
    ap.add_argument('--warmup', type=int, default=3, help='untimed repetitions before each timed block')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--flush-mib', type=int, default=512)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    B.require_idle()
    tel = Telemetry(Path(str(args.out) + '.telemetry.csv'))
    try:
        run(args, tel)
    finally:
        tel.close()


def run(args, tel):
    torch.backends.cuda.matmul.allow_tf32 = False
    arts = dict(spec.split('=', 1) for spec in args.artifact)
    kernels = dict(stock=KernelSet('stock'), mixed=KernelSet('mixed'), stock_wB=Kernel.load('stock_wB'),
                   n8k64_wB=Kernel.load('n8k64_wB'))
    cfgs = [c for c, (kind, *_) in CONFIGS.items() if kind in arts and (not args.configs or c in args.configs.split(','))]
    res = dict(status='running', gpu=B.gpu_info(), power_limit_w=tel.power_limit_w(), l2_bytes=torch.cuda.get_device_properties(0).L2_cache_size,
               model=args.model, artifacts=arts,
               protocol=dict(method='isolated launches, cold weights (results/paper/PROTOCOL_GEMM_ISOLATED.md)',
                             reps=args.reps, warmup=args.warmup, rounds=args.rounds, flush_mib=args.flush_mib,
                             order='rotated: round r starts at position r * len / rounds',
                             value='median over all rounds of the CUPTI device time of isolated launches'),
               kernels={k: (v.sha256 if isinstance(v, Kernel) else v.describe()) for k, v in kernels.items()},
               configs={c: dict(artifact=CONFIGS[c][0], tags=CONFIGS[c][1], kernel=CONFIGS[c][2], activation_quantizer=CONFIGS[c][3])
                        for c in cfgs},
               projections={}, rows=[], checks=dict(bitwise=[], counts_ok=True, other_processes=[]))
    if args.flush_mib * 2 ** 20 < 4 * res['l2_bytes']:
        raise SystemExit(f'--flush-mib {args.flush_mib} is below 4x the L2 ({res["l2_bytes"]} bytes)')
    tags, pw = {}, {}
    for kind, path in arts.items():
        meta, weights = A.load(path, device='cpu')
        tags[kind] = tag_modules(meta, weights)
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
        lins = {}
        for c in cfgs:
            kind, variant, kname, act = CONFIGS[c]
            w = pw[(kind, variant, proj)]
            w = dataclasses.replace(w, packed=w.packed.cuda(), scales=w.scales.cuda(), bias=None if w.bias is None else w.bias.cuda())
            lins[c] = NativeLinear(w, kernels[kname], act, name=f'{proj}@{c}')
            lins[c].share_input = False
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

                def gemm(q, kern=kern, lin=lin, y=y, bptr=bptr):
                    xp, xsf, gs = q
                    if lin.weights_on_a:
                        kern.gemm_ptr(lin.packed.data_ptr(), lin.sf.data_ptr(), xp.data_ptr(), xsf.data_ptr(), n, t, k,
                                      None, lin.global_scale, gs.data_ptr(), 1.0, bptr, y, stream.cuda_stream)
                    else:
                        kern.gemm_ptr(xp.data_ptr(), xsf.data_ptr(), lin.packed.data_ptr(), lin.sf.data_ptr(), t, n, k,
                                      gs.data_ptr(), 1.0, None, lin.global_scale, bptr, y, stream.cuda_stream)
                    return y
                fns[c] = (quant, gemm, kern)
                # registered check: the isolated path equals the deployment path (NativeLinear's fused forward) bitwise
                x.copy_(pool[0])
                ref = lin(x)
                got = gemm(quant()).clone()
                ok = bool(torch.equal(ref.view(torch.int16), got.view(torch.int16)))
                res['checks']['bitwise'].append(dict(proj=proj, tokens=t, config=c, equal=ok))
                if not ok:
                    res['status'] = 'failed'
                    B.write(args.out, res)
                    raise CheckFailed(f'{proj} T={t} {c}: the isolated GEMM differs from NativeLinear')
                ys[c] = y

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

            for r in range(args.rounds):
                shift = (r * len(cfgs) // args.rounds) % len(cfgs)
                for pos, c in enumerate(cfgs[shift:] + cfgs[:shift]):
                    others = tel.others()
                    if others:
                        res['checks']['other_processes'].append(dict(proj=proj, tokens=t, config=c, round=r, pids=others))
                        res['status'] = 'failed'
                        B.write(args.out, res)
                        raise CheckFailed(f'another compute process on the GPU: {others}')
                    for i in range(args.warmup):
                        rep(c, i, False)
                    before = tel.snap()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        ev = [rep(c, i, True) for i in range(args.reps)]
                    after = tel.snap()
                    kt = {'gemm': [], 'quant': []}
                    for e in prof.events():
                        if e.device_type.name == 'CUDA':
                            cls = classify(e.name)
                            if cls in kt:
                                kt[cls].append(e.device_time_total if hasattr(e, 'device_time_total') else e.cuda_time_total)
                    if len(kt['gemm']) != args.reps or len(kt['quant']) != args.reps:
                        res['checks']['counts_ok'] = False
                        res['status'] = 'failed'
                        B.write(args.out, res)
                        raise CheckFailed(f'{proj} T={t} {c} round {r}: {len(kt["gemm"])} GEMM / {len(kt["quant"])} quantizer '
                                          f'launches profiled, expected {args.reps}')
                    per[c].append(dict(round=r, position=pos, gemm_us=kt['gemm'], quant_us=kt['quant'], event_us=ev,
                                       telemetry=Telemetry.block(before, after)))
            for c in cfgs:
                kind, variant, kname, act = CONFIGS[c]
                kern = fns[c][2]
                g_all = [v for blk in per[c] for v in blk['gemm_us']]
                q_all = [v for blk in per[c] for v in blk['quant_us']]
                e_all = [v for blk in per[c] for v in blk['event_us']]
                res['rows'].append(dict(
                    proj=proj, out=n, inp=k, tokens=t, config=c, kind=kind, tags=variant, kernel=kern.cfg.name, act=act,
                    width=lins[c].kernel_set.width(n, k, t) if lins[c].kernel_set is not None else None,
                    gemm_us=statistics.median(g_all), gemm=stats(g_all), quant_us=statistics.median(q_all), quant=stats(q_all),
                    event_us=statistics.median(e_all), event=stats(e_all),
                    rounds=[dict(round=b['round'], position=b['position'], gemm_us=statistics.median(b['gemm_us']),
                                 quant_us=statistics.median(b['quant_us']), event_us=statistics.median(b['event_us']),
                                 telemetry=b['telemetry']) for b in per[c]]))
            print(f"{args.model} {proj} T={t} " + ' '.join(f"{r['config']}={r['gemm_us']:.1f}" for r in res['rows']
                                                           if r['proj'] == proj and r['tokens'] == t), flush=True)
            del pool, x, ys, fns
            B.write(args.out, res)
        del lins
    res['kernel_sets_after'] = {k: v.describe() for k, v in kernels.items() if isinstance(v, KernelSet)}
    res['gpu_end'] = B.gpu_info()
    res['status'] = 'complete'
    B.write(args.out, res)


if __name__ == '__main__':
    main()
