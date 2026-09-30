#!/usr/bin/env python3
"""E0M3 investigation, test A: OMMA throughput and power per format in a register-resident loop.

    python experiments/kernel_opt/e0m3/test_a.py --build-dir DIR --out JSON

Protocol: results/kernel_opt/e0m3/PROTOCOL.md (test A).
- Build: omma_loop.cu with nvcc (sm_120a, CUDA 13.1), once per site (-DOMMA_SITE), then the kernel's SASS patcher on
  each library (sites 1-3; site 0 is native). The patcher sets the OMMA format bits from the PRMT site tag and reports
  a census; every site must have the same OMMA count, and the unpatched instruction streams must be identical apart from
  the PRMT selector. So the four kernels issue the same instructions; only the format bits differ:
    site 0  E2M1 x E2M1 (native)        site 1  E0M3 x E2M1 (A in E0M3: the weights-on-A kernels' E0M3 tiles)
    site 2  E2M1 x E0M3 (B in E0M3: n8k64_wB)     site 3  E0M3 x E0M3
- Launch: 2 CTAs of 256 threads per SM (8 independent accumulator chains per thread), --iters iterations per launch.
- Operand data, the same bits for every site:
    random     every nibble and every ue4m3 scale byte (in [2^-4, 2^2]) drawn uniformly (seeded)
    constant   every nibble = 0x2, every scale byte = 0x38 (1.0)
    zero       every nibble = 0, every scale byte = 0x38
- Two timing modes, per (data, site), in a rotated site order over --rounds rounds:
    isolated   single launches after a 50 ms idle gap (GPU unthrottled), CUPTI kernel time, median of 10
    sustained  back-to-back launches for about --sustain-s seconds (the power cap engages if it will), CUDA-event time
               per launch over the last half, with NVML (SM clock, power, software power-cap reason) sampled every 5 ms
Reported: OMMA/s and each site's time relative to site 0 per mode and data; clocks and power per site.
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import ctypes
from pathlib import Path

import pynvml
import torch
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CUDA = Path('/home/dev/.conda/envs/mixfp4-cuda131')
PATCHER = REPO / 'sm120' / 'kernel' / 'scripts' / 'patch_mixed_nvfp4_gemm.py'


def sass_ops(path):
    """The kernel's instruction mnemonics with the PRMT selector blanked (to compare the four builds' streams)."""
    import re
    text = subprocess.run([str(CUDA / 'bin' / 'cuobjdump'), '--dump-sass', str(path)], check=True, capture_output=True,
                          text=True).stdout
    ops = []
    for ln in text.splitlines():
        m = re.match(r'\s*/\*[0-9a-f]+\*/\s+(.*?);', ln)
        if m:
            ops.append(re.sub(r'0x3[0-9a-f]{3}\b', 'SEL', m.group(1)))
    return ops


def build(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    libs, logs, streams = {}, {}, {}
    for site in range(4):
        raw, lib = out_dir / f'libomma{site}.unpatched.so', out_dir / f'libomma{site}.so'
        subprocess.run([str(CUDA / 'bin' / 'nvcc'), '-O3', '-std=c++17', '--generate-code=arch=compute_120a,code=[sm_120a]',
                        f'-DOMMA_SITE={site}', '-Xcompiler=-fPIC', '-shared', '-ccbin', '/usr/bin/g++',
                        str(HERE / 'omma_loop.cu'), '-o', str(raw)], check=True)
        streams[site] = sass_ops(raw)
        if site == 0:
            import shutil
            shutil.copy(raw, lib)
            logs[site] = 'site 0: native E2M1 x E2M1, not patched'
        else:
            logs[site] = subprocess.run([sys.executable, str(PATCHER), '--cuobjdump', str(CUDA / 'bin' / 'cuobjdump'),
                                         '--allow-missing-sites', str(raw), str(lib)],
                                        check=True, capture_output=True, text=True).stdout
        libs[site] = lib
    same = all(streams[s] == streams[0] for s in range(4))
    if not same:
        raise SystemExit('the four unpatched instruction streams differ beyond the PRMT selector')
    return libs, logs, len([o for o in streams[0] if o.startswith('OMMA')])


class Sampler:
    def __init__(self):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.rows, self.stop = [], threading.Event()
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            r = pynvml.nvmlDeviceGetCurrentClocksEventReasons(self.h)
            self.rows.append((time.time(), pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM),
                              pynvml.nvmlDeviceGetPowerUsage(self.h) / 1e3, bool(r & pynvml.nvmlClocksEventReasonSwPowerCap)))
            time.sleep(0.005)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join()

    def summary(self, since=None):
        rows = [r for r in self.rows if since is None or r[0] >= since]
        if not rows:
            return {}
        return dict(samples=len(rows), sm_mhz_median=statistics.median(r[1] for r in rows), sm_mhz_min=min(r[1] for r in rows),
                    power_w_median=statistics.median(r[2] for r in rows), power_w_max=max(r[2] for r in rows),
                    power_cap_share=sum(r[3] for r in rows) / len(rows))


def operands(kind, threads, seed=0):
    g = torch.Generator('cpu').manual_seed(seed)
    if kind == 'random':
        a = torch.randint(0, 2 ** 32, (threads * 4,), generator=g, dtype=torch.int64)
        b = torch.randint(0, 2 ** 32, (threads * 2,), generator=g, dtype=torch.int64)
        # ue4m3 scale bytes with exponent field 3..9 (2^-4 .. 2^2) and a random mantissa
        e = torch.randint(3, 10, (threads * 2, 4), generator=g)
        m = torch.randint(0, 8, (threads * 2, 4), generator=g)
        byte = (e << 3) | m
        sf = (byte[:, 0] | (byte[:, 1] << 8) | (byte[:, 2] << 16) | (byte[:, 3] << 24))
    else:
        nib = 0x22222222 if kind == 'constant' else 0
        a = torch.full((threads * 4,), nib, dtype=torch.int64)
        b = torch.full((threads * 2,), nib, dtype=torch.int64)
        sf = torch.full((threads * 2,), 0x38383838, dtype=torch.int64)
    to32 = lambda v: (v & 0xFFFFFFFF).to(torch.int64).numpy().astype('uint32')  # noqa: E731
    return [torch.from_numpy(to32(v).view('int32')).cuda() for v in (a, b, sf)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--build-dir', type=Path, required=True)
    ap.add_argument('--iters', type=int, default=4096)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--sustain-s', type=float, default=2.0)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    lib_paths, plogs, static_ommas = build(args.build_dir)
    census = {s: [ln for ln in log.splitlines() if 'census' in ln or 'per-site' in ln or 'site 0' in ln] for s, log in plogs.items()}
    libs = {}
    for s, pth in lib_paths.items():
        lib = ctypes.CDLL(str(pth))
        lib.omma_loop.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 5
        lib.omma_loop.restype = ctypes.c_int
        assert lib.omma_site() == s
        libs[s] = lib
    per_iter = libs[0].omma_per_iter()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    grid, block = 2 * sms, 256
    threads = grid * block
    warps = threads // 32
    omma_per_launch = warps * per_iter * args.iters
    out = torch.empty(threads, dtype=torch.float32, device='cuda')
    stream = torch.cuda.current_stream().cuda_stream
    res = dict(protocol='results/kernel_opt/e0m3/PROTOCOL.md (test A)', gpu=torch.cuda.get_device_name(0),
               patcher_log=census, static_ommas_per_kernel=static_ommas, instruction_streams_identical=True,
               grid=grid, block=block, iters=args.iters, omma_per_launch=omma_per_launch, rows=[])
    sites = [0, 1, 2, 3]
    for kind in ('random', 'constant', 'zero'):
        a, b, sf = operands(kind, threads)

        def launch(site):
            rc = libs[site].omma_loop(grid, block, args.iters, a.data_ptr(), b.data_ptr(), sf.data_ptr(), out.data_ptr(), stream)
            assert rc == 0, rc
        for s in sites:                       # warm-up and a check that every site runs
            launch(s)
        torch.cuda.synchronize()
        for r in range(args.rounds):
            order = sites[r % 4:] + sites[:r % 4]
            for s in order:
                # isolated
                iso = []
                for _ in range(10):
                    time.sleep(0.05)
                    torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        launch(s)
                        torch.cuda.synchronize()
                    iso += [e.device_time_total for e in prof.events() if e.device_type.name == 'CUDA' and 'omma_kernel' in e.name]
                # sustained
                t_end = time.time() + args.sustain_s
                evs = []
                with Sampler() as smp:
                    t_half = time.time() + args.sustain_s / 2
                    while time.time() < t_end:
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        launch(s)
                        e1.record()
                        evs.append((time.time(), e0, e1))
                    torch.cuda.synchronize()
                sus = [e0.elapsed_time(e1) * 1e3 for (tt, e0, e1) in evs if tt >= t_half]
                row = dict(data=kind, site=s, round=r, isolated_us=statistics.median(iso), sustained_us=statistics.median(sus),
                           sustained_launches=len(sus), telemetry=smp.summary(since=t_half))
                row['isolated_omma_per_s'] = omma_per_launch / (row['isolated_us'] * 1e-6)
                row['sustained_omma_per_s'] = omma_per_launch / (row['sustained_us'] * 1e-6)
                res['rows'].append(row)
                t = row['telemetry']
                print(f"{kind:8s} site {s} r{r}: isolated {row['isolated_us']:8.1f} us  sustained {row['sustained_us']:8.1f} us  "
                      f"SM {t.get('sm_mhz_median')} MHz (min {t.get('sm_mhz_min')})  {t.get('power_w_median', 0):.0f} W  "
                      f"cap {100 * t.get('power_cap_share', 0):.0f} %", flush=True)
        del a, b, sf
    summ = {}
    for kind in ('random', 'constant', 'zero'):
        for s in sites:
            rows = [r for r in res['rows'] if r['data'] == kind and r['site'] == s]
            summ[f'{kind}/site{s}'] = dict(isolated_us=statistics.median(r['isolated_us'] for r in rows),
                                           sustained_us=statistics.median(r['sustained_us'] for r in rows),
                                           sm_mhz=statistics.median(r['telemetry'].get('sm_mhz_median', 0) for r in rows),
                                           power_w=statistics.median(r['telemetry'].get('power_w_median', 0) for r in rows),
                                           power_cap_share=statistics.median(r['telemetry'].get('power_cap_share', 0) for r in rows))
        base = summ[f'{kind}/site0']
        for s in sites:
            e = summ[f'{kind}/site{s}']
            e['isolated_vs_site0_pct'] = 100 * (e['isolated_us'] / base['isolated_us'] - 1)
            e['sustained_vs_site0_pct'] = 100 * (e['sustained_us'] / base['sustained_us'] - 1)
    res['summary'] = summ
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    for k, v in summ.items():
        print(f"{k:16s} isolated {v['isolated_vs_site0_pct']:+5.1f} %  sustained {v['sustained_vs_site0_pct']:+5.1f} %  "
              f"SM {v['sm_mhz']:.0f} MHz  {v['power_w']:.0f} W  cap {100 * v['power_cap_share']:.0f} %")


if __name__ == '__main__':
    main()
