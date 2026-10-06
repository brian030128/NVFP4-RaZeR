"""topk-cal's results/topk_cal/flipquant_parity/measure.py (its sampling unchanged) plus every process's exact lifetime
peak RSS, /proc/<pid>/status VmHWM (the quantity ru_maxrss reports), read at every sample. Used for the one-time data
preparation runs of the release maps.

    python measure_vmhwm.py --out JSON --log LOG --cwd DIR -- COMMAND ...

- **Wall time:** from the start of the command to its exit.
- **Host RSS:** sampled at 10 Hz with psutil, for the top process and for each of its descendants separately, and for
  their sum.
- **VmHWM:** per process, the last reading before it exits (the high-water mark is monotonic).
- **The largest process's lifetime peak:** `ru_maxrss` of RUSAGE_CHILDREN.
- **GPU memory per process:** nvidia-smi's `used_memory` per PID, every second.
"""
import argparse
import json
import resource
import subprocess
import time
from pathlib import Path

import psutil

GIB = 2 ** 30


def gpu_by_pid():
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    return {int(p): int(m) for p, m in (line.split(',') for line in out.strip().splitlines() if line.strip())}


def vmhwm(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmHWM:'):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--cwd', default='.')
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ['--'] else args.cmd
    peak = dict(top=0, descendants_sum=0, tree=0)
    by_pid, hwm, gpu, samples, last_gpu = {}, {}, {}, 0, 0.0
    started = time.time()
    with open(args.log, 'w') as log:
        proc = subprocess.Popen(cmd, cwd=args.cwd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        top = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                r_top = top.memory_info().rss
                kids = {}
                for k in top.children(recursive=True):
                    try:
                        kids[k.pid] = (k.memory_info().rss, ' '.join(k.cmdline()[:4]))
                    except psutil.Error:
                        pass
            except psutil.Error:
                break
            samples += 1
            peak['top'] = max(peak['top'], r_top)
            peak['descendants_sum'] = max(peak['descendants_sum'], sum(r for r, _ in kids.values()))
            peak['tree'] = max(peak['tree'], r_top + sum(r for r, _ in kids.values()))
            for pid, (r, name) in kids.items():
                if r > by_pid.get(pid, (0, ''))[0]:
                    by_pid[pid] = (r, name)
            for pid, name in [(proc.pid, 'top: ' + ' '.join(cmd[:4]))] + [(p, n) for p, (_, n) in kids.items()]:
                h = vmhwm(pid)
                if h is not None and h > hwm.get(pid, (0, ''))[0]:
                    hwm[pid] = (h, name)
            if time.time() - last_gpu >= 1.0:
                last_gpu = time.time()
                for pid, mib in gpu_by_pid().items():
                    role = 'top' if pid == proc.pid else 'descendant' if pid in kids else 'other'
                    if mib > gpu.get(pid, dict(mib=0))['mib']:
                        gpu[pid] = dict(mib=mib, role=role)
            time.sleep(0.1)
        rc = proc.wait()
    wall = time.time() - started
    rec = dict(command=cmd, cwd=args.cwd, returncode=rc, wall_seconds=wall, rss_samples=samples, rss_interval_seconds=0.1,
               peak_rss_gib=dict(top=peak['top'] / GIB, descendants_sum=peak['descendants_sum'] / GIB,
                                 tree=peak['tree'] / GIB),
               descendants=[dict(pid=pid, peak_rss_gib=r / GIB, cmd=name) for pid, (r, name) in sorted(by_pid.items())],
               vmhwm_gib=[dict(pid=pid, vmhwm_gib=h / GIB, cmd=name) for pid, (h, name) in sorted(hwm.items())],
               largest_process_ru_maxrss_gib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20,
               gpu_used_mib_by_pid={str(k): v for k, v in gpu.items()})
    with open(args.out, 'w') as fh:
        json.dump(rec, fh, indent=1)
        fh.write('\n')
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
