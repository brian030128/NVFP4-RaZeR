"""Runs one command and records its end-to-end wall time and memory (results/topk_cal/flipquant_parity/NOTE.md).

    python measure.py --out JSON --log LOG --cwd DIR -- COMMAND ...

- **Wall time:** from the start of the command to its exit.
- **Host RSS:** sampled at 10 Hz with psutil, for the top process (for flipquant, the train_map_razer wrapper) and for
  each of its descendants (the trainer subprocess) separately, and for their sum.
- **The largest process's lifetime peak:** `ru_maxrss` of RUSAGE_CHILDREN, i.e. of the largest waited-for descendant.
- **GPU memory per process:** nvidia-smi's `used_memory` per PID, every second. It includes the CUDA context and the
  caching allocator's reserve; the trainer's own torch peaks are in its report.json.
"""
import argparse
import json
import resource
import subprocess
import time

import psutil

GIB = 2 ** 30


def gpu_by_pid():
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    return {int(p): int(m) for p, m in (line.split(',') for line in out.strip().splitlines() if line.strip())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--cwd', default='.')
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ['--'] else args.cmd
    peak = dict(top=0, descendants_sum=0, tree=0)
    by_pid, gpu, samples, last_gpu = {}, {}, 0, 0.0
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
               largest_process_ru_maxrss_gib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20,
               gpu_used_mib_by_pid={str(k): v for k, v in gpu.items()})
    with open(args.out, 'w') as fh:
        json.dump(rec, fh, indent=1)
        fh.write('\n')
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
