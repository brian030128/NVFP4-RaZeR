"""Process-tree measurement of every calibration.train_map run, from outside the release driver (which is already
running): the method of topk-cal's results/topk_cal/flipquant_parity/measure.py, attached to each wrapper process as it
appears (polled every 0.1 s).

- Host RSS sampled at 10 Hz with psutil: the wrapper (calibration.train_map), each descendant (the preparation and
  trainer subprocesses), and their sum (the tree).
- The exact lifetime peak RSS of every process: /proc/<pid>/status VmHWM (the kernel's high-water mark, the same
  quantity as ru_maxrss), read at every sample; the last reading before the process exits is kept.
- GPU memory per process: nvidia-smi's used_memory per PID, every second.
- Wall time from the wrapper's start (its create_time) to its exit.

One JSON per run in measure/<out directory name>.json. A run that was already going when the watcher started is
marked attached_late (its VmHWM peaks are still exact; its sampled tree peak covers only the attached part).

    python watch_tree.py            (runs until killed)
"""
import json
import subprocess
import time
from pathlib import Path

import psutil

GIB = 2 ** 30
OUT = Path("/home/dev/n16k64_campaign/fqrel/measure")


def vmhwm(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def gpu_by_pid():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(p): int(m) for p, m in (l.split(",") for l in out.strip().splitlines() if l.strip())}


class Run:
    def __init__(self, proc, started_watch):
        self.proc = proc
        self.cmd = proc.cmdline()
        self.out_dir = Path(self.cmd[self.cmd.index("--out") + 1]).name if "--out" in self.cmd else str(proc.pid)
        self.created = proc.create_time()
        self.attached = time.time()
        self.late = self.attached - self.created > 1.0 or started_watch
        self.peak = dict(top=0, descendants_sum=0, tree=0)
        self.procs = {}                       # pid -> dict(cmd, sampled_peak, vmhwm)
        self.gpu = {}
        self.samples = 0
        self.last_gpu = 0.0

    def sample(self):
        try:
            top_rss = self.proc.memory_info().rss
            kids = self.proc.children(recursive=True)
        except psutil.Error:
            return False
        rows = {self.proc.pid: (top_rss, "wrapper: " + " ".join(self.cmd[1:4]))}
        for k in kids:
            try:
                rows[k.pid] = (k.memory_info().rss, " ".join(k.cmdline()[:3]))
            except psutil.Error:
                pass
        self.samples += 1
        kid_sum = sum(r for pid, (r, _) in rows.items() if pid != self.proc.pid)
        self.peak["top"] = max(self.peak["top"], top_rss)
        self.peak["descendants_sum"] = max(self.peak["descendants_sum"], kid_sum)
        self.peak["tree"] = max(self.peak["tree"], top_rss + kid_sum)
        for pid, (rss, name) in rows.items():
            p = self.procs.setdefault(pid, dict(cmd=name, sampled_peak=0, vmhwm=0))
            p["sampled_peak"] = max(p["sampled_peak"], rss)
            h = vmhwm(pid)
            if h is not None:
                p["vmhwm"] = max(p["vmhwm"], h)
        if time.time() - self.last_gpu >= 1.0:
            self.last_gpu = time.time()
            for pid, mib in gpu_by_pid().items():
                if pid in rows and mib > self.gpu.get(pid, 0):
                    self.gpu[pid] = mib
        return True

    def finish(self):
        ended = time.time()
        wrapper = self.procs.get(self.proc.pid, {})
        rec = dict(out_dir=self.out_dir, command=self.cmd, wrapper_pid=self.proc.pid, attached_late=self.late,
                   attached_after_seconds=self.attached - self.created, wall_seconds=ended - self.created,
                   rss_samples=self.samples, rss_interval_seconds=0.1,
                   peak_rss_sampled_gib={k: v / GIB for k, v in self.peak.items()},
                   wrapper_vmhwm_gib=wrapper.get("vmhwm", 0) / GIB,
                   largest_process_vmhwm_gib=max((p["vmhwm"] for p in self.procs.values()), default=0) / GIB,
                   processes=[dict(pid=pid, cmd=p["cmd"], sampled_peak_rss_gib=p["sampled_peak"] / GIB,
                                   vmhwm_gib=p["vmhwm"] / GIB, gpu_used_peak_mib=self.gpu.get(pid))
                              for pid, p in sorted(self.procs.items())])
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{self.out_dir}.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(f"{time.strftime('%H:%M:%S')} wrote {self.out_dir}: tree {rec['peak_rss_sampled_gib']['tree']:.2f} GiB, "
              f"wrapper VmHWM {rec['wrapper_vmhwm_gib']:.2f} GiB, wall {rec['wall_seconds']:.0f} s, late={self.late}",
              flush=True)


def main():
    runs, first = {}, True
    while True:
        for p in psutil.process_iter(["pid", "cmdline"]):
            cmd = p.info["cmdline"] or []
            if p.pid not in runs and "-m" in cmd and "calibration.train_map" in cmd:
                try:
                    runs[p.pid] = Run(p, first)
                    print(f"{time.strftime('%H:%M:%S')} attached to {runs[p.pid].out_dir} (pid {p.pid}, "
                          f"late={runs[p.pid].late})", flush=True)
                except psutil.Error:
                    pass
        first = False
        for pid in list(runs):
            if not runs[pid].sample():
                runs.pop(pid).finish()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
