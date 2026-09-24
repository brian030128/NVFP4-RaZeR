"""Per-phase GPU and host memory for the calibration-cost study.

Phases are contiguous: `enter(name)` closes the running phase and opens the next, so
the whole run is covered and the run-wide peak is the maximum of the phase peaks
(the same quantity torch.cuda.max_memory_allocated() gives without resets). At each
phase start the CUDA peak statistics are reset (torch.cuda.reset_peak_memory_stats);
at its end max_memory_allocated / max_memory_reserved are read for every device.
Host memory is the process RSS, sampled by a background thread (default 10 Hz) and at
every phase boundary; resource.getrusage's ru_maxrss (the method of run_multiround.py's
resource block) is reported alongside as the lifetime peak.
"""
import resource
import threading
import time

import psutil
import torch

GIB = 2 ** 30


class PhaseMonitor:
    def __init__(self, interval=0.1):
        self.process = psutil.Process()
        self.interval = interval
        self.started = time.time()
        self.phases, self.current, self.samples = [], None, 0
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.devices = torch.cuda.device_count()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _rss(self):
        return self.process.memory_info().rss

    def _sample(self):
        while not self.stopped.wait(self.interval):
            rss = self._rss()
            with self.lock:
                self.samples += 1
                if self.current is not None and rss > self.current['host_peak_rss']:
                    self.current['host_peak_rss'] = rss

    def _sync(self):
        for i in range(self.devices):
            torch.cuda.synchronize(i)

    def enter(self, name, sync=True):
        """Close the running phase and open `name` (None: close only)."""
        if sync:
            self._sync()
        now, rss = time.time(), self._rss()
        with self.lock:
            if self.current is not None:
                c = self.current
                c.update(seconds=now - self.started - c['start'], host_rss_end=rss,
                         host_peak_rss=max(c['host_peak_rss'], rss),
                         gpu_peak_allocated=[torch.cuda.max_memory_allocated(i) for i in range(self.devices)],
                         gpu_peak_reserved=[torch.cuda.max_memory_reserved(i) for i in range(self.devices)],
                         gpu_allocated_end=[torch.cuda.memory_allocated(i) for i in range(self.devices)],
                         device_used_end=[(lambda f, t: t - f)(*torch.cuda.mem_get_info(i)) for i in range(self.devices)])
                self.phases.append(c)
                self.current = None
            if name is not None:
                for i in range(self.devices):
                    torch.cuda.reset_peak_memory_stats(i)
                self.current = dict(name=name, start=now - self.started, host_rss_start=rss, host_peak_rss=rss,
                                    gpu_allocated_start=[torch.cuda.memory_allocated(i) for i in range(self.devices)])

    def close(self, sync=True):
        self.enter(None, sync=sync)
        self.stopped.set()
        self.thread.join()

    def summary(self):
        """JSON-ready record: every phase in order, per-name aggregates, run-wide peaks (GiB)."""
        def gib(v):
            return [x / GIB for x in v] if isinstance(v, list) else v / GIB
        phases = []
        for p in self.phases:
            phases.append({k: (gib(v) if k.startswith(('gpu_', 'host_', 'device_')) else v) for k, v in p.items()})
        by_name = {}
        for p in phases:
            a = by_name.setdefault(p['name'], dict(count=0, seconds=0.0, gpu_peak_allocated_gib=[0.0] * self.devices,
                                                   gpu_peak_reserved_gib=[0.0] * self.devices, host_peak_rss_gib=0.0))
            a['count'] += 1
            a['seconds'] += p['seconds']
            a['gpu_peak_allocated_gib'] = [max(x, y) for x, y in zip(a['gpu_peak_allocated_gib'], p['gpu_peak_allocated'])]
            a['gpu_peak_reserved_gib'] = [max(x, y) for x, y in zip(a['gpu_peak_reserved_gib'], p['gpu_peak_reserved'])]
            a['host_peak_rss_gib'] = max(a['host_peak_rss_gib'], p['host_peak_rss'])
        return dict(
            phases=phases, by_phase=by_name,
            gpu_peak_allocated_gib=[max((p['gpu_peak_allocated'][i] for p in phases), default=0.0) for i in range(self.devices)],
            gpu_peak_reserved_gib=[max((p['gpu_peak_reserved'][i] for p in phases), default=0.0) for i in range(self.devices)],
            host_peak_rss_sampled_gib=max((p['host_peak_rss'] for p in phases), default=0.0),
            cpu_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
            host_rss_samples=self.samples, host_rss_interval_seconds=self.interval,
            method='torch.cuda.max_memory_allocated/reserved after reset_peak_memory_stats at each phase start; '
                   'host RSS sampled by psutil every interval and at phase boundaries; cpu_peak_rss_gib is ru_maxrss')
