"""PROTOCOL.md section 3: pick the C and D training settings from the probe reports.

The C setting is the fitting FP32-state (adamw_fp32) probe with the lowest steady step
time; only if none fits, the fallbacks in order (adamw_8bit, cpu_offload, lora). D is the
fastest fitting scale probe. Live-teacher and reference probes are not candidates.

python results/cost_comparison/choose_config.py RUNS_DIR
"""
import json
import sys
from pathlib import Path


def probes(runs):
    for p in sorted((Path(runs) / 'probes').glob('*/report.json')):
        r = json.loads(p.read_text())
        c = r['config']
        yield p.parent.name, r, c


def fastest(rows):
    ok = [(r['steady_step_seconds'], name, c) for name, r, c in rows if r['status'] == 'complete']
    return min(ok, key=lambda x: x[0]) if ok else None


def main():
    runs = sys.argv[1]
    rows = [x for x in probes(runs) if not x[2]['live_teacher'] and not x[0].startswith('ref_')]
    out = {}
    for arm, order in (('qat', ('adamw_fp32', 'adamw_8bit', 'cpu_offload')), ('scale', ('torch_adamw',))):
        for opt in order:
            best = fastest([x for x in rows if x[2]['arm'] == arm and x[2]['optimizer'] == opt])
            if best:
                out[arm] = dict(probe=best[1], optimizer=opt, micro_batch=best[2]['micro_batch'],
                                checkpointing=best[2]['checkpointing'], steady_step_seconds=best[0])
                break
        if arm == 'qat' and 'qat' not in out:
            best = fastest([x for x in rows if x[2]['arm'] == 'lora'])
            if best:
                out['qat'] = dict(probe=best[1], arm='lora', optimizer='torch_adamw', micro_batch=best[2]['micro_batch'],
                                  checkpointing=best[2]['checkpointing'], steady_step_seconds=best[0])
    for name, r, c in rows:
        status = r['status'] if r['status'] != 'oom' else f"OOM at step {r['oom']['step']} ({r['oom']['gpu_peak_allocated_gib']:.1f} GiB)"
        print(f"{name:28s} {status:34s} step {r.get('steady_step_seconds') or float('nan'):.2f}s")
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
