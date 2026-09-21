"""In-container job wrapper: GPU preflight before/after, run the target, record status.

The target is `-m package.module args...`. It can reach the shared PhaseGate through
`campaign.runtime.gate(phase)` to write gpu_preflight_phase_<phase>.json records.
"""
import argparse
import json
import os
import runpy
import sys
import time
import traceback
from pathlib import Path

from campaign import gpu_preflight as gp
from campaign import runtime


def _write(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--gpu-run', action='store_true')
    g.add_argument('--cpu-run', action='store_true')
    ap.add_argument('rest', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.rest[1:] if args.rest[:1] == ['--'] else args.rest
    run_dir = Path(os.environ['CAMPAIGN_RUN_DIR'])
    status_path = run_dir / 'job_status.json'
    gate = gp.PhaseGate(run_dir, gpu_run=args.gpu_run)
    runtime.install(run_dir, gate, gpu_run=args.gpu_run)
    status = dict(status='running', started_utc=gp._utc(), command=cmd, pid=os.getpid(), gpu_run=args.gpu_run)
    _write(status_path, status)
    t0 = time.time()
    try:
        gate('before', filename='gpu_preflight_before.json')
        if cmd[:1] == ['-m']:
            sys.argv = [cmd[1]] + cmd[2:]
            runpy.run_module(cmd[1], run_name='__main__', alter_sys=True)
        else:
            sys.argv = list(cmd)
            runpy.run_path(cmd[0], run_name='__main__')
        status['status'] = 'complete'
    except SystemExit as exc:
        if exc.code in (0, None):
            status['status'] = 'complete'
        else:
            status.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    except BaseException as exc:  # includes OOM; recorded, never swallowed silently
        status.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        if 'OutOfMemoryError' in repr(exc) or 'out of memory' in repr(exc).lower():
            status['oom'] = True
    finally:
        status['wall_seconds'] = time.time() - t0
        try:
            status['torch_peak_memory_bytes'] = runtime.peak_memory()
        except Exception as exc:
            status['torch_peak_memory_bytes'] = dict(error=repr(exc))
        try:
            gate('after', filename='gpu_preflight_after.json')
        except SystemExit as exc:
            status['after_preflight_failed'] = str(exc)
            if status['status'] == 'complete':
                status['status'] = 'invalid'
        status['preflight_records'] = gate.records
        status['finished_utc'] = gp._utc()
        _write(status_path, status)
    sys.exit(0 if status['status'] == 'complete' else 1)


if __name__ == '__main__':
    main()
