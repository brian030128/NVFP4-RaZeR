"""Single-machine replacement for campaign.job_wrapper/launcher (no Slurm, no GPU lease).

Creates $CAMPAIGN_ROOT/runs/<name>_attempt<N>, writes a launch_record.json whose
source_manifest_sha256 hashes every campaign/support Python file, installs the run
context without the cluster GPU-preflight gate, and runs the target module unchanged.

    python repro_local/run_local.py <name> -m campaign.evaluate_ppl --model llama8b ...
"""
import hashlib
import json
import os
import runpy
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOFTWARE = REPO / 'research/n16k64/software/primary'


def source_manifest():
    files = sorted(p for d in ('campaign', 'support') for p in (SOFTWARE / d).rglob('*.py'))
    entries = [(str(p.relative_to(SOFTWARE)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    blob = json.dumps(entries, separators=(',', ':')).encode()
    return hashlib.sha256(blob).hexdigest(), len(entries)


def write(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n')
    tmp.replace(path)


def main():
    name, cmd = sys.argv[1], sys.argv[2:]
    root = Path(os.environ['CAMPAIGN_ROOT'])
    attempt = 1 + len(list((root / 'runs').glob(f'{name}_attempt*')))
    run_dir = root / 'runs' / f'{name}_attempt{attempt}'
    run_dir.mkdir(parents=True)
    os.environ['CAMPAIGN_RUN_DIR'] = str(run_dir)
    os.environ['CAMPAIGN_RUN_ID'] = run_dir.name
    digest, count = source_manifest()
    record = dict(status='running', run_id=run_dir.name, command=cmd, source_manifest_sha256=digest,
                  source_files=count, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  host='local single RTX PRO 6000 (no Slurm, no GPU lease)', image_id=None)
    write(run_dir / 'launch_record.json', record)

    from campaign import runtime
    runtime.install(run_dir, None, gpu_run=True)
    t0 = time.time()
    try:
        sys.argv = [cmd[1]] + cmd[2:]
        runpy.run_module(cmd[1], run_name='__main__', alter_sys=True)
        record['status'] = 'complete'
    except SystemExit as exc:
        record['status'] = 'complete' if exc.code in (0, None) else 'failed'
        if record['status'] == 'failed':
            record.update(error=repr(exc), traceback=traceback.format_exc())
    except BaseException as exc:
        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    finally:
        record['wall_seconds'] = time.time() - t0
        record['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        record['torch_peak_memory_bytes'] = runtime.peak_memory()
        write(run_dir / 'launch_record.json', record)
    print(f'RUN {run_dir.name} {record["status"]} {record["wall_seconds"]:.0f}s', flush=True)
    sys.exit(0 if record['status'] == 'complete' else 1)


if __name__ == '__main__':
    main()
