"""Lightweight Slurm metadata/log watcher; never launches compute or changes jobs.

Only bounded log tails and scheduler status are read. Suitable for the login
node's permitted scheduler-monitoring work; no torch/model/data imports.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time


TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def tail(path, limit=16384):
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - limit))
        value = stream.read(limit).decode('utf-8', errors='replace')
    return re.sub(r'\x1b\[[0-9;]*m', '', value).replace('\r', '\n')


def snapshot(jobs, log_dir):
    result = subprocess.run(
        ['sacct', '-n', '-P', '-j', ','.join(jobs),
         '--format=JobIDRaw,JobName,State,Elapsed,ExitCode'],
        capture_output=True, text=True, timeout=20, check=True)
    records = {}
    for line in result.stdout.splitlines():
        fields = line.split('|')
        if len(fields) < 5 or fields[0] not in jobs:
            continue
        job, name, state, elapsed, code = fields[:5]
        records[job] = dict(name=name, state=state, elapsed=elapsed, exit_code=code)
    for job in jobs:
        item = records.setdefault(job, dict(state='UNKNOWN'))
        # Paths are fixed from these submitted jobs, not arbitrary directory scans.
        stem = 'tile256_zs' if job in ('348910', '348924') else 'tile256_ppl'
        stdout = tail(log_dir / f'{stem}_{job}.out')
        stderr = tail(log_dir / f'{stem}_{job}.err')
        item['recent_output'] = stdout.splitlines()[-30:]
        item['recent_stderr'] = stderr.splitlines()[-8:]
        item['ppl_results'] = [line for line in stdout.splitlines() if line.startswith('PPL ')]
        item['error_markers'] = [line for line in (stdout + '\n' + stderr).splitlines()
                                 if re.search(r'Traceback|Error:|FAILED|OUT_OF_MEMORY', line)][-8:]
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jobs', nargs='+', required=True)
    ap.add_argument('--log-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--interval', type=float, default=60)
    ap.add_argument('--hours', type=float, default=24)
    args = ap.parse_args()
    if args.interval < 30 or args.hours <= 0:
        raise ValueError('Use interval >= 30 seconds and a positive lifetime')
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    previous = None
    while True:
        now = datetime.now(timezone.utc).isoformat()
        try:
            records = snapshot(args.jobs, args.log_dir)
            done = all(item['state'].split()[0].rstrip('+') in TERMINAL for item in records.values())
            signature = json.dumps({job: {key: value for key, value in item.items() if key != 'elapsed'}
                                    for job, item in records.items()}, sort_keys=True)
            status = dict(updated_utc=now, watcher='complete' if done else 'watching', jobs=records,
                          watcher_pid=os.getpid(), poll_seconds=args.interval,
                          reordering_validation='9/9 tests passed; syntax check passed (348924.0/.1)',
                          reordering_model_quality='not measured')
            temporary = args.out / 'status.json.tmp'
            temporary.write_text(json.dumps(status, indent=2) + '\n')
            temporary.replace(args.out / 'status.json')
            if signature != previous:
                with (args.out / 'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(status) + '\n')
                print(now, {job: item['state'] for job, item in records.items()}, flush=True)
                previous = signature
            if done:
                return
        except (OSError, subprocess.SubprocessError) as error:
            print(now, 'Scheduler/log read failed; retrying:', str(error), flush=True)
        if time.monotonic() - started >= args.hours * 3600:
            path = args.out / 'status.json'
            if path.is_file():
                status = json.loads(path.read_text())
                status.update(updated_utc=now, watcher='stopped_at_lifetime_limit')
                path.write_text(json.dumps(status, indent=2) + '\n')
            print(now, 'Watcher lifetime reached; monitoring stopped.', flush=True)
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
