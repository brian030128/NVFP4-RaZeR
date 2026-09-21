"""Lightweight Slurm watcher with optional notifications to the same Codex session.

Only bounded log tails and scheduler status are read. Suitable for the login
node's permitted scheduler-monitoring work; no torch/model/data imports.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time


TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def terminal(item):
    return item['state'].split()[0].rstrip('+') in TERMINAL


def notify_terminal(records, out, thread, codex='codex'):
    """Queue newly terminal jobs once; failed queue attempts remain retryable.

    A successful CLI response means queued, not confirmed read by the agent.
    The caller holds the watcher-directory lock across this operation.
    """
    path = out / 'notifications.json'
    ledger = json.loads(path.read_text()) if path.exists() else dict(thread=thread, jobs={})
    if ledger['thread'] != thread:
        raise ValueError('Notification ledger belongs to another session; use a new output directory')
    fresh = {job: {key: item.get(key) for key in ('name', 'state', 'exit_code')}
             for job, item in records.items() if terminal(item) and job not in ledger['jobs']}
    if not fresh:
        return
    message = ('Slurm completion notification for the current MixFP4 research. '
               + json.dumps(fresh, sort_keys=True)
               + f'. Status and bounded log tails: {(out / "status.json").resolve()}. '
               'Read the results and continue the next justified step under the current user instructions. '
               'Reuse completed work, enforce fresh-data gates, and do not repeat jobs on notification replay.')
    try:
        result = subprocess.run([codex, 'queue', '--thread', thread, '--message', message],
                                capture_output=True, text=True, check=True, timeout=20)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or 'No CLI diagnostic output').strip()[-2000:]
        raise OSError(f'Codex queue failed (exit {error.returncode}): {detail}') from error
    receipt = dict(queued_utc=datetime.now(timezone.utc).isoformat(),
                   queue_response=result.stdout.strip(), delivery='queued_not_yet_acknowledged')
    ledger['jobs'].update({job: {**item, **receipt} for job, item in fresh.items()})
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(ledger, indent=2) + '\n')
    temporary.replace(path)
    print('COMPLETION_QUEUED', json.dumps(fresh, sort_keys=True), flush=True)


def tail(path, limit=16384):
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - limit))
        value = stream.read(limit).decode('utf-8', errors='replace')
    return re.sub(r'\x1b\[[0-9;]*m', '', value).replace('\r', '\n')


def snapshot(jobs, log_dir, log_stems=None):
    result = subprocess.run(
        ['sacct', '--array', '-n', '-P', '-j', ','.join(sorted({job.split('_')[0] for job in jobs})),
         '--format=JobID%40,JobName%80,State,Elapsed,ExitCode'],
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
        # Explicit stems support new job names and %A_%a array log paths.
        legacy = {'348910': 'tile256_zs', '348924': 'tile256_zs'}
        stems = log_stems or {}
        stem = stems.get(job, stems.get(job.split('_')[0],
                         legacy.get(job, item.get('name', 'tile256_ppl'))))
        stdout = tail(log_dir / f'{stem}_{job}.out')
        stderr = tail(log_dir / f'{stem}_{job}.err')
        for shard in range(4):
            shard_out = tail(log_dir / f'{stem}_{job}_shard{shard}.out')
            shard_err = tail(log_dir / f'{stem}_{job}_shard{shard}.err')
            if shard_out:
                stdout += f'\nGPU SHARD {shard}\n' + shard_out
            if shard_err:
                stderr += f'\nGPU SHARD {shard}\n' + shard_err
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
    ap.add_argument('--log-stems', nargs='*', default=[], help='JOBID=log_stem (array parent allowed)')
    ap.add_argument('--interval', type=float, default=60)
    ap.add_argument('--hours', type=float, default=24)
    delivery = ap.add_mutually_exclusive_group()
    delivery.add_argument('--notify-thread', help='Queue completion/failure messages to this existing Codex session')
    delivery.add_argument('--attached', action='store_true', help='Return completion directly through an attached agent tool wait')
    ap.add_argument('--codex-bin', default='codex', help='Codex CLI executable used for session notifications')
    args = ap.parse_args()
    stems = dict(value.split('=', 1) for value in args.log_stems)
    if args.interval < 30 or args.hours <= 0:
        raise ValueError('Use interval >= 30 seconds and a positive lifetime')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / '.watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    previous = None
    while True:
        now = datetime.now(timezone.utc).isoformat()
        try:
            records = snapshot(args.jobs, args.log_dir, stems)
            done = all(terminal(item) for item in records.values())
            signature = json.dumps({job: {key: value for key, value in item.items() if key != 'elapsed'}
                                    for job, item in records.items()}, sort_keys=True)
            status = dict(updated_utc=now, watcher='complete' if done else 'watching', jobs=records,
                          watcher_pid=os.getpid(), poll_seconds=args.interval)
            if args.attached:
                status['completion_delivery'] = 'attached_tool_result'
            if args.notify_thread:
                status.update(notify_thread=args.notify_thread, notifications='checking')
            temporary = args.out / 'status.json.tmp'
            temporary.write_text(json.dumps(status, indent=2) + '\n')
            temporary.replace(args.out / 'status.json')
            if signature != previous:
                with (args.out / 'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(status) + '\n')
                print(now, {job: item['state'] for job, item in records.items()}, flush=True)
                previous = signature
            if args.notify_thread:
                try:
                    notify_terminal(records, args.out, args.notify_thread, args.codex_bin)
                    status['notifications'] = 'terminal_events_queued'
                except (OSError, subprocess.SubprocessError) as error:
                    status.update(watcher='notification_retry_pending', notifications='queue_failed',
                                  notification_error=str(error))
                    print(now, 'Completion notification failed; retrying:', str(error), flush=True)
                temporary.write_text(json.dumps(status, indent=2) + '\n')
                temporary.replace(args.out / 'status.json')
                if status['notifications'] == 'queue_failed':
                    done = False
            if done:
                if args.attached:
                    print('COMPLETION_EVENT', json.dumps({job: {key: item.get(key) for key in
                          ('name', 'state', 'exit_code')} for job, item in records.items()}), flush=True)
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
