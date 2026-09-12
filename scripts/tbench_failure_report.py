"""
    Why did each Terminal-Bench trial not produce a score?

    A pass rate of zero can mean the model tried twenty tasks and solved none, or that nineteen
    trials never reached the verifier. Those call for opposite responses, and the harbor progress
    bar shows the same 0.000 for both -- which is how a 900-second agent cap and a broken
    container prelude got recorded as "the benchmark returns no signal".

    This reads the per-trial results and splits the zeros by cause.

        python scripts/tbench_failure_report.py results/terminal_bench/kse_job_*
"""

import collections
import glob
import json
import os
import sys


def phase_minutes(rec, phase):
    a = (rec.get(phase) or {}).get('started_at')
    b = (rec.get(phase) or {}).get('finished_at')
    if not a or not b:
        return None
    try:
        from datetime import datetime
        fix = lambda t: datetime.fromisoformat(t.replace('Z', '+00:00'))
        return (fix(b) - fix(a)).total_seconds() / 60.0
    except Exception:
        return None


def main(dirs):
    for run in dirs:
        for jobs_dir in sorted(glob.glob(os.path.join(run, 'jobs_*'))):
            policy = os.path.basename(jobs_dir)[len('jobs_'):]
            causes = collections.Counter()
            solved = scored = 0
            mins = []
            for path in glob.glob(os.path.join(jobs_dir, '*', 'result.json')):
                try:
                    rec = json.load(open(path))
                except Exception:
                    continue
                if 'task_name' not in rec:
                    continue
                exc = rec.get('exception_info')
                m = phase_minutes(rec, 'agent_execution')
                if m is not None:
                    mins.append(m)
                if exc:
                    causes[exc.get('exception_type') or 'unknown'] += 1
                    continue
                rewards = (rec.get('verifier_result') or {}).get('rewards') or {}
                if 'reward' in rewards:
                    scored += 1
                    if float(rewards['reward']) > 0:
                        solved += 1
                        causes['SOLVED'] += 1
                    else:
                        causes['scored 0 (real attempt)'] += 1
                else:
                    causes['no reward recorded'] += 1
            total = sum(causes.values())
            if not total:
                continue
            med = sorted(mins)[len(mins) // 2] if mins else None
            print(f'=== {os.path.basename(run)}  policy={policy}  '
                  f'{total} trial(s), {scored} scored, {solved} solved'
                  + (f', median agent {med:.0f} min' if med is not None else ''))
            for cause, n in causes.most_common():
                print(f'    {n:3d}  {cause}')
            infra = sum(n for c, n in causes.items()
                        if c not in ('SOLVED', 'scored 0 (real attempt)'))
            if infra:
                print(f'    -> {infra} of {total} trials never reached the verifier; a pass rate '
                      f'computed over {total} understates the model by that much')
            print()


if __name__ == '__main__':
    main(sys.argv[1:] or sorted(glob.glob('results/terminal_bench/kse_job_*')))
