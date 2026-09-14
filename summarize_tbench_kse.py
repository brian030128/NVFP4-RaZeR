"""
    Summarize the Qwen3.8-27B Terminal-Bench comparison across quantization policies.

    Terminal-Bench is an agentic benchmark: the model drives a real shell in a container and is
    scored by the task's own verifier, so it sees things the multiple-choice panel in §1a cannot
    -- instruction following over many turns, tool syntax, and recovery from its own mistakes.
    That also makes it a small-n benchmark. Twelve tasks at one trial each is 12 Bernoulli draws
    per policy, so this reports the counts and the paired disagreement rather than implying a
    resolution the sample size does not support.

        python summarize_tbench_kse.py --runs results/terminal_bench/kse_job_339084 ... \
            --out results/terminal_bench/SECTION_27b.md
"""

import argparse
import glob
import json
import os
from math import comb

POLICY_LABEL = {'bf16': 'BF16', 'nvfp4': 'NVFP4 W4A4',
                'four_over_six': 'NVFP4 FourOverSix W4A4', 'k3': 'MixFP4 k=3 W4A4'}
ORDER = ['bf16', 'nvfp4', 'four_over_six', 'k3']


def trials(run_dir, policy):
    """Per-task reward for one policy, keyed by task name.

    The job-level result.json aggregates, but it buckets trials by reward value rather than
    naming the reward per task, and it drops tasks that raised before the verifier ran. Reading
    the per-trial files keeps errored tasks visible as errors instead of silently as zeros.
    """
    out, errors, minutes, truncated = {}, {}, {}, set()
    root = os.path.join(run_dir, f'jobs_{policy}')
    for path in glob.glob(os.path.join(root, '*', 'result.json')):
        try:
            r = json.load(open(path))
        except Exception:
            continue
        if 'task_name' not in r:
            continue                       # the job-level result.json sits alongside the trials
        task = r['task_name'].split('/')[-1]
        exc = r.get('exception_info')
        rewards = (r.get('verifier_result') or {}).get('rewards') or {}
        # An exception does not mean the trial went unscored: the agent phase can time out
        # after the task is already solved, and the verifier still runs. Only a trial with no
        # reward at all is an error for counting purposes.
        if exc and 'reward' not in rewards:
            errors[task] = (exc.get('exception_type') or str(exc))[:60]
            continue
        if 'reward' in rewards:
            out[task] = float(rewards['reward'])
            if (exc or {}).get('exception_type') == 'AgentTimeoutError':
                truncated.add(task)
            secs = _duration(r.get('agent_execution'))
            if secs is not None:
                minutes[task] = secs / 60.0
    return out, errors, minutes, truncated


def _duration(phase):
    """Seconds a trial phase took, or None when it is missing or unparseable."""
    if not isinstance(phase, dict):
        return None
    a, b = phase.get('started_at'), phase.get('finished_at')
    if not a or not b:
        return None
    try:
        from datetime import datetime
        fix = lambda t: datetime.fromisoformat(t.replace('Z', '+00:00'))
        return (fix(b) - fix(a)).total_seconds()
    except Exception:
        return None


def sign_test(a, b):
    """Two-sided exact binomial test on the tasks where the two policies disagree.

    The pairing is what makes 12 tasks worth testing at all: task difficulty is the dominant
    source of variance and it cancels, so only the discordant tasks carry information.
    """
    wins = sum(1 for t in a if t in b and a[t] > b[t])
    losses = sum(1 for t in a if t in b and a[t] < b[t])
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return wins, losses, min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True,
                    help='kse_job_* directories, one per policy (or several policies each)')
    ap.add_argument('--reference', default='four_over_six',
                    help='Policy the others are tested against. FourOverSix is the base the '
                         'method switches from, so it is the comparison that matters.')
    ap.add_argument('--out', default='results/terminal_bench/SECTION_27b.md')
    args = ap.parse_args()

    got, errs, mins, trunc, provenance = {}, {}, {}, {}, {}
    for run in args.runs:
        for path in glob.glob(os.path.join(run, 'provenance_*.json')):
            policy = os.path.basename(path)[len('provenance_'):-len('.json')]
            rewards, errors, minutes, cut = trials(run, policy)
            if not rewards and not errors:
                continue
            got[policy], errs[policy], mins[policy] = rewards, errors, minutes
            trunc[policy] = cut
            provenance[policy] = json.load(open(path))
    assert got, 'no completed trials found in ' + ' '.join(args.runs)

    policies = [p for p in ORDER if p in got] + sorted(set(got) - set(ORDER))
    tasks = sorted({t for r in got.values() for t in r})
    common = [t for t in tasks if all(t in got[p] for p in policies)]

    L = ['## Terminal-Bench (Qwen3.8-27B)', '',
         'Terminal-Bench runs the model as an agent in a real container: it issues shell '
         'commands over many turns and the task\'s own verifier decides pass or fail. It '
         'therefore probes what the §1a multiple-choice panel cannot -- multi-turn instruction '
         'following, tool syntax, and recovery from the model\'s own errors -- on the one model '
         'in the panel large enough to attempt the tasks at all.', '']

    prec = {p: provenance[p].get('precision', '?') for p in policies}
    L += [f'Served in process from the same weights the perplexity and accuracy runs use, so '
          f'the activation-quantization hooks are live '
          f'({", ".join(f"{POLICY_LABEL.get(p, p)} = {prec[p]}" for p in policies)}). '
          f'{len(tasks)} tasks, 1 trial each, GPU tasks excluded.', '',
          '| policy | precision | resolved | of | pass rate | errored | median agent min |',
          '|---|---|---:|---:|---:|---:|---:|']
    for p in policies:
        r, e, mm = got[p], errs[p], mins[p]
        solved = sum(1 for v in r.values() if v > 0)
        med = '—'
        if mm:
            vals = sorted(mm.values())
            med = f'{vals[len(vals) // 2]:.0f}'
        L.append(f'| {POLICY_LABEL.get(p, p)} | {prec[p]} | {solved} | {len(r)} | '
                 f'{solved / len(r):.1%} | {len(e) or "—"} | {med} |')
    cut_n = {p: len(trunc.get(p, ())) for p in policies}
    worst = max(cut_n.values()) if cut_n else 0
    L += ['',
          'The last column is the median minutes the agent phase ran. `errored` counts trials '
          'that never reached the verifier at all, so the pass rate is over the trials that did.',
          '']
    if worst:
        L += [f'**These are not capability numbers.** The agent phase is capped, and the cap '
              f'fires on most trials that get scored -- up to {worst} of them here, which the '
              f'verifier then grades on whatever state the agent had reached. Serving runs in '
              f'process through Transformers so the W4A4 activation hooks stay live, at roughly '
              f'2 tokens per second, and a multi-turn task needs far longer than the cap allows. '
              f'The cap is identical across policies, so the comparison between them is fair; '
              f'what it measures is what each policy achieves within a fixed budget, and every '
              f'pass rate here is a lower bound on the model.', '']

    ref = args.reference
    if ref in got:
        L += [f'### Paired against {POLICY_LABEL.get(ref, ref)}', '',
              'Only tasks both policies completed are compared, and only the tasks they '
              'disagree on carry information -- an exact two-sided sign test on those.', '',
              '| policy | wins | losses | ties | sign-test p |', '|---|---:|---:|---:|---:|']
        for p in policies:
            if p == ref:
                continue
            w, l, pv = sign_test(got[p], got[ref])
            shared = sum(1 for t in got[p] if t in got[ref])
            L.append(f'| {POLICY_LABEL.get(p, p)} | {w} | {l} | {shared - w - l} | {pv:.3g} |')
        L.append('')

    if common:
        L += ['### Per task', '',
              '| task | ' + ' | '.join(POLICY_LABEL.get(p, p) for p in policies) + ' |',
              '|---|' + '---:|' * len(policies)]
        for t in common:
            marks = ' | '.join('pass' if got[p][t] > 0 else 'fail' for p in policies)
            L.append(f'| `{t}` | {marks} |')
        L.append('')

    dropped = [t for t in tasks if t not in common]
    if dropped:
        L += [f'**{len(dropped)} of {len(tasks)} tasks are not in the per-task table** because '
              f'at least one policy did not produce a verifier result for them '
              f'(`{"`, `".join(sorted(dropped))}`). They are excluded from the paired test '
              f'rather than counted as failures.', '']

    # The detection floor is computed, not quoted: the smallest clean sweep (no losses) whose
    # two-sided sign test clears 0.05, capped by how many tasks there actually are.
    n = len(common) or len(tasks)
    need = next((w for w in range(1, n + 1) if sign_test(
        {f't{i}': 1.0 for i in range(n)},
        {f't{i}': (0.0 if i < w else 1.0) for i in range(n)})[2] < 0.05), None)
    floor = (f'a policy would have to win {need} tasks against zero losses before a two-sided '
             f'sign test cleared 0.05' if need else
             f'no outcome at all on {n} tasks can clear 0.05 in a two-sided sign test')
    L += [f'**What {n} tasks can settle.** At one trial per task, {floor}, so this is sized to '
          f'catch a large regression, not to certify equivalence. Read it as a smoke test of '
          f'agentic capability under W4A4 rather than as a ranking.', '']

    out = '\n'.join(L) + '\n'
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(out)
    print(out)
    print(f'written to {args.out}')


if __name__ == '__main__':
    main()
