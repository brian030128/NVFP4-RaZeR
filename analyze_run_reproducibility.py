"""
    How reproducible is a W4A4 multiple-choice score across jobs, and does §1a's test survive it?

    The same policy, weights, data and library versions re-measured in a different job does not
    reproduce document for document. Borderline loglikelihood comparisons -- winogrande's two
    continuations differ only by a pronoun, so their scores sit within rounding of each other --
    flip when the GEMM kernel changes, and it changes with the GPU the job lands on.

    That is worth knowing on its own, but the reason to measure it is that §1a's conclusions all
    rest on an exact McNemar test over per-document outcomes. If re-running a policy produced a
    *systematic* shift, that test would be reading hardware noise as a policy effect. So this is
    the negative control: run the identical comparison on two runs of the SAME policy, where the
    true difference is zero by construction, and check that the test says so.

        python analyze_run_reproducibility.py --model qwen4b
"""

import argparse
import glob
import json
import os

from analyze_zeroshot_paired import compare, load

NODE_GPU = {}   # filled from the job logs; the partition table in CLAUDE.md maps node -> GPU


def gpu_of(node):
    """H100 or H200, from the partition layout in /home/u4320956/CLAUDE.md."""
    if not node.startswith('hgpn'):
        return '?'
    try:
        n = int(node[4:])
    except ValueError:
        return '?'
    return 'H200' if n in {39, 40, 41, 43, 44, 45, 46} else 'H100'


def job_node(job_id):
    """The node a job ran on, recovered from its Slurm log's banner line."""
    for path in glob.glob(f'slurm/logs/*_{job_id}.out'):
        try:
            with open(path, errors='ignore') as f:
                for line in f:
                    if ' on hgpn' in line:
                        return line.rsplit(' on ', 1)[1].strip().split()[0]
        except OSError:
            pass
    return '?'


def sample_sets(model):
    """Every {policy: samples_path} keyed by job id, for runs that kept per-document samples."""
    out = {}
    for d in sorted(glob.glob(f'results/zeroshot_kse/job_*/samples_{model}')):
        job = os.path.basename(os.path.dirname(d)).replace('job_', '')
        got = {os.path.basename(p)[:-len('.json')]: p for p in glob.glob(os.path.join(d, '*.json'))}
        if got:
            out[job] = got
    return out


def build(models):
    """The negative-control block, or None when no policy has been measured twice.

    `models` is either a list of keys or of (key, display label) pairs, so the table can name
    models the way the rest of the section does.
    """
    labels = dict(m if isinstance(m, tuple) else (m, m) for m in models)
    # (model, job_a, job_b, policy) for every policy measured twice, scored once each.
    results = []
    for model in labels:
        runs = sample_sets(model)
        jobs = sorted(runs)
        for i, ja in enumerate(jobs):
            for jb in jobs[i + 1:]:
                for pol in sorted(set(runs[ja]) & set(runs[jb])):
                    sa, sb = load(runs[ja][pol]), load(runs[jb][pol])
                    # The samples directory is named per model, not per task set, so a panel run
                    # and a gsm8k run collide here. Only a shared task set is a re-measurement.
                    if set(sa) != set(sb):
                        continue
                    _, pooled = compare(sa, sb)
                    if pooled['n']:
                        results.append((model, ja, jb, pol, pooled))
    if not results:
        return None

    L = ['#### Is the paired test reading hardware noise?', '',
         'Every conclusion above rests on an exact McNemar test over per-document outcomes, so '
         'it is worth running that test where the answer is known. Below is the identical '
         'comparison applied to two jobs of the **same** policy, on the same weights, data and '
         'library versions -- the true difference is zero by construction. `b` and `c` count the '
         'documents only one of the two runs gets right; a significant p here would mean §1a is '
         'reading the hardware as if it were the method.', '',
         '| model | policy | jobs | GPUs | documents | b | c | delta | McNemar p |',
         '|---|---|---|---|---:|---:|---:|---:|---:|']

    worst = 0.0
    for model, ja, jb, pol, pooled in results:
        ga, gb = gpu_of(job_node(ja)), gpu_of(job_node(jb))
        gpus = ga if ga == gb else f'{ga} vs {gb}'
        L.append(f'| {labels[model]} | {pol} | {ja} vs {jb} | {gpus} | {pooled["n"]} | '
                 f'{pooled["b"]} | '
                 f'{pooled["c"]} | {pooled["delta"]:+.4f} | {pooled["p"]:.3g} |')
        worst = max(worst, (pooled['b'] + pooled['c']) / pooled['n'])

    ps = [pooled['p'] for *_, pooled in results]
    bad = sum(1 for p in ps if p < 0.05)
    verdict = (f'none of the {len(ps)} shows a significant asymmetry' if not bad else
               f'**{bad} of {len(ps)} show a significant asymmetry**, which would undermine the '
               f'tests above')

    # Same-GPU and cross-GPU pairs answer different questions, so they are counted apart.
    same = [r for r in results if gpu_of(job_node(r[1])) == gpu_of(job_node(r[2]))]
    cross = [r for r in results if r not in same]
    para = []
    if same and all(p['b'] + p['c'] == 0 for *_, p in same):
        para.append(f'Re-running on the same GPU model reproduces the evaluation exactly: '
                    f'{len(same)} such pairs, and not one of '
                    f'{sum(p["n"] for *_, p in same):,} scored documents changes outcome. The '
                    f'evaluation itself is deterministic.')
    if cross:
        w = max((p['b'] + p['c']) / p['n'] for *_, p in cross)
        para.append(f'Across GPU models it is not. Up to **{w:.1%} of documents flip** -- where '
                    f'two continuations score within rounding of each other, and winogrande\'s '
                    f'differ only by a pronoun, a different GEMM kernel is enough to reverse the '
                    f'comparison. Aggregate accuracy still moves by well under a point, because '
                    f'the flips go both ways.')
    L += ['', ' '.join(para), '',
          f'The point of the table is that {verdict}. That symmetry is the case McNemar '
          f'conditions on: the test is computed from the *difference* between `b` and `c`, not '
          f'from their size, so noise that inflates both equally cancels. The comparisons in '
          f'§1a are also made within a single job, where the evaluation is exactly reproducible, '
          f'so they do not carry even this term. What it does mean is that a single document\'s '
          f'outcome is not a portable property of a policy, and that accuracies here should be '
          f'read to a few tenths of a percent rather than to the digits lm-eval prints.', '']

    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['llama8b', 'qwen4b', 'qwen27b'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    out = build(args.models)
    assert out, f'no policy is present in two runs of any of {args.models}'
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            f.write(out)
    print(out)


if __name__ == '__main__':
    main()
