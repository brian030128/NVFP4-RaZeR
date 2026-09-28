#!/usr/bin/env python3
"""Step 04 (Experiment 2): downstream accuracy through lm-eval, run_lmeval_deploy.py, NativeLinear (c) only.

    PAPER_PYTHON experiments/paper/04_downstream.py [--models ...] [--units ...] [--policies ...] [--tasks ...] [--smoke]

- **Tasks:** MMLU 5-shot; ARC-Challenge, ARC-Easy, HellaSwag and PIQA with lm-eval's defaults (0-shot).
- **Harness:** lm-eval 0.4.11 with the default BOS handling, and the datasets pinned by lm_eval_datasets.py
  (cais/mmlu at a fixed revision). Batch 16 (Qwen3.8-27B: 8).
- **MMLU:** its score is lm-eval's group aggregate: weight_by_size, the micro-average over all 14,042 questions. The
  57 subjects and 4 categories are recorded as well.
- **Policies:** step 03's (paper_common.ACCURACY_POLICIES), same artifacts and kernels, no fake quantization.
- **Recorded:** per example, the correctness under the primary metric (acc_norm for ARC, HellaSwag and PIQA; acc for
  MMLU) and the per-choice log-likelihoods, for the paired comparisons of step 07.
- **Resumable:** one process per (model, policy), into <out>/lmeval/<model>/<policy>/report.json. A finished
  (model, policy) is skipped; an unfinished one resumes at the task level (run_lmeval_deploy.py --resume).
- **--smoke:** Llama-3.1-8B, every policy but 256x64, --limit 5 (MMLU: 5 questions per subject).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as P  # noqa: E402

TASKS = 'mmlu,arc_challenge,arc_easy,hellaswag,piqa'
FEWSHOT = ['mmlu=5']


def finished(report, pol, tasks, limit):
    try:
        r = json.loads(report.read_text())
    except (OSError, ValueError):
        return False
    done = r.get('evaluations', {}).get(pol, {}).get('tasks', {})
    return r.get('status') == 'complete' and r.get('limit') == limit and all(done.get(t, {}).get('examples') for t in tasks)


def main():
    ap = P.parser(__doc__)
    ap.add_argument('--tasks', default=TASKS)
    ap.add_argument('--batch-size', type=int, default=None,
                    help='override the batch (16; Qwen 8), e.g. after an out-of-memory error: a deviation, record it')
    args = P.setup(ap.parse_args())
    if args.smoke:
        args.models, args.units = ['llama8b'], ['8x64', '16x64']
    limit = 5 if args.smoke else None
    tasks = args.tasks.split(',')
    names = [p for p in P.ACCURACY_POLICIES if P.unit_of(p) is None or P.unit_of(p) in args.units]
    if args.policies:
        names = [p for p in names if p in args.policies.split(',')]
    for model in args.models:
        for pol in names:
            out = args.out / 'lmeval' / model / pol
            report = out / 'report.json'
            if finished(report, pol, tasks, limit) and not args.force:
                print(f'{model} {pol}: done')
                continue
            kind, kernel = P.ACCURACY_POLICIES[pol]
            if kind is None:
                spec = f'{pol}=bf16'
            else:
                art = P.artifact(args.out, model, kind)
                if not (art / 'artifact.json').exists():
                    P.die(f'{art} missing: run step 02 first')
                spec = f'{pol}=native:{art}:{kernel}'
            cmd = [P.PY, 'run_lmeval_deploy.py', '--model', model, '--data-root', P.DATA[model], *P.DEVIATION.get(model, []),
                   '--evaluate', spec, '--tasks', ','.join(tasks), '--batch-size', str(args.batch_size or P.LMEVAL_BATCH.get(model, 16)),
                   *[a for f in FEWSHOT for a in ('--num-fewshot', f)], '--out', out]
            if limit is not None:
                cmd += ['--limit', str(limit)]
            if report.exists() and not args.force:
                cmd.append('--resume')
            rc = P.run(args.out, f'04_lmeval_{model}_{pol}', cmd, extra_env=P.ACCURACY_ENV)
            if rc != 0 or not finished(report, pol, tasks, limit):
                P.die(f'lm-eval failed: {model} {pol} (log: {args.out}/logs/04_lmeval_{model}_{pol}.log); '
                      're-running this step resumes at the task level')
            ev = json.loads(report.read_text())['evaluations'][pol]['tasks']
            print(f'{model} {pol}: ' + ', '.join(f"{t} {ev[t]['metrics'].get(('acc' if t == 'mmlu' else 'acc_norm') + ',none'):.4f}"
                                                  for t in tasks), flush=True)


if __name__ == '__main__':
    main()
