"""Strict paired accuracy analysis of the four frozen evaluation shards."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from analyze_zeroshot_paired import mcnemar_exact


def paired_accuracy(left, right, metric):
    a = {r['doc_id']: r for r in left}
    b = {r['doc_id']: r for r in right}
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError('Duplicate or unmatched question IDs')
    wins = losses = correct_a = correct_b = 0
    for key, r in a.items():
        s = b[key]
        if r['content_sha256'] != s['content_sha256']:
            raise ValueError('Question, target, or prompt mismatch')
        if r[metric] not in (0, 1) or s[metric] not in (0, 1):
            raise ValueError('Expected binary task accuracy')
        correct_a += int(r[metric]); correct_b += int(s[metric])
        wins += int(s[metric] > r[metric]); losses += int(s[metric] < r[metric])
    n = len(a)
    return dict(n=n, raw_correct=correct_a, arranged_correct=correct_b,
                wins=wins, losses=losses, unchanged=n-wins-losses)


def statistics(counts):
    n, wins, losses = counts['n'], counts['wins'], counts['losses']
    delta = (wins - losses) / n
    se = math.sqrt(max(0., (wins + losses - n * delta * delta) / (n - 1) / n))
    return dict(counts, raw_accuracy=counts['raw_correct'] / n,
        arranged_accuracy=counts['arranged_correct'] / n, delta=delta, two_se=2 * se,
        interval=[delta - 2 * se, delta + 2 * se], p_exact=mcnemar_exact(losses, wins))


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Summarize on a Slurm worker'
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    args = ap.parse_args()
    root = args.root
    plan_path = root / 'plan.json'
    plan = json.loads(plan_path.read_text())
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    expected = {'arc_challenge', *plan['mmlu_subjects']}
    counts = {}; reports = {}
    for shard in range(4):
        directory = root / 'shards' / str(shard)
        path = directory / 'report.json'
        report = json.loads(path.read_text())
        assert report['status'] == 'complete' and not report['smoke']
        assert report['plan_sha256'] == digest(plan_path)
        assert report['raw_restore_bitwise_audit'] and report['source_weight_hashes_checked']
        assert report['raw_tiles'] == 195 and report['arranged_tiles'] == 212
        reports[str(path)] = digest(path)
        samples = {}
        for policy in ('raw256', 'arranged'):
            sample_path = directory / f'{policy}_samples.json'
            assert digest(sample_path) == report['sample_sha256'][policy]
            samples[policy] = json.loads(sample_path.read_text())
            assert set(samples[policy]) == set(plan['shards'][str(shard)])
        for task in report['tasks']:
            assert task not in counts
            metric = 'acc_norm' if task == 'arc_challenge' else 'acc'
            values = paired_accuracy(samples['raw256'][task], samples['arranged'][task], metric)
            assert values['n'] == report['datasets'][task]['test_documents']
            for policy, field in [('raw256', 'raw_correct'), ('arranged', 'arranged_correct')]:
                assert abs(values[field] / values['n'] - report['results'][policy][task][metric + ',none']) < 1e-12
            counts[task] = values
    assert set(counts) == expected
    mmlu = {key: sum(counts[t][key] for t in plan['mmlu_subjects']) for key in counts['arc_challenge']}
    endpoints = {'mmlu': statistics(mmlu), 'arc_challenge': statistics(counts['arc_challenge'])}
    cumulative = 0.
    for rank, name in enumerate(sorted(endpoints, key=lambda t: endpoints[t]['p_exact'])):
        cumulative = max(cumulative, (len(endpoints) - rank) * endpoints[name]['p_exact'])
        endpoints[name]['p_holm'] = min(1., cumulative)
        endpoints[name]['evidence_of_improvement'] = endpoints[name]['delta'] > 0 and cumulative < .05
    result = dict(status='complete', diagnostic_only=True, no_fitting=True, native_backend=False,
        plan_sha256=digest(plan_path), source_reports=reports, endpoints=endpoints,
        subjects={task: statistics(values) for task, values in counts.items()})
    (root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Frozen Qwen 256×64 task accuracy', '',
        'Raw 195-tile map versus accepted 212-tile reordered map. Zero-shot, no chat template, '
        'same H200 and batch ordering for each paired comparison. No benchmark fitting.', '',
        '| Task | Questions | Raw | Reordered | Δ percentage points | Wins / losses | Holm p |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, value in endpoints.items():
        lines.append(f'| {name} | {value["n"]} | {100*value["raw_accuracy"]:.3f}% | '
            f'{100*value["arranged_accuracy"]:.3f}% | {100*value["delta"]:+.3f} | '
            f'{value["wins"]} / {value["losses"]} | {value["p_holm"]:.4g} |')
    lines += ['', 'MMLU is document-weighted across all 57 subjects; ARC uses acc_norm. '
        'Exact two-sided McNemar tests are Holm-adjusted across these two endpoints. '
        'Per-subject results are descriptive. A nonsignificant positive delta is not established improvement. '
        'This measures the fake-quantized model, not the unverified native backend.']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('ACCURACY PAIRED ' + json.dumps(endpoints), flush=True)


if __name__ == '__main__':
    main()
