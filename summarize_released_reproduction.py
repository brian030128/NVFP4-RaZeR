"""Validate every direct-release case and publish all Table 3 residuals."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def compare_environments(root, reports):
    plan_path = root.parent / 'environment_comparison_plan.json'
    if not plan_path.exists():
        return ''
    plan = json.loads(plan_path.read_text())
    if root.name != plan['second']:
        return ''
    first = root.parent / plan['first']
    rows = []
    for new in reports:
        old = json.loads((first / new['case']['id'] / 'report.json').read_text())
        assert old['status'] == 'complete'
        assert old['case'] == new['case']
        assert [w['input_sha256'] for w in old['windows']] == [w['input_sha256'] for w in new['windows']]
        old_weights, new_weights = old['quantized_weight_sha256'], new['quantized_weight_sha256']
        assert old_weights.keys() == new_weights.keys()
        changed = sum(old_weights[k] != new_weights[k] for k in old_weights)
        rows.append(dict(case=new['case']['id'], changed_weight_matrices=changed,
                         weight_matrices=len(old_weights),
                         wiki_delta=new['ppl']['wikitext']-old['ppl']['wikitext'],
                         c4_delta=new['ppl']['c4']-old['ppl']['c4'],
                         max_window_nll_delta=max(abs(a['nll']-b['nll']) for a,b in zip(old['windows'],new['windows']))))
    data = dict(plan=plan, inputs_identical=True, rows=rows,
                max_abs_ppl_delta=max(abs(r[k]) for r in rows for k in ('wiki_delta','c4_delta')))
    (root / 'environment_comparison.json').write_text(json.dumps(data, indent=2)+'\n')
    lines = ['### Fixed environment diagnostic', '',
             'Both complete attempts are retained. The second uses Python 3.10.18, the released core package pins, '
             'and Torch 2.7.1/CUDA 12.6 inferred from the listed Triton/CUDA dependencies. '
             'The first uses Python 3.11/Torch 2.9. This does not establish the authors’ exact environment.', '',
             f"All input windows match exactly between attempts. Maximum absolute PPL change: **{data['max_abs_ppl_delta']:.6f}**. "
             'Δ below is second environment minus first; every case is shown.', '',
             '| Case | Δ Wiki | Δ C4 | Changed weight matrices |',
             '|---|---:|---:|---:|']
    lines += [f"| {r['case']} | {r['wiki_delta']:+.6f} | {r['c4_delta']:+.6f} | {r['changed_weight_matrices']}/{r['weight_matrices']} |" for r in rows]
    lines += ['', 'Qwen3-4B NVFP4 weight-only WikiText is listed as **13.63 in Table 1** and '
              '**13.83 in Table 3**. The primary table above keeps the Table 3 target. '
              'This internal difference is documented separately rather than changing the reproduction target.', '']
    return '\n'.join(lines)


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run aggregation through Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--update-report', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    cases = json.loads((root / 'cases.json').read_text())
    reports = []
    reference_inputs = {}
    rows = []
    matches = 0
    for case in cases:
        path = root / case['id'] / 'report.json'
        report = json.loads(path.read_text())
        assert report['status'] == 'complete', (path, report.get('error'))
        assert report['case'] == case
        assert report['release_commit'] == manifest['release_commit']
        assert report['c4_windows'] == 256
        assert report['dtype'] == 'torch.bfloat16'
        assert all('/source/' in p for p in report['imported_source'].values())
        inputs = [v['input_sha256'] for v in report['windows']]
        assert all(v['tokens'] == 2048 for v in report['windows'])
        previous = reference_inputs.setdefault(case['model'], inputs)
        assert previous == inputs, f"Different evaluation inputs: {case['id']}"
        reports.append(report)
        for dataset in ('wikitext', 'c4'):
            actual = report['ppl'][dataset]
            target = case['paper'][dataset]
            match = f'{actual:.2f}' == f'{target:.2f}'
            assert match == report['displayed_match'][dataset]
            matches += match
            rows.append(dict(model=case['model'], policy=case['policy'], dataset=dataset,
                             paper=target, reproduced=actual, delta=actual-target,
                             displayed_match=match, report_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    summary = dict(status='complete', validation='All cases complete; identical token windows across policies within each model',
                   source_manifest=manifest, cases=len(cases), cells=len(rows), displayed_matches=matches,
                   rows=rows, aggregation_job=os.environ['SLURM_JOB_ID'])
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['## Direct reproduction with the February RaZeR release', '',
        f"The archived released evaluator at commit `{manifest['release_commit'][:7]}` was run directly for "
        f"{len(cases)} cases ({len(rows)} separate WikiText-2/C4 cells). **{matches}/{len(rows)} cells match "
        'Table 3 at its published two-decimal precision.** No calibration or parameter search was used. '
        'The evaluator uses 2048-token windows, seed 0, 256 sampled C4 windows, and its original float32 PPL aggregation.', '',
        'Signed Δ is reproduced PPL minus published PPL; negative means lower perplexity. '
        'A displayed-precision match is not a claim of bitwise agreement with unpublished author outputs.', '',
        '| Model | Method | Paper Wiki | Reproduced Wiki | Δ Wiki | Paper C4 | Reproduced C4 | Δ C4 | Matches |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    names = {'bf16': 'BF16 (paper: FP16)', 'nvfp4': 'NVFP4', 'four_over_six': 'FourOverSix', 'razer': 'RaZeR'}
    for r in reports:
        c = r['case']; policy = c['policy']
        name = names['bf16'] if policy == 'bf16' else names[policy.rsplit('_', 1)[0]] + ' ' + policy.rsplit('_', 1)[1].upper()
        w, cw = r['ppl']['wikitext'], r['ppl']['c4']
        pw, pc = c['paper']['wikitext'], c['paper']['c4']
        matched = ', '.join(d for d in ('wikitext', 'c4') if r['displayed_match'][d]) or 'none'
        lines.append(f"| {c['model']} | {name} | {pw:.2f} | {w:.6f} | {w-pw:+.6f} | {pc:.2f} | {cw:.6f} | {cw-pc:+.6f} | {matched} |")
    versions = reports[0]['packages']
    backends = ', '.join(sorted({r['attention_backend'] for r in reports}))
    gpus = ', '.join(sorted({r['gpu'] for r in reports}))
    lines += ['', f"Execution: job `{manifest['job_id']}`, account `gov113008`; {gpus}. "
              'Independent one-GPU Slurm steps share one eight-GPU allocation. '
              f"Python {reports[0]['python']}, Torch {versions['torch']}, Transformers {versions['transformers']}, "
              f"datasets {versions['datasets']}; recorded attention backend(s): `{backends}`. "
              'All policies within each model passed exact input-token-hash equality checks; '
              'recorded losses reproduce the original evaluator’s PPL exactly.', '',
              '**Historical behavior and limits.** The archived Qwen wrapper leaves `o_proj` inputs '
              'unquantized, despite calculating a quantized copy. Its W4A4 labels therefore describe the '
              'release’s command-line setting, with this omission; they must not replace a corrected full-W4A4 '
              'baseline without disclosure. The original NVFP4 midpoint lookup also differs from the current '
              'arithmetic quantizer. The first January evaluator could load author-local cached C4 tokens; '
              'those tokens are unavailable, and the February release always regenerates seed-0 windows. '
              'The released environment omits a Torch version, so exact environment reconstruction is not established. '
              'These are reproduction limits, not explanations proven to account for every residual.', '',
              'This audit does not change or validate the existing 512-token E0M3 calibration gains at 2048 tokens.', '',
              '[Published Table 3](https://arxiv.org/html/2501.04052v2#S4.T3).', '']
    body = '\n'.join(lines) + '\n' + compare_environments(root, reports)
    (root / 'REPORT.md').write_text(body)
    if args.update_report:
        project = Path(__file__).resolve().parent
        report_path = project / 'MIXFP4_REPORT.md'
        start, end = '<!-- RELEASED_REPRODUCTION_START -->', '<!-- RELEASED_REPRODUCTION_END -->'
        link = str((root / 'REPORT.md').relative_to(project))
        block = start + '\n' + body + f'\n[Full reproduction record]({link}) · [Protocol](results/released_reproduction/PROTOCOL.md).\n\n' + end
        text = report_path.read_text()
        if start in text:
            before, remaining = text.split(start, 1)
            _, after = remaining.split(end, 1)
            text = before + block + after
        else:
            anchor = '<!-- BASELINE_PROTOCOL_AUDIT_END -->'
            assert anchor in text
            text = text.replace(anchor, anchor + '\n\n' + block, 1)
        report_path.write_text(text)
    print(f"VALIDATED {len(cases)} cases, {matches}/{len(rows)} displayed matches", flush=True)


if __name__ == '__main__':
    main()
