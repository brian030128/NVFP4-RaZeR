"""Aggregate completed frozen diagnostics without selecting any new policy."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('results/task_reorder/llama_diagnosis_20260920')


def main():
    assert os.environ.get('SLURM_JOB_ID')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    plan_hash = digest(ROOT / 'plan.json')
    sources = {}
    reports = {}
    for name in ('fit', 'depth_0', 'depth_15', 'depth_31'):
        p = ROOT / name / 'report.json'
        r = json.loads(p.read_text())
        assert r['status'] == 'complete' and not r['smoke'] and r['plan_sha256'] == plan_hash
        sources[str(p)] = digest(p)
        reports[name] = r
    fit = reports['fit']
    assert fit['completed_bf16'] == fit['completed_quant'] == 208
    assert fit['audits']['bf16_replay'] == fit['audits']['quant_raw_replay'] == 208
    assert len(fit['audits']['changed_full']) == 45
    assert fit['records_sha256'] == digest(ROOT / 'records.pt')
    for name in ('depth_0', 'depth_15', 'depth_31'):
        r = reports[name]
        assert len(r['sequences']) == 32
        assert all(v == 32 for v in r['audits'].values())
    reference = reports['depth_0']['sequences']
    for name in ('depth_15', 'depth_31'):
        assert [(r['document_sha256'], r['token_sha256']) for r in reference] == [
            (r['document_sha256'], r['token_sha256']) for r in reports[name]['sequences']]
    result = dict(status='complete', diagnostic_only=True, plan_sha256=plan_hash,
        source_reports=sources, fit=fit['paired'], weight_diagnostics=fit['weight_diagnostics'],
        depth={name: r['groups'] for name, r in reports.items() if name.startswith('depth')},
        all_audits_passed=True)
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    def estimate(value):
        return f'{value["mean"]:+.6f} ± {value["two_se"]:.6f}'
    lines = ['# Frozen Llama mechanism diagnosis', '',
        'All reported intervals are descriptive mean ±2SE across documents, without multiplicity adjustment. '
        'No search, candidate promotion, or accuracy-benchmark fitting occurred.', '',
        '## Quantization context and data transfer', '',
        '| Documents | Quantized arranged−raw ΔCE | Δteacher KL | BF16+delta ΔCE | BF16−delta ΔCE |',
        '|---|---:|---:|---:|---:|']
    for group in ('fit', 'election', 'development', 'fresh_in_domain', 'fresh_general'):
        q = fit['paired']['quant_both - quant_raw'][group]
        plus = fit['paired']['bf16_plus_both - bf16'][group]['ce']
        minus = fit['paired']['bf16_minus_both - bf16'][group]['ce']
        lines.append(f'| {group} | {estimate(q["ce"])} | {estimate(q["kl"])} | {estimate(plus)} | {estimate(minus)} |')
    lines += ['', 'The BF16 counterfactual adds/subtracts the effective arranged−raw quantized weight change '
        'to/from original BF16 weights. This is not a deployment policy. Original fit/election and reused '
        'development documents are not independent validation. Fresh math/code64 and general-text32 '
        'documents exclude prior recorded calibration and confirmation documents.', '',
        '## Projection interactions', '',
        '| Fresh documents | Gate-only ΔCE | Up-only ΔCE | Down-only ΔCE | Joint minus sum of isolated ΔCE |',
        '|---|---:|---:|---:|---:|']
    for group in ('fresh_in_domain', 'fresh_general'):
        values = [fit['paired'][f'quant_{p} - quant_raw'][group]['ce'] for p in ('gate', 'up', 'down')]
        values.append(fit['paired']['joint_minus_sum_isolated'][group]['ce'])
        lines.append('| ' + group + ' | ' + ' | '.join(estimate(v) for v in values) + ' |')
    lines += ['', '## Depth: joint fixed-template perturbation on24 fresh documents', '',
        '| Layer index | Predicted ΔCE | Actual ΔCE | Frozen residual ΔCE | Actual prediction MAE | Frozen prediction MAE | Actual / frozen sign agreement |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for layer in (0, 15, 31):
        v = reports[f'depth_{layer}']['groups']['fresh']['joint']
        lines.append(f'| {layer} | {estimate(v["predicted"])} | {estimate(v["actual"])} | '
            f'{estimate(v["frozen"])} | {v["actual_prediction_mae"]:.6f} | {v["frozen_prediction_mae"]:.6f} | '
            f'{v["actual_sign_agreement"]:.1%} / {v["frozen_sign_agreement"]:.1%} |')
    lines += ['', 'The same accepted final-layer permutation/mask is transferred diagnostically to earlier '
        'layers, which have different weights and perturbation norms. This is not an optimized early-layer '
        'candidate and cannot prove earlier layers are unhelpful. The frozen control uses q0+(x−x0) at '
        'activation quantizers: its raw baseline is bitwise exact but it is not a legal FP4 execution. '
        'STE predictions use the actual BF16-rounded weight delta. BF16 rounding, nonlinear layers and '
        'interactions remain in the control. Quarter-size and isolated-projection results are in summary.json.', '',
        'All416 suffix baseline audits,45 changed full-model audits, and288 depth baseline/restore/STE '
        'audits passed. Source weights, frozen files and cross-depth question identities were checked.']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('DIAGNOSIS SUMMARY COMPLETE', flush=True)


if __name__ == '__main__':
    main()
