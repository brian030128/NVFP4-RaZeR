"""V20/V21/V22/V23 deliverables from historical runs, judged against the frozen numerical tolerances."""
import json
import math
import os
from pathlib import Path

from campaign import runtime
from campaign.policies import latest_complete_run

CR = Path(os.environ['CAMPAIGN_ROOT'])
FREEZE = json.loads((CR / 'freeze' / 'PROTOCOL_FREEZE.json').read_text())
TOL = FREEZE['numerical_tolerances']['historical']['tierB']


def run_or_none(job):
    try:
        return latest_complete_run(CR, job)
    except FileNotFoundError:
        return None


def attempts(job):
    out = []
    for d in sorted((CR / 'runs').glob(f'{job}_attempt*')):
        lr = d / 'launch_record.json'
        st = json.loads(lr.read_text()).get('status') if lr.exists() else None
        out.append(dict(run=d.name, status=st))
    return out


def mean_abs(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def main():
    out = runtime.out_dir('analysis_historical')
    src = CR / 'source' / 'NVFP4-RaZeR-main'
    score_manifest, anchors, evals = {}, {}, {}
    for m in ('qwen4b', 'llama8b', 'qwen27b'):
        run = run_or_none(f'V21_hist_calib_{m}')
        spec = json.loads((src / {'qwen4b': 'results/math_code_adaptive/calibration_333779_qwen4b', 'llama8b': 'results/math_code_adaptive/calibration_333779_llama8b',
                                  'qwen27b': 'results/math_code_adaptive/calibration_333787_qwen27b'}[m] / 'report.json').read_text())
        if run is None:
            score_manifest[m] = dict(status='not complete', attempts=attempts(f'V21_hist_calib_{m}'))
            anchors[m] = dict(status='not complete', attempts=attempts(f'V21_hist_calib_{m}'))
            continue
        r = json.loads((run / 'historical' / 'historical_report.json').read_text())
        score_manifest[m] = dict(run=run.name, regenerated=True, archived_score_shards_available=False,
                                 shard_sha256=r.get('shard_sha256'), shard_sha256_equal_archived=r['shard_sha256_equal_archived'],
                                 upper_file_sha256=r.get('upper_file_sha256'), weight_mse_sha256=r['weight_mse_sha256'],
                                 maps_json_sha256=r['maps_json_sha256'], calibration_tokens_equal_archived=r['calibration_tokens_equal_archived'],
                                 source_weights_verified=r['source_weights_verified'], stream_mode=r['stream'],
                                 completeness=dict(modules=len(r.get('shard_sha256') or {}) or None, sequences=128, objectives=['ce', 'kl', 'sampled_ce']))
        bf = [a for a in r['bf16_fit_nll']]
        ce_new = [x['ce'] for x in r['initial_fit_losses']]
        ce_old = [x['ce'] for x in spec['initial_fit_losses']]
        kl_new = [x['kl'] for x in r['initial_fit_losses']]
        kl_old = [x['kl'] for x in spec['initial_fit_losses']]
        k3 = r['kse_election']['k3']
        fx = r['map_comparison']['fixed256_math_code128']
        checks = {
            'N8_k2_score_identity (tierA)': dict(value=r['k2_equals_common_descent_scores'], passed=r['k2_equals_common_descent_scores'] is True),
            'fixed256_map_bitwise_vs_archived (tierA)': dict(value=fx['bitwise_equal'], passed=bool(fx['bitwise_equal'])),
            'fixed256_jaccard (tierB >= %s)' % TOL['fixed256_jaccard']: dict(value=fx['jaccard'], passed=fx['jaccard'] >= TOL['fixed256_jaccard']),
            'k3_count_relative (tierB <= %s)' % TOL['k3_count_rel']: dict(value=k3['selected'] / k3['archived'] - 1, regenerated=k3['selected'], archived=k3['archived'],
                                                                       passed=abs(k3['selected'] / k3['archived'] - 1) <= TOL['k3_count_rel']),
            'bf16_fit_nll_max_abs (tierB <= %s)' % TOL['bf16_fit_nll_max_abs']: dict(value=r['bf16_fit_nll_vs_archived']['max_abs'], passed=r['bf16_fit_nll_vs_archived']['max_abs'] <= TOL['bf16_fit_nll_max_abs']),
            'bf16_fit_nll_mean_abs (tierB <= %s)' % TOL['bf16_fit_nll_mean_abs']: dict(value=mean_abs(bf, spec['bf16_fit_nll']), passed=mean_abs(bf, spec['bf16_fit_nll']) <= TOL['bf16_fit_nll_mean_abs']),
            'w4a4_fit_ce_max_abs (tierB <= %s)' % TOL['w4a4_fit_ce_max_abs']: dict(value=max(abs(a - b) for a, b in zip(ce_new, ce_old)), passed=max(abs(a - b) for a, b in zip(ce_new, ce_old)) <= TOL['w4a4_fit_ce_max_abs']),
            'w4a4_fit_ce_mean_abs (tierB <= %s)' % TOL['w4a4_fit_ce_mean_abs']: dict(value=mean_abs(ce_new, ce_old), passed=mean_abs(ce_new, ce_old) <= TOL['w4a4_fit_ce_mean_abs']),
            'fit_kl_max_abs (tierB <= %s)' % TOL['fit_kl_max_abs']: dict(value=max(abs(a - b) for a, b in zip(kl_new, kl_old)), passed=max(abs(a - b) for a, b in zip(kl_new, kl_old)) <= TOL['fit_kl_max_abs']),
            'fit_kl_mean_abs (tierB <= %s)' % TOL['fit_kl_mean_abs']: dict(value=mean_abs(kl_new, kl_old), passed=mean_abs(kl_new, kl_old) <= TOL['fit_kl_mean_abs']),
            'weight_mse_map_bitwise (tierA)': dict(value=r['weight_mse_equal_archived'], passed=bool(r['weight_mse_equal_archived'])),
            'adaptive_maps_bitwise (tierA, count)': dict(value=sum(v['bitwise_equal'] for k, v in r['map_comparison'].items() if k.startswith('adaptive')),
                                                         total=sum(1 for k in r['map_comparison'] if k.startswith('adaptive')), passed=None),
            'n256_prefix_equals_regenerated_fixed256 (internal)': dict(value=r['n256_prefix_equals_regenerated_fixed256'], passed=bool(r['n256_prefix_equals_regenerated_fixed256'])),
        }
        inv = run_or_none(f'V21_hist_calib_{m}_sdpa_investigation')
        anchors[m] = dict(run=run.name, checks=checks, kse_election=r['kse_election'], archived_fixed256_ranks=r['archived_fixed256_ranks_in_regenerated'],
                          all_fixed256_maps={k: v for k, v in r['map_comparison'].items() if k.startswith('fixed256')},
                          same_gpu_kernel_perturbation=(json.loads((inv / 'historical' / 'historical_report.json').read_text()).get('same_gpu_comparison_with') if inv else 'not run'))
        ev = run_or_none(f'V22_hist_eval_{m}')
        if ev is None:
            evals[m] = dict(status='not complete', attempts=attempts(f'V22_hist_eval_{m}'))
            continue
        e = json.loads((ev / 'historical_eval' / 'historical_eval_report.json').read_text())
        ec = {}
        for pol, c in e['comparison'].items():
            ec[pol] = {}
            for dom, v in c.items():
                ec[pol][dom] = dict(v, tierA_exact=v['exact_equal'], tierB_ppl_rel=abs(v['ppl_rel_diff']) <= TOL['ppl_rel_w4a4'])
        pe = {}
        for pol, doms in e.get('paired_effects', {}).items():
            pe[pol] = {d: dict(v, tierC=(v['sign_equal'] and abs(v['difference']) <= max(0.25 * abs(v['archived_dlogppl']), 0.002))) for d, v in doms.items()}
        evals[m] = dict(run=ev.name, comparison=ec, paired_effects=pe, gpu=e['gpu'])
    ada = run_or_none('V23_hist_eval_qwen4b_ada')
    a6 = run_or_none('V22_hist_eval_qwen4b')
    cross = dict(status='not complete', attempts=attempts('V23_hist_eval_qwen4b_ada'))
    if ada and a6:
        ra = json.loads((ada / 'historical_eval' / 'historical_eval_report.json').read_text())
        rb = json.loads((a6 / 'historical_eval' / 'historical_eval_report.json').read_text())
        cross = dict(ada_run=ada.name, a6000_run=a6.name, gpus=dict(ada=ra['gpu'], a6000=rb['gpu']), policies={})
        for pol in ra['evaluation']:
            if pol not in rb['evaluation']:
                continue
            cross['policies'][pol] = {}
            for dom in ('wiki', 'c4'):
                x, y = ra['evaluation'][pol][dom]['nll'], rb['evaluation'][pol][dom]['nll']
                d = [p - q for p, q in zip(x, y)]
                cross['policies'][pol][dom] = dict(ppl_ada=ra['evaluation'][pol][dom]['ppl'], ppl_a6000=rb['evaluation'][pol][dom]['ppl'],
                                                   ppl_rel=ra['evaluation'][pol][dom]['ppl'] / rb['evaluation'][pol][dom]['ppl'] - 1,
                                                   window_nll_max_abs=max(abs(v) for v in d), window_nll_mean_abs=sum(abs(v) for v in d) / len(d), exact_equal=x == y)
    runtime.atomic_json(out / 'SCORE_MANIFEST.json', score_manifest)
    runtime.atomic_json(out / 'N8_ANCHOR_RESULTS.json', anchors)
    runtime.atomic_json(out / 'HISTORICAL_PPL_ANCHORS.json', evals)
    runtime.atomic_json(out / 'CROSS_GPU_ARCHIVE.json', cross)
    lines = ['# N8 anchor report (V21)', '', f'Frozen tolerances: `PROTOCOL_FREEZE.json` sha256 `{(CR / "freeze" / "PROTOCOL_FREEZE.sha256").read_text().split()[0]}`.', '']
    for m, a in anchors.items():
        lines.append(f'## {m}')
        if 'checks' not in a:
            lines += [f'Not complete: {a}', '']
            continue
        lines += [f'Run `{a["run"]}`.', '', '| check | value | passed |', '|---|---|---|']
        for k, v in a['checks'].items():
            val = v['value'] if not isinstance(v['value'], float) else f'{v["value"]:.6g}'
            lines.append(f'| {k} | {val}{" / " + str(v["total"]) if "total" in v else ""} | {v["passed"]} |')
        lines += ['', f'k-SE election counts (regenerated vs archived): `{json.dumps(a["kse_election"])}`', '',
                  f'Archived fixed-256 tiles inside the regenerated ranking: `{json.dumps(a["archived_fixed256_ranks"])}`', '',
                  f'Same-GPU kernel-path perturbation (eager -> SDPA): `{json.dumps(a["same_gpu_kernel_perturbation"])}`', '']
    (out / 'N8_ANCHOR_REPORT.md').write_text('\n'.join(lines) + '\n')
    src_manifest = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(
        protocol_id='historical', protocol_freeze_sha256=(CR / 'freeze' / 'PROTOCOL_FREEZE.sha256').read_text().split()[0],
        source=dict(model_id=None, model_revision=None, tokenizer_revision=None, model_class=None, module_manifest_sha256=None, source_manifest_sha256=src_manifest),
        environment=runtime.environment(), data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=None, token_hashes={}, overlap_audit=None),
        policies=[], results=dict(raw_outputs=[str(p) for p in sorted(out.iterdir())], summary=dict(models=list(anchors)), uncertainty={},
                                  attempted_endpoints=['V20', 'V21', 'V22', 'V23'], missing_endpoints=[m for m, v in evals.items() if 'run' not in v]),
        logs=[], failures=[]))


if __name__ == '__main__':
    main()
