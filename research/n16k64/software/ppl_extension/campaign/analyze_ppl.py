"""Aligned PPL analyses: V31/V62 primary tables, V40/V41/V42/V50/V51/V52/V71/V80/V81 comparisons (CPU)."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from campaign import runtime
from campaign import stats as S
from campaign.policies import latest_complete_run

CR = Path(os.environ['CAMPAIGN_ROOT'])
FREEZE = json.loads((CR / 'freeze' / 'PROTOCOL_FREEZE.json').read_text())
FSHA = (CR / 'freeze' / 'PROTOCOL_FREEZE.sha256').read_text().split()[0]
MARGIN = FREEZE['success_criteria']['sesoi']['ppl_noninferiority_margin_dlogppl']


def load_ppl(job):
    run = latest_complete_run(CR, job)
    rep = json.loads((run / 'ppl' / 'ppl_report.json').read_text())
    wins = {d: json.loads((run / 'ppl' / f'windows_{d}.json').read_text()) for d in rep['domains']}
    return run, rep, wins


def compare(rep, wins, a, b, B=10000, margin=None, block=True):
    out = {}
    for dom in rep['domains']:
        na, ta = S.window_table(rep, a, dom)
        nb, tb = S.window_table(rep, b, dom)
        assert np.array_equal(ta, tb)
        cl = S.clusters_for(wins[dom], dom)
        r = S.paired_dlogppl(na, nb, ta, cl, B=B, margin=margin)
        if dom == 'wiki' and block:
            r['block5_sensitivity'] = S.paired_dlogppl(na, nb, ta, S.clusters_for(wins[dom], 'wiki', 'block5'), B=min(B, 2000))
        r['ppl_a'], r['ppl_b'] = rep['evaluation'][a][dom]['ppl'], rep['evaluation'][b][dom]['ppl']
        out[dom] = r
    return out


def diagnostics(run, rep, wins, policies, ref='four_over_six', B=2000):
    """Token-level diagnostics and paired deltas vs ref (teacher metrics on the windows where they exist)."""
    res = {}
    for dom in rep['domains']:
        res[dom] = {}
        arrays = {}
        for p in policies:
            ta = rep['evaluation'][p][dom].get('token_arrays')
            if ta:
                arrays[p] = S.token_arrays(ta['path'])
        if ref not in arrays:
            continue
        toks = np.array([w['tokens'] for w in rep['evaluation'][ref][dom]['windows']], dtype=np.int64)
        teacher_windows = [w['window'] for w in rep['evaluation'][ref][dom]['windows'] if w.get('teacher_diagnostics') or 'kl_mean' in w]
        cl = S.clusters_for(wins[dom], dom)
        for p, arr in arrays.items():
            e = dict(token_accuracy=float(arr['correct'].mean()), entropy=float(arr['entropy'].mean()), confidence=float(arr['confidence'].mean()),
                     ece_10bin=S.ece(arr['confidence'], arr['correct']), tokens=int(arr['nll'].size))
            if 'kl' in arr:
                e.update(kl_to_bf16=float(arr['kl'].mean()), top1_agreement_with_bf16=float(arr['top1_agree'].mean()), teacher_windows=len(teacher_windows),
                         teacher_tokens=int(arr['kl'].size))
            if p != ref:
                for metric in ('correct', 'entropy', 'confidence'):
                    e[f'delta_{metric}_vs_{ref}'] = S.paired_token_metric(arr[metric].astype(np.float64), arrays[ref][metric].astype(np.float64), toks, cl, B=B)
                if 'kl' in arr and 'kl' in arrays[ref] and arr['kl'].size == arrays[ref]['kl'].size:
                    if arr['kl'].size == arr['nll'].size:
                        tt, cc = toks, cl
                    else:
                        tt = toks[:len(teacher_windows)]
                        cc = cl[:len(teacher_windows)]
                        cc = np.unique(cc, return_inverse=True)[1]
                    e['delta_kl_vs_ref'] = S.paired_token_metric(arr['kl'].astype(np.float64), arrays[ref]['kl'].astype(np.float64), tt, cc, B=B)
                    e['delta_top1_vs_ref'] = S.paired_token_metric(arr['top1_agree'].astype(np.float64), arrays[ref]['top1_agree'].astype(np.float64), tt, cc, B=B)
            res[dom][p] = e
    return res


def primary(models, job_of, deliverable, matrix):
    out = dict(matrix_id=matrix, freeze_sha256=FSHA, margin_dlogppl=MARGIN, models={})
    endpoints = []
    for m in models:
        try:
            run, rep, wins = load_ppl(job_of(m))
        except FileNotFoundError as exc:
            out['models'][m] = dict(status='missing', error=str(exc))
            continue
        pols = [p['name'] for p in rep['plan']]
        mm = dict(run=run.name, windows=rep['windows'], teacher=rep['teacher'], reinstall_check=rep['reinstall_check'],
                  ppl={p: {d: rep['evaluation'][p][d]['ppl'] for d in rep['domains']} for p in pols},
                  selected_tiles={i['name']: i.get('selected_tiles') for i in rep['installs']},
                  map_sha256={i['name']: i.get('map_sha256') for i in rep['installs']}, contrasts={})
        for a, b in (('n16_k3', 'four_over_six'), ('n8_k3', 'four_over_six'), ('n16_k3', 'n8_k3'), ('four_over_six', 'bf16'), ('n16_k3', 'bf16'),
                     ('nvfp4', 'four_over_six'), ('all_e0m3', 'four_over_six'), ('n8_k3', 'bf16')):
            mm['contrasts'][f'{a}-{b}'] = compare(rep, wins, a, b, margin=(MARGIN if (a, b) == ('n16_k3', 'four_over_six') else None))
        mm['diagnostics'] = diagnostics(run, rep, wins, pols)
        n16 = sum(mm['contrasts']['n16_k3-four_over_six'][d]['estimate'] for d in rep['domains'])
        n8 = sum(mm['contrasts']['n8_k3-four_over_six'][d]['estimate'] for d in rep['domains'])
        mm['retained_fraction_n16_over_n8'] = (n16 / n8) if n8 < 0 else None
        out['models'][m] = mm
        for d in rep['domains']:
            c = mm['contrasts']['n16_k3-four_over_six'][d]
            endpoints.append(dict(model=m, domain=d, estimate=c['estimate'], ci95=c['ci95'], p_noninferiority=c['noninferiority']['p_one_sided'],
                                  upper_ci_below_margin=c['noninferiority']['upper_ci_below_margin']))
    if endpoints:
        adj, rej = S.holm([e['p_noninferiority'] for e in endpoints])
        for e, a, r in zip(endpoints, adj, rej):
            e.update(holm_adjusted_p=a, noninferior_holm=r)
    out['endpoints'] = endpoints
    runtime.atomic_json(runtime.out_dir('analysis_ppl') / deliverable, out)
    return out


def secondary(models, job_of, deliverable, matrix, pairs_fn, B=2000):
    out = dict(matrix_id=matrix, freeze_sha256=FSHA, models={})
    for m in models:
        try:
            run, rep, wins = load_ppl(job_of(m))
        except FileNotFoundError as exc:
            out['models'][m] = dict(status='missing', error=str(exc))
            continue
        pols = [p['name'] for p in rep['plan']]
        mm = dict(run=run.name, ppl={p: {d: rep['evaluation'][p][d]['ppl'] for d in rep['domains']} for p in pols},
                  selected_tiles={i['name']: i.get('selected_tiles') for i in rep['installs']},
                  selected_weights={i['name']: (i.get('selected_tiles') or 0) * (i.get('type_block') or [0, 0])[0] * (i.get('type_block') or [0, 0])[1] for i in rep['installs']},
                  map_sha256={i['name']: i.get('map_sha256') for i in rep['installs']}, installed_weight_sha256={i['name']: i['installed_weight_sha256'] for i in rep['installs']},
                  contrasts={})
        for a, b in pairs_fn(pols):
            mm['contrasts'][f'{a}-{b}'] = compare(rep, wins, a, b, B=B, block=False)
        out['models'][m] = mm
    runtime.atomic_json(runtime.out_dir('analysis_ppl') / deliverable, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--what', required=True, choices=('primary', 'secondary', 'all'))
    args = ap.parse_args()
    dev = lambda m: f'V31_ppl_primary_{m}'
    conf = lambda m: f'V62_ppl_primary_{m}'
    if args.what in ('primary', 'all'):
        primary(('llama8b', 'qwen4b', 'qwen27b'), dev, 'LEGACY_PANEL_PPL.json', 'V31')
        primary(('mistral7b', 'phi4', 'olmo2_13b'), conf, 'CONFIRMATORY_PPL.json', 'V62')
    if args.what in ('secondary', 'all'):
        vs_base = lambda pols: [(p, 'four_over_six') for p in pols if p != 'four_over_six']
        secondary(('llama8b', 'qwen4b', 'mistral7b'), lambda m: f'V40_ppl_ksweep_{m}', 'K_SENSITIVITY_PPL.json', 'V40', vs_base)
        secondary(('llama8b', 'qwen4b', 'mistral7b'), lambda m: f'V42_ppl_controls_{m}', 'SELECTOR_CONTROLS_PPL.json', 'V42',
                  lambda pols: vs_base(pols))
        secondary(('llama8b', 'qwen4b', 'mistral7b'), lambda m: f'V80_ppl_baselines_{m}', 'ADDITIONAL_BASELINES_PPL.json', 'V80',
                  lambda pols: vs_base(pols) + [(p, 'n16_k3') for p in pols if p not in ('n16_k3', 'four_over_six')])
        secondary(('llama8b', 'qwen4b', 'mistral7b'), lambda m: f'V50_ppl_draws_{m}', 'CALIBRATION_SEED_STABILITY_PPL.json', 'V50', vs_base)
        secondary(('qwen4b', 'mistral7b'), lambda m: f'V51_ppl_sizedomain_{m}', 'CALIBRATION_SIZE_DOMAIN_PPL.json', 'V51', vs_base)
        secondary(('qwen4b', 'mistral7b'), lambda m: f'V52_ppl_crossdomain_{m}', 'CALIBRATION_CROSS_DOMAIN_PPL.json', 'V52',
                  lambda pols: vs_base(pols) + [(f'{r}_k3_math64', f'{r}_k3_code64') for r in ('n8', 'n16')] +
                  [(f'{r}_k3_{s}', f'{r}_k3') for r in ('n8', 'n16') for s in ('math64', 'code64', 'heldout')])
        secondary(('llama8b', 'qwen4b', 'mistral7b'), lambda m: f'V71_ppl_long_{m}', 'LONG_CONTEXT.json', 'V71',
                  lambda pols: [(p, 'four_over_six') for p in pols if p != 'four_over_six'] + [('n16_k3', 'n8_k3')])
    job = runtime.run_dir / 'job_result.json'
    src = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    runtime.atomic_json(job, dict(protocol_id='statistics', protocol_freeze_sha256=FSHA,
        source=dict(model_id=None, model_revision=None, tokenizer_revision=None, model_class=None, module_manifest_sha256=None, source_manifest_sha256=src),
        environment=runtime.environment(), data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=None, token_hashes={}, overlap_audit=None),
        policies=[], results=dict(raw_outputs=[str(p) for p in sorted(runtime.out_dir('analysis_ppl').iterdir())], summary={}, uncertainty={},
                                  attempted_endpoints=[args.what], missing_endpoints=runtime.collect_missing(runtime.out_dir('analysis_ppl'))), logs=[], failures=[]))


if __name__ == '__main__':
    main()
