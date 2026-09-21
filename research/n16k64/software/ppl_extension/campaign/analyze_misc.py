"""V43 first-order fidelity, V81 cross-GPU aligned portability, V82 determinism, V70 Qwen smoothing assembly (CPU)."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy import stats as ss

from campaign import mapio as MIO
from campaign import runtime
from campaign import tiles as T
from campaign.policies import latest_complete_run

CR = Path(os.environ['CAMPAIGN_ROOT'])
FREEZE = json.loads((CR / 'freeze' / 'PROTOCOL_FREEZE.json').read_text())
FSHA = (CR / 'freeze' / 'PROTOCOL_FREEZE.sha256').read_text().split()[0]


def maybe(job):
    try:
        return latest_complete_run(CR, job)
    except FileNotFoundError:
        return None


def attempts(job):
    return [dict(run=d.name, status=json.loads((d / 'launch_record.json').read_text()).get('status') if (d / 'launch_record.json').exists() else None)
            for d in sorted((CR / 'runs').glob(f'{job}_attempt*'))]


def paired_aggregate(rs, obj):
    """Mean over sequences of the average per-sequence delta across these tiles, with the SE taken
    over sequences. Individual tile effects are far below the measurement noise (see V43 noise
    control), so the aggregate over a stratum is the only single-tile quantity with any resolution."""
    key = 'actual_%s_per_seq' % obj
    rows = [r[key] for r in rs if r.get(key)]
    if not rows:
        return None
    M = np.asarray(rows, dtype=np.float64)
    per_seq = M.mean(0)
    se = float(per_seq.std(ddof=1) / math.sqrt(per_seq.size)) if per_seq.size > 1 else None
    return dict(tiles=int(M.shape[0]), sequences=int(M.shape[1]), mean=float(per_seq.mean()), se=se,
                t=(float(per_seq.mean() / se) if se else None), mean_pred=float(np.mean([r['pred_' + obj] for r in rs])))


def stratum_contrast(records, obj, a_name='selected', b_name='rejected'):
    """Difference in measured effect between two strata. Read with two caveats. (1) V43 measures each
    stratum as one contiguous block, so a between-stratum difference confounds the tile population
    with the measurement block. (2) There is no null pair here: `rejected` is the upper half of the
    positive-U3 tiles and `random` samples the whole population, so rejected-vs-random is a
    dose-response contrast, not a null comparison. The confound-free ordering test is
    `dose_response`, which works inside a single stratum."""
    x = np.array([r['actual_' + obj] for r in records if r['stratum'] == a_name], dtype=np.float64)
    y = np.array([r['actual_' + obj] for r in records if r['stratum'] == b_name], dtype=np.float64)
    if x.size < 2 or y.size < 2 or (x.var() == 0 and y.var() == 0):
        return None
    t, p = ss.ttest_ind(x, y, equal_var=False)
    return dict(a=a_name, b=b_name, n_a=int(x.size), n_b=int(y.size), mean_a=float(x.mean()), mean_b=float(y.mean()),
                difference=float(x.mean() - y.mean()), welch_t=float(t), welch_p=float(p),
                mean_pred_a=float(np.mean([r['pred_' + obj] for r in records if r['stratum'] == a_name])),
                mean_pred_b=float(np.mean([r['pred_' + obj] for r in records if r['stratum'] == b_name])))


def order_confound(records, obj, stratum='random'):
    """V43 measures the strata consecutively (selected, near_threshold, rejected, random), so stratum
    is aliased with measurement order. Two estimates of what that aliasing could inject: splitting one
    stratum in half along measurement order (its tiles are drawn in random order, so the two halves
    are exchangeable and any difference is drift or chance), and the trend of the measured effect
    across all records. Note the forward pass itself is bitwise deterministic under a fixed batch
    composition (V43_fidelity_noise), so these detect block-level drift, not per-measurement noise."""
    a = np.array([r['actual_' + obj] for r in records if r['stratum'] == stratum], dtype=np.float64)
    out = dict(stratum=stratum, note='between-stratum differences in this arm are confounded with drift')
    if a.size >= 4:
        h = a.size // 2
        first, second = a[:h], a[h:]
        if not (first.var() == 0 and second.var() == 0):
            t, p = ss.ttest_ind(first, second, equal_var=False)
            out.update(first_half_mean=float(first.mean()), second_half_mean=float(second.mean()),
                       half_difference=float(first.mean() - second.mean()), half_welch_p=float(p))
    allv = np.array([r['actual_' + obj] for r in records], dtype=np.float64)
    if allv.size > 2 and allv.var() > 0:
        idx = np.arange(allv.size, dtype=np.float64)
        rho, rho_p = ss.spearmanr(idx, allv)
        sl = ss.linregress(idx, allv)
        out.update(trend_spearman=float(rho), trend_p=float(rho_p), trend_total_over_all_records=float(sl.slope * allv.size))
    return out


def dose_response(records, obj, stratum='random'):
    """Does a larger predicted effect produce a larger measured one, WITHIN one stratum? The `random`
    stratum samples the whole tile population and is measured as a single contiguous block, so this is
    the one ordering test in the V43 singles that is free of any between-block confound. Reported as
    rank correlations against both the election statistic U3 and the predicted mean, plus a split at
    the median U3."""
    rs = [r for r in records if r['stratum'] == stratum]
    if len(rs) < 6:
        return None
    u = np.array([r['U3'] for r in rs], dtype=np.float64)
    p = np.array([r['pred_' + obj] for r in rs], dtype=np.float64)
    a = np.array([r['actual_' + obj] for r in rs], dtype=np.float64)
    if a.var() == 0 or u.var() == 0:
        return None
    ru, rp = ss.spearmanr(u, a), ss.spearmanr(p, a)
    out = dict(stratum=stratum, n=len(rs), spearman_u3_vs_actual=float(ru.correlation), p_u3=float(ru.pvalue),
               spearman_pred_vs_actual=float(rp.correlation), p_pred=float(rp.pvalue))
    hi = u > np.median(u)
    if hi.sum() >= 2 and (~hi).sum() >= 2:
        t, pv = ss.ttest_ind(a[hi], a[~hi], equal_var=False)
        out.update(high_u3_mean=float(a[hi].mean()), low_u3_mean=float(a[~hi].mean()),
                   high_minus_low=float(a[hi].mean() - a[~hi].mean()), welch_p=float(pv))
    return out


def fidelity_metrics(records):
    pred = np.array([[r['pred_ce'], r['pred_kl']] for r in records])
    act = np.array([[r['actual_ce'], r['actual_kl']] for r in records])
    out = {}
    for j, obj in enumerate(('ce', 'kl')):
        p, a = pred[:, j], act[:, j]
        pos_pred, pos_act = p < 0, a < 0
        tp = int((pos_pred & pos_act).sum())
        out[obj] = dict(n=len(p), pearson=float(ss.pearsonr(p, a)[0]) if len(p) > 2 else None, spearman=float(ss.spearmanr(p, a).correlation) if len(p) > 2 else None,
                        sign_precision=(tp / int(pos_pred.sum())) if pos_pred.sum() else None, sign_recall=(tp / int(pos_act.sum())) if pos_act.sum() else None,
                        slope_actual_on_pred=float(np.polyfit(p, a, 1)[0]) if len(p) > 2 and np.ptp(p) > 0 else None,
                        calibration_bins=[dict(pred_mean=float(p[b].mean()), actual_mean=float(a[b].mean()), n=int(b.sum()))
                                          for b in [(p >= lo) & (p <= hi) for lo, hi in zip(*[np.quantile(p, q) for q in (np.linspace(0, 0.8, 5), np.linspace(0.2, 1.0, 5))])] if b.any()])
        # Resolution context. A correlation between predicted and actual is only meaningful if the
        # actual is resolved at all; here the predicted per-tile effect is orders of magnitude below
        # the SE of the measurement that is supposed to detect it.
        se = np.array([r['actual_%s_se' % obj] for r in records], dtype=np.float64)
        se_med = float(np.median(se))
        out[obj].update(pred_abs_median=float(np.median(np.abs(p))), actual_abs_median=float(np.median(np.abs(a))),
                        actual_se_median=se_med,
                        predicted_over_actual_se_median=(float(np.median(np.abs(p)) / se_med) if se_med else None),
                        resolvable_fraction_3se=float((np.abs(a) > 3 * se).mean()),
                        resolvable_fraction_1p96se=float((np.abs(a) > 1.96 * se).mean()),
                        paired_aggregate_by_stratum={s: paired_aggregate([r for r in records if r['stratum'] == s], obj)
                                                     for s in dict.fromkeys(r['stratum'] for r in records)},
                        selected_vs_rejected=stratum_contrast(records, obj),
                        rejected_vs_random=stratum_contrast(records, obj, 'rejected', 'random'),
                        dose_response_within_random=dose_response(records, obj),
                        order_confound=order_confound(records, obj))
    sel = [r for r in records if r['stratum'] == 'selected']
    if sel:
        fp_any = sum(1 for r in sel if r['actual_ce'] >= 0 or r['actual_kl'] >= 0)
        fp_both = sum(1 for r in sel if r['actual_ce'] >= 0 and r['actual_kl'] >= 0)
        sig = sum(1 for r in sel if r['actual_ce'] + 1.96 * r['actual_ce_se'] < 0 and r['actual_kl'] + 1.96 * r['actual_kl_se'] < 0)
        out['selected'] = dict(n=len(sel), false_positive_rate_any_objective=fp_any / len(sel), false_positive_rate_both=fp_both / len(sel),
                               actual_both_significantly_negative=sig / len(sel))
    out['by_stratum'] = {s: dict(n=sum(1 for r in records if r['stratum'] == s), mean_pred_ce=float(np.mean([r['pred_ce'] for r in records if r['stratum'] == s])),
                                 mean_actual_ce=float(np.mean([r['actual_ce'] for r in records if r['stratum'] == s])),
                                 mean_pred_kl=float(np.mean([r['pred_kl'] for r in records if r['stratum'] == s])),
                                 mean_actual_kl=float(np.mean([r['actual_kl'] for r in records if r['stratum'] == s])))
                         for s in dict.fromkeys(r['stratum'] for r in records)}
    out['by_layer_third'] = {str(t): fidelity_small([r for r in records if r['layer_third'] == t]) for t in (0, 1, 2)}
    out['by_module_type'] = {mt: fidelity_small([r for r in records if r['module_type'] == mt]) for mt in dict.fromkeys(r['module_type'] for r in records)}
    return out


def fidelity_small(rs):
    if len(rs) < 3:
        return dict(n=len(rs))
    p = np.array([r['pred_ce'] for r in rs])
    a = np.array([r['actual_ce'] for r in rs])
    return dict(n=len(rs), spearman_ce=float(ss.spearmanr(p, a).correlation), sign_agree_ce=float(((p < 0) == (a < 0)).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--what', default='all')
    args = ap.parse_args()
    out = runtime.out_dir('analysis_misc')
    # ---- V43
    fid = dict(matrix_id='V43', freeze_sha256=FSHA, models={})
    for m in ('llama8b', 'qwen4b', 'mistral7b'):
        run = maybe(f'V43_fidelity_{m}')
        if run is None:
            fid['models'][m] = dict(status='missing', attempts=attempts(f'V43_fidelity_{m}'))
            continue
        rep = json.loads((run / 'fidelity' / 'fidelity_report.json').read_text())
        mm = dict(run=run.name, baseline=rep['baseline'])
        singles = {}
        for res in ('n8', 'n16'):
            r = rep['results'][res]
            mm[res] = dict(singles=fidelity_metrics(r['singles']), thresholds=r['thresholds'], baseline_restored_exact=r['baseline_restored_exact'])
            b = r['batched']
            single_sum = {}
            mm[res]['batched'] = [dict(x, interaction_error_ce=x['actual_ce'] - x['pred_ce_sum'], interaction_error_kl=x['actual_kl'] - x['pred_kl_sum'],
                                       ratio_actual_to_pred_ce=(x['actual_ce'] / x['pred_ce_sum']) if x['pred_ce_sum'] else None) for x in b]
            singles[res] = {(s['module'], s['tile']): s for s in r['singles']}
        # N8 vs N16 fidelity where two N8 children cancel: compare parent actual with the children predicted sum when sampled
        # noise-floor control for this instrument (V43_fidelity_noise): without it the singles
        # cannot be told apart from run-to-run numerical noise.
        nrun = maybe(f'V43_fidelity_noise_{m}')
        if nrun is None:
            mm['noise_control'] = dict(status='missing', attempts=attempts(f'V43_fidelity_noise_{m}'))
        else:
            nrep = json.loads((nrun / 'fidelity_noise' / 'fidelity_noise_report.json').read_text())
            mm['noise_control'] = dict(run=nrun.name, status=nrep.get('status'), config=nrep.get('config'),
                                       measurements=len(nrep.get('measurements', [])), summary=nrep.get('summary'))
        fid['models'][m] = mm
    runtime.atomic_json(out / 'FIRST_ORDER_FIDELITY.json', fid)
    # ---- V82 determinism
    a = maybe('V30_calib_qwen4b_seed0')
    b = maybe('V82_calib_qwen4b_seed0_repeat')
    det = dict(matrix_id='V82', freeze_sha256=FSHA, status='missing', attempts=attempts('V82_calib_qwen4b_seed0_repeat'))
    if a and b:
        ra = json.loads((a / 'calibration' / 'calibration_report.json').read_text())
        rb = json.loads((b / 'calibration' / 'calibration_report.json').read_text())
        la = json.loads((a / 'launch_record.json').read_text())
        lb = json.loads((b / 'launch_record.json').read_text())
        ma = {e['policy']: e for e in ra['maps']}
        mb = {e['policy']: e for e in rb['maps']}
        per = {}
        for p in ('n8_k3', 'n16_k3', 'n8_k2', 'n16_k2', 'n16_k4'):
            ha, xa, _ = MIO.read_map(ma[p]['path'], ma[p]['sha256'])
            hb, xb, _ = MIO.read_map(mb[p]['path'], mb[p]['sha256'])
            inter = sum(int((xa[n] & xb[n]).sum()) for n in xa)
            union = sum(int((xa[n] | xb[n]).sum()) for n in xa)
            per[p] = dict(sha_equal=ma[p]['sha256'] == mb[p]['sha256'], payload_equal=all(torch.equal(xa[n], xb[n]) for n in xa),
                          header_equal_except_source=({k: v for k, v in ha.items() if k != 'source_manifest_sha256'} == {k: v for k, v in hb.items() if k != 'source_manifest_sha256'}),
                          selected=(ma[p]['selected_tiles'], mb[p]['selected_tiles']), jaccard=(inter / union if union else 1.0))
        det = dict(matrix_id='V82', freeze_sha256=FSHA, runs=[a.name, b.name], gpus=[la.get('leased_uuids'), lb.get('leased_uuids')],
                   same_gpu_uuid=la.get('leased_uuids') == lb.get('leased_uuids'), same_source_manifest=la['source_manifest_sha256'] == lb['source_manifest_sha256'],
                   score_stream_sha256_equal=ra['score_stream_sha256_all'] == rb['score_stream_sha256_all'], bf16_fit_nll_equal=ra['bf16_fit_nll'] == rb['bf16_fit_nll'],
                   fit_losses_equal=ra['fit_losses'] == rb['fit_losses'], maps=per,
                   tierA=all(v['payload_equal'] for v in per.values()) and ra['score_stream_sha256_all'] == rb['score_stream_sha256_all'],
                   tierB=all(v['jaccard'] >= 0.99 for v in per.values()),
                   note='map file SHA includes the generating source manifest; payload equality isolates the masks when the source tree changed between runs')
    runtime.atomic_json(out / 'DETERMINISM_REPORT.json', det)
    # ---- V81 cross-GPU aligned
    port = dict(matrix_id='V81', freeze_sha256=FSHA, status='missing',
                attempts=dict(calibration=attempts('V81_calib_qwen4b_seed0_ada'), ppl_ada=attempts('V81_ppl_portability_qwen4b_ada'), ppl_a6000=attempts('V81_ppl_portability_qwen4b_a6000')))
    ca, cb = maybe('V30_calib_qwen4b_seed0'), maybe('V81_calib_qwen4b_seed0_ada')
    pa, pb = maybe('V81_ppl_portability_qwen4b_a6000'), maybe('V81_ppl_portability_qwen4b_ada')
    if ca and cb:
        ra = json.loads((ca / 'calibration' / 'calibration_report.json').read_text())
        rb = json.loads((cb / 'calibration' / 'calibration_report.json').read_text())
        maps = {}
        for p in ('n8_k3', 'n16_k3'):
            ea = {e['policy']: e for e in ra['maps']}[p]
            eb = {e['policy']: e for e in rb['maps']}[p]
            _, xa, _ = MIO.read_map(ea['path'], ea['sha256'])
            _, xb, _ = MIO.read_map(eb['path'], eb['sha256'])
            inter = sum(int((xa[n] & xb[n]).sum()) for n in xa)
            union = sum(int((xa[n] | xb[n]).sum()) for n in xa)
            maps[p] = dict(a6000=ea['selected_tiles'], ada=eb['selected_tiles'], intersection=inter, jaccard=inter / union if union else 1.0,
                           payload_equal=all(torch.equal(xa[n], xb[n]) for n in xa), tierB=(inter / union if union else 1.0) >= 0.9)
        port.update(status='calibration compared', calibration_runs=[ca.name, cb.name], maps=maps,
                    fit_ce_max_abs=max(abs(x['ce'] - y['ce']) for x, y in zip(ra['fit_losses'], rb['fit_losses'])))
    if pa and pb:
        ra = json.loads((pa / 'ppl' / 'ppl_report.json').read_text())
        rb = json.loads((pb / 'ppl' / 'ppl_report.json').read_text())
        ppl = {}
        for pol in ra['evaluation']:
            ppl[pol] = {}
            for d in ra['domains']:
                x = [w['nll_mean'] for w in ra['evaluation'][pol][d]['windows']]
                y = [w['nll_mean'] for w in rb['evaluation'][pol][d]['windows']]
                ppl[pol][d] = dict(a6000=ra['evaluation'][pol][d]['ppl'], ada=rb['evaluation'][pol][d]['ppl'],
                                   rel=rb['evaluation'][pol][d]['ppl'] / ra['evaluation'][pol][d]['ppl'] - 1,
                                   window_nll_max_abs=max(abs(p - q) for p, q in zip(x, y)), exact_equal=x == y,
                                   tierB=abs(rb['evaluation'][pol][d]['ppl'] / ra['evaluation'][pol][d]['ppl'] - 1) <= FREEZE['numerical_tolerances']['cross_gpu']['ppl_rel_w4a4'])
        eff = {}
        for pol in ('n16_k3', 'n16_k3_ada'):
            if pol in ra['evaluation']:
                eff[pol] = {d: dict(a6000=math.log(ra['evaluation'][pol][d]['ppl'] / ra['evaluation']['four_over_six'][d]['ppl']),
                                    ada=math.log(rb['evaluation'][pol][d]['ppl'] / rb['evaluation']['four_over_six'][d]['ppl'])) for d in ra['domains']}
                for d in eff[pol]:
                    e = eff[pol][d]
                    e['sign_equal'] = (e['a6000'] < 0) == (e['ada'] < 0)
                    e['tier'] = e['sign_equal'] and abs(e['a6000'] - e['ada']) <= max(0.25 * abs(e['a6000']), 0.002)
        port.update(status='complete' if (ca and cb) else 'ppl only', ppl_runs=[pa.name, pb.name], ppl=ppl, paired_effects=eff)
    runtime.atomic_json(out / 'CROSS_GPU_ALIGNED.json', port)
    src = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(protocol_id='statistics', protocol_freeze_sha256=FSHA,
        source=dict(model_id=None, model_revision=None, tokenizer_revision=None, model_class=None, module_manifest_sha256=None, source_manifest_sha256=src),
        environment=runtime.environment(), data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=None, token_hashes={}, overlap_audit=None),
        policies=[], results=dict(raw_outputs=[str(p) for p in sorted(out.iterdir())], summary={}, uncertainty={}, attempted_endpoints=['V43', 'V81', 'V82'],
                                  missing_endpoints=runtime.collect_missing(out)), logs=[], failures=[]))


if __name__ == '__main__':
    main()
