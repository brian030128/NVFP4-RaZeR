"""V90 assembly (CPU): matrix-named deliverables that combine the per-analysis outputs, map-level overlap statistics,
cross-run paired contrasts, the frozen V70 smoothing rule, compute disclosure and EXPERIMENT_MATRIX coverage.

Nothing here re-runs a model. Every map is re-read from disk and digest-verified; every cross-run paired contrast
first checks that both runs evaluated identical token windows and reports the per-window NLL agreement of the shared
FourOverSix anchor policy, so exact pairing across runs is demonstrated rather than assumed."""
import csv
import json
import math
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats as ss

from campaign import mapio as MIO
from campaign import runtime
from campaign import stats as S
from campaign.policies import latest_complete_run

CR = Path(os.environ['CAMPAIGN_ROOT'])
FREEZE = json.loads((CR / 'freeze' / 'PROTOCOL_FREEZE.json').read_text())
FSHA = (CR / 'freeze' / 'PROTOCOL_FREEZE.sha256').read_text().split()[0]
MATRIX = CR / 'handoff' / 'agent_handoff' / 'EXPERIMENT_MATRIX.csv'
DEV = ('llama8b', 'qwen4b', 'qwen27b')
CONF = ('mistral7b', 'phi4', 'olmo2_13b')
REP3 = ('llama8b', 'qwen4b', 'mistral7b')
CALIB = lambda m, draw='seed0': f'{"V30" if m in DEV else "V61"}_calib_{m}_{draw}'
PRIMARY_PPL = lambda m: f'{"V31" if m in DEV else "V62"}_ppl_primary_{m}'


def maybe(job):
    try:
        return latest_complete_run(CR, job)
    except FileNotFoundError:
        return None


def attempts(job):
    out = []
    for d in sorted((CR / 'runs').glob(f'{job}_attempt*'), key=lambda p: int(p.name.rsplit('attempt', 1)[1])):
        lr = d / 'launch_record.json'
        out.append(dict(run=d.name, status=json.loads(lr.read_text()).get('status') if lr.exists() else None))
    return out


def analysis(job, rel):
    run = maybe(job)
    p = run / rel if run else None
    if p is None or not p.exists():
        return None, None
    return p, json.loads(p.read_text())


def cite(p):
    if not p:
        return None
    try:
        rel = str(Path(p).relative_to(CR))
    except ValueError:      # a path outside the campaign root (only possible with an overridden out_dir)
        rel = str(p)
    return dict(path=rel, sha256=runtime.sha256_file(p))


# ------------------------------------------------------------------------------------------------ maps
_MAPS = {}


def manifest(job, sub='calibration'):
    run = maybe(job)
    if run is None or not (run / sub / 'map_manifest.json').exists():
        return None, {}
    return run, {e['policy']: e for e in json.loads((run / sub / 'map_manifest.json').read_text())}


def masks(entry):
    if entry['sha256'] not in _MAPS:
        header, m, digest = MIO.read_map(entry['path'], expected_sha256=entry['sha256'])
        _MAPS[entry['sha256']] = (header, m)
    return _MAPS[entry['sha256']]


def overlap(ea, eb):
    _, xa = masks(ea)
    _, xb = masks(eb)
    if set(xa) != set(xb):
        raise ValueError('maps cover different modules')
    inter = sum(int((xa[n] & xb[n]).sum()) for n in xa)
    na = sum(int(xa[n].sum()) for n in xa)
    nb = sum(int(xb[n].sum()) for n in xb)
    union = na + nb - inter
    return dict(a=na, b=nb, intersection=inter, jaccard=(inter / union) if union else 1.0,
                fraction_of_a_in_b=(inter / na) if na else None, fraction_of_b_in_a=(inter / nb) if nb else None)


def nested(e_small, e_large):
    """True iff every tile of e_small is selected in e_large."""
    o = overlap(e_small, e_large)
    return o['intersection'] == o['a']


def module_count_spearman(ea, eb):
    ka, kb = ea['per_module_selected'], eb['per_module_selected']
    names = sorted(set(ka) & set(kb))
    a = [ka[n] for n in names]
    b = [kb[n] for n in names]
    return float(ss.spearmanr(a, b).correlation) if len(names) > 2 and np.ptp(a) > 0 and np.ptp(b) > 0 else None


# ------------------------------------------------------------------------------------------------ PPL across runs
def ppl_run(job):
    run = maybe(job)
    if run is None:
        return None
    rep = json.loads((run / 'ppl' / 'ppl_report.json').read_text())
    wins = {d: json.loads((run / 'ppl' / f'windows_{d}.json').read_text()) for d in rep['domains']}
    return run, rep, wins


def cross_run_contrast(ja, pa, jb, pb, anchor='four_over_six', B=2000):
    A, Bn = ppl_run(ja), ppl_run(jb)
    if A is None or Bn is None:
        return dict(status='missing', a=attempts(ja)[-1:] if A is None else A[0].name, b=attempts(jb)[-1:] if Bn is None else Bn[0].name)
    (ra, repa, wa), (rb, repb, wb) = A, Bn
    out = dict(run_a=ra.name, policy_a=pa, run_b=rb.name, policy_b=pb, domains={})
    for d in repa['domains']:
        if d not in repb['domains']:
            continue
        if wa[d]['token_sha256'] != wb[d]['token_sha256']:
            out['domains'][d] = dict(status='windows differ')
            continue
        na, ta = S.window_table(repa, pa, d)
        nb, tb = S.window_table(repb, pb, d)
        r = S.paired_dlogppl(na, nb, ta, S.clusters_for(wa[d], d), B=B)
        if anchor in repa['evaluation'] and anchor in repb['evaluation']:
            xa, _ = S.window_table(repa, anchor, d)
            xb, _ = S.window_table(repb, anchor, d)
            r['anchor_check'] = dict(policy=anchor, window_nll_sum_max_abs_diff=float(np.abs(xa - xb).max()), bitwise_equal=bool(np.array_equal(xa, xb)))
        out['domains'][d] = r
    return out


# ------------------------------------------------------------------------------------------------ deliverables
def smoke_report():
    rows = {}
    for job in sorted({d.name.rsplit('_attempt', 1)[0] for d in (CR / 'runs').glob('V14_*_attempt*')}):
        run = maybe(job)
        files = sorted(str(p.relative_to(CR)) for p in (run.rglob('*.json') if run else []) if p.parent != run and 'preflight' not in p.parts)
        rows[job] = dict(attempts=attempts(job), complete_run=run.name if run else None, outputs=files[:50])
    p, e2e = analysis('V14_cpu_mini_e2e', 'tests/SMOKE_REPORT_CPU_E2E.json')
    return dict(matrix_id='V14', freeze_sha256=FSHA, cpu_end_to_end=cite(p), cpu_end_to_end_summary=e2e, gpu_smoke_and_diagnostics=rows,
                note='pre-freeze compatibility/memory smoke (no quantized quality inspected) plus the causality diagnostic; see PROTOCOL_FREEZE.json#panels.compatibility_smoke')


def map_manifest(models, matrix):
    out = dict(matrix_id=matrix, freeze_sha256=FSHA, models={})
    for m in models:
        jobs = [CALIB(m)] + ([CALIB(m, f'draw{k}') for k in (1, 2, 3, 4)] if m in REP3 else []) + ([CALIB(m, 'heldout')] if m in ('qwen4b', 'mistral7b') else [])
        jobs += (['V82_calib_qwen4b_seed0_repeat', 'V81_calib_qwen4b_seed0_ada'] if m == 'qwen4b' else [])
        mm = {}
        for job in jobs:
            run, man = manifest(job)
            if run is None:
                mm[job] = dict(status='missing', attempts=attempts(job))
                continue
            rep = json.loads((run / 'calibration' / 'calibration_report.json').read_text())
            rr = json.loads((run / 'run_record.json').read_text())
            maps = {}
            for pol, e in man.items():
                header, _ = masks(e)
                maps[pol] = dict(path=str(Path(e['path']).relative_to(CR)), sha256=e['sha256'], selected_tiles=e['selected_tiles'], total_tiles=e.get('total_tiles'),
                                 type_block=e.get('type_block'), reloaded_and_verified=True, header_protocol=header.get('protocol_id'))
            mm[job] = dict(run=run.name, draw=rep['draw'], sequences=rep['sequences'], calibration_manifest_sha256=rep['calibration_manifest_sha256'],
                           module_manifest_sha256=rep['module_manifest_sha256'], tokenizer_manifest_sha256=rep['tokenizer_manifest_sha256'],
                           score_stream_sha256_all=rep['score_stream_sha256_all'], n8_total=rep['n8_total'], n16_total=rep['n16_total'],
                           moment_files=rep.get('moment_files'), score_seconds=rep.get('score_seconds'), gpu_hours=rr['artifacts']['compute_usage']['gpu_hours'],
                           peak_gpu_memory_bytes=rr['artifacts']['compute_usage']['peak_gpu_memory_bytes'], gpus=rr['gpu_allocation']['gpu_names'], maps=maps)
        derive = f'V51_derive_subset_maps_{m}'
        run, man = manifest(derive, 'derived_maps')
        if run is not None:
            mm[derive] = dict(run=run.name, maps={pol: dict(path=str(Path(e['path']).relative_to(CR)), sha256=e['sha256'], selected_tiles=e['selected_tiles'])
                                                   for pol, e in man.items() if masks(e)})
        out['models'][m] = mm
    return out


def k_sensitivity():
    pp, ppl = analysis('V90_analyze_ppl', 'analysis_ppl/K_SENSITIVITY_PPL.json')
    pa, acc = analysis('V90_analyze_accuracy', 'analysis_accuracy/K_SENSITIVITY_ACCURACY.json')
    maps = {}
    for m in REP3:
        _, man = manifest(CALIB(m))
        if not man:
            maps[m] = 'missing'
            continue
        maps[m] = dict(counts={p: man[p]['selected_tiles'] for p in man if p.startswith(('n8_k', 'n16_k')) and len(p) <= 6},
                       nested_n16=all(nested(man[f'n16_k{k + 1}'], man[f'n16_k{k}']) for k in (2, 3, 4, 5)),
                       nested_n8=all(nested(man[f'n8_k{k + 1}'], man[f'n8_k{k}']) for k in (2, 3, 4, 5)),
                       jaccard_to_k3={f'{r}_k{k}': overlap(man[f'{r}_k{k}'], man[f'{r}_k3'])['jaccard'] for r in ('n8', 'n16') for k in (2, 4, 5, 6)})
    return dict(matrix_id='V40', freeze_sha256=FSHA, rule=FREEZE['success_criteria']['k_rule'], ppl=cite(pp), accuracy=cite(pa), maps=maps,
                ppl_models=(ppl or {}).get('models'), accuracy_models=(acc or {}).get('models'))


def objective_ablation():
    pp, ppl = analysis('V90_analyze_ppl', 'analysis_ppl/SELECTOR_CONTROLS_PPL.json')
    pa, acc = analysis('V90_analyze_accuracy', 'analysis_accuracy/SELECTOR_CONTROLS_ACCURACY.json')
    rules = ('n16_k3', 'n16_k3_ce_only', 'n16_k3_kl_only', 'n16_k3_mean_only')
    out = dict(matrix_id='V41', freeze_sha256=FSHA, definitions=dict(n16_k3='intersection: max(mean_CE+3SE, mean_KL+3SE) < 0 (primary)', n16_k3_ce_only='mean_CE+3SE < 0',
               n16_k3_kl_only='mean_KL+3SE < 0', n16_k3_mean_only='max(mean_CE, mean_KL) < 0 (no SE term)'), ppl=cite(pp), accuracy=cite(pa), models={})
    for m in REP3:
        _, man = manifest(CALIB(m))
        if not man:
            out['models'][m] = dict(status='missing')
            continue
        mm = dict(counts={r: man[r]['selected_tiles'] for r in rules}, overlaps={f'{a}|{b}': overlap(man[a], man[b]) for a, b in combinations(rules, 2)},
                  intersection_subset_of_ce_only=nested(man['n16_k3'], man['n16_k3_ce_only']), intersection_subset_of_kl_only=nested(man['n16_k3'], man['n16_k3_kl_only']))
        mm['ppl_vs_four_over_six'] = {r: ((ppl or {}).get('models', {}).get(m, {}).get('contrasts', {}).get(f'{r}-four_over_six')) for r in rules[1:]}
        mm['ppl_vs_primary_n16_k3'] = {r: cross_run_contrast(f'V42_ppl_controls_{m}', r, PRIMARY_PPL(m), 'n16_k3') for r in rules[1:]}
        mm['representative_accuracy_vs_four_over_six'] = {r: (acc or {}).get('models', {}).get(m, {}).get(r) for r in rules}
        out['models'][m] = mm
    return out


def selector_controls():
    pp, ppl = analysis('V90_analyze_ppl', 'analysis_ppl/SELECTOR_CONTROLS_PPL.json')
    pa, acc = analysis('V90_analyze_accuracy', 'analysis_accuracy/SELECTOR_CONTROLS_ACCURACY.json')
    controls = [f'n16_random_s{s}' for s in range(5)] + ['n16_weight_mse', 'n16_magnitude', 'n16_change_norm', 'n16_density_matched_n8k3']
    out = dict(matrix_id='V42', freeze_sha256=FSHA, ppl=cite(pp), accuracy=cite(pa), models={},
               note='random/weight-MSE/magnitude/change-norm controls are per-module count matched to N16 k3; density_matched_n8k3 matches the N8 k3 selected-weight count per module')
    for m in REP3:
        _, man = manifest(CALIB(m))
        if not man:
            out['models'][m] = dict(status='missing')
            continue
        mm = dict(counts={c: man[c]['selected_tiles'] for c in controls + ['n16_k3', 'n8_k3']},
                  per_module_count_match={c: man[c]['per_module_selected'] == man['n16_k3']['per_module_selected'] for c in controls if c != 'n16_density_matched_n8k3'},
                  overlap_with_n16_k3={c: overlap(man[c], man['n16_k3']) for c in controls})
        mm['ppl_vs_four_over_six'] = {c: (ppl or {}).get('models', {}).get(m, {}).get('contrasts', {}).get(f'{c}-four_over_six') for c in controls}
        mm['ppl_n16_k3_minus_control'] = {c: cross_run_contrast(PRIMARY_PPL(m), 'n16_k3', f'V42_ppl_controls_{m}', c) for c in controls}
        rnd = [mm['ppl_n16_k3_minus_control'][f'n16_random_s{s}'] for s in range(5)]
        mm['random_summary'] = {d: dict(n16_k3_better_than_all_random=all(r.get('domains', {}).get(d, {}).get('ci95', [0, 0])[1] < 0 for r in rnd),
                                        estimates=[r.get('domains', {}).get(d, {}).get('estimate') for r in rnd]) for d in ('wiki', 'c4')}
        mm['representative_accuracy_vs_four_over_six'] = {c: (acc or {}).get('models', {}).get(m, {}).get(c) for c in controls + ['n16_k3']}
        out['models'][m] = mm
    return out


def seed_stability():
    pp, ppl = analysis('V90_analyze_ppl', 'analysis_ppl/CALIBRATION_SEED_STABILITY_PPL.json')
    pa, acc = analysis('V90_analyze_accuracy', 'analysis_accuracy/CALIBRATION_SEED_STABILITY_ACCURACY.json')
    out = dict(matrix_id='V50', freeze_sha256=FSHA, ppl=cite(pp), accuracy=cite(pa), models={})
    for m in REP3:
        draws = {'seed0': manifest(CALIB(m))[1]}
        draws.update({f'draw{k}': manifest(CALIB(m, f'draw{k}'))[1] for k in (1, 2, 3, 4)})
        have = {k: v for k, v in draws.items() if v}
        mm = dict(available_draws=sorted(have), missing_draws=sorted(set(draws) - set(have)), maps={})
        for r in ('n8_k3', 'n16_k3'):
            pairs = {f'{a}|{b}': overlap(have[a][r], have[b][r]) for a, b in combinations(sorted(have), 2)}
            counts = {k: have[k][r]['selected_tiles'] for k in sorted(have)}
            union = None
            for k in have:
                _, x = masks(have[k][r])
                union = {n: x[n].clone() for n in x} if union is None else {n: union[n] | x[n] for n in x}
            inter_all = None
            for k in have:
                _, x = masks(have[k][r])
                inter_all = {n: x[n].clone() for n in x} if inter_all is None else {n: inter_all[n] & x[n] for n in x}
            mm['maps'][r] = dict(counts=counts, count_cv=(float(np.std(list(counts.values())) / np.mean(list(counts.values()))) if len(counts) > 1 else None),
                                 pairwise=pairs, mean_pairwise_jaccard=(float(np.mean([p['jaccard'] for p in pairs.values()])) if pairs else None),
                                 selected_in_every_draw=(sum(int(v.sum()) for v in inter_all.values()) if inter_all else None),
                                 selected_in_any_draw=(sum(int(v.sum()) for v in union.values()) if union else None),
                                 module_count_spearman={f'{a}|{b}': module_count_spearman(have[a][r], have[b][r]) for a, b in combinations(sorted(have), 2)})
        mp = (ppl or {}).get('models', {}).get(m, {})
        mm['ppl_vs_four_over_six'] = mp.get('contrasts')
        for r in ('n8', 'n16'):
            prim = cross_run_contrast(PRIMARY_PPL(m), f'{r}_k3', PRIMARY_PPL(m), 'four_over_six')
            ests = {d: [prim.get('domains', {}).get(d, {}).get('estimate')] + [(mp.get('contrasts', {}).get(f'{r}_k3_draw{k}-four_over_six', {}).get(d, {}) or {}).get('estimate')
                                                                            for k in (1, 2, 3, 4)] for d in ('wiki', 'c4')}
            mm[f'{r}_k3_effect_across_draws'] = {d: dict(seed0_and_draws=v, all_negative=all(x is not None and x < 0 for x in v),
                                                         between_draw_sd=(float(np.std([x for x in v if x is not None], ddof=1)) if sum(x is not None for x in v) > 1 else None))
                                                 for d, v in ests.items()}
        mm['representative_accuracy_vs_four_over_six'] = (acc or {}).get('models', {}).get(m)
        out['models'][m] = mm
    return out


def size_and_domain():
    pp, ppl = analysis('V90_analyze_ppl', 'analysis_ppl/CALIBRATION_SIZE_DOMAIN_PPL.json')
    px, xdom = analysis('V90_analyze_ppl', 'analysis_ppl/CALIBRATION_CROSS_DOMAIN_PPL.json')
    pa, acc = analysis('V90_analyze_accuracy', 'analysis_accuracy/CALIBRATION_SIZE_DOMAIN_ACCURACY.json')
    size = dict(matrix_id='V51', freeze_sha256=FSHA, ppl=cite(pp), accuracy=cite(pa), models={},
                settings=dict(mc32='16 math + 16 code', mc64='32 math + 32 code', seed0='64 math + 64 code (primary)'))
    domain = dict(matrix_id='V52', freeze_sha256=FSHA, ppl_wiki_c4=cite(pp), ppl_in_domain_heldout=cite(px), accuracy=cite(pa), models={},
                  settings=dict(math64='64 OpenWebMath', code64='64 CodeParrot', seed0='balanced 64+64', heldout='32 arXiv + 32 GovReport (keyed draw heldout)'),
                  amendment='provenance/PROTOCOL_AMENDMENTS.jsonl seq 5 (in-domain held-out PPL)')
    for m in ('qwen4b', 'mistral7b'):
        seed_run, seed_man = manifest(CALIB(m))
        drun, dman = manifest(f'V51_derive_subset_maps_{m}', 'derived_maps')
        hrun, hman = manifest(CALIB(m, 'heldout'))
        crep = json.loads((seed_run / 'calibration' / 'calibration_report.json').read_text()) if seed_run else None
        per_seq_seconds = (crep['score_seconds'] / crep['sequences']) if crep else None
        s, d = {}, {}
        for r in ('n8', 'n16'):
            if seed_man and dman:
                s[r] = {sub: dict(sequences=seqs, selected_tiles=(dman[f'{r}_k3_{sub}']['selected_tiles'] if sub != 'seed0' else seed_man[f'{r}_k3']['selected_tiles']),
                                  overlap_with_full=(overlap(dman[f'{r}_k3_{sub}'], seed_man[f'{r}_k3']) if sub != 'seed0' else None),
                                  estimated_scoring_seconds=(per_seq_seconds * seqs if per_seq_seconds else None))
                        for sub, seqs in (('mc32', 32), ('mc64', 64), ('seed0', 128))}
                d[r] = dict(counts={sub: dman[f'{r}_k3_{sub}']['selected_tiles'] for sub in ('math64', 'code64')},
                            math_vs_code=overlap(dman[f'{r}_k3_math64'], dman[f'{r}_k3_code64']),
                            math_vs_balanced=overlap(dman[f'{r}_k3_math64'], seed_man[f'{r}_k3']), code_vs_balanced=overlap(dman[f'{r}_k3_code64'], seed_man[f'{r}_k3']))
                if hman:
                    d[r]['heldout_vs_balanced'] = overlap(hman[f'{r}_k3'], seed_man[f'{r}_k3'])
                    d[r]['counts']['heldout'] = hman[f'{r}_k3']['selected_tiles']
        mp = (ppl or {}).get('models', {}).get(m, {})
        size['models'][m] = dict(maps=s, ppl_vs_four_over_six={k: v for k, v in (mp.get('contrasts') or {}).items() if 'mc32' in k or 'mc64' in k},
                                 ppl_full_vs_subset={f'{r}_k3_{sub}': cross_run_contrast(PRIMARY_PPL(m), f'{r}_k3', f'V51_ppl_sizedomain_{m}', f'{r}_k3_{sub}')
                                                     for r in ('n8', 'n16') for sub in ('mc32', 'mc64')},
                                 representative_accuracy={k: v for k, v in ((acc or {}).get('models', {}).get(m) or {}).items() if 'mc32' in k or 'mc64' in k or k == 'n16_k3'})
        xm = (xdom or {}).get('models', {}).get(m, {})
        domain['models'][m] = dict(maps=d, ppl_wiki_c4_vs_four_over_six={k: v for k, v in (mp.get('contrasts') or {}).items() if any(x in k for x in ('math64', 'code64', 'heldout'))},
                                   ppl_in_domain=dict(run=xm.get('run'), ppl=xm.get('ppl'), contrasts=xm.get('contrasts')),
                                   representative_accuracy={k: v for k, v in ((acc or {}).get('models', {}).get(m) or {}).items() if any(x in k for x in ('math64', 'code64', 'heldout')) or k == 'n16_k3'})
    return size, domain


def interval(c):
    return (c or {}).get('ci95') if isinstance(c, dict) else None


def qwen_smoothing():
    """Frozen PROTOCOL_FREEZE.json#success_criteria.qwen_smoothing_rule applied mechanically."""
    rule = FREEZE['success_criteria']['qwen_smoothing_rule']
    pl, lp = analysis('V90_analyze_ppl', 'analysis_ppl/LEGACY_PANEL_PPL.json')
    pa, la = analysis('V90_analyze_accuracy', 'analysis_accuracy/LEGACY_PANEL_ACCURACY.json')
    pg, lg = analysis('V90_analyze_accuracy', 'analysis_accuracy/LEGACY_GENERATION.json')
    out = dict(matrix_id='V70', freeze_sha256=FSHA, rule=rule, sources=dict(ppl=cite(pl), accuracy=cite(pa), generation=cite(pg)), contrasts={})
    q = (lp or {}).get('models', {}).get('qwen4b', {})
    for pol in ('n8_k3', 'n16_k3'):
        key = f'{pol}-four_over_six'
        acc = ((la or {}).get('models', {}).get('qwen4b', {}) or {}).get(key, {})
        gen = ((lg or {}).get('models', {}).get('qwen4b', {}) or {}).get(key, {})
        macro_ci = (acc.get('macro') or {}).get('ci95')
        gsm_ci = (gen.get('gsm8k_flexible') or {}).get('ci95')
        per = {}
        for dom in ('wiki', 'c4'):
            c = (q.get('contrasts', {}).get(key, {}) or {}).get(dom)
            dg = (q.get('diagnostics', {}).get(dom, {}) or {}).get(pol, {})
            base = (q.get('diagnostics', {}).get(dom, {}) or {}).get('four_over_six', {})
            bf = (q.get('diagnostics', {}).get(dom, {}) or {}).get('bf16', {})
            if c is None or macro_ci is None or gsm_ci is None or not dg:
                per[dom] = dict(classification='undetermined (missing inputs)', nll=bool(c), macro=bool(macro_ci), gsm8k=bool(gsm_ci), diagnostics=bool(dg))
                continue
            nll_neg = c['ci95'][1] < 0
            top1_ci = (dg.get('delta_top1_vs_ref') or {}).get('ci95')
            ent_ci = (dg.get('delta_entropy_vs_four_over_six') or {}).get('ci95')
            capability = nll_neg and (macro_ci[0] > 0 or gsm_ci[0] > 0) and (top1_ci is not None and top1_ci[1] >= 0)
            smoothing = nll_neg and macro_ci[0] <= 0 and gsm_ci[0] <= 0 and (ent_ci is not None and ent_ci[0] > 0)
            label = 'capability_gain' if capability else 'confidence_smoothing' if smoothing else 'mixed_or_inconclusive'
            per[dom] = dict(classification=label, dlogppl=dict(estimate=c['estimate'], ci95=c['ci95']), macro_accuracy_diff_ci95=macro_ci, gsm8k_flexible_diff_ci95=gsm_ci,
                            top1_agreement_delta=dg.get('delta_top1_vs_ref'), entropy_delta=dg.get('delta_entropy_vs_four_over_six'), kl_to_bf16_delta=dg.get('delta_kl_vs_ref'),
                            token_accuracy_delta=dg.get('delta_correct_vs_four_over_six'), confidence_delta=dg.get('delta_confidence_vs_four_over_six'),
                            ece=dict(bf16=bf.get('ece_10bin'), four_over_six=base.get('ece_10bin'), policy=dg.get('ece_10bin')),
                            entropy=dict(bf16=bf.get('entropy'), four_over_six=base.get('entropy'), policy=dg.get('entropy')),
                            ppl=dict(bf16=q['ppl']['bf16'][dom], four_over_six=q['ppl']['four_over_six'][dom], policy=q['ppl'][pol][dom]))
        labels = {v['classification'] for v in per.values()}
        out['contrasts'][key] = dict(per_dataset=per, overall=(labels.pop() if len(labels) == 1 else 'mixed_or_inconclusive (datasets disagree)'),
                                     macro_accuracy=acc.get('macro'), gsm8k=dict(flexible=gen.get('gsm8k_flexible'), strict=gen.get('gsm8k_strict')))
    out['statement'] = 'PPL below BF16 is never reported as capability exceeding BF16 (frozen rule).'
    return out


def baselines():
    pp, ppl = analysis('V90_analyze_ppl', 'analysis_ppl/ADDITIONAL_BASELINES_PPL.json')
    pa, acc = analysis('V90_analyze_accuracy', 'analysis_accuracy/ADDITIONAL_BASELINES_ACCURACY.json')
    return dict(matrix_id='V80', freeze_sha256=FSHA, ppl=cite(pp), accuracy=cite(pa), definitions=dict(policies=FREEZE['policies']['baselines'], note=FREEZE['policies']['baselines_note']),
                ppl_models=(ppl or {}).get('models'), accuracy_models=(acc or {}).get('models'),
                primary_run_anchor={m: cross_run_contrast(f'V80_ppl_baselines_{m}', 'n16_k3', PRIMARY_PPL(m), 'n16_k3') for m in REP3})


def compute_disclosure():
    per_matrix = defaultdict(lambda: dict(gpu_hours=0.0, runs=0, by_status=defaultdict(int), peak_gpu_memory_bytes=0))
    rows = []
    for d in sorted((CR / 'runs').iterdir()):
        lr = d / 'launch_record.json'
        if not d.is_dir() or not lr.exists():
            continue
        L = json.loads(lr.read_text())
        rec = json.loads((d / 'run_record.json').read_text()) if (d / 'run_record.json').exists() else None
        cu = (rec or {}).get('artifacts', {}).get('compute_usage', {}) or {}
        gh = float(cu.get('gpu_hours') or L.get('gpu_hours') or 0.0)
        mid = L.get('matrix_id') or 'unknown'
        pm = per_matrix[mid]
        pm['gpu_hours'] += gh
        pm['runs'] += 1
        pm['by_status'][L.get('status')] += 1
        pm['peak_gpu_memory_bytes'] = max(pm['peak_gpu_memory_bytes'], int(cu.get('peak_gpu_memory_bytes') or 0))
        rows.append(dict(run=d.name, matrix_id=mid, status=L.get('status'), gpus=L.get('gpus') or (rec or {}).get('gpu_allocation', {}).get('gpu_count'),
                         gpu_names=(rec or {}).get('gpu_allocation', {}).get('gpu_names'), gpu_hours=gh, wall_seconds=cu.get('wall_seconds', L.get('wall_seconds')),
                         peak_gpu_memory_bytes=cu.get('peak_gpu_memory_bytes'), exit_code=L.get('exit_code')))
    total = sum(r['gpu_hours'] for r in rows)
    by_gpu = defaultdict(float)
    for r in rows:
        by_gpu[', '.join(sorted(set(r['gpu_names'] or ['cpu-only'])))] += r['gpu_hours']
    return dict(matrix_id='C20', freeze_sha256=FSHA, total_gpu_hours_all_attempts=total,
                gpu_hours_complete_runs=sum(r['gpu_hours'] for r in rows if r['status'] == 'complete'),
                gpu_hours_failed_invalid_or_stopped=sum(r['gpu_hours'] for r in rows if r['status'] != 'complete'), gpu_hours_by_device=dict(by_gpu),
                per_matrix={k: dict(v, by_status=dict(v['by_status'])) for k, v in sorted(per_matrix.items())}, runs=rows,
                note='gpu_hours = leased GPU count x container wall time; includes model loading and every failed, invalid or stopped attempt')


DELIVERABLE_SOURCES = {
    'V00': [('V00_inventory', 'inventory/INVENTORY.json')], 'V01': [('V90_report_v01', 'report_v01/GPU_PREFLIGHT_TEST_REPORT.md')],
    'V02': [(None, 'freeze/PROTOCOL_FREEZE.json')], 'V10': [('V10_cpu_quantizer', 'tests/QUANTIZER_CORRECTNESS.json')],
    'V11': [('V11_cpu_tiles', 'tests/TILE_LAYOUT_TESTS.json')], 'V12': [('V12_cpu_scores', 'tests/SCORE_AGGREGATION_TESTS.json')],
    'V13': [('V13_cpu_maps', 'tests/MAP_SERIALIZATION_TESTS.json')], 'V14': [('@', 'SMOKE_REPORT.json')],
    'V20': [('V90_analyze_historical', 'analysis_historical/SCORE_MANIFEST.json')], 'V21': [('V90_analyze_historical', 'analysis_historical/N8_ANCHOR_REPORT.md')],
    'V22': [('V90_final_reports', 'final/REPRODUCTION_REPORT.md'), ('V90_analyze_historical', 'analysis_historical/HISTORICAL_PPL_ANCHORS.json')],
    'V23': [('V90_analyze_historical', 'analysis_historical/CROSS_GPU_ARCHIVE.json')], 'V30': [('@', 'ALIGNED_MAP_MANIFEST.json')],
    'V31': [('V90_analyze_ppl', 'analysis_ppl/LEGACY_PANEL_PPL.json')], 'V32': [('V90_analyze_accuracy', 'analysis_accuracy/LEGACY_PANEL_ACCURACY.json')],
    'V33': [('V90_analyze_accuracy', 'analysis_accuracy/LEGACY_GENERATION.json')], 'V40': [('@', 'K_SENSITIVITY.json')], 'V41': [('@', 'OBJECTIVE_ABLATION.json')],
    'V42': [('@', 'SELECTOR_CONTROLS.json')], 'V43': [('V90_analyze_misc', 'analysis_misc/FIRST_ORDER_FIDELITY.json')], 'V50': [('@', 'CALIBRATION_SEED_STABILITY.json')],
    'V51': [('@', 'CALIBRATION_SIZE.json')], 'V52': [('@', 'CALIBRATION_DOMAIN.json')], 'V53': [('V53_data_overlap', 'overlap/DATA_OVERLAP_REPORT.json')],
    'V60': [('V90_final_reports', 'final/CONFIRMATORY_FREEZE.json')], 'V61': [('@', 'CONFIRMATORY_MAP_MANIFEST.json')],
    'V62': [('V90_analyze_ppl', 'analysis_ppl/CONFIRMATORY_PPL.json')], 'V63': [('V90_analyze_accuracy', 'analysis_accuracy/CONFIRMATORY_ACCURACY.json')],
    'V64': [('V90_analyze_accuracy', 'analysis_accuracy/CONFIRMATORY_GENERATION.json')], 'V70': [('@', 'QWEN_SMOOTHING_DIAGNOSTIC.json')],
    'V71': [('V90_analyze_ppl', 'analysis_ppl/LONG_CONTEXT.json')], 'V72': [('V90_analyze_selection', 'analysis_selection/N8_N16_STRUCTURE.json')],
    'V73': [('V90_final_reports', 'final/STATISTICAL_VALIDITY_REPORT.md'), ('V90_analyze_selection', 'analysis_selection/SELECTION_STATISTICS.json')],
    'V80': [('@', 'ADDITIONAL_BASELINES.json')], 'V81': [('V90_analyze_misc', 'analysis_misc/CROSS_GPU_ALIGNED.json')],
    'V82': [('V90_analyze_misc', 'analysis_misc/DETERMINISM_REPORT.json')], 'V83': [('V83_validate_artifacts', 'artifact_validation/ARTIFACT_VALIDATION.md')],
    'V90': [('V90_final_reports', 'final/FINAL_SUBMISSION_RISK_AUDIT.md')], 'V91': [('V90_final_reports', 'final/N16_DECISION.md')],
}
SEED0_CALIB = lambda j: '_calib_' in j and j.endswith('_seed0') and j.startswith(('V30_', 'V61_'))
EXTRA_JOBS = {'V41': lambda j: j.startswith('V42_ppl_controls_') or (j.startswith('V42_acc_rep_') and j.endswith(('_ce_only', '_kl_only', '_mean_only'))) or SEED0_CALIB(j),
              'V60': lambda j: j == 'V02_protocol_freeze',
              'V70': lambda j: j in ('V31_ppl_primary_qwen4b', 'V30_calib_qwen4b_seed0') or j.startswith(('V32_acc_full8_qwen4b_', 'V33_gsm8k_qwen4b_')),
              'V72': SEED0_CALIB, 'V73': lambda j: SEED0_CALIB(j) or j.startswith(('V31_ppl_primary_', 'V62_ppl_primary_', 'V30_calib_qwen4b_draw', 'V30_calib_llama8b_draw', 'V61_calib_mistral7b_draw')),
              'V90': lambda j: j.startswith('V90_'), 'V91': lambda j: j.startswith('V90_')}


def jobs_by_matrix(exclude=()):
    by = defaultdict(list)
    for p in (CR / 'queue' / 'jobs').glob('*.json'):
        j = json.loads(p.read_text())
        by[j['matrix_id']].append(j['job_id'])
    all_jobs = [j for js in by.values() for j in js]
    for mid, prefixes in EXTRA_JOBS.items():
        by[mid] = sorted(set(by[mid]) | {j for j in all_jobs if prefixes(j)})
    return {k: sorted(set(v) - set(exclude)) for k, v in by.items()}


def matrix_coverage(resolve, jobs):
    """resolve(job, rel) -> Path or None. A row is complete only if every job has a complete attempt and every deliverable exists."""
    rows = list(csv.DictReader(open(MATRIX, newline='', encoding='utf-8-sig')))
    cov = []
    for r in rows:
        mid = r['id']
        states = {}
        for j in jobs.get(mid, []):
            att = attempts(j)
            states[j] = dict(complete=maybe(j) is not None, attempts=len(att), last_status=att[-1]['status'] if att else 'not_launched')
        dels = []
        for job, rel in DELIVERABLE_SOURCES[mid]:
            p = resolve(job, rel)
            dels.append(dict(name=rel, source_job=job, **(cite(p) if p is not None and p.exists() else dict(path=None, sha256=None, status='missing'))))
        incomplete = sorted(j for j, s in states.items() if not s['complete'])
        cov.append(dict(id=mid, phase=r['phase'], models=r['models'], deliverable_required=r['deliverable'], publication_risk=r['publication_risk'],
                        jobs=states, jobs_incomplete=incomplete, deliverables=dels,
                        status=('complete' if not incomplete and all(d.get('sha256') for d in dels) else 'partial' if any(d.get('sha256') for d in dels) else 'missing')))
    return dict(freeze_sha256=FSHA, rows=cov, summary={s: sum(1 for c in cov if c['status'] == s) for s in ('complete', 'partial', 'missing')})


def main():
    out = runtime.out_dir('deliverables')
    written = {}

    def put(name, obj):
        runtime.atomic_json(out / name, obj)
        written[name] = runtime.sha256_file(out / name)

    put('SMOKE_REPORT.json', smoke_report())
    put('ALIGNED_MAP_MANIFEST.json', map_manifest(DEV, 'V30'))
    put('CONFIRMATORY_MAP_MANIFEST.json', map_manifest(CONF, 'V61'))
    put('K_SENSITIVITY.json', k_sensitivity())
    put('OBJECTIVE_ABLATION.json', objective_ablation())
    put('SELECTOR_CONTROLS.json', selector_controls())
    put('CALIBRATION_SEED_STABILITY.json', seed_stability())
    size, domain = size_and_domain()
    put('CALIBRATION_SIZE.json', size)
    put('CALIBRATION_DOMAIN.json', domain)
    put('QWEN_SMOOTHING_DIAGNOSTIC.json', qwen_smoothing())
    put('ADDITIONAL_BASELINES.json', baselines())
    put('COMPUTE_DISCLOSURE.json', compute_disclosure())
    src = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    runtime.atomic_json(runtime.run_dir / 'job_result.json', dict(protocol_id='synthesis', protocol_freeze_sha256=FSHA,
        source=dict(model_id=None, model_revision=None, tokenizer_revision=None, model_class=None, module_manifest_sha256=None, source_manifest_sha256=src),
        environment=runtime.environment(), data=dict(calibration_manifest_sha256=None, evaluation_manifest_sha256=None, token_hashes={}, overlap_audit=None),
        policies=[], results=dict(raw_outputs=[str(out / n) for n in written], summary=written, uncertainty={}, attempted_endpoints=list(written), missing_endpoints=runtime.collect_missing(out)),
        logs=[], failures=[]))
    print(json.dumps(written, indent=1))


if __name__ == '__main__':
    main()
