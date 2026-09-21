"""Frozen paired-cluster analysis for the three MixFP4 mechanism questions."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as ss

from campaign import mapio, runtime


B = 10_000
BOOT_SEED = 20260917
_workspace = Path(os.environ.get('MIXFP4_WORKSPACE_ROOT', Path.cwd()))
PARENT = Path(os.environ.get(
    'MIXFP4_PRIMARY_CAMPAIGN',
    _workspace / 'research_runs/mixfp4_n16k64_full_validation_20260911T065444Z',
))
FIDELITY = PARENT / 'runs/V90_analyze_misc_attempt7/analysis_misc/FIRST_ORDER_FIDELITY.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def seed_for(*parts):
    return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:16], 16)


def cluster_labels(meta, domain):
    if domain == 'wiki':
        return [f'a{w["first_article"]}' for w in meta['window_articles']]
    if domain == 'c4':
        per = int(meta['windows']) // len(meta['documents'])
        return [d['document_sha256'] for d in meta['documents'] for _ in range(per)]
    raise ValueError(domain)


def cluster_table(report, meta, domain):
    labels = cluster_labels(meta, domain)
    unique = list(dict.fromkeys(labels)); at = {x: i for i, x in enumerate(unique)}
    policies = [p['name'] for p in report['plan']]
    tokens = np.zeros(len(unique), np.float64)
    nll = np.zeros((len(policies), len(unique)), np.float64)
    for pi, policy in enumerate(policies):
        rows = report['evaluation'][policy][domain]['windows']
        if len(rows) != len(labels): raise ValueError('window/cluster length mismatch')
        for row, lab in zip(rows, labels):
            ci = at[lab]
            nll[pi, ci] += float(row['nll_sum'])
            if pi == 0: tokens[ci] += float(row['tokens'])
    return policies, np.asarray(unique), tokens, nll


def bootstrap_delta(dc, nc, label):
    est = float(dc.sum() / nc.sum())
    centered = dc - est * nc
    rng = np.random.default_rng(seed_for(BOOT_SEED, label))
    idx = rng.integers(0, len(dc), size=(B, len(dc)))
    noise = centered[idx].sum(1) / nc[idx].sum(1)
    boot = est + noise
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = (1 + int((np.abs(noise) >= abs(est)).sum())) / (B + 1)
    rates = dc / nc
    return {'estimate': est, 'ci95': [float(lo), float(hi)], 'p_two_sided_plus_one': float(p),
            'clusters': int(len(dc)), 'tokens': int(nc.sum()), 'bootstrap_replicates': B,
            'relative_ppl_change': float(math.expm1(est)),
            'cluster_regression_probability': float((rates > 0).mean()),
            'cluster_worst': float(rates.max()),
            'cluster_quantiles': {f'q{q}': float(np.quantile(rates, q / 100)) for q in (90, 95, 99)}}


def bootstrap_scalar_pair(rate_a, rate_b, functional, label):
    obs = float(functional(rate_a) - functional(rate_b))
    rng = np.random.default_rng(seed_for(BOOT_SEED, label))
    idx = rng.integers(0, len(rate_a), size=(B, len(rate_a)))
    vals = np.empty(B, np.float64)
    for s in range(0, B, 500):
        j = idx[s:s + 500]
        vals[s:s + len(j)] = np.asarray([functional(rate_a[x]) - functional(rate_b[x]) for x in j])
    centered = vals - vals.mean()
    lo, hi = np.percentile(obs + centered, [2.5, 97.5])
    p = (1 + int((np.abs(centered) >= abs(obs)).sum())) / (B + 1)
    return {'estimate': obs, 'ci95': [float(lo), float(hi)], 'p_two_sided_plus_one': float(p), 'bootstrap_replicates': B}


def holm(rows):
    if not rows: return
    order = sorted(range(len(rows)), key=lambda i: rows[i]['p_two_sided_plus_one'])
    running = 0.0; m = len(rows)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[i]['p_two_sided_plus_one']))
        rows[i]['holm_adjusted_p'] = float(running)
        rows[i]['holm_reject_0p05'] = bool(running < 0.05)


def load_run(path):
    p = Path(path) / 'ppl'
    rep = json.loads((p / 'ppl_report.json').read_text())
    meta = {d: json.loads((p / f'windows_{d}.json').read_text()) for d in rep['domains']}
    return p.parent, rep, meta


def contrast(nll, at, a, b):
    return nll[at[a]] - nll[at[b]]


def analyze_model(model, run_path, root, family_rows):
    run, rep, metas = load_run(run_path)
    model_out = {'run': str(run), 'run_launch_sha256': sha(run / 'launch_record.json'),
                 'evaluation_manifest_sha256': rep['evaluation_manifest_sha256'], 'domains': {}}
    arrays_dir = root / 'arrays'; arrays_dir.mkdir(exist_ok=True)
    for domain in ('wiki', 'c4'):
        policies, ids, tokens, nll = cluster_table(rep, metas[domain], domain)
        at = {x: i for i, x in enumerate(policies)}
        arr_path = arrays_dir / f'{model}_{domain}_paired_cluster_nll.npz'
        np.savez(arr_path, policies=np.asarray(policies), cluster_ids=ids, cluster_tokens=tokens,
                 cluster_nll_sum=nll, evaluation_manifest_sha256=np.asarray(rep['evaluation_manifest_sha256']))
        d = {'paired_array': {'path': str(arr_path), 'sha256': sha(arr_path), 'clusters': len(ids)},
             'absolute_ppl': {p: float(rep['evaluation'][p][domain]['ppl']) for p in policies},
             'ranking': {}, 'veto': {}, 'interaction': {}}
        base = 'four_over_six'
        groups = ('strongest', 'weakest') if model == 'mistral7b' else ('strongest', 'weakest', 'random')
        for g in groups:
            go = contrast(nll, at, f'group_only_{g}', base)
            marginal = contrast(nll, at, 'full', f'full_minus_{g}')
            d['ranking'][g] = {'group_only_vs_baseline': bootstrap_delta(go, tokens, f'{model}:{domain}:A:group:{g}'),
                               'full_context_marginal': bootstrap_delta(marginal, tokens, f'{model}:{domain}:A:marginal:{g}')}
        for mode in ('group_only', 'full_context_marginal'):
            key = 'group_only_vs_baseline' if mode == 'group_only' else 'full_context_marginal'
            pairs = [('strongest', 'weakest')]
            if model != 'mistral7b': pairs += [('strongest', 'random'), ('weakest', 'random')]
            for x, y in pairs:
                ax = contrast(nll, at, f'group_only_{x}', base) if mode == 'group_only' else contrast(nll, at, 'full', f'full_minus_{x}')
                ay = contrast(nll, at, f'group_only_{y}', base) if mode == 'group_only' else contrast(nll, at, 'full', f'full_minus_{y}')
                result = bootstrap_delta(ax - ay, tokens, f'{model}:{domain}:A:{mode}:{x}-{y}')
                result.update(model=model, domain=domain, endpoint=f'{mode}:{x}-{y}', family=('A_validation' if model == 'mistral7b' else 'A_development'))
                d['ranking'][f'{mode}:{x}-{y}'] = result; family_rows[result['family']].append(result)

        veto_defs = [('kl_vetoed_ce_approved', 'full_plus_kl_vetoed_ce_approved', 'full_plus_kl_vetoed_matched_random')]
        if model != 'mistral7b':
            veto_defs.append(('ce_vetoed_kl_approved', 'full_plus_ce_vetoed_kl_approved', 'full_plus_ce_vetoed_matched_random'))
        for label, actual, random in veto_defs:
            da = contrast(nll, at, actual, 'full'); dr = contrast(nll, at, random, 'full')
            av = bootstrap_delta(da, tokens, f'{model}:{domain}:B:{label}:actual-full')
            rv = bootstrap_delta(dr, tokens, f'{model}:{domain}:B:{label}:random-full')
            ar = bootstrap_delta(da - dr, tokens, f'{model}:{domain}:B:{label}:actual-random')
            rate_a, rate_r = da / tokens, dr / tokens
            tails = {
                'q90_actual_minus_random': bootstrap_scalar_pair(rate_a, rate_r, lambda x: np.quantile(x, .90), f'{model}:{domain}:B:{label}:q90'),
                'q95_actual_minus_random': bootstrap_scalar_pair(rate_a, rate_r, lambda x: np.quantile(x, .95), f'{model}:{domain}:B:{label}:q95'),
                'q99_actual_minus_random': bootstrap_scalar_pair(rate_a, rate_r, lambda x: np.quantile(x, .99), f'{model}:{domain}:B:{label}:q99'),
                'worst_actual_minus_random': bootstrap_scalar_pair(rate_a, rate_r, np.max, f'{model}:{domain}:B:{label}:worst'),
                'regression_probability_actual_minus_random': bootstrap_scalar_pair(rate_a, rate_r, lambda x: np.mean(x > 0), f'{model}:{domain}:B:{label}:prob'),
            }
            fam = 'B_validation' if model == 'mistral7b' else 'B_development'
            for nm, rr in [('mean_actual_minus_full', av), ('mean_actual_minus_random', ar), *tails.items()]:
                rr.update(model=model, domain=domain, endpoint=f'{label}:{nm}', family=fam); family_rows[fam].append(rr)
            d['veto'][label] = {'actual_vs_full': av, 'matched_random_vs_full': rv, 'actual_vs_matched_random': ar, 'tails': tails}

        full = contrast(nll, at, 'full', base)
        attn = contrast(nll, at, 'attention_only', base); mlp = contrast(nll, at, 'mlp_only', base)
        rfull = contrast(nll, at, 'matched_random_union', base)
        rattn = contrast(nll, at, 'matched_random_attention', base); rmlp = contrast(nll, at, 'matched_random_mlp', base)
        actual_i = full - attn - mlp; random_i = rfull - rattn - rmlp
        ai = bootstrap_delta(actual_i, tokens, f'{model}:{domain}:C:actual')
        ri = bootstrap_delta(random_i, tokens, f'{model}:{domain}:C:random')
        diff = bootstrap_delta(actual_i - random_i, tokens, f'{model}:{domain}:C:actual-random')
        fam = 'C_validation' if model == 'mistral7b' else 'C_development'
        for nm, rr in (('actual_residual', ai), ('actual_minus_random_residual', diff)):
            rr.update(model=model, domain=domain, endpoint=nm, family=fam); family_rows[fam].append(rr)
        d['interaction'] = {'actual_residual': ai, 'matched_random_residual': ri, 'actual_minus_matched_random': diff}
        model_out['domains'][domain] = d
    return model_out


def component_analysis(root, map_run, results):
    map_root = Path(map_run) / 'derived_maps'
    manifest = json.loads((map_root / 'map_manifest.json').read_text())
    entries = {(e['model'], e['policy']): e for e in manifest}
    fidelity = json.loads(FIDELITY.read_text())
    out = {'source_first_order_fidelity': str(FIDELITY), 'source_sha256': sha(FIDELITY), 'models': {}}
    for model in results:
        npz_path = map_root / model / f'{model}_selected_score_components.npz'
        z = np.load(npz_path)
        module_names = z['module_names'].tolist()
        per_group = {}
        groups = ('strongest', 'weakest') if model == 'mistral7b' else ('strongest', 'weakest', 'random')
        for g in groups:
            e = entries[(model, f'group_only_{g}')]
            _, masks, _ = mapio.read_map(e['path'], e['sha256'])
            keep = np.zeros(len(z['tile_index']), bool)
            for i, (mi, ti) in enumerate(zip(z['module_index'], z['tile_index'])):
                keep[i] = bool(masks[module_names[int(mi)]].view(-1)[int(ti)])
            per_group[g] = {'tiles': int(keep.sum()), 'sum_ce_mean': float(z['ce_mean'][keep].sum()),
                            **{f'mean_{k}': float(z[k][keep].mean()) for k in ('ce_mean','ce_se','ce_standardized','ce_margin','kl_mean','kl_se','kl_standardized','kl_margin','combined_margin')}}
        correlations = {}
        for observed_kind, observed_key in (('group_only', 'group_only_vs_baseline'), ('marginal', 'full_context_marginal')):
            obs, group_labels = [], []
            for domain in ('wiki', 'c4'):
                for g in groups:
                    obs.append(results[model]['domains'][domain]['ranking'][g][observed_key]['estimate']); group_labels.append(g)
            obs = np.asarray(obs)
            corr = {}
            for component in ('sum_ce_mean','mean_ce_mean','mean_ce_se','mean_ce_standardized','mean_ce_margin','mean_kl_mean','mean_kl_se','mean_kl_standardized','mean_kl_margin','mean_combined_margin'):
                pred = np.asarray([per_group[g][component] for g in group_labels])
                corr[component] = {'pearson': float(ss.pearsonr(pred, obs).statistic),
                                   'spearman': float(ss.spearmanr(pred, obs).statistic)}
            correlations[observed_kind] = corr
        f = fidelity['models'][model]['n16']
        single = f['singles']['ce']
        out['models'][model] = {'group_components': per_group, 'component_correlations': correlations,
            'individual_tile_existing_frozen_evidence': {
                'n': single['n'], 'sign_precision': single['sign_precision'], 'sign_recall': single['sign_recall'],
                'pearson': single['pearson'], 'spearman': single['spearman'],
                'resolvable_fraction_1p96se': single['resolvable_fraction_1p96se'],
                'resolvable_fraction_3se': single['resolvable_fraction_3se'],
                'full_map_actual_to_predicted_ce_ratio': f['batched'][-1]['ratio_actual_to_pred_ce'],
                'full_map_actual_ce': f['batched'][-1]['actual_ce'], 'full_map_predicted_ce': f['batched'][-1]['pred_ce_sum'],
            }, 'score_components_npz': {'path': str(npz_path), 'sha256': sha(npz_path)}}
    return out


def classify(results, family_rows):
    dev = ('llama8b', 'qwen4b')
    a = [r for r in family_rows['A_development'] if r['endpoint'] == 'full_context_marginal:strongest-weakest']
    a_group = [r for r in family_rows['A_development'] if r['endpoint'] == 'group_only:strongest-weakest']
    a_ok = len(a) == 4 and all(r['estimate'] < 0 for r in a) and sum(r['holm_reject_0p05'] for r in a) >= 3 and sum(r['estimate'] < 0 for r in a_group) >= 3
    # Both veto classes must meet the predeclared development rule independently.
    b_class = {}
    for label in ('kl_vetoed_ce_approved', 'ce_vetoed_kl_approved'):
        full = [r for r in family_rows['B_development'] if r['endpoint'] == f'{label}:mean_actual_minus_full']
        rand = [r for r in family_rows['B_development'] if r['endpoint'] == f'{label}:mean_actual_minus_random']
        b_class[label] = (len(full) == 4 and len(rand) == 4 and sum(r['estimate'] > 0 for r in full) >= 3 and
                          sum(r['estimate'] > 0 for r in rand) >= 3 and sum(r['holm_reject_0p05'] for r in full + rand) >= 2)
    b_ok = all(b_class.values())
    c = [r for r in family_rows['C_development'] if r['endpoint'] == 'actual_residual']
    signs = [np.sign(r['estimate']) for r in c]
    c_ok = len(c) == 4 and (signs.count(1) == 4 or signs.count(-1) == 4) and sum(r['holm_reject_0p05'] for r in c) >= 3
    return {
        'ranking_not_calibration': {'classification': 'supported' if a_ok else 'not_supported_under_strict_gate',
            'gate': 'all 4 marginal strongest-minus-weakest estimates negative; >=3 Holm-significant; >=3/4 group-only directions negative',
            'passed': a_ok},
        'ce_kl_veto': {'classification': 'supported' if b_ok else 'not_supported_under_strict_gate',
            'gate': 'for both veto classes, actual add-back worse than full and matched random in >=3/4 development endpoints, with >=2 Holm-significant endpoints',
            'per_class_passed': b_class, 'passed': b_ok},
        'attention_mlp_interaction': {'classification': 'supported' if c_ok else 'not_supported_under_strict_gate',
            'gate': 'actual interaction residual has one sign in all 4 development endpoints and >=3 are Holm-significant', 'passed': c_ok},
    }


def make_figures(root, results):
    figdir = root / 'figures'; figdir.mkdir(exist_ok=True)
    specs = [
        ('score_margin_ranking', 'Ranking: strongest minus weakest', lambda d: d['ranking']['full_context_marginal:strongest-weakest']),
        ('ce_kl_veto', 'Veto: add-back minus matched random', lambda d: d['veto']['kl_vetoed_ce_approved']['actual_vs_matched_random']),
        ('module_interaction', 'Attention/MLP interaction residual', lambda d: d['interaction']['actual_residual']),
    ]
    for name, title, getter in specs:
        labels, vals, lo, hi = [], [], [], []
        for model, mr in results.items():
            for domain in ('wiki', 'c4'):
                x = getter(mr['domains'][domain]); labels.append(f'{model}\n{domain}')
                vals.append(x['estimate']); lo.append(x['estimate']-x['ci95'][0]); hi.append(x['ci95'][1]-x['estimate'])
        xloc = np.arange(len(vals)); fig, ax = plt.subplots(figsize=(8.2, 4.4))
        ax.axhline(0, color='#333333', lw=.8); ax.errorbar(xloc, vals, yerr=[lo, hi], fmt='o', color='#155e75', capsize=3)
        ax.set_xticks(xloc, labels); ax.set_ylabel('delta log(PPL) = delta NLL'); ax.set_title(title); ax.grid(axis='y', alpha=.25)
        fig.tight_layout()
        for ext in ('png', 'svg'): fig.savefig(figdir / f'{name}.{ext}', dpi=240)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', required=True); ap.add_argument('--protocol-sha256', required=True)
    ap.add_argument('--map-run', required=True); ap.add_argument('--llama-run', required=True)
    ap.add_argument('--qwen-run', required=True); ap.add_argument('--mistral-run', required=True)
    args = ap.parse_args()
    if sha(args.protocol) != args.protocol_sha256: raise SystemExit('protocol hash mismatch')
    root = Path(os.environ['CAMPAIGN_ROOT'])
    family_rows = {k: [] for k in ('A_development','B_development','C_development','A_validation','B_validation','C_validation')}
    results = {}
    for model, run in (('llama8b', args.llama_run), ('qwen4b', args.qwen_run), ('mistral7b', args.mistral_run)):
        results[model] = analyze_model(model, run, root, family_rows)
    for rows in family_rows.values(): holm(rows)
    score = component_analysis(root, args.map_run, results)
    verdict = classify(results, family_rows)
    top = {'schema': 'mixfp4-mechanism-results/v1', 'protocol_sha256': args.protocol_sha256,
           'bootstrap_replicates': B, 'models': results, 'holm_families': family_rows,
           'hypothesis_classification': verdict,
           'claims_prohibited': ['individual-tile causality', 'native FP4/E0M3 Tensor Core execution', 'latency or speedup', 'area', 'power']}
    runtime.atomic_json(root / 'MECHANISM_RESULTS.json', top)
    runtime.atomic_json(root / 'SCORE_MARGIN_ANALYSIS.json', {'protocol_sha256': args.protocol_sha256, **score,
        'ranking_results': {m: {d: results[m]['domains'][d]['ranking'] for d in ('wiki','c4')} for m in results},
        'classification': verdict['ranking_not_calibration']})
    runtime.atomic_json(root / 'CE_KL_VETO_ANALYSIS.json', {'protocol_sha256': args.protocol_sha256,
        'models': {m: {d: results[m]['domains'][d]['veto'] for d in ('wiki','c4')} for m in results},
        'classification': verdict['ce_kl_veto']})
    runtime.atomic_json(root / 'MODULE_INTERACTION_ANALYSIS.json', {'protocol_sha256': args.protocol_sha256,
        'models': {m: {d: results[m]['domains'][d]['interaction'] for d in ('wiki','c4')} for m in results},
        'classification': verdict['attention_mlp_interaction']})
    rows = []
    for model, mr in results.items():
        for domain, d in mr['domains'].items():
            groups = ('strongest','weakest') if model == 'mistral7b' else ('strongest','weakest','random')
            for g in groups:
                for endpoint, r in d['ranking'][g].items():
                    rows.append({'model':model,'domain':domain,'family':'ranking','variant':g,'endpoint':endpoint,**{k:r[k] for k in ('estimate','ci95','p_two_sided_plus_one','relative_ppl_change')}})
            for label, v in d['veto'].items():
                for endpoint, r in (('actual_vs_full',v['actual_vs_full']),('actual_vs_random',v['actual_vs_matched_random'])):
                    rows.append({'model':model,'domain':domain,'family':'veto','variant':label,'endpoint':endpoint,**{k:r[k] for k in ('estimate','ci95','p_two_sided_plus_one','relative_ppl_change')}})
            for endpoint, r in d['interaction'].items():
                rows.append({'model':model,'domain':domain,'family':'interaction','variant':'attention_mlp','endpoint':endpoint,**{k:r[k] for k in ('estimate','ci95','p_two_sided_plus_one','relative_ppl_change')}})
    with open(root / 'MECHANISM_RESULTS.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model','domain','family','variant','endpoint','estimate','ci95','p_two_sided_plus_one','relative_ppl_change']); w.writeheader(); w.writerows(rows)
    make_figures(root, results)
    lines = ['# Statistical report','',f'- Frozen protocol SHA-256: `{args.protocol_sha256}`',f'- Bootstrap: {B:,} deterministic paired natural-cluster replicates.',
             '- P-values use finite Monte Carlo plus-one correction; Holm adjustment is applied within each frozen family.','']
    for fam, rs in family_rows.items():
        lines += [f'## {fam}','', '| Model | Corpus | Endpoint | delta-log-PPL | 95% CI | p | Holm p |', '|---|---|---|---:|---:|---:|---:|']
        for r in rs: lines.append(f'| {r["model"]} | {r["domain"]} | {r["endpoint"]} | {r["estimate"]:+.7f} | [{r["ci95"][0]:+.7f}, {r["ci95"][1]:+.7f}] | {r["p_two_sided_plus_one"]:.6g} | {r["holm_adjusted_p"]:.6g} |')
        lines.append('')
    (root / 'STATISTICAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    vlines=['# Mechanism verdict','']
    for k,v in verdict.items(): vlines += [f'## {k.replace("_"," ").title()}','',f'**{v["classification"]}.** {v["gate"]}','']
    vlines += ['## Scope boundary','', 'The campaign tests aggregate ranking, veto risk control, and module interaction. It does not establish individual-tile causality. Native FP4/E0M3 execution and latency, speedup, area, or power remain out of scope.','']
    (root/'MECHANISM_VERDICT.md').write_text('\n'.join(vlines))
    (root/'NEXT_STEP_RECOMMENDATION.md').write_text('# Next-step recommendation\n\nDo not launch another broad selector search. Use the frozen results to decide whether a narrowly powered replication of only the supported mechanism contrast is warranted. Preserve negative and null contrasts as first-class evidence. Native-kernel performance remains a separate project.\n')
    launch=json.loads((runtime.run_dir/'launch_record.json').read_text())
    runtime.atomic_json(runtime.run_dir/'job_result.json', {'protocol_id':'aligned-analysis','protocol_freeze_sha256':args.protocol_sha256,
      'source':{'model_id':None,'model_revision':None,'tokenizer_revision':None,'model_class':None,'module_manifest_sha256':None,'source_manifest_sha256':launch['source_manifest_sha256']},
      'environment':runtime.environment(),'data':{'calibration_manifest_sha256':None,'evaluation_manifest_sha256':None,'token_hashes':{},'overlap_audit':None},'policies':[],
      'results':{'raw_outputs':[str(root/x) for x in ('MECHANISM_RESULTS.json','MECHANISM_RESULTS.csv','SCORE_MARGIN_ANALYSIS.json','CE_KL_VETO_ANALYSIS.json','MODULE_INTERACTION_ANALYSIS.json','STATISTICAL_REPORT.md','MECHANISM_VERDICT.md','NEXT_STEP_RECOMMENDATION.md')],
                 'summary':verdict,'uncertainty':{'bootstrap_replicates':B},'attempted_endpoints':list(family_rows),'missing_endpoints':[]},'logs':[],'failures':[]})


if __name__ == '__main__': main()
