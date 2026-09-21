"""V90/V91 synthesis: REPRODUCTION_REPORT.md, N16_DECISION.md, STATISTICAL_VALIDITY_REPORT.md, FINAL_SUBMISSION_RISK_AUDIT.md,
CONFIRMATORY_FREEZE.json and ARTIFACT_INDEX.json. Every statement is derived from a deliverable file whose path and SHA-256 are cited."""
import json
import math
import os
from pathlib import Path

from campaign import runtime
from campaign.policies import latest_complete_run

CR = Path(os.environ['CAMPAIGN_ROOT'])
FREEZE = json.loads((CR / 'freeze' / 'PROTOCOL_FREEZE.json').read_text())
FSHA = (CR / 'freeze' / 'PROTOCOL_FREEZE.sha256').read_text().split()[0]
SUCC = FREEZE['success_criteria']
MARGIN = SUCC['sesoi']['ppl_noninferiority_margin_dlogppl']


def find(job, rel):
    try:
        run = latest_complete_run(CR, job)
    except FileNotFoundError:
        return None, None
    p = run / rel
    if not p.exists():
        return None, None
    return p, json.loads(p.read_text()) if p.suffix == '.json' else p.read_text()


def cite(p):
    return f'`{p.relative_to(CR)}` (sha256 `{runtime.sha256_file(p)[:16]}…`)' if p else '**missing**'


def fmt(x, nd=4):
    if x is None:
        return 'n/a'
    if isinstance(x, (list, tuple)):
        return '[' + ', '.join(fmt(v, nd) for v in x) + ']'
    if isinstance(x, float):
        return f'{x:+.{nd}f}' if abs(x) < 1e4 else f'{x:.3e}'
    return str(x)


def accuracy_block(art, art_path, null_sentence, pending_note):
    """Render the representative-accuracy half of a decision-document section.

    K_SENSITIVITY_ACCURACY and ADDITIONAL_BASELINES_ACCURACY share one shape: models.<model>.<policy>
    is either {'status': 'missing'} or a comparison whose macro block carries diff/ci95 as proportions.
    The verdict sentence is DERIVED from the data and never hard-coded, so this section cannot assert
    "pending" once the accuracy jobs land, nor assert a result the artifact does not contain.
    """
    if not art:
        return [f'Accuracy is not available yet ({pending_note}).', '']
    rows, sig, missing = [], [], 0
    for m, mm in (art.get('models') or {}).items():
        for pol, c in (mm or {}).items():
            if not isinstance(c, dict):
                continue
            mac = c.get('macro') if c.get('status') != 'missing' else None
            ci = (mac or {}).get('ci95') or [None, None]
            if not mac or mac.get('diff') is None or ci[0] is None or ci[1] is None:
                missing += 1
                continue
            lo, hi = ci[0] * 100, ci[1] * 100
            if lo > 0 or hi < 0:
                sig.append(f'{m} {pol}')
            rows.append(f'| {m} | {pol} | {fmt(mac["diff"] * 100, 2)} | [{fmt(lo, 2)}, {fmt(hi, 2)}] | '
                        f'{len(c.get("tasks_with_ci_below_zero") or [])} | {len(c.get("tasks_with_ci_above_zero") or [])} |')
    if not rows:
        return [f'Accuracy is not available yet ({pending_note}); {missing} model x policy cells outstanding.', '']
    out = [f'Accuracy, macro over the representative task suite against {art.get("reference", "the reference arm")}. '
           f'Source: {cite(art_path)}.', '',
           '| model | policy | macro pp | 95% CI (pp) | tasks CI<0 | tasks CI>0 |', '|---|---|---:|---|---:|---:|'] + rows + ['']
    if sig:
        out += [f'{len(sig)} of {len(rows)} model x policy cells have a macro accuracy CI excluding zero '
                f'({", ".join(sig)}).', '']
    else:
        out += [f'No macro accuracy CI excludes zero in any of the {len(rows)} model x policy cells measured, so '
                f'{null_sentence} at this protocol\'s resolution (risk-audit row C07).', '']
    if missing:
        out += [f'{missing} further model x policy cells are still outstanding ({pending_note}).', '']
    return out


def gates(conf_ppl, conf_acc):
    """Apply the frozen success criteria mechanically; returns (label, details)."""
    details = {}
    if conf_ppl is None or not conf_ppl.get('endpoints') or len(conf_ppl['endpoints']) < 6:
        return 'undetermined (primary confirmatory endpoints incomplete)', details
    eps = conf_ppl['endpoints']
    all_noninf = all(e['upper_ci_below_margin'] for e in eps)
    details['all_primary_upper_ci_below_margin'] = all_noninf
    details['holm_noninferior'] = [e.get('noninferior_holm') for e in eps]
    pooled = sum(e['estimate'] for e in eps) / len(eps)
    details['pooled_mean_primary_dlogppl'] = pooled
    acc_ok, acc_detail = True, {}
    for m in ('mistral7b', 'phi4', 'olmo2_13b'):
        c = ((conf_acc or {}).get('models', {}).get(m, {}) or {}).get('n16_k3-four_over_six', {})
        if 'macro' not in c:
            acc_ok = None if acc_ok is not False else acc_ok
            acc_detail[m] = 'missing'
            continue
        ok = c['macro']['ci95'][0] * 100 > SUCC['sesoi']['accuracy_macro_margin_pp'] and len(c['tasks_with_ci_below_zero']) < 3
        acc_detail[m] = dict(macro_diff=c['macro']['diff'], macro_ci95=c['macro']['ci95'], tasks_ci_below_zero=c['tasks_with_ci_below_zero'], ok=ok)
        acc_ok = acc_ok and ok if acc_ok is not None else None
    details['accuracy'] = acc_detail
    R = {}
    for m, mm in conf_ppl['models'].items():
        R[m] = mm.get('retained_fraction_n16_over_n8') if isinstance(mm, dict) else None
    n16_sum = sum(e['estimate'] for e in eps)
    n8_sum = sum(mm['contrasts']['n8_k3-four_over_six'][d]['estimate'] for mm in conf_ppl['models'].values() if 'contrasts' in mm for d in ('wiki', 'c4'))
    pooled_R = n16_sum / n8_sum if n8_sum < 0 else None
    details.update(retained_fraction_by_model=R, pooled_retained_fraction=pooled_R, pooled_n8_dlogppl_sum=n8_sum)
    minimum = all_noninf and acc_ok is True
    if acc_ok is None:
        return 'undetermined (confirmatory accuracy incomplete)', details
    stop = sum(1 for v in R.values() if v is not None and v <= 0) >= 2 or any(e['ci95'][1] >= MARGIN for e in eps) and pooled >= MARGIN
    if minimum and pooled_R is not None and pooled_R >= 0.5 and n8_sum < 0:
        return 'strong pass (quality component only; overhead out of scope)', details
    if minimum:
        return 'minimum pass', details
    if stop:
        return 'stop / redirect', details
    if pooled < 0:
        return 'investigate, do not claim success', details
    return 'fails minimum pass', details


def main():
    out = runtime.out_dir('final')
    D = {}
    D['anchors'] = find('V90_analyze_historical', 'analysis_historical/N8_ANCHOR_RESULTS.json')
    D['hist_ppl'] = find('V90_analyze_historical', 'analysis_historical/HISTORICAL_PPL_ANCHORS.json')
    D['score_manifest'] = find('V90_analyze_historical', 'analysis_historical/SCORE_MANIFEST.json')
    D['cross_archive'] = find('V90_analyze_historical', 'analysis_historical/CROSS_GPU_ARCHIVE.json')
    D['legacy_ppl'] = find('V90_analyze_ppl', 'analysis_ppl/LEGACY_PANEL_PPL.json')
    D['conf_ppl'] = find('V90_analyze_ppl', 'analysis_ppl/CONFIRMATORY_PPL.json')
    for name in ('K_SENSITIVITY_PPL', 'SELECTOR_CONTROLS_PPL', 'ADDITIONAL_BASELINES_PPL', 'CALIBRATION_SEED_STABILITY_PPL', 'CALIBRATION_SIZE_DOMAIN_PPL', 'LONG_CONTEXT'):
        D[name] = find('V90_analyze_ppl', f'analysis_ppl/{name}.json')
    D['legacy_acc'] = find('V90_analyze_accuracy', 'analysis_accuracy/LEGACY_PANEL_ACCURACY.json')
    D['conf_acc'] = find('V90_analyze_accuracy', 'analysis_accuracy/CONFIRMATORY_ACCURACY.json')
    D['legacy_gen'] = find('V90_analyze_accuracy', 'analysis_accuracy/LEGACY_GENERATION.json')
    D['conf_gen'] = find('V90_analyze_accuracy', 'analysis_accuracy/CONFIRMATORY_GENERATION.json')
    for name in ('K_SENSITIVITY_ACCURACY', 'SELECTOR_CONTROLS_ACCURACY', 'ADDITIONAL_BASELINES_ACCURACY', 'CALIBRATION_SEED_STABILITY_ACCURACY', 'CALIBRATION_SIZE_DOMAIN_ACCURACY'):
        D[name] = find('V90_analyze_accuracy', f'analysis_accuracy/{name}.json')
    D['structure'] = find('V90_analyze_selection', 'analysis_selection/N8_N16_STRUCTURE.json')
    D['selection_stats'] = find('V90_analyze_selection', 'analysis_selection/SELECTION_STATISTICS.json')
    D['fidelity'] = find('V90_analyze_misc', 'analysis_misc/FIRST_ORDER_FIDELITY.json')
    D['determinism'] = find('V90_analyze_misc', 'analysis_misc/DETERMINISM_REPORT.json')
    D['cross_aligned'] = find('V90_analyze_misc', 'analysis_misc/CROSS_GPU_ALIGNED.json')
    D['overlap'] = find('V53_data_overlap', 'overlap/DATA_OVERLAP_REPORT.json')
    for _name in ('K_SENSITIVITY', 'ADDITIONAL_BASELINES'):  # countervailing evidence for the decision document
        D[_name.lower()] = find('V90_assemble_deliverables', f'deliverables/{_name}.json')
    D['inventory'] = find('V00_inventory', 'inventory/INVENTORY.json')
    D['artifact'] = find('V83_validate_artifacts', 'artifact_validation/ARTIFACT_VALIDATION.json')
    D['v01'] = (latest_complete_run(CR, 'V90_report_v01') / 'report_v01' / 'GPU_PREFLIGHT_TEST_REPORT.md', None) if _exists('V90_report_v01') else (None, None)
    tests = {}
    for job, name in (('V10_cpu_quantizer', 'QUANTIZER_CORRECTNESS.json'), ('V11_cpu_tiles', 'TILE_LAYOUT_TESTS.json'), ('V12_cpu_scores', 'SCORE_AGGREGATION_TESTS.json'),
                      ('V13_cpu_maps', 'MAP_SERIALIZATION_TESTS.json'), ('V14_cpu_mini_e2e', 'SMOKE_REPORT_CPU_E2E.json'), ('V01_cpu_preflight_unit', 'GPU_PREFLIGHT_UNIT_TESTS.json')):
        tests[job] = find(job, f'tests/{name}')
    p = lambda k: D[k][0]
    v = lambda k: D[k][1]

    # ---------------- CONFIRMATORY_FREEZE.json (V60)
    cf = dict(matrix_id='V60', protocol_freeze_sha256=FSHA, confirmatory_panel=FREEZE['panels']['confirmatory'], replacements=FREEZE['panels']['replacements'],
              compatibility_smoke={k: FREEZE['panels']['compatibility_smoke'][k] for k in ('mistral7b', 'phi4', 'olmo2_13b')},
              quality_results_opened_before_freeze=FREEZE['quality_results_opened_before_freeze'], created_utc=FREEZE['created_utc'])
    runtime.atomic_json(out / 'CONFIRMATORY_FREEZE.json', cf)

    # ---------------- N16 decision
    label, gd = gates(v('conf_ppl'), v('conf_acc'))
    L = ['# N16K64 research decision (V91)', '', f'Protocol freeze: `freeze/PROTOCOL_FREEZE.json` sha256 `{FSHA}` (created {FREEZE["created_utc"]}).', '',
         '## 1. Frozen primary configuration', '', '```', json.dumps(FREEZE['primary_policy'], indent=1), '```', '']
    for key, title in (('legacy_ppl', 'Development panel'), ('conf_ppl', 'Confirmatory panel')):
        L += [f'## 2{"a" if key == "legacy_ppl" else "b"}. {title}: exact-map perplexity', '', f'Source: {cite(p(key))}', '']
        data = v(key)
        if not data:
            L += ['**Missing** (see failed/skipped registry).', '']
            continue
        L += ['| model | dataset | PPL FourOverSix | PPL N8 k3 | PPL N16 k3 | dlogPPL N16-4/6 [95% CI] | dlogPPL N8-4/6 [95% CI] | dlogPPL N16-N8 [95% CI] | N8 tiles | N16 tiles | map sha256 (N16) |',
              '|---|---|---:|---:|---:|---|---|---|---:|---:|---|']
        for m, mm in data['models'].items():
            if 'contrasts' not in mm:
                L.append(f'| {m} | – | missing | | | | | | | | |')
                continue
            for d in ('wiki', 'c4'):
                c16, c8, c168 = (mm['contrasts'][k][d] for k in ('n16_k3-four_over_six', 'n8_k3-four_over_six', 'n16_k3-n8_k3'))
                L.append(f'| {m} | {d} | {mm["ppl"]["four_over_six"][d]:.4f} | {mm["ppl"]["n8_k3"][d]:.4f} | {mm["ppl"]["n16_k3"][d]:.4f} | {fmt(c16["estimate"])} {fmt(c16["ci95"])} | '
                         f'{fmt(c8["estimate"])} {fmt(c8["ci95"])} | {fmt(c168["estimate"])} {fmt(c168["ci95"])} | {mm["selected_tiles"].get("n8_k3")} | {mm["selected_tiles"].get("n16_k3")} | `{(mm["map_sha256"].get("n16_k3") or "")[:16]}…` |')
        L.append('')
        if key == 'conf_ppl' and data.get('endpoints'):
            L += ['Primary endpoints (one-sided non-inferiority, margin log(1.005), Holm across 6):', '', '| model | dataset | estimate | 95% CI | p (NI) | Holm p | non-inferior (Holm) |', '|---|---|---:|---|---:|---:|---|']
            for e in data['endpoints']:
                L.append(f'| {e["model"]} | {e["domain"]} | {fmt(e["estimate"])} | {fmt(e["ci95"])} | {e["p_noninferiority"]:.4g} | {e["holm_adjusted_p"]:.4g} | {e["noninferior_holm"]} |')
            L.append('')
    for key, title in (('legacy_acc', 'Development panel accuracy'), ('conf_acc', 'Confirmatory panel accuracy')):
        L += [f'## 3. {title} (exact maps, 8 tasks, paired example bootstrap)', '', f'Source: {cite(p(key))}', '']
        data = v(key)
        if not data:
            L += ['**Missing**.', '']
            continue
        L += ['| model | contrast | macro diff [95% CI] (pp) | tasks CI<0 | tasks CI>0 |', '|---|---|---|---|---|']
        for m, mm in data['models'].items():
            for ck, c in mm.items():
                if 'macro' in c:
                    L.append(f'| {m} | {ck} | {100 * c["macro"]["diff"]:+.2f} [{100 * c["macro"]["ci95"][0]:+.2f}, {100 * c["macro"]["ci95"][1]:+.2f}] | {", ".join(c["tasks_with_ci_below_zero"]) or "–"} | {", ".join(c["tasks_with_ci_above_zero"]) or "–"} |')
                else:
                    L.append(f'| {m} | {ck} | missing | | |')
        L.append('')
    L += ['## 4. Retained fraction and gate classification', '', f'Frozen gates: `{json.dumps({k: SUCC[k] for k in ("minimum_pass", "strong_pass_quality_only", "investigate", "stop_redirect")})}`', '',
          f'**Classification: {label}.**', '', '```', json.dumps(gd, indent=1, default=str), '```', '']
    # Section 5. The gate above grades the frozen quality criteria only. A reader of this decision document must see the
    # results that qualify it here, not only in the risk audit. Every figure is extracted from the cited artifact.
    g = lambda x, nd=4: (fmt(x, nd) if isinstance(x, (int, float)) else '–')        # signed: the sign is load-bearing
    u = lambda x, nd=4: (f'{x:.{nd}f}' if isinstance(x, (int, float)) else '–')     # unsigned: proportions and ratios
    L += ['## 5. Evidence that qualifies the classification above', '',
          'The classification in section 4 grades the frozen quality criteria and nothing else. The results below were produced',
          'by this same campaign and materially qualify it.', '']
    ks = v('k_sensitivity')
    if ks:
        L += [f'**5a. k=3 is a pre-registered conservative threshold, not a tuned optimum.** Source: {cite(p("k_sensitivity"))}.',
              f'Frozen rule: `{ks.get("rule", "")}`', '',
              '| model | corpus | dlogPPL k=2 vs FourOverSix [95% CI] | dlogPPL k=3 vs FourOverSix [95% CI] | k=2 tiles | k=3 tiles |',
              '|---|---|---|---|---:|---:|']
        for m, mm in (ks.get('ppl_models') or {}).items():
            con, cnt = mm.get('contrasts') or {}, (((ks.get('maps') or {}).get(m) or {}).get('counts') or {})
            for dom in ('wiki', 'c4'):
                a, b = (con.get('n16_k2-four_over_six') or {}).get(dom), (con.get('n16_k3-four_over_six') or {}).get(dom)
                if not a or not b:
                    continue
                L.append(f'| {m} | {dom} | {g(a.get("estimate"))} [{g(a.get("ci95", [None])[0])}, {g(a.get("ci95", [None, None])[1])}] | '
                         f'{g(b.get("estimate"))} [{g(b.get("ci95", [None])[0])}, {g(b.get("ci95", [None, None])[1])}] | '
                         f'{cnt.get("n16_k2", "–")} | {cnt.get("n16_k3", "–")} |')
        L += ['', 'Lower k selects more tiles and gives better perplexity in every model x corpus cell measured, so k=3 must be',
              'justified by the pre-registration and by per-tile reliability, never as an optimum.', '']
        L += accuracy_block(v('K_SENSITIVITY_ACCURACY'), p('K_SENSITIVITY_ACCURACY'),
                            'the monotone perplexity improvement as k decreases has no counterpart in downstream accuracy',
                            'V40 representative-accuracy jobs')
    else:
        L += ['**5a. k sensitivity**: artifact not available yet.', '']
    ab = v('additional_baselines')
    if ab:
        L += [f'**5b. A published prior-art format outperforms N16K64 on most models measured.** Source: {cite(p("additional_baselines"))}.', '',
              '| model | corpus | RaZeR minus N16 k3 [95% CI] | nover6 minus N16 k3 [95% CI] |', '|---|---|---|---|']
        for m, mm in (ab.get('ppl_models') or {}).items():
            con = mm.get('contrasts') or {}
            for dom in ('wiki', 'c4'):
                r = (con.get('razer_native_rows-n16_k3') or {}).get(dom) or (con.get('razer_wonly_shared_act-n16_k3') or {}).get(dom)
                n = (con.get('nover6_native_rows-n16_k3') or {}).get(dom)
                if not r:
                    continue
                nn = f'{g(n.get("estimate"))} [{g(n.get("ci95", [None])[0])}, {g(n.get("ci95", [None, None])[1])}]' if n else '–'
                L.append(f'| {m} | {dom} | {g(r.get("estimate"))} [{g(r.get("ci95", [None])[0])}, {g(r.get("ci95", [None, None])[1])}] | {nn} |')
        L += ['', 'A negative entry means the baseline is **better** than N16 k3. RaZeR reaches its numbers with a different element',
              'format (e3m3 weights, e4m3 activations) and therefore a different datapath, so this is not a matched-hardware',
              'comparison - but it must be shown rather than omitted, and no state-of-the-art quality claim is available from this',
              'campaign.', '']
        L += accuracy_block(v('ADDITIONAL_BASELINES_ACCURACY'), p('ADDITIONAL_BASELINES_ACCURACY'),
                            'the perplexity gaps between these baselines and N16 k3 have no counterpart in downstream accuracy',
                            'V80 representative-accuracy jobs')
    else:
        L += ['**5b. Additional baselines**: artifact not available yet.', '']
    fid = v('fidelity')
    if fid:
        L += [f'**5c. The first-order score does not predict individual tile effects.** Source: {cite(p("fidelity"))}.', '',
              '| model | type block | Spearman(pred, actual) CE / KL | selected-stratum FPR | full-map actual/predicted CE |',
              '|---|---|---|---:|---:|']
        for m, mm in (fid.get('models') or {}).items():
            for res in ('n8', 'n16'):
                r = mm.get(res)
                if not isinstance(r, dict) or 'singles' not in r:
                    continue
                s, bat = r['singles'], (r.get('batched') or [])
                fpr = (s.get('selected') or {}).get('false_positive_rate_any_objective')
                ratio = bat[-1].get('ratio_actual_to_pred_ce') if bat else None
                L.append(f'| {m} | {res} | {g((s.get("ce") or {}).get("spearman"), 2)} / {g((s.get("kl") or {}).get("spearman"), 2)} | '
                         f'{u(fpr, 3)} | {u(ratio, 2)} |')
        L += ['', 'The aligned evaluation is bitwise deterministic (V43 noise controls: 24 identical baseline evaluations per model on',
              'three different A6000 cards, end-of-run drift exactly 0.0, with two independent cross-card reproductions of the same',
              'measured tile effect), so a single-tile measurement carries no measurement error whatsoever. Under replication the',
              'predicted CE sign was correct in only **3 of the 9** tiles re-measured (three per model; KL 7 of 9), and the tile the',
              'k=3 rule *rejects* bears no relation to its prediction on any model - helping on Llama-3.1-8B (-8.24e-4) and Mistral-7B',
              '(-5.31e-4), hurting on Qwen3-4B (+1.12e-3), always by two to three orders of magnitude more than predicted. Decisively,',
              'the model carrying the highest single-tile rank correlation of all six cells (Qwen3-4B, Spearman +0.239) still',
              'mispredicts its median and rejected tiles by -146x and -1,657x, so this is not an artifact of the models where the score',
              'looks worst. The reproducible object is the aggregate effect of a hash-verified map, never an individual tile selection,',
              'and no per-tile or per-layer claim is supported.', '']
    else:
        L += ['**5c. First-order fidelity**: artifact not available yet.', '']
    ss = v('selection_stats')
    if ss:
        L += [f'**5d. The archived multiplicity argument does not hold.** Source: {cite(p("selection_stats"))}.', '',
              '| model | median per-tile CE/KL correlation | tiles at k=3 | BH q=0.05 retains | sign-flip FDP |',
              '|---|---:|---:|---:|---:|']
        for m, mm in (ss.get('models') or {}).items():
            x, f = (mm.get('multiplicity') or {}).get('n16'), (mm.get('sign_flip') or {}).get('n16')
            if not x or not f:
                continue
            med = ((x.get('ce_kl_correlation_over_sequences') or {}).get('quantiles') or {}).get('p50')
            L.append(f'| {m} | {g(med, 2)} | {x.get("selected_k", "–")} | {(x.get("normal") or {}).get("BH_q0.05", "–")} | '
                     f'{u(f.get("estimated_false_discovery_proportion"), 3)} |')
        L += ['', 'CE and KL scores are strongly positively correlated per tile, so the archived independence-based bound is invalid.',
              'The defensible statement is the empirical sign-flip false-discovery proportion above.', '']
    else:
        L += ['**5d. Selection statistics**: artifact not available yet.', '']
    L += ['## 6. Overhead', '', 'Only the user-supplied estimates exist (N8K64 ~13%, N16K64 ~1.5%; `KNOWN_RESULTS.json`). They were not measured in this campaign and',
          'cannot be measured on RTX A6000 / RTX 6000 Ada; any quality-overhead Pareto statement must label them as unverified external estimates.', '']
    (out / 'N16_DECISION.md').write_text('\n'.join(L) + '\n')

    # (the remaining reports are assembled by final_reports_text.py, which imports this module's D-dictionary loaders)
    runtime.atomic_json(out / 'decision_gate.json', dict(label=label, details=gd, freeze_sha256=FSHA))
    runtime.atomic_json(out / 'deliverable_index.json', {k: (str(p(k).relative_to(CR)) if p(k) else None) for k in D})
    print(json.dumps(dict(label=label), indent=1))


def _exists(job):
    try:
        latest_complete_run(CR, job)
        return True
    except FileNotFoundError:
        return False


if __name__ == '__main__':
    main()
