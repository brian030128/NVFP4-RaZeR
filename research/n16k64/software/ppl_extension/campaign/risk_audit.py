"""FINAL_SUBMISSION_RISK_AUDIT.md and RISK_AUDIT.json (V90): classify every PAPER_RISK_REGISTER.md row.

Classification is mechanical from the rules below (campaign audit rules, recorded in provenance/PROTOCOL_AMENDMENTS.jsonl
before the exploratory results they grade were computed). Missing evidence never counts as support. An authored note in
reports/authored/risk_audit_notes.json may add context, rejection risks and a recommendation per row; an authored
override of the mechanical class is printed next to the mechanical class with its reason, never silently.

Classes: supported | partially_supported | unsupported | out_of_scope."""
import json
import os
from pathlib import Path

from campaign import runtime
from campaign.policies import latest_complete_run

CR = Path(os.environ['CAMPAIGN_ROOT'])
FREEZE = json.loads((CR / 'freeze' / 'PROTOCOL_FREEZE.json').read_text())
FSHA = (CR / 'freeze' / 'PROTOCOL_FREEZE.sha256').read_text().split()[0]
MARGIN = FREEZE['success_criteria']['sesoi']['ppl_noninferiority_margin_dlogppl']
CONF = ('mistral7b', 'phi4', 'olmo2_13b')
REP3 = ('llama8b', 'qwen4b', 'mistral7b')
P0 = {'C02', 'C03', 'C04', 'C05', 'C06', 'C13', 'C15', 'C16', 'H01'}
REGISTER = [
    ('C01', 'Quantizer arithmetic is correct'), ('C02', 'N8/N16 map layout is correct'), ('C03', 'Archived N8 results are reproducible'),
    ('C04', 'k=3 is confirmatory for N16'), ('C05', 'N16 improves or preserves PPL'), ('C06', 'N16 preserves downstream accuracy'),
    ('C07', 'PPL gains reflect capability'), ('C08', 'Proposed selector adds value'), ('C09', 'Results are calibration-stable'),
    ('C10', 'Results are not domain-specific'), ('C11', 'Rule generalizes'), ('C12', 'First-order scores predict actual changes'),
    ('C13', 'Multiplicity argument is valid'), ('C14', 'Evaluation uncertainty is valid'), ('C15', 'Activation/backend mismatch is not driving results'),
    ('C16', 'Result generation is deterministic/auditable'), ('C17', 'Results port across available GPUs'), ('C18', 'Data leakage does not explain results'),
    ('C19', 'Baseline comparison is fair'), ('C20', 'Compute disclosure is complete'),
    ('H01', 'Native E0M3/MixFP4 executes as claimed'), ('H02', 'N8 overhead is ~13% and N16 is ~1.5%'), ('H03', 'Mixed-format hardware has acceptable area/power'),
]


class Ev:
    """Evidence loader that records every file it reads (path + sha256)."""

    def __init__(self, final_dir=None):
        self.cited = []
        self.final_dir = final_dir

    def run(self, job):
        try:
            return latest_complete_run(CR, job)
        except FileNotFoundError:
            return None

    def get(self, job, rel):
        if job == '@final':
            p = self.final_dir / rel if self.final_dir else None
        elif job is None:
            p = CR / rel
        else:
            r = self.run(job)
            p = r / rel if r else None
        if p is None or not p.exists():
            self.cited.append(dict(job=job, rel=rel, path=None, sha256=None))
            return None
        try:
            shown = str(Path(p).relative_to(CR))
        except ValueError:      # a path outside the campaign root (only possible with an overridden out_dir)
            shown = str(p)
        self.cited.append(dict(job=job, rel=rel, path=shown, sha256=runtime.sha256_file(p)))
        return json.loads(p.read_text()) if p.suffix == '.json' else p.read_text()

    def take(self):
        c, self.cited = self.cited, []
        return c


def ci_lt(c, x=0.0):
    return c is not None and c.get('ci95') is not None and c['ci95'][1] < x


def res(cls, basis, metrics=None):
    return dict(classification=cls, basis=basis, metrics=metrics or {})


def missing(what):
    return res('unsupported', [f'insufficient evidence: {w} missing' for w in what])


# ------------------------------------------------------------------------------------------------ rules
def c01(E):
    q = E.get('V10_cpu_quantizer', 'tests/QUANTIZER_CORRECTNESS.json')
    a = E.get('V10_cpu_archived_repo_tests_main', 'tests/ARCHIVED_REPOSITORY_TESTS.json')
    if q is None:
        return missing(['V10 quantizer tests'])
    ok = q['all_passed'] and (a is None or a['all_passed'])
    return res('supported' if ok else 'unsupported', [f'independent scalar/reference quantizer tests all_passed={q["all_passed"]}',
                                                    f'archived repository tests all_passed={a and a["all_passed"]}'])


def c02(E):
    t = E.get('V11_cpu_tiles', 'tests/TILE_LAYOUT_TESTS.json')
    m = E.get('V13_cpu_maps', 'tests/MAP_SERIALIZATION_TESTS.json')
    s = E.get('V12_cpu_scores', 'tests/SCORE_AGGREGATION_TESTS.json')
    chk = {k: E.get(f'V42_checksum_controls_{k}', 'checksum/CHECKSUM_CONTROLS.json') for k in ('qwen4b', 'llama8b', 'mistral7b', 'phi4', 'olmo2_13b')}
    if t is None or m is None:
        return missing([n for n, v in (('V11', t), ('V13', m)) if v is None])
    unit = t['all_passed'] and m['all_passed'] and (s is None or s['all_passed'])
    real = {k: (v or {}).get('passed') for k, v in chk.items()}
    basis = [f'V11 tile layout all_passed={t["all_passed"]}; V12 aggregation all_passed={s and s["all_passed"]}; V13 serialization all_passed={m["all_passed"]}',
             f'real-model all-false/all-true checksum controls: {real}']
    if not unit or any(v is False for v in real.values()):
        return res('unsupported', basis)
    return res('supported' if all(v is True for v in real.values()) else 'partially_supported', basis +
               ([] if all(v is True for v in real.values()) else ['real-model checksum controls incomplete']))


def c03(E):
    an = E.get('V90_analyze_historical', 'analysis_historical/N8_ANCHOR_RESULTS.json')
    hp = E.get('V90_analyze_historical', 'analysis_historical/HISTORICAL_PPL_ANCHORS.json')
    inv = E.get('V00_inventory', 'inventory/INVENTORY.json')
    if an is None or hp is None:
        return missing(['historical anchors'])
    failed = {m: sorted(k for k, c in a.get('checks', {}).items() if c.get('passed') is False) for m, a in an.items()}
    not_run = sorted(m for m, a in an.items() if not a.get('checks'))
    ppl_fail = {m: sorted(f'{p}:{d}' for p, doms in e.get('comparison', {}).items() for d, c in doms.items()
                          if c.get('tierB_ppl_rel') is False or (e.get('paired_effects', {}).get(p, {}).get(d) or {}).get('tierC') is False)
                for m, e in hp.items()}
    k3_ok = all((e.get('paired_effects', {}).get('hist_n8_k3', {}).get(d) or {}).get('tierC') for e in hp.values() if 'comparison' in e for d in ('wiki', 'c4'))
    basis = [f'archived score shards present: {inv and inv["archived"]["calibrations"]["qwen4b"]["score_shards_present"]} (regenerated instead)',
             f'failed anchor checks: {failed}', f'models without anchor run: {not_run}', f'PPL anchor tier-B/C failures: {ppl_fail}',
             f'N8 k3 paired effect reproduces (tier C) on every completed model/dataset: {k3_ok}']
    if not k3_ok:
        return res('unsupported', basis)
    return res('supported' if not any(failed.values()) and not any(ppl_fail.values()) and not not_run else 'partially_supported', basis)


def c04(E):
    fr = E.get(None, 'freeze/PROTOCOL_FREEZE.json')
    reg = [json.loads(l) for l in (CR / 'registry' / 'attempts.jsonl').read_text().splitlines()]
    conf_created = sorted(e['logged_utc'] for e in reg if e.get('event') == 'created' and e.get('run_id', '').startswith(('V61_', 'V62_', 'V63_', 'V64_')))
    amend = [json.loads(l) for l in (CR / 'provenance' / 'PROTOCOL_AMENDMENTS.jsonl').read_text().splitlines()] if (CR / 'provenance' / 'PROTOCOL_AMENDMENTS.jsonl').exists() else []
    conf_changes = [a['title'] for a in amend if a['affects_confirmatory']]
    before = bool(conf_created) and fr['created_utc'] < conf_created[0]
    basis = [f'freeze sha256 {FSHA} created {fr["created_utc"]}; first confirmatory run created {conf_created[0] if conf_created else None}',
             f'primary k fixed: {fr["success_criteria"]["k_rule"]}', f'amendments affecting confirmatory endpoints: {conf_changes or "none"}',
             f'quality results opened before freeze: {json.dumps(fr["quality_results_opened_before_freeze"])[:300]}']
    return res('supported' if before and not conf_changes else 'unsupported', basis)


def primary(E):
    return E.get('V90_analyze_ppl', 'analysis_ppl/CONFIRMATORY_PPL.json'), E.get('V90_analyze_ppl', 'analysis_ppl/LEGACY_PANEL_PPL.json')


def c05(E):
    cp, lp = primary(E)
    if cp is None or len(cp.get('endpoints', [])) < 6:
        return missing(['6 primary confirmatory endpoints'])
    eps = cp['endpoints']
    noninf = all(e['noninferior_holm'] and e['upper_ci_below_margin'] for e in eps)
    improve = all(e['ci95'][1] < 0 for e in eps)
    dev = {m: {d: mm['contrasts']['n16_k3-four_over_six'][d]['ci95'] for d in ('wiki', 'c4')} for m, mm in (lp or {}).get('models', {}).items() if 'contrasts' in mm}
    basis = [f'{e["model"]}/{e["domain"]}: {e["estimate"]:+.5f} [{e["ci95"][0]:+.5f}, {e["ci95"][1]:+.5f}] Holm p={e["holm_adjusted_p"]:.3g}' for e in eps]
    basis += [f'all non-inferior (Holm, margin log 1.005): {noninf}; all upper CI < 0: {improve}', f'development panel N16k3-4/6 CIs: {dev}']
    return res('supported' if noninf else 'unsupported', basis, dict(all_improve=improve))


def c06(E):
    ca = E.get('V90_analyze_accuracy', 'analysis_accuracy/CONFIRMATORY_ACCURACY.json')
    la = E.get('V90_analyze_accuracy', 'analysis_accuracy/LEGACY_PANEL_ACCURACY.json')
    if ca is None:
        return missing(['confirmatory accuracy'])
    per, basis = {}, []
    for m in CONF:
        c = (ca['models'].get(m) or {}).get('n16_k3-four_over_six', {})
        if 'macro' not in c:
            per[m] = None
            basis.append(f'{m}: missing ({c.get("error", "")[:120]})')
            continue
        ok = c['noninferior_macro'] and c['fewer_than_3_tasks_clear_loss']
        per[m] = ok
        basis.append(f'{m}: macro {100 * c["macro"]["diff"]:+.2f} pp [{100 * c["macro"]["ci95"][0]:+.2f}, {100 * c["macro"]["ci95"][1]:+.2f}], tasks CI<0 {c["tasks_with_ci_below_zero"]}, pass={ok}')
    for m, mm in (la or {}).get('models', {}).items():
        c = (mm or {}).get('n16_k3-four_over_six', {})
        if 'macro' in c:
            basis.append(f'development {m}: macro {100 * c["macro"]["diff"]:+.2f} pp [{100 * c["macro"]["ci95"][0]:+.2f}, {100 * c["macro"]["ci95"][1]:+.2f}], tasks CI<0 {c["tasks_with_ci_below_zero"]}')
    if any(v is False for v in per.values()):
        return res('unsupported', basis)
    if any(v is None for v in per.values()):
        return res('partially_supported' if any(per.values()) else 'unsupported', basis + ['confirmatory accuracy incomplete'])
    return res('supported', basis)


def c07(E):
    """Rule: capability claim needs positive accuracy/GSM8K evidence where PPL improves, and no smoothing verdict on Qwen."""
    ca = E.get('V90_analyze_accuracy', 'analysis_accuracy/CONFIRMATORY_ACCURACY.json')
    la = E.get('V90_analyze_accuracy', 'analysis_accuracy/LEGACY_PANEL_ACCURACY.json')
    cg = E.get('V90_analyze_accuracy', 'analysis_accuracy/CONFIRMATORY_GENERATION.json')
    lg = E.get('V90_analyze_accuracy', 'analysis_accuracy/LEGACY_GENERATION.json')
    qs = E.get('V90_assemble_deliverables', 'deliverables/QWEN_SMOOTHING_DIAGNOSTIC.json')
    cp, lp = primary(E)
    if ca is None or qs is None:
        return missing([n for n, v in (('confirmatory accuracy', ca), ('V70 smoothing diagnostic', qs)) if v is None])
    pos, null, neg, basis = [], [], [], []
    for panel, data in (('confirmatory', ca), ('development', la or {'models': {}})):
        for m, mm in data['models'].items():
            c = (mm or {}).get('n16_k3-four_over_six', {})
            if 'macro' not in c:
                continue
            lo, hi = c['macro']['ci95']
            (pos if lo > 0 else neg if hi < 0 else null).append(f'{panel}:{m}')
    for data in (cg, lg):
        for m, mm in (data or {}).get('models', {}).items():
            g = (mm or {}).get('n16_k3-four_over_six', {}).get('gsm8k_flexible')
            if g:
                basis.append(f'GSM8K {m}: {100 * g["diff"]:+.2f} pp [{100 * g["ci95"][0]:+.2f}, {100 * g["ci95"][1]:+.2f}]')
    diag = []
    for data, name in ((cp, 'confirmatory'), (lp, 'development')):
        for m, mm in (data or {}).get('models', {}).items():
            for d in ('wiki', 'c4'):
                x = (mm.get('diagnostics', {}).get(d, {}) or {}).get('n16_k3', {})
                if 'delta_kl_vs_ref' in x:
                    diag.append(f'{name}:{m}:{d} dKL(BF16||.) {x["delta_kl_vs_ref"]["estimate"]:+.4f} CI {[round(v, 4) for v in x["delta_kl_vs_ref"]["ci95"]]}, '
                                f'dTop1 {x["delta_top1_vs_ref"]["estimate"]:+.4f}, dEntropy {x["delta_entropy_vs_four_over_six"]["estimate"]:+.3f}')
    q = {k: v['overall'] for k, v in qs['contrasts'].items()}
    basis = [f'macro accuracy N16k3-4/6: CI>0 {pos}, CI includes 0 {null}, CI<0 {neg}', f'V70 Qwen3-4B verdicts: {q}'] + basis + diag
    smoothing = any('smoothing' in v for v in q.values())
    if pos and not neg and not smoothing and len(pos) >= 2:
        return res('supported', basis)
    if neg and not pos:
        return res('unsupported', basis)
    return res('partially_supported' if (pos or null) and not smoothing else 'unsupported', basis +
               ['PPL gains are small relative to accuracy resolution; accuracy is preserved rather than shown to improve' if null and not pos else ''])


def c08(E):
    sc = E.get('V90_assemble_deliverables', 'deliverables/SELECTOR_CONTROLS.json')
    if sc is None:
        return missing(['V42 selector controls'])
    basis, better_random, not_worse_heur, missing_m = [], [], [], []
    for m in REP3:
        mm = sc['models'].get(m, {})
        if 'ppl_n16_k3_minus_control' not in mm:
            missing_m.append(m)
            continue
        rnd = [mm['ppl_n16_k3_minus_control'][f'n16_random_s{s}'] for s in range(5)]
        br = all(ci_lt(r.get('domains', {}).get(d)) for r in rnd for d in ('wiki', 'c4'))
        heur = {c: mm['ppl_n16_k3_minus_control'][c].get('domains', {}) for c in ('n16_weight_mse', 'n16_magnitude', 'n16_change_norm')}
        nw = all((v.get(d) or {}).get('ci95', [1, 1])[0] <= 0 for v in heur.values() for d in ('wiki', 'c4'))
        sig_better_heur = {c: {d: (v.get(d) or {}).get('ci95') for d in ('wiki', 'c4')} for c, v in heur.items()}
        better_random.append(br)
        not_worse_heur.append(nw)
        basis.append(f'{m}: N16k3 better than all 5 random count-matched maps on Wiki and C4: {br}; N16k3 minus heuristic CIs {sig_better_heur}')
    if missing_m:
        return res('unsupported' if len(missing_m) == len(REP3) else 'partially_supported', basis + [f'missing models {missing_m}'])
    if all(better_random) and all(not_worse_heur):
        return res('supported', basis)
    if sum(better_random) >= 2:
        return res('partially_supported', basis)
    return res('unsupported', basis)


def c09(E):
    st = E.get('V90_assemble_deliverables', 'deliverables/CALIBRATION_SEED_STABILITY.json')
    if st is None:
        return missing(['V50 seed stability'])
    basis, allneg, noninf, miss = [], True, True, []
    for m in REP3:
        mm = st['models'].get(m, {})
        if mm.get('missing_draws'):
            miss.append(f'{m}:{mm["missing_draws"]}')
        eff = mm.get('n16_k3_effect_across_draws', {})
        cons = mm.get('ppl_vs_four_over_six') or {}
        for d in ('wiki', 'c4'):
            e = eff.get(d, {})
            allneg &= bool(e.get('all_negative'))
            uppers = [((cons.get(f'n16_k3_draw{k}-four_over_six') or {}).get(d) or {}).get('ci95', [None, None])[1] for k in (1, 2, 3, 4)]
            noninf &= all(u is not None and u < MARGIN for u in uppers)
            basis.append(f'{m} {d}: N16k3 effects seed0+draws {[round(x, 5) if x is not None else None for x in e.get("seed0_and_draws", [])]}, sd {e.get("between_draw_sd")}, draw upper CIs {uppers}')
        maps = mm.get('maps', {}).get('n16_k3', {})
        basis.append(f'{m}: N16k3 counts {maps.get("counts")}, mean pairwise Jaccard {maps.get("mean_pairwise_jaccard")}, tiles in every draw {maps.get("selected_in_every_draw")}')
    if miss:
        return res('partially_supported' if allneg else 'unsupported', basis + [f'missing draws: {miss}'])
    return res('supported' if allneg and noninf else 'partially_supported' if noninf else 'unsupported', basis)


def c10(E):
    dm = E.get('V90_assemble_deliverables', 'deliverables/CALIBRATION_DOMAIN.json')
    sz = E.get('V90_assemble_deliverables', 'deliverables/CALIBRATION_SIZE.json')
    if dm is None:
        return missing(['V52 domain ablation'])
    basis, generic_ok, indomain_ok, cross_penalty, miss = [], True, True, False, []
    for m in ('qwen4b', 'mistral7b'):
        mm = dm['models'].get(m, {})
        g = mm.get('ppl_wiki_c4_vs_four_over_six') or {}
        x = (mm.get('ppl_in_domain') or {}).get('contrasts') or {}
        if not g or not x:
            miss.append(m)
        for s in ('math64', 'code64', 'heldout'):
            for d in ('wiki', 'c4'):
                c = (g.get(f'n16_k3_{s}-four_over_six') or {}).get(d)
                generic_ok &= c is not None and c['ci95'][1] < MARGIN
                basis.append(f'{m} N16k3[{s}] {d}: {c and round(c["estimate"], 5)} CI {c and [round(v, 5) for v in c["ci95"]]}')
            for d in ('math_eval', 'code_eval'):
                c = (x.get(f'n16_k3_{s}-four_over_six') or {}).get(d)
                indomain_ok &= c is not None and c['ci95'][1] < MARGIN
                basis.append(f'{m} N16k3[{s}] held-out {d}: {c and round(c["estimate"], 5)} CI {c and [round(v, 5) for v in c["ci95"]]}')
        for d, worse in (('math_eval', 'n16_k3_code64-n16_k3_math64'), ('code_eval', 'n16_k3_math64-n16_k3_code64')):
            a = (x.get('n16_k3_math64-n16_k3_code64') or {}).get(d)
            if a:
                pen = a['estimate'] if d == 'code_eval' else -a['estimate']
                lo_pen = a['ci95'][0] if d == 'code_eval' else -a['ci95'][1]
                cross_penalty |= lo_pen > MARGIN
                basis.append(f'{m} off-domain penalty on {d} ({worse}): {pen:+.5f} (lower CI {lo_pen:+.5f})')
    for m in ('qwen4b', 'mistral7b'):
        s = (sz or {}).get('models', {}).get(m, {}).get('maps', {}).get('n16')
        if s:
            basis.append(f'{m} calibration size N16 counts {{mc32: {s["mc32"]["selected_tiles"]}, mc64: {s["mc64"]["selected_tiles"]}, full: {s["seed0"]["selected_tiles"]}}}')
    if miss:
        return res('unsupported' if len(miss) == 2 else 'partially_supported', basis + [f'missing models {miss}'])
    if generic_ok and indomain_ok and not cross_penalty:
        return res('supported', basis)
    return res('partially_supported' if generic_ok else 'unsupported', basis)


def c11(E):
    gate = E.get('@final', 'decision_gate.json')
    if gate is None:
        return missing(['N16 decision gate'])
    lab = gate['label']
    basis = [f'frozen gate classification on 3 unseen families (Mistral, Phi-4, OLMo-2): {lab}', f'details: {json.dumps(gate["details"], default=str)[:600]}',
             f'families: {FREEZE["panels"]["confirmatory"]}; replacement: {FREEZE["panels"]["replacements"]}']
    if lab.startswith(('minimum pass', 'strong pass')):
        return res('supported', basis)
    if lab.startswith(('investigate', 'undetermined')):
        return res('partially_supported', basis)
    return res('unsupported', basis)


def c12(E):
    """Rule: supported if, for every model, single-intervention Spearman(pred, actual) >= 0.5 for CE and KL at N16 and the selected
    stratum false-positive rate (either objective not negative) <= 0.2; unsupported if any model has Spearman <= 0.2 or FPR > 0.5."""
    f = E.get('V90_analyze_misc', 'analysis_misc/FIRST_ORDER_FIDELITY.json')
    if f is None:
        return missing(['V43 fidelity'])
    basis, good, bad, miss = [], [], [], []
    for m in REP3:
        mm = f['models'].get(m, {})
        if 'n16' not in mm:
            miss.append(m)
            continue
        for r in ('n8', 'n16'):
            s = mm[r]['singles']
            sel = s.get('selected', {})
            batched = mm[r]['batched']
            ratios = [round(b['ratio_actual_to_pred_ce'], 3) for b in batched if b.get('ratio_actual_to_pred_ce') is not None]
            basis.append(f'{m} {r}: Spearman CE {s["ce"]["spearman"]:.2f} KL {s["kl"]["spearman"]:.2f}; sign precision CE {s["ce"]["sign_precision"]}; '
                         f'selected FPR(any) {sel.get("false_positive_rate_any_objective")}; batched actual/pred CE ratios {ratios}')
        s16 = mm['n16']['singles']
        fpr = s16.get('selected', {}).get('false_positive_rate_any_objective', 1.0)
        good.append(min(s16['ce']['spearman'], s16['kl']['spearman']) >= 0.5 and fpr <= 0.2)
        bad.append(min(s16['ce']['spearman'], s16['kl']['spearman']) <= 0.2 or fpr > 0.5)
    if miss:
        return res('unsupported' if len(miss) == len(REP3) else 'partially_supported', basis + [f'missing {miss}'])
    return res('supported' if all(good) else 'unsupported' if any(bad) else 'partially_supported', basis)


def c13(E):
    """Rule: the archived Phi(-3)^2 argument is valid only if CE/KL scores are ~independent (median per-tile correlation < 0.1).
    Otherwise the argument is unsupported; the dependency-robust replacements (IUT bound, BH/BY, sign-flip FDP) are reported."""
    s = E.get('V90_analyze_selection', 'analysis_selection/SELECTION_STATISTICS.json')
    if s is None:
        return missing(['selection statistics'])
    basis, medians = [], []
    for m, mm in s['models'].items():
        if 'multiplicity' not in mm:
            basis.append(f'{m}: missing')
            continue
        x = mm['multiplicity']['n16']
        f = mm['sign_flip']['n16']
        med = x['ce_kl_correlation_over_sequences']['quantiles']['p50']
        medians.append(med)
        basis.append(f'{m} N16: median CE/KL corr {med:.2f}; k=3 selects {x["selected_k"]}; IUT any-dependence null bound {x["null_expected_false_positive_bounds"]["iut_any_dependence"]["all_tiles"]:.0f}; '
                     f'BY q=.05 {x["normal"]["BY_q0.05"]}; sign-flip FDP {f["estimated_false_discovery_proportion"]:.3f} (global p {f["p_value_global"]:.4f})')
    return res('supported' if medians and max(medians) < 0.1 else 'unsupported', basis +
               ['independence argument invalid; a valid statement must use the sign-flip FDP estimate / IUT bound instead' if medians and max(medians) >= 0.1 else ''])


def c14(E):
    cp, lp = primary(E)
    st = E.get('V90_assemble_deliverables', 'deliverables/CALIBRATION_SEED_STABILITY.json')
    if cp is None:
        return missing(['confirmatory PPL'])
    basis, agree = [], True
    for data, name in ((cp, 'confirmatory'), (lp, 'development')):
        for m, mm in (data or {}).get('models', {}).items():
            if 'contrasts' not in mm:
                continue
            c = mm['contrasts']['n16_k3-four_over_six']['wiki']
            b = c.get('block5_sensitivity')
            if b:
                same = (c['ci95'][1] < 0) == (b['ci95'][1] < 0) and (c['ci95'][0] > 0) == (b['ci95'][0] > 0)
                agree &= same
                basis.append(f'{name} {m} wiki: article-cluster CI {[round(v, 5) for v in c["ci95"]]} ({c["clusters"]} clusters) vs block-5 CI {[round(v, 5) for v in b["ci95"]]}; conclusions agree {same}')
    var_sep = bool(st) and all('n16_k3_effect_across_draws' in st['models'].get(m, {}) and not st['models'][m].get('missing_draws') for m in REP3)
    basis.append(f'calibration-draw vs evaluation variance separated on all 3 draw models: {var_sep}')
    return res('supported' if agree and var_sep else 'partially_supported' if agree else 'unsupported', basis)


def c15(E):
    hp = E.get('V90_analyze_historical', 'analysis_historical/HISTORICAL_PPL_ANCHORS.json')
    an = E.get('V90_analyze_historical', 'analysis_historical/N8_ANCHOR_RESULTS.json')
    cp, lp = primary(E)
    if hp is None or lp is None:
        return missing(['historical anchors or aligned development PPL'])
    basis, same_sign = [], True
    for m in ('llama8b', 'qwen4b'):
        h = (hp.get(m, {}).get('paired_effects', {}).get('hist_n8_k3') or {})
        a = (lp['models'].get(m, {}).get('contrasts', {}) or {}).get('n8_k3-four_over_six', {})
        for d in ('wiki', 'c4'):
            he = (h.get(d) or {}).get('regenerated_dlogppl')
            ae = (a.get(d) or {}).get('estimate')
            if he is None or ae is None:
                same_sign = False
                basis.append(f'{m} {d}: missing (historical {he}, aligned {ae})')
                continue
            same_sign &= (he < 0) == (ae < 0)
            basis.append(f'{m} {d}: N8k3 dlogPPL historical (tensor-wide act, eager) {he:+.5f} vs aligned causal SDPA {ae:+.5f}')
    inv = (an or {}).get('qwen4b', {}).get('same_gpu_kernel_perturbation')
    basis.append(f'same-GPU eager->SDPA selection perturbation (qwen4b): {json.dumps(inv)[:300]}')
    basis.append('confirmatory panel calibrated and evaluated only under the aligned causal protocol')
    return res('supported' if same_sign and cp else 'partially_supported' if same_sign else 'unsupported', basis +
               (['selected tile sets are backend-sensitive (Jaccard well below 1) although effects reproduce'] if inv else []))


def c16(E):
    det = E.get('V90_analyze_misc', 'analysis_misc/DETERMINISM_REPORT.json')
    av = E.get('V83_validate_artifacts', 'artifact_validation/ARTIFACT_VALIDATION.json')
    if det is None or det.get('status') == 'missing' or av is None:
        return missing([n for n, v in (('V82 determinism', det if det and det.get('status') != 'missing' else None), ('V83 artifact validation', av)) if v is None])
    # Superseded attempts (already remediated by a later complete attempt) and still-running runs must not count as
    # problems; V83 separates them (amendment 17) and this rule now consumes that split instead of the raw list.
    complete_problems = av.get('blocking_runs_with_problems')
    if complete_problems is None:  # validation artifacts written before the split
        complete_problems = [r for r in av['runs_with_problems'] if r['status'] == 'complete']
    basis = [f'V82 repeat (same GPU: {det.get("same_gpu_uuid")}, same source manifest: {det.get("same_source_manifest")}): tierA {det["tierA"]}, tierB {det["tierB"]}, '
             f'score stream equal {det["score_stream_sha256_equal"]}, forward fit losses equal {det.get("fit_losses_equal")} and BF16 fit NLL equal {det.get("bf16_fit_nll_equal")}, '
             f'maps {json.dumps({k: (v["payload_equal"], round(v["jaccard"], 4)) for k, v in det["maps"].items()})}',
             f'V83: {av["runs"]} runs, by status {av["by_status"]}, blocking runs with problems {len(complete_problems)} '
             f'(superseded {len(av.get("superseded_runs_with_problems", []))}, in flight {len(av.get("in_flight_runs_with_problems", []))}); manifest entries {av["entries"]}']
    if complete_problems:
        return res('unsupported', basis + [f'problems: {[r["run"] for r in complete_problems][:10]}'])
    return res('supported' if det['tierA'] else 'partially_supported' if det['tierB'] else 'unsupported', basis)


def c17(E):
    x = E.get('V90_analyze_misc', 'analysis_misc/CROSS_GPU_ALIGNED.json')
    xa = E.get('V90_analyze_historical', 'analysis_historical/CROSS_GPU_ARCHIVE.json')
    if x is None or x.get('status') == 'missing':
        att = (x or {}).get('attempts')
        return res('unsupported', ['insufficient evidence: A6000-vs-RTX 6000 Ada aligned portability runs did not complete', f'attempts: {json.dumps(att)[:600]}',
                                   f'archived-anchor cross-GPU (V23): {json.dumps(xa)[:300] if xa else "missing"}'])
    maps = x.get('maps', {})
    eff = x.get('paired_effects', {})
    ppl_ok = all(v['tierB'] for pol in x.get('ppl', {}).values() for v in pol.values())
    eff_ok = all(v['tier'] for pol in eff.values() for v in pol.values())
    map_ok = all(v['tierB'] for v in maps.values())
    basis = [f'maps A6000 vs Ada: {json.dumps(maps)[:500]}', f'PPL tier: {ppl_ok}; paired effect tier: {eff_ok}']
    if x.get('status') != 'complete':
        return res('partially_supported', basis + [f'status {x.get("status")}'])
    return res('supported' if ppl_ok and eff_ok and map_ok else 'partially_supported' if eff_ok else 'unsupported', basis)


def c18(E):
    o = E.get('V53_data_overlap', 'overlap/DATA_OVERLAP_REPORT.json')
    if o is None:
        return missing(['V53 overlap audit'])
    exact = {k: v.get('exact_normalized_overlap') for k, v in o['pairs'].items()}
    frac = {k: v.get('13gram', {}).get('calibration_shingle_fraction') for k, v in o['pairs'].items()}
    basis = [f'exact normalized document overlap: {exact}', f'13-gram calibration shingle fraction: {frac}', f'limits: {o.get("limits")}',
             f'task load error: {o.get("task_load_error")}']
    if any(v for v in exact.values()) or any((f or 0) > 0.01 for f in frac.values()):
        return res('partially_supported', basis)
    return res('supported' if not o.get('task_load_error') else 'partially_supported', basis)


def c19(E):
    b = E.get('V90_assemble_deliverables', 'deliverables/ADDITIONAL_BASELINES.json')
    cp, lp = primary(E)
    if b is None:
        return missing(['V80 baselines'])
    basis, complete = [], True
    for m in REP3:
        pm = (b.get('ppl_models') or {}).get(m, {})
        am = (b.get('accuracy_models') or {}).get(m, {})
        have_ppl = 'contrasts' in pm
        have_acc = all('macro' in (am.get(p) or {}) for p in FREEZE['policies']['baselines'])
        complete &= have_ppl and have_acc
        if have_ppl:
            for p in FREEZE['policies']['baselines']:
                c = pm['contrasts'].get(f'{p}-n16_k3', {})
                basis.append(f'{m} {p} minus N16k3: ' + ', '.join(f'{d} {c[d]["estimate"]:+.4f} [{c[d]["ci95"][0]:+.4f}, {c[d]["ci95"][1]:+.4f}]' for d in ('wiki', 'c4') if d in c))
        anc = (b.get('primary_run_anchor') or {}).get(m, {})
        basis.append(f'{m}: baseline PPL present {have_ppl}, representative accuracy present {have_acc}, n16_k3 reproduced across runs '
                     f'{ {d: (v or {}).get("anchor_check", {}).get("bitwise_equal") for d, v in anc.get("domains", {}).items()} }')
    basis.append('all methods share source weights, tokenizer, module scope, evaluation tokens, attention backend and (except *_native_rows) the causal FourOverSix activation quantizer')
    return res('supported' if complete else 'partially_supported', basis)


def c20(E):
    cd = E.get('V90_assemble_deliverables', 'deliverables/COMPUTE_DISCLOSURE.json')
    fs = E.get('V83_validate_artifacts', 'artifact_validation/FAILED_OR_SKIPPED_RUNS.json')
    if cd is None or fs is None:
        return missing([n for n, v in (('compute disclosure', cd), ('failed/skipped registry', fs)) if v is None])
    basis = [f'total GPU-hours (all attempts) {cd["total_gpu_hours_all_attempts"]:.1f}; complete {cd["gpu_hours_complete_runs"]:.1f}; failed/invalid/stopped {cd["gpu_hours_failed_invalid_or_stopped"]:.1f}',
             f'by device {cd["gpu_hours_by_device"]}', f'failed/invalid/not-started attempts registered: {len(fs["attempts"])}']
    return res('supported', basis)


def hardware(cid):
    txt = {'H01': 'All results use dequantized BF16 matmul (fake quantization) on RTX A6000 / RTX 6000 Ada; no native E0M3/MixFP4 kernel exists in this campaign.',
           'H02': 'Overhead figures are external team estimates (KNOWN_RESULTS.json); not measurable on the available GPUs and not re-measured.',
           'H03': 'No RTL, simulator or physical evaluation was in scope.'}[cid]
    return res('out_of_scope', [txt, 'excluded by PROTOCOL_FREEZE.json#scope.excluded'])


RULES = dict(C01=c01, C02=c02, C03=c03, C04=c04, C05=c05, C06=c06, C07=c07, C08=c08, C09=c09, C10=c10, C11=c11, C12=c12, C13=c13, C14=c14,
             C15=c15, C16=c16, C17=c17, C18=c18, C19=c19, C20=c20, H01=lambda E: hardware('H01'), H02=lambda E: hardware('H02'), H03=lambda E: hardware('H03'))


def main(final_dir=None):
    out = final_dir or runtime.out_dir('final')
    E = Ev(final_dir=out)
    notes_p = CR / 'reports' / 'authored' / 'risk_audit_notes.json'
    notes = json.loads(notes_p.read_text()) if notes_p.exists() else {}
    rows = []
    for cid, claim in REGISTER:
        try:
            r = RULES[cid](E)
        except Exception as exc:  # a broken rule is reported, never hidden
            r = res('unsupported', [f'audit rule error: {exc!r}'])
        r['basis'] = [b for b in r['basis'] if b]
        n = notes.get(cid, {})
        final = n.get('override_classification') or r['classification']
        rows.append(dict(id=cid, claim=claim, p0=cid in P0, mechanical_classification=r['classification'], classification=final,
                         override_reason=n.get('override_reason') if n.get('override_classification') else None, basis=r['basis'], metrics=r['metrics'],
                         evidence=E.take(), rejection_risks=n.get('rejection_risks', []), recommendation=n.get('recommendation'),
                         note=n.get('note')))
    for r in rows:
        if r['p0'] and r['classification'] not in ('supported', 'out_of_scope') and not r['recommendation']:
            r['recommendation'] = 'UNRESOLVED: P0 row below supported requires an authored recommendation (narrow / limitation / later experiment / do not submit)'
    runtime.atomic_json(out / 'RISK_AUDIT.json', dict(freeze_sha256=FSHA, rows=rows, notes_file=(str(notes_p.relative_to(CR)) if notes_p.exists() else None),
                                                      notes_sha256=(runtime.sha256_file(notes_p) if notes_p.exists() else None)))
    L = ['# Final submission risk audit', '', f'Protocol freeze `{FSHA}`. Classes follow `PAPER_RISK_REGISTER.md`; the mechanical rule for each row is in '
         '`campaign/risk_audit.py` (recorded in `provenance/PROTOCOL_AMENDMENTS.jsonl` before the graded exploratory results existed). '
         'Evidence files are listed with SHA-256; "missing" evidence never counts as support.', '',
         '| ID | P0 | claim | classification | mechanical | recommendation |', '|---|---|---|---|---|---|']
    for r in rows:
        L.append(f'| {r["id"]} | {"yes" if r["p0"] else ""} | {r["claim"]} | **{r["classification"]}** | {r["mechanical_classification"]} | {(r["recommendation"] or "").split(chr(10))[0][:160]} |')
    L.append('')
    for r in rows:
        L += [f'## {r["id"]} — {r["claim"]}', '', f'**Classification: {r["classification"]}**' + (f' (authored override of mechanical `{r["mechanical_classification"]}`: {r["override_reason"]})' if r['override_reason'] else ''), '']
        L += ['Basis:', ''] + [f'- {b}' for b in r['basis']] + ['']
        L += ['Evidence:', ''] + [f'- `{e["path"]}` sha256 `{e["sha256"]}`' if e['path'] else f'- **missing**: `{e["job"]}` / `{e["rel"]}`' for e in r['evidence']] + ['']
        if r['note']:
            L += [r['note'], '']
        if r['rejection_risks']:
            L += ['Remaining rejection risks:', ''] + [f'- {x}' for x in r['rejection_risks']] + ['']
        if r['recommendation']:
            L += [f'Recommendation: {r["recommendation"]}', '']
    scope = CR / 'reports' / 'authored' / 'FINAL_SUBMISSION_RISK_AUDIT_scope.md'
    L += ['## Recommended claim scope and venue', '', scope.read_text() if scope.exists() else '_Authored scope section missing (`reports/authored/FINAL_SUBMISSION_RISK_AUDIT_scope.md`)._', '']
    (out / 'FINAL_SUBMISSION_RISK_AUDIT.md').write_text('\n'.join(L) + '\n')
    return rows


if __name__ == '__main__':
    main()
