"""REPRODUCTION_REPORT.md and STATISTICAL_VALIDITY_REPORT.md (data-driven sections; narrative findings appended from findings.json)."""
import json
import os
from pathlib import Path

from campaign import runtime
from campaign.final_reports import CR, FREEZE, FSHA, cite, find, fmt
from campaign.policies import latest_complete_run


def notes(name):
    p = CR / 'reports' / 'authored' / f'{name}.md'
    return p.read_text() if p.exists() else f'_No authored findings section yet (`reports/authored/{name}.md`)._\n'


def reproduction(out):
    inp = json.loads((CR / 'provenance' / '00_input_resolution.json').read_text())
    hv = json.loads((CR / 'provenance' / 'handoff_sha256_verify.json').read_text())
    inv_p, inv = find('V00_inventory', 'inventory/INVENTORY.json')
    an_p, an = find('V90_analyze_historical', 'analysis_historical/N8_ANCHOR_RESULTS.json')
    hp_p, hp = find('V90_analyze_historical', 'analysis_historical/HISTORICAL_PPL_ANCHORS.json')
    sm_p, sm = find('V90_analyze_historical', 'analysis_historical/SCORE_MANIFEST.json')
    cx_p, cx = find('V90_analyze_historical', 'analysis_historical/CROSS_GPU_ARCHIVE.json')
    L = ['# Reproduction report (V20-V23)', '', f'Protocol freeze `{FSHA}`.', '',
         '## Inputs', '', f'- Handoff ZIP `{inp["zip_canonical_path"]}` sha256 `{inp["zip_sha256"]}` ({inp["zip_size_bytes"]} bytes); unchanged after extraction.',
         f'- `SHA256SUMS.txt` (CRLF line endings) verified {hv["verified_ok"]}/{hv["entries"]} files; failures: {hv["failures"]}.',
         f'- Inventory: {cite(inv_p)}; archived score shards present: {inv and inv["archived"]["calibrations"]["qwen4b"]["score_shards_present"]}; original cluster paths available: {inv and inv["archived"]["original_cluster_paths_available"]}.',
         '- Models were fetched at the archived revisions (Llama-3.1-8B d04e592, Qwen3-4B 1cfa9a7, Qwen3.8-27B 1d4bf0f) and every matrix weight was checked against the archived per-matrix SHA-256 before use.',
         '- Calibration crops (64 OpenWebMath + 64 CodeParrot, 512 tokens) and WikiText/C4 evaluation windows were rebuilt from pinned local files; their token SHA-256 lists equal the archived lists for all three development models.', '',
         '## Environment differences from the archived jobs', '',
         '| item | archived | this campaign |', '|---|---|---|',
         '| GPU | NVIDIA H100 / H200 (Slurm, gov113008) | RTX A6000 (Ampere sm86); RTX 6000 Ada for portability |',
         '| scheduler | Slurm | campaign-local lease + Docker device cgroup (D01) |',
         '| Python / torch | 3.11.11 / 2.9.0+cu128 | 3.11.11 / 2.9.0+cu128 (uv lock `env/hist.lock.txt`, `env/main.lock.txt`) |',
         '| transformers | 4.57.3 (4B/8B), 5.16.1 (27B) | same, by environment |',
         '| datasets | 4.8.5 (released reproduction) | 4.8.5; files read locally and token-hash verified |',
         '| driver | not recorded | 565.57.01 (CUDA 12.7 API) |',
         '| deviations | – | D-H1..D-H5 in `campaign/historical.py` (inputs from local files, local snapshot, BF16 teacher logits in RAM, no SLURM_JOB_ID, 27B streaming moments) |', '',
         '## Score shards and N8 anchors (V20/V21)', '', f'Score manifest: {cite(sm_p)}. Anchor checks: {cite(an_p)}.', '']
    for m, a in (an or {}).items():
        if 'checks' not in a:
            L += [f'- **{m}**: not complete ({a.get("attempts")}).']
            continue
        L += [f'### {m} (`{a["run"]}`)', '', '| check | value | passed |', '|---|---|---|']
        for k, c in a['checks'].items():
            val = c['value']
            L.append(f'| {k} | {fmt(val, 6) if isinstance(val, float) else val}{" / " + str(c["total"]) if "total" in c else ""} | {c["passed"]} |')
        L += ['', f'Election counts regenerated vs archived: `{json.dumps(a["kse_election"])}`.', f'Archived fixed-256 tiles in the regenerated k=2 ranking: `{json.dumps(a["archived_fixed256_ranks"])}`.',
              f'Same-GPU kernel perturbation (eager to SDPA, identical inputs/weights): `{json.dumps(a["same_gpu_kernel_perturbation"])}`.', '']
    L += ['## PPL anchors (V22)', '', f'Source: {cite(hp_p)}.', '', '| model | policy | dataset | archived PPL | regenerated PPL | rel. diff | window NLL max abs diff | tier B (<=0.5%) | tier C effect |', '|---|---|---|---:|---:|---:|---:|---|---|']
    for m, e in (hp or {}).items():
        if 'comparison' not in e:
            L.append(f'| {m} | – | – | missing | | | | | |')
            continue
        for pol, doms in e['comparison'].items():
            for d, c in doms.items():
                tc = (e.get('paired_effects', {}).get(pol, {}).get(d, {}) or {}).get('tierC', '–')
                L.append(f'| {m} | {pol} | {d} | {c["archived_ppl"]:.6f} | {c["regenerated_ppl"]:.6f} | {c["ppl_rel_diff"]:+.5f} | {c["window_nll_max_abs_diff"]:.4f} | {c["tierB_ppl_rel"]} | {tc} |')
    L += ['', '## Cross-GPU archived anchor (V23)', '', f'Source: {cite(cx_p)}.', '', '```', json.dumps(cx, indent=1)[:4000] if cx else 'missing', '```', '',
          '## Findings', '', notes('REPRODUCTION_REPORT')]
    (out / 'REPRODUCTION_REPORT.md').write_text('\n'.join(L) + '\n')


def statistics(out):
    ss_p, ss = find('V90_analyze_selection', 'analysis_selection/SELECTION_STATISTICS.json')
    st_p, st = find('V90_analyze_selection', 'analysis_selection/N8_N16_STRUCTURE.json')
    cp_p, cp = find('V90_analyze_ppl', 'analysis_ppl/CONFIRMATORY_PPL.json')
    lp_p, lp = find('V90_analyze_ppl', 'analysis_ppl/LEGACY_PANEL_PPL.json')
    dr_p, dr = find('V90_analyze_ppl', 'analysis_ppl/CALIBRATION_SEED_STABILITY_PPL.json')
    L = ['# Statistical validity report (V73)', '', f'Protocol freeze `{FSHA}`; statistics plan: `PROTOCOL_FREEZE.json#statistics`.', '',
         '## 1. Sampling units', '', '- WikiText-2: 2048-token windows of the joined test split; cluster = article containing the window\'s first token (sensitivity: contiguous blocks of 5 windows).',
         '- C4: 256 crops from documents drawn with replacement (Random(0)); cluster = document SHA-256 (231-235 unique documents per tokenizer).',
         '- Accuracy/GSM8K: examples (MMLU: questions pooled over 57 subjects); paired example bootstrap; macro = mean of task differences with independent task bootstraps.',
         '- Calibration: 128 sequences; tile statistics use per-sequence scores; N16 per-sequence scores are sums of the two N8 children.',
         f'- Bootstrap: B=10000 for primary endpoints, 2000 exploratory, percentile intervals, seed {FREEZE["statistics"]["bootstrap"]["seed"]}.', '',
         '## 2. Evaluation-sample uncertainty (primary endpoints)', '', f'Sources: {cite(lp_p)}, {cite(cp_p)}.', '']
    for name, data in (('development', lp), ('confirmatory', cp)):
        if not data:
            L.append(f'- {name}: missing')
            continue
        for m, mm in data['models'].items():
            if 'contrasts' not in mm:
                L.append(f'- {name} {m}: missing')
                continue
            for d in ('wiki', 'c4'):
                c = mm['contrasts']['n16_k3-four_over_six'][d]
                extra = f' block-5 CI {fmt(c["block5_sensitivity"]["ci95"])}' if 'block5_sensitivity' in c else ''
                L.append(f'- {name} {m} {d}: dlogPPL(N16k3-4/6) {fmt(c["estimate"])} CI {fmt(c["ci95"])} over {c["clusters"]} clusters / {c["windows"]} windows;{extra}')
    if cp and cp.get('endpoints'):
        L += ['', 'Holm-adjusted one-sided non-inferiority (margin log 1.005):', '']
        for e in cp['endpoints']:
            L.append(f'- {e["model"]}/{e["domain"]}: p={e["p_noninferiority"]:.4g}, Holm p={e["holm_adjusted_p"]:.4g}, non-inferior={e["noninferior_holm"]}')
    L += ['', '## 3. Calibration-draw variability versus evaluation variance', '', f'Source: {cite(dr_p)}.', '']
    if dr:
        import statistics as stt
        for m, mm in dr['models'].items():
            if 'contrasts' not in mm:
                L.append(f'- {m}: missing')
                continue
            for res in ('n8', 'n16'):
                for d in ('wiki', 'c4'):
                    ests = [mm['contrasts'][f'{res}_k3_draw{k}-four_over_six'][d]['estimate'] for k in (1, 2, 3, 4) if f'{res}_k3_draw{k}-four_over_six' in mm['contrasts']]
                    ws = [((mm['contrasts'][f'{res}_k3_draw{k}-four_over_six'][d]['ci95'][1] - mm['contrasts'][f'{res}_k3_draw{k}-four_over_six'][d]['ci95'][0]) / 3.92) ** 2
                          for k in (1, 2, 3, 4) if f'{res}_k3_draw{k}-four_over_six' in mm['contrasts']]
                    seed0 = ((lp or {}).get('models', {}).get(m) or (cp or {}).get('models', {}).get(m) or {}).get('contrasts', {}).get(f'{res}_k3-four_over_six', {}).get(d, {}).get('estimate')
                    allv = ests + ([seed0] if seed0 is not None else [])
                    if len(allv) >= 2:
                        L.append(f'- {m} {res} {d}: draws(+seed0) estimates {fmt(allv)}; between-draw variance {stt.variance(allv):.3e}; mean within-draw bootstrap variance {sum(ws) / len(ws):.3e}')
    L += ['', '## 4. Selection statistics: dependence, joint null and multiplicity', '', f'Source: {cite(ss_p)}.', '']
    for m, mm in (ss or {}).get('models', {}).items():
        if 'multiplicity' not in mm:
            L.append(f'- {m}: missing')
            continue
        for res in ('n8', 'n16'):
            x = mm['multiplicity'][res]
            b = x['null_expected_false_positive_bounds']
            f = mm['sign_flip'][res]
            L.append(f'- **{m} {res.upper()}**: tiles {x["tiles"]:,}, selected k=3 {x["selected_k"]:,}; per-tile CE/KL correlation median {x["ce_kl_correlation_over_sequences"]["quantiles"]["p50"]:.2f} '
                     f'(p05 {x["ce_kl_correlation_over_sequences"]["quantiles"]["p05"]:.2f}, p95 {x["ce_kl_correlation_over_sequences"]["quantiles"]["p95"]:.2f}); across-tile corr(t_CE,t_KL) {x["across_tile_correlation_of_t"]["pearson"]:.2f}. '
                     f'Null false-positive bounds: independence Phi(-3)^2 {b["independence_phi_squared"]["all_tiles"]:.1f} (invalid), measured-rho {b["measured_rho"]["all_tiles"]:.1f}, IUT any-dependence {b["iut_any_dependence"]["all_tiles"]:.0f}. '
                     f'BH q=.05 {x["normal"]["BH_q0.05"]:,} / BY q=.05 {x["normal"]["BY_q0.05"]:,} (t-dist BH {x["t"]["BH_q0.05"]:,}). '
                     f'Sign-flip (R={1000}, {f["tiles"]:,} sampled tiles): observed {f["observed"]} vs permuted mean {f["permuted_mean"]:.2f} (p95 {f["permuted_p95"]}), est. FDP {fmt(f["estimated_false_discovery_proportion"], 3)}, global p {f["p_value_global"]:.4f}.')
    L += ['', '## 5. N8 versus N16 structure', '', f'Source: {cite(st_p)}.', '']
    for m, mm in (st or {}).get('models', {}).items():
        if 'totals' not in mm:
            L.append(f'- {m}: missing')
            continue
        t = mm['totals']
        L.append(f'- {m}: N8 {t["n8"]:,} tiles ({mm["selected_weights"]["n8"]:,} weights), N16 {t["n16"]:,} ({mm["selected_weights"]["n16"]:,}); N8-any {t["n8_any"]:,}, N8-both {t["n8_both"]:,}; '
                 f'Jaccard(N16,any) {mm["jaccard_n16_vs_n8_any"]:.3f}, (N16,both) {mm["jaccard_n16_vs_n8_both"]:.3f}; selected children per selected parent {mm["selected_children_per_selected_n16_parent"]}; '
                 f'cancellation/amplification {json.dumps({k: v for k, v in mm["cancellation_amplification"].items() if k != "note"})}')
    L += ['', '## 6. Numerical noise floor', '', 'W4A4 fake quantization amplifies kernel-level floating-point differences into discrete rounding flips: on Qwen3-4B the same prompt',
          'evaluated as a 48- vs 64-token prefix differs by up to 13.7 logits (bf16) and 6.1 logits (float32) with no KV cache (`runs/V14_causality_diag_qwen4b_attempt3/diag/causality_diag.json`).',
          'Per-window and per-tile quantities are therefore not portable across kernels/GPUs; aggregate paired endpoints are the unit of inference.', '',
          '## 7. Confirmatory versus exploratory', '', '- Confirmatory: the 6 primary endpoints listed in the freeze (confirmatory panel x {WikiText, C4}, N16k3 vs FourOverSix).',
          '- Secondary (pre-declared, not multiplicity-controlled): accuracy, GSM8K, retained fraction, development-panel endpoints.', '- Exploratory: k sweep, objective ablations, selector controls, draws, size/domain, fidelity, long context, baselines, structure.', '',
          '## Findings', '', notes('STATISTICAL_VALIDITY_REPORT')]
    (out / 'STATISTICAL_VALIDITY_REPORT.md').write_text('\n'.join(L) + '\n')


def main():
    out = runtime.out_dir('final')
    reproduction(out)
    statistics(out)


if __name__ == '__main__':
    main()
