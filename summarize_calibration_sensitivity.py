"""Validate and publish all fixed calibration settings, including block counts."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
from calibration_sensitivity import calibration_subsets, validate_losses
from run_c4_frozen import digest_file
from run_conditional_model import paired

DOMAINS = ('c4', 'wiki', 'literature', 'science', 'government')
MODELS = ('qwen4b', 'llama8b', 'qwen27b')
LABELS = dict(qwen4b='Qwen3-4B', llama8b='Llama-3.1-8B', qwen27b='Qwen3.8-27B')
POLICIES = ('four_over_six', *calibration_subsets(), 'weight_mse')
EQUAL64 = ('c4_64', 'math_64', 'code_64', 'c4_math64', 'c4_code64', 'math_code64', 'pooled64')
START = '<!-- CALIBRATION_SENSITIVITY_START -->'
END = '<!-- CALIBRATION_SENSITIVITY_END -->'


def table(model, r, delta=False):
    lines = [f'### {LABELS[model]}' + (' — baseline-relative PPL changes' if delta else ''), '',
             '| Calibration | Sequences | E0M3 blocks | % of type blocks | C4 | WikiText-2 | Literature | Science | Government | Dataset average |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for policy in POLICIES:
        stats = r['block_statistics'][policy]
        vals = [r['evaluation'][policy][d]['ppl'] - (r['evaluation']['four_over_six'][d]['ppl'] if delta else 0)
                for d in DOMAINS]
        fmt = '+.6f' if delta else '.6f'
        lines.append(f'| {policy} | {stats.get("calibration_sequences", "—")} | {stats["selected_blocks"]:,} | '
                     f'{100*stats["selected_fraction"]:.6f}% | ' + ' | '.join(format(v, fmt) for v in vals)
                     + f' | {format(sum(vals)/len(vals), fmt)} |')
    return lines


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--reports', nargs=3, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--update-report', action='store_true')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    reports, frozen_maps, cells = {}, {}, []
    for path_string in args.reports:
        path = Path(path_string)
        r = json.loads(path.read_text())
        model = r['model']; assert model in MODELS and model not in reports
        assert r['status'] == 'complete' and r['maps_unchanged'] and r['source_weights_verified']
        assert set(r['evaluation']) == set(POLICIES)
        assert digest_file(r['prepared']) == r['prepared_sha256']
        frozen = json.loads(Path(r['prepared']).read_text())
        assert frozen['original_maps_reproduced_exactly'] and frozen['subsets'] == calibration_subsets()
        assert frozen['block_statistics'] == r['block_statistics']
        for policy in POLICIES:
            assert r['suffix_intervention'][policy]['equal']
            assert r['suffix_intervention'][policy]['max_logit_difference'] == 0
            assert set(r['evaluation'][policy]) == set(DOMAINS)
            stats = r['block_statistics'][policy]
            if policy in frozen['maps']:
                assert sum(len(v) for v in frozen['maps'][policy].values()) == stats['selected_blocks'] <= 256
            for domain in DOMAINS:
                v = r['evaluation'][policy][domain]
                windows = r['data'][domain]['windows']
                assert len(r['data'][domain]['token_sha256']) == windows
                assert v['scored_tokens'] == windows * 511
                assert abs(validate_losses(v['nll'], windows) - v['ppl']) < 1e-12
                contrast = paired(v['nll'], r['evaluation']['four_over_six'][domain]['nll'])
                if policy != 'four_over_six':
                    assert all(abs(contrast[k] - r['contrasts'][policy][domain][k]) < 1e-12 for k in contrast)
                origin = v['origin']
                if origin['reused']:
                    assert digest_file(origin['report']) == origin['report_sha256']
                    assert origin['first_window_absolute_error'] <= 1e-6
                cells.append(dict(model=model, policy=policy, dataset=domain, ppl=v['ppl'],
                    baseline_ppl=r['evaluation']['four_over_six'][domain]['ppl'], **contrast,
                    selected_e0m3_blocks=stats['selected_blocks'], selected_fraction=stats['selected_fraction'],
                    calibration_sequences=stats.get('calibration_sequences', 0),
                    windows=windows, reused=origin['reused'], report=str(path)))
        reports[model], frozen_maps[model] = r, frozen
    assert set(reports) == set(MODELS) and len(cells) == 3*19*5
    all_capped = all(reports[m]['block_statistics'][p]['cap_reached']
                     for m in MODELS for p in calibration_subsets())
    with (out / 'cells.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0])); writer.writeheader(); writer.writerows(cells)
    lines = ['# Calibration sensitivity and cross-dataset generalization', '',
             'Three models; 17 prespecified calibration settings plus two controls; five evaluation datasets. '
             'One existing 192-sequence score table per model, no seed replication or destination-based selection.', '',
             'All tables use causal per-token activation factors. Source score tables retain the original '
             'window-wide convention. Every row is a frozen map evaluated on the same destination inputs. '
             'A lower PPL is better. The average is an unweighted arithmetic mean of five dataset PPLs, '
             'not a pooled-corpus perplexity. Model tokenizers differ.', '',
             'E0M3 counts are 8x64 **type blocks** (512 weights / 32 scale blocks each), '
             'not 16-element scale blocks. Percentages use all quantized type blocks as denominator.', '',
             '## Separate dataset PPL and selected block counts', '']
    compact = ['## Calibration sensitivity and generalization', '',
               'The fixed source/sample-count study covers Qwen3-4B, Llama-3.1-8B and Qwen3.8-27B. '
               'Each model reuses one shared score table from 64 C4 + 64 OpenWebMath + 64 CodeParrot '
               'sequences. All 17 subsets use the unchanged CE/KL two-SE rule and at-most-256-block cap; '
               'no dataset-specific selection, recalibration, or seed replication is performed.', '',
               'Rows report causal evaluation PPL and the actual selected **8x64 E0M3 type-block count**. '
               'Each type block contains 512 weights and 32 scale blocks. The percentage denominator is '
               'all quantized type blocks. Dataset averages are unweighted arithmetic means of all five '
               'displayed PPLs. These sensitivity settings do not replace the frozen pooled192 primary rule.', '']
    if all_capped:
        note = ('All 51 calibrated maps select exactly 256 E0M3 type blocks: the common cap binds '
                'in every setting. Constant counts reflect the budget, not identical selections. '
                'Eligible-block counts and pairwise Jaccard overlap below measure changes in the candidate '
                'pool and selected block identities.')
        lines += [note, '']
        compact += [note.replace(' below', ' in the full study'), '']
    for model in MODELS:
        lines += table(model, reports[model]) + ['']
        compact += table(model, reports[model]) + ['']
    lines += ['## Baseline-relative PPL changes', '', 'Negative values favor the selected map.', '']
    for model in MODELS:
        lines += table(model, reports[model], delta=True) + ['']
    lines += ['## Eligibility before the cap and map stability', '']
    for model in MODELS:
        frozen = frozen_maps[model]
        lines += [f'### {LABELS[model]}', '',
                  '| Setting | C4/math/code sequences | Eligible blocks | Selected | Selected / eligible | Cap reached |',
                  '|---|---|---:|---:|---:|---|']
        for policy in calibration_subsets():
            s = frozen['block_statistics'][policy]
            lines.append(f'| {policy} | {"/".join(map(str,s["source_counts"]))} | '
                         f'{s["eligible_negative_score_blocks"]:,} | {s["selected_blocks"]} | '
                         f'{100*s["selected_fraction_of_eligible"]:.6f}% | {s["cap_reached"]} |')
        lines += ['', 'Jaccard overlap at equal 64-sequence budget:', '',
                  '| Setting | ' + ' | '.join(EQUAL64) + ' |', '|---|' + '---:|'*len(EQUAL64)]
        for p in EQUAL64:
            lines.append('| ' + p + ' | ' + ' | '.join(f'{frozen["overlap"][p][q]["jaccard"]:.4f}' for q in EQUAL64) + ' |')
        lines += ['']
        all_overlap = {p: frozen['overlap'][p] for p in calibration_subsets()}
        (out / f'{model}_selection.json').write_text(json.dumps(dict(
            statistics=frozen['block_statistics'], pairwise_overlap=all_overlap,
            per_module_counts={p: {n: len(v) for n, v in ms.items()} for p, ms in frozen['maps'].items()}), indent=2)+'\n')
    lines += ['## Calibration sensitivity contrasts', '',
              'Mean ΔNLL below averages the four non-C4 destinations equally. All four are outside '
              'every calibration-source pool. No configuration is elected from these measurements.', '',
              '| Model | Setting | Mean out-of-source ΔNLL vs baseline | Point gains / 4 destinations | Descriptive 2SE supported gains / 4 |',
              '|---|---|---:|---:|---:|']
    sensitivity = {}
    for model, r in reports.items():
        sensitivity[model] = {}
        for policy in calibration_subsets():
            cs = [r['contrasts'][policy][d] for d in DOMAINS[1:]]
            avg = sum(c['mean_nll'] for c in cs) / 4
            gains = sum(c['mean_nll'] < 0 for c in cs)
            supported = sum(c['mean_nll'] + c['two_se'] < 0 for c in cs)
            sensitivity[model][policy] = dict(mean_oos_delta_nll=avg, point_gains=gains, supported_gains=supported)
            lines.append(f'| {LABELS[model]} | {policy} | {avg:+.6f} | {gains} | {supported} |')
    comparisons = [(p, 'c4_64') for p in EQUAL64 if p != 'c4_64']
    comparisons += [(f'{source}_{b}', f'{source}_{a}') for source in ('c4','math','code') for a,b in ((16,32),(32,64))]
    comparisons += [(f'pooled{b}', f'pooled{a}') for a,b in ((16,32),(32,64),(64,128),(128,192))]
    comparisons += [('pooled64', p) for p in EQUAL64 if p not in ('c4_64', 'pooled64')]
    paired_comparisons = {m: {f'{a}_minus_{b}': {d: paired(r['evaluation'][a][d]['nll'], r['evaluation'][b][d]['nll'])
                            for d in DOMAINS} for a,b in comparisons} for m,r in reports.items()}
    lines += ['', 'All prespecified equal-budget source and nested sample-count contrasts, including '
              'per-dataset paired two-SE values, are in `summary.json`. All per-cell baseline '
              'contrasts are in `cells.csv`. All 17x17 overlap matrices and per-module counts are '
              'in the model selection JSONs; sparse block indices are in each prepared `maps.json`.', '',
              '## Interpretation limits', '',
              'This studies calibration source and sample count for one realized pool per model. '
              'Nested subsets share examples; no uncertainty across calibration draws is measured. '
              'The five target families were previously inspected, so this is not untouched confirmation. '
              'C4 is within-source when calibration includes C4; otherwise it is a source-transfer target. '
              'Source separation does not establish semantic-domain, near-duplicate, or pretraining independence.', '',
              'Paired two-SE intervals are descriptive evaluation-window summaries, not seed intervals '
              'or simultaneous guarantees. Adjacent WikiText windows can share articles. Counts of gains '
              'are descriptive, not independent replications. Reused cells retain their original provenance. '
              'The earlier failed source-diversity screen and inconclusive 27B pooled C4 result remain unchanged. '
              'No best setting is chosen, and the primary pooled192 rule is not retuned.', '',
              'Simulated nonhead-text-linear W4A4 reference-text loss; other operations remain native. '
              'The 27B model retains its native hybrid architecture and unquantized recurrent/conv/vision weights.']
    reused = sum(c['reused'] for c in cells)
    summary = dict(models=3, calibration_settings=17, policies_including_controls=19, datasets=list(DOMAINS),
                   total_cells=len(cells), calibration_setting_cells=3*17*5, reused_cells=reused,
                   newly_evaluated_cells=len(cells)-reused, seed_replication=False,
                   sensitivity=sensitivity, paired_comparisons=paired_comparisons,
                   source_reports=args.reports)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    compact += ['C4 is within-source for settings containing C4. WikiText, literature, science and government '
                'are source-transfer targets for all settings. The study uses previously inspected dataset '
                'families and one realized calibration pool per model; it does not establish seed robustness '
                'or a universally preferable calibration source. Evaluation-window two-SE intervals are '
                'descriptive and do not account for WikiText article dependence or multiple comparisons.', '',
                f'[Full study, eligibility and overlap tables]({out.as_posix()}/REPORT.md) · '
                f'[All per-dataset contrasts]({out.as_posix()}/cells.csv) · '
                f'[Sensitivity figure]({out.as_posix()}/sensitivity.pdf)', '']
    if args.update_report:
        path = Path('MIXFP4_REPORT.md'); text = path.read_text()
        block = START + '\n' + '\n'.join(compact) + '\n' + END
        if START in text:
            assert text.count(START) == text.count(END) == 1
            text = text[:text.index(START)] + block + text[text.index(END)+len(END):]
        else:
            anchor = '<!-- END POOLED PERFORMANCE TABLES -->'
            pos = text.index(anchor) + len(anchor)
            text = text[:pos] + '\n\n' + block + '\n' + text[pos:]
        path.write_text(text)
    # Export standard scientific figures; no uncertainty band without seed repeats.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 13), constrained_layout=True)
    for row, model in enumerate(MODELS):
        ax = axes[row, 0]
        for source, ns in [('c4', (16,32,64)), ('math', (16,32,64)), ('code', (16,32,64)),
                           ('pooled', (16,32,64,128,192))]:
            policies = [f'pooled{n}' if source == 'pooled' else f'{source}_{n}' for n in ns]
            ax.plot(ns, [sensitivity[model][p]['mean_oos_delta_nll'] for p in policies], marker='o', label=source)
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set_title(LABELS[model]); ax.set_xlabel('Calibration sequences'); ax.set_ylabel('Mean out-of-source ΔNLL')
        ax.set_xticks((16,32,64,128,192)); ax.grid(alpha=.2); ax.legend(fontsize=8)
        ax = axes[row, 1]
        matrix = [[frozen_maps[model]['overlap'][p][q]['jaccard'] for q in EQUAL64] for p in EQUAL64]
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(7), EQUAL64, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(7), EQUAL64, fontsize=8)
        ax.set_title('Selected-block Jaccard at 64 sequences')
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle('Fixed calibration subsets; causal evaluation; no seed replication', fontsize=13)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out / f'sensitivity.{ext}', dpi=180)
    plt.close(fig)
    print(json.dumps({k: v for k,v in summary.items() if k not in ('sensitivity','paired_comparisons')}), flush=True)


if __name__ == '__main__':
    main()
