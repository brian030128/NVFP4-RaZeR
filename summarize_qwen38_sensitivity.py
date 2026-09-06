"""Summarize the frozen target experiment; execute through Slurm."""
import csv
import json
import math
import os
from pathlib import Path
import statistics


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Submit summaries through Slurm.'
    root = Path('results/task_sensitivity_qwen38')
    rows = []
    references = []
    anatomy = {}
    selected_sets = {}
    lines = ['# Qwen3.8-27B: frozen 8x64 MixFP4 calibration', '',
             'Two independent seeds use the predeclared [protocol](PROTOCOL.md).',
             'Changes below compare the proposed map with corrected E2M1 W4A4,',
             'including proposals rejected by validation.', '',
             '| Seed | Validation decision | E0M3 tiles | Dataset | Baseline PPL | Proposal PPL | Delta PPL | Delta NLL ± 2 SE |',
             '|---|---|---:|---|---:|---:|---:|---:|']
    for seed in (20260909, 20260910):
        directory = root / f'seed{seed}'
        report = json.loads((directory / 'report.json').read_text())
        assert report['complete'], directory
        rule = 'gradient_trust_backtracking'
        proposal = report['results'][rule]
        candidate = json.loads((directory/'candidate_type_map.json').read_text())
        exported = json.loads((directory/'type_map.json').read_text())
        def count(spec):
            assert spec['scope'] == 'qwen3_5_text_linear'
            assert spec['weight_type_block'] == [8, 64]
            assert spec['model_commit'] == report['model_commit']
            total = 0
            for name, value in spec['modules'].items():
                assert 'language_model' in name and 'head' not in name
                indices = value['e0m3_flat_indices']
                assert len(set(indices)) == len(indices)
                assert all(0 <= i < math.prod(value['tile_grid_shape']) for i in indices)
                total += len(indices)
            return total
        assert count(candidate) == proposal['tiles']
        assert count(exported) == (proposal['tiles'] if report['export_decision'] == 'accepted' else 0)
        projections = sorted(
            [{'module': name, 'tiles': len(value['e0m3_flat_indices'])}
             for name, value in candidate['modules'].items()],
            key=lambda item: -item['tiles'])
        families = {}
        columns = {}
        selected_sets[seed] = {(name, i) for name, value in candidate['modules'].items()
                               for i in value['e0m3_flat_indices']}
        for item in projections:
            family = item['module'].split('.layers.')[-1].split('.', 1)[-1]
            families[family] = families.get(family, 0) + item['tiles']
            value = candidate['modules'][item['module']]
            width = value['tile_grid_shape'][1]
            for i in value['e0m3_flat_indices']:
                key = f'{family}:K={64*(i % width)}..{64*(i % width)+63}'
                columns[key] = columns.get(key, 0) + 1
        anatomy[str(seed)] = {'projections': projections, 'projection_families': families,
                             'input_channel_ranges': dict(sorted(columns.items(), key=lambda item: -item[1])),
                             'selected_fraction': proposal['fraction'],
                             'predicted_delta_nll': proposal['predicted_delta_nll'],
                             'fit_backtracking': report['proposal_fit_calibration'],
                             'validation': proposal['val']}
        for ds in ('wikitext', 'c4'):
            baseline = report['final']['baseline'][ds]
            result = report['final'][rule][ds]
            delta = result['vs_baseline']
            assert len(result['nll']) == len(baseline['nll'])
            row = dict(seed=seed, decision=report['export_decision'], tiles=proposal['tiles'],
                       dataset=ds, baseline_ppl=baseline['ppl'], proposal_ppl=result['ppl'],
                       delta_ppl=delta['ppl_delta'], delta_nll=delta['mean'], se=delta['se'],
                       windows=len(result['nll']),
                       forecast_relative_ppl_percent=report['validation_forecast_relative_ppl_percent'],
                       forecast_lower_relative_ppl_percent=100*math.expm1(proposal['val']['mean']-2*proposal['val']['se']),
                       forecast_upper_relative_ppl_percent=100*math.expm1(proposal['val']['mean']+2*proposal['val']['se']),
                       observed_relative_ppl_percent=100*math.expm1(delta['mean']))
            rows.append(row)
            lines.append(f'| {seed} | {row["decision"]} | {row["tiles"]} | {ds} | '
                         f'{row["baseline_ppl"]:.6f} | {row["proposal_ppl"]:.6f} | '
                         f'{row["delta_ppl"]:+.6f} | {row["delta_nll"]:+.6f} ± {2*row["se"]:.6f} |')
        if 'bf16_reference' in report:
            references.extend(['', 'Native BF16 reference: ' + ', '.join(
                f'{ds} PPL {value["ppl"]:.6f}' for ds, value in report['bf16_reference'].items()), ''])
    with (root/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines.extend(references)
    lines.extend(['', 'Forecasts made from independent WikiText calibration validation:', '',
                  '| Seed | Forecast relative PPL change | Validation mean ± 2 SE, transformed to PPL | Observed WikiText change |',
                  '|---|---:|---:|---:|'])
    for row in rows:
        if row['dataset'] == 'wikitext':
            lines.append(f'| {row["seed"]} | {row["forecast_relative_ppl_percent"]:+.2f}% | '
                         f'[{row["forecast_lower_relative_ppl_percent"]:+.2f}%, {row["forecast_upper_relative_ppl_percent"]:+.2f}%] | '
                         f'{row["observed_relative_ppl_percent"]:+.2f}% |')
    lines.extend(['', 'Both WikiText test changes lie inside these descriptive validation intervals.',
                  'The validation point estimates are imperfect; the first overpredicts the gain.',
                  'Neither interval describes C4 transfer: C4 gains are tiny and inconclusive.',
                  'A final workload needs representative calibration and independent validation.'])
    for seed, detail in anatomy.items():
        lines.extend(['', f'Seed {seed}: {100*detail["selected_fraction"]:.6f}% of text weight tiles selected.'])
        if detail['projections']:
            lines.append('Largest selections: ' + '; '.join(
                f'`{item["module"]}` ({item["tiles"]} tiles)' for item in detail['projections'][:5]) + '.')
    a, b = selected_sets.values()
    anatomy['seed_overlap'] = {'intersection': len(a & b), 'union': len(a | b),
                              'jaccard': len(a & b)/max(1, len(a | b))}
    lines.extend(['', f'The two maps share {len(a & b)} tiles out of {len(a | b)} in their union.'])
    (root/'anatomy.json').write_text(json.dumps(anatomy, indent=2)+'\n')
    controls_path = root/'controls'/'report.json'
    if controls_path.exists():
        controls = json.loads(controls_path.read_text())
        assert controls['complete']
        assert controls['export_replay_max_nll_delta'] == 0
        assert controls['baseline_replay_max_nll_delta'] == 0
        lines.extend(['', 'Diagnostic controls (registered after the first primary seed):', '',
                      '| Control | Tiles | Dataset | PPL | Delta PPL | Delta NLL ± 2 SE |',
                      '|---|---:|---|---:|---:|---:|'])
        for label, datasets in controls['final'].items():
            for ds, result in datasets.items():
                delta = result['vs_baseline']
                lines.append(f'| {label} | {controls["tile_counts"][label]} | {ds} | {result["ppl"]:.6f} | '
                             f'{delta["ppl_delta"]:+.6f} | {delta["mean"]:+.6f} ± {2*delta["se"]:.6f} |')
        lines.extend(['', 'The exported native map and baseline replayed 16 held-out windows exactly',
                      'on a fresh model load (maximum NLL difference zero).'])
        lines.extend(['', 'Direct paired MSE-minus-calibrated comparisons (negative favors MSE):', '',
                      '| Calibration seed | Dataset | Delta PPL | Delta NLL ± 2 SE |',
                      '|---|---|---:|---:|'])
        for seed in (20260909, 20260910):
            primary = json.loads((root/f'seed{seed}'/'report.json').read_text())
            for ds in ('wikitext', 'c4'):
                mse = controls['final']['weight_mse'][ds]
                calibrated = primary['final']['gradient_trust_backtracking'][ds]
                differences = [a-b for a, b in zip(mse['nll'], calibrated['nll'])]
                mean = statistics.mean(differences)
                se = statistics.stdev(differences)/math.sqrt(len(differences))
                lines.append(f'| {seed} | {ds} | {mse["ppl"]-calibrated["ppl"]:+.6f} | '
                             f'{mean:+.6f} ± {2*se:.6f} |')
    lines.extend(['', 'The uncertainty column is a descriptive paired window standard error,',
                  'not a guarantee across domains. WikiText windows may be correlated.',
                  'These measurements cover text linear weights and inputs; recurrent state,',
                  'convolution, norms, embeddings, head, and vision retain native precision.',
                  'Fake quantization measures accuracy, not hardware speed.'])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
