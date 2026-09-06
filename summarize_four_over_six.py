"""Summarize matched FourOverSix comparisons and mechanism evidence in Slurm."""
import os
assert os.environ.get('SLURM_JOB_ID'), 'Submit through Slurm.'
import csv
import json
import math
from pathlib import Path
import statistics
import torch
from quantize.risk_certificate import retention_certificate, bayes_sequence_bound
import analyze_task_sensitivity as task


def load(path):
    value = json.loads(Path(path).read_text())
    assert value.get('complete'), path
    return value


def paired(a, b):
    assert len(a) == len(b)
    diff = [x-y for x, y in zip(a, b)]
    return statistics.mean(diff), statistics.stdev(diff)/math.sqrt(len(diff))


def main():
    torch.set_num_threads(12)
    root = Path('results/task_sensitivity_four_over_six')
    old = load('results/task_sensitivity_qwen38/seed20260909/report.json')
    old2 = load('results/task_sensitivity_qwen38/seed20260910/report.json')
    controls = load('results/task_sensitivity_qwen38/controls/report.json')
    primary = {seed: load(root/f'seed{seed}'/'report.json') for seed in (20260912, 20260913)}
    rule = 'gradient_trust_backtracking'
    rows, certificates = [], {}
    lines = ['# Gains beyond FourOverSix', '',
             'Matched native Qwen3.8-27B, W4A4, 8x64 weight type tiles. The new maps',
             'preserve FourOverSix E2M1 scaling and add E0M3 alpha=1 switches.',
             'All test results include the proposed map even if validation rejects it.', '',
             '| Seed | Validation | Tiles | Dataset | FourOverSix PPL | Proposed PPL | Delta PPL | Delta NLL ± 2 SE |',
             '|---|---|---:|---|---:|---:|---:|---:|']
    for seed, report in primary.items():
        entry = report['results'][rule]
        for ds in ('wikitext', 'c4'):
            base = report['final']['baseline'][ds]
            candidate = report['final'][rule][ds]
            delta = candidate['vs_baseline']
            rows.append({'model': 'Qwen/Qwen3.8-27B', 'seed': seed, 'dataset': ds,
                         'decision': report['export_decision'], 'tiles': entry['tiles'],
                         'baseline_ppl': base['ppl'], 'candidate_ppl': candidate['ppl'],
                         'delta_ppl': delta['ppl_delta'], 'delta_nll': delta['mean'], 'se': delta['se']})
            lines.append(f'| {seed} | {report["export_decision"]} | {entry["tiles"]} | {ds} | '
                         f'{base["ppl"]:.6f} | {candidate["ppl"]:.6f} | {delta["ppl_delta"]:+.6f} | '
                         f'{delta["mean"]:+.6f} ± {2*delta["se"]:.6f} |')
        # A deliberately conservative diagnostic, NOT a deployment certificate:
        # clip mean window NLL to [0,10], and explicitly do not assert IID.
        fitval = report['baseline']['val']
        chosenval = entry['val_nll']
        cert = retention_certificate([min(x, 10.) for x in chosenval], [min(x, 10.) for x in fitval],
                                     {'four_over_six': [min(x, 10.) for x in fitval]},
                                     bound=10., rho=.5, delta=.05)
        cert['deployable_certificate'] = False
        cert['reason'] = 'IID assumptions not established; clipped window loss is not raw perplexity.'
        certificates[str(seed)] = cert
    lines.extend(['', 'All target configurations on the same held-out data:', '',
                  '| Configuration | WikiText PPL | C4 PPL |', '|---|---:|---:|'])
    references = {'plain_nvfp4': old['final']['baseline'],
                  'four_over_six': primary[20260912]['final']['baseline'],
                  'old_sparse_seed09': old['final'][rule], 'old_sparse_seed10': old2['final'][rule],
                  'weight_mse_a1': controls['final']['weight_mse']}
    for name, values in references.items():
        lines.append(f'| {name} | {values["wikitext"]["ppl"]:.6f} | {values["c4"]["ppl"]:.6f} |')
    lines.extend(['', 'Paired comparisons against the reference bank (negative favors the new map):', '',
                  '| New seed | Dataset | Reference | Delta NLL ± 2 SE |',
                  '|---|---|---|---:|'])
    for seed, report in primary.items():
        for ds in ('wikitext', 'c4'):
            for name, values in references.items():
                mean, se = paired(report['final'][rule][ds]['nll'], values[ds]['nll'])
                lines.append(f'| {seed} | {ds} | {name} | {mean:+.6f} ± {2*se:.6f} |')
    lines.extend(['', 'Descriptive retention of each positive reference gain, measured in NLL',
                  'relative to plain NVFP4. Ratios are unstable for very small reference gains;',
                  'these held-out ratios are not universal certificates.', '',
                  '| New seed | Dataset | Reference | Fraction of reference gain retained |',
                  '|---|---|---|---:|'])
    mixture = {}
    for seed, report in primary.items():
        for ds in ('wikitext', 'c4'):
            baseline_mean = statistics.mean(old['final']['baseline'][ds]['nll'])
            gain = baseline_mean-statistics.mean(report['final'][rule][ds]['nll'])
            for name, values in references.items():
                refgain = baseline_mean-statistics.mean(values[ds]['nll'])
                if refgain > 0:
                    lines.append(f'| {seed} | {ds} | {name} | {gain/refgain:.3f} |')
    for ds in ('wikitext', 'c4'):
        # Diagnostic theoretical ensemble likelihood, not a single-map result.
        totals = [sum(v[ds]['nll'])*2047 for v in references.values()]
        result = bayes_sequence_bound(totals)
        result['valid_online_ensemble_measurement'] = False
        result['qualification'] = ('Algebraic comparison of stored prefill losses only. No causal ensemble was run; '
                                   'dynamic tensor activation scales can depend on later positions in a window.')
        result['average_nll_regret_bound'] = result['universal_regret_bound']/(len(old['final']['baseline'][ds]['nll'])*2047)
        mixture[ds] = result
    lines.extend(['', 'Four-model transfer panel, with the same frozen procedure:', '',
                  '| Model | Validation | Dataset | FourOverSix PPL | Proposed PPL | Delta NLL ± 2 SE |',
                  '|---|---|---|---:|---:|---:|'])
    for model in ('qwen3-4b', 'llama-3.1-8b-local', 'qwen3-8b', 'llama-3.1-8b-ins-local'):
        directory = Path('results/task_sensitivity_four_panel')/model
        report = load(directory/'report.json')
        entry = report['results'][rule]
        decision = 'accepted' if entry['val']['mean']+2*entry['val']['se'] < 0 else 'fallback_four_over_six'
        masks = torch.load(directory/'policies.pt', map_location='cpu', weights_only=True)[rule]
        task.export_type_map(directory/'candidate_type_map.json', model, masks, report['data_sha256']['fit'],
                             report['model_commit'], rule, 'nvfp4_4over6')
        safe = masks if decision == 'accepted' else {n: torch.zeros_like(m) for n, m in masks.items()}
        task.export_type_map(directory/'type_map.json', model, safe, report['data_sha256']['fit'],
                             report['model_commit'], rule, 'nvfp4_4over6')
        for ds in ('wikitext', 'c4'):
            base = report['final']['baseline'][ds]
            value = report['final'][rule][ds]
            delta = value['vs_baseline']
            rows.append({'model': model, 'seed': 20260912, 'dataset': ds, 'decision': decision,
                         'tiles': entry['tiles'], 'baseline_ppl': base['ppl'], 'candidate_ppl': value['ppl'],
                         'delta_ppl': delta['ppl_delta'], 'delta_nll': delta['mean'], 'se': delta['se']})
            lines.append(f'| {model} | {decision} | {ds} | {base["ppl"]:.6f} | {value["ppl"]:.6f} | '
                         f'{delta["mean"]:+.6f} ± {2*delta["se"]:.6f} |')
    mechanism = load(root/'mechanism'/'report.json')
    assert mechanism['four_over_six_export_all_weights_exact']
    lines.extend(['', 'Prespecified eight-window probe ablations:', '',
                  '| Baseline | Map origin | Intervention | Tiles | Delta NLL ± 2 SE |',
                  '|---|---|---|---:|---:|'])
    for baseline, entry in mechanism['results'].items():
        for origin, experiments in entry['maps'].items():
            for name, value in experiments.items():
                delta = value['vs_baseline']
                lines.append(f'| {baseline} | {origin} | {name} | {value["tiles"]} | '
                             f'{delta["mean"]:+.6f} ± {2*delta["se"]:.6f} |')
    geometry = json.loads((root/'mechanism'/'local_error_geometry.json').read_text())
    lines.extend(['', 'Paired contrasts on the same probe windows (negative favors the full map):', '',
                  '| Baseline | Map origin | Full minus intervention | Delta NLL ± 2 SE |',
                  '|---|---|---|---:|'])
    for baseline, entry in mechanism['results'].items():
        for origin, experiments in entry['maps'].items():
            for name in ('channels_3968_4031_only', 'random_rows_same_columns'):
                mean, se = paired(experiments['full']['nll'], experiments[name]['nll'])
                lines.append(f'| {baseline} | {origin} | {name} | {mean:+.6f} ± {2*se:.6f} |')
    anatomy = mechanism['activation_anatomy']
    region_peaks = sum(3968 <= v['peak_channel'] < 4032 for v in anatomy.values())
    median_energy = statistics.median(v['channel_region_energy_fraction'] for v in anatomy.values())
    lines.extend(['', f'Among {len(anatomy)} projections with 5120 input channels, {region_peaks}',
                  'have their largest measured input energy in channels 3968–4031.',
                  f'The median energy fraction in this 64-channel region is {median_energy:.6f}.',
                  'These are measurements on eight NVFP4-reference probe windows, not a universal channel index rule.'])
    lines.extend(['', 'Isolated-tile reconstruction geometry (common NVFP4-reference probe inputs):', '',
                  '| Error baseline | Tiles examined | Clear adverse-input witnesses | Lower calibration output error |',
                  '|---|---:|---:|---:|'])
    for baseline in ('nvfp4', 'four_over_six'):
        items = [v for v in geometry if v['baseline'] == baseline]
        adverse = sum(v['adverse_input_witness'] > 1e-8*v['gram_nuclear_norm'] for v in items)
        improving = sum(v['calibration_output_error_change'] < 0 for v in items)
        lines.append(f'| {baseline} | {len(items)} | {adverse} | {improving} |')
    lines.extend(['', 'These local witnesses refute all-input reconstruction dominance for the',
                  'identified tiles. They do not imply that those directions are common in',
                  'deployment, or that local reconstruction error alone determines task loss.'])
    lines.extend(['', 'Read [GUARANTEE.md](GUARANTEE.md) for the precise gain-retention property,',
                  'conditional finite-sample certificate, unconditional fixed-map obstruction,',
                  'and online-mixture regret guarantee. The latter requires multiple predictive',
                  'configurations and is not a static MixFP4 implementation.',
                  'All numerical 2-SE intervals here are descriptive, not universal guarantees.'])
    (root/'certificate_diagnostics.json').write_text(json.dumps(certificates, indent=2)+'\n')
    (root/'ensemble_bound_diagnostic.json').write_text(json.dumps(mixture, indent=2)+'\n')
    with (root/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
