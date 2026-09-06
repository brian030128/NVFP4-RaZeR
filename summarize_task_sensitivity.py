"""Summarize and plot the calibration study; run in a Slurm allocation."""
import csv
import argparse
import json
import math
import os
from pathlib import Path

if __name__ == '__main__' and not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Submit tensor processing and plotting through Slurm.')

import torch


MODELS = ['qwen3-4b', 'llama-3.1-8b-local', 'qwen3-8b',
          'llama-3.2-1b-ins-local', 'qwen3-14b', 'llama-3.1-8b-ins-local']
RULE = 'gradient_trust_0p1'


def load(path):
    with open(path) as f:
        value = json.load(f)
    assert value.get('complete'), f'Incomplete experiment: {path}'
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--more-calibration', action='store_true')
    parser.add_argument('--backtracking', action='store_true')
    parser.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit tensor processing and plotting through Slurm.')
    torch.set_num_threads(12)
    root = Path('results/task_sensitivity')
    root.mkdir(exist_ok=True)
    reports = {m: load(Path('results/task_sensitivity_sparse')/m/'report.json') for m in MODELS}
    repeats = {m: load(Path('results/task_sensitivity_replication')/m/'report.json') for m in MODELS[:2]}
    more = {seed: load(Path('results/task_sensitivity_morecalib')/
                      f'llama-3.1-8b-local_seed{seed}'/'report.json')
            for seed in [20260906, 20260907]} if args.more_calibration else {}
    backtracked = {seed: load(Path('results/task_sensitivity_backtrack')/
                             f'llama-3.1-8b-local_seed{seed}'/'report.json')
                   for seed in [20260906, 20260907]} if args.backtracking else {}
    confirmed = {m: load(Path('results/task_sensitivity_confirmation')/m/'report.json')
                 for m in MODELS[:2]} if args.confirmation else {}
    rows, anatomy = [], {}
    for model, report in reports.items():
        masks = torch.load(Path('results/task_sensitivity_sparse')/model/'policies.pt',
                           map_location='cpu', weights_only=True)
        selected = masks[RULE]
        type_map = {'model': model, 'model_commit': report.get('model_commit'), 'rule': RULE, 'weight_type_block': [8, 64],
                    'scale_block': 16, 'alpha': 1., 'default': 'E2M1',
                    'fit_sha256': report['data_sha256']['fit'],
                    'modules': {n: {'tile_grid_shape': list(v.shape),
                                    'e0m3_flat_indices': v.flatten().nonzero().flatten().tolist()}
                                for n, v in selected.items() if v.any()}}
        with open(Path('results/task_sensitivity_sparse')/model/'type_map.json', 'w') as f:
            json.dump(type_map, f, indent=2)
        parts = sorted([{'module': n, 'tiles': int(v.sum())} for n, v in selected.items() if v.any()],
                       key=lambda x: -x['tiles'])
        count = sum(p['tiles'] for p in parts)
        overlap = {r: sum(int((v & masks[r][n]).sum()) for n, v in selected.items()) / max(count, 1)
                   for r in ['hess_h1.5', 'hess_impg16_h10']}
        anatomy[model] = {'selected_tiles': count, 'overlap_fraction_with_controls': overlap,
                          'projections': parts}
        for ds in ['wikitext', 'c4']:
            for policy, values in report['final'].items():
                value = values[ds]
                diff = value.get('vs_baseline', {'mean': 0., 'se': 0., 'ppl_delta': 0.})
                rows.append({'model': model, 'dataset': ds, 'policy': policy,
                             'ppl': value['ppl'], 'delta_ppl': diff['ppl_delta'],
                             'delta_nll': diff['mean'], 'se_delta_nll': diff['se'],
                             'windows': len(value['nll']), 'selected_tiles': count if policy == RULE else ''})
    with open(root/'summary.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(root/'anatomy.json', 'w') as f:
        json.dump(anatomy, f, indent=2)

    lines = ['# Task-sensitive MixFP4 at 8x64: measured results', '',
             '**8x64 supports substantial gains, but the 16-sequence proposal needs independent validation.**',
             'A second Llama calibration seed regresses; its validation check predicts that regression',
             'before full evaluation. The calibration command can reject such a map and export NVFP4 instead.', '',
             *(['Finite-step calibration on fitting data corrects both subsequent 64-sequence Llama failures:',
                'the same acceptance rule chooses different step sizes and improves WikiText and C4 on both seeds.',
                'The tables below retain every failed fixed-budget proposal and separate the follow-up evidence.', '']
               if backtracked else []),
             'One frozen calibration rule: 16 training sequences, identity-STE final-NLL gradients,',
             'mean + 2 SE < 0 eligibility, and a total predicted reduction budget of 0.1 nats/token.',
             'Weights use 8x64 type tiles, 1x16 scale blocks, alpha=1 on both grids.',
             'Activations use nvfp4_4over6. These are W4A4 prefill evaluations at sequence length 2048,',
             'with use_cache=False for all rows. No reordering or scale search is used.', '',
             'The first two models are development models; the remaining four test the unchanged rule.',
             'All rows have contemporaneous paired baselines. Historical absolute perplexities are',
             'not substituted for these baselines. C4 uses a fixed 64-window subset.', '',
             '| Model | Tiles switched | WikiText baseline | WikiText delta | C4 baseline | C4 delta |',
             '|---|---:|---:|---:|---:|---:|']
    for m, r in reports.items():
        f = r['final']
        lines.append(f"| {m} | {r['results'][RULE]['tiles']:,} | {f['baseline']['wikitext']['ppl']:.4f} | "
                     f"{f[RULE]['wikitext']['vs_baseline']['ppl_delta']:+.4f} | "
                     f"{f['baseline']['c4']['ppl']:.4f} | {f[RULE]['c4']['vs_baseline']['ppl_delta']:+.4f} |")
    lines += ['', '## Controls', '',
              'Perplexity deltas against the same baseline, WikiText / C4.', '',
              '| Model | Frozen task rule | Fixed impg16_h10 | hess_h1.5 | Matched random |',
              '|---|---:|---:|---:|---:|']
    for m, r in reports.items():
        vals = [' / '.join(f"{r['final'][p][ds]['vs_baseline']['ppl_delta']:+.4f}" for ds in ['wikitext', 'c4'])
                for p in [RULE, 'hess_impg16_h10', 'hess_h1.5', 'matched_random']]
        lines.append('| ' + m + ' | ' + ' | '.join(vals) + ' |')
    lines += ['', '## Forecast before full evaluation', '',
              'A single check of the selected map against baseline on 16 separate calibration windows',
              'estimates its NLL change. The relative perplexity forecast is exp(delta NLL) - 1.',
              'This evaluates one chosen map, not a sweep of configurations. It forecasts WikiText',
              'relative change; it does not assume that the same shift holds on C4.', '',
              '| Model | Calibration forecast (%) | Actual WikiText change (%) | Calibration NLL SE |',
              '|---|---:|---:|---:|']
    for m, r in reports.items():
        val = r['results'][RULE]['val']
        actual = r['final'][RULE]['wikitext']['vs_baseline']['mean']
        lines.append(f"| {m} | {100*math.expm1(val['mean']):+.2f} | {100*math.expm1(actual):+.2f} | {val['se']:.5f} |")
    lines += ['', '## Independent calibration seed', '',
              'Same rule and final evaluation text, seed 20260907 instead of 20260906.', '',
              '| Model | First seed WikiText / C4 | Second seed WikiText / C4 | Second-seed tiles |',
              '|---|---:|---:|---:|']
    for m, r in repeats.items():
        a = ' / '.join(f"{reports[m]['final'][RULE][ds]['vs_baseline']['ppl_delta']:+.4f}" for ds in ['wikitext','c4'])
        b = ' / '.join(f"{r['final'][RULE][ds]['vs_baseline']['ppl_delta']:+.4f}" for ds in ['wikitext','c4'])
        lines.append(f"| {m} | {a} | {b} | {r['results'][RULE]['tiles']:,} |")
    if more:
        lines += ['', '## More calibration, unchanged rule', '',
                  'Llama-3.1-8B uses 64 fit sequences, retaining each original 16-sequence prefix',
                  'and exactly the same validation/probe windows. The score rule and 0.1 budget stay fixed.', '',
                  '| Seed | Tiles | Validation delta NLL | Validation supports improvement? | WikiText delta | C4 delta |',
                  '|---|---:|---:|---|---:|---:|']
        for seed, r in more.items():
            old = reports['llama-3.1-8b-local'] if seed == 20260906 else repeats['llama-3.1-8b-local']
            assert r['data_sha256']['val'] == old['data_sha256']['val']
            assert r['data_sha256']['probe'] == old['data_sha256']['probe']
            v = r['results'][RULE]['val']
            supported = v['mean'] + 2*v['se'] < 0
            lines.append(f"| {seed} | {r['results'][RULE]['tiles']:,} | {v['mean']:+.6f} | "
                         f"{'yes' if supported else 'no'} | "
                         f"{r['final'][RULE]['wikitext']['vs_baseline']['ppl_delta']:+.4f} | "
                         f"{r['final'][RULE]['c4']['vs_baseline']['ppl_delta']:+.4f} |")
    if backtracked:
        lines += ['', '## Calibrating the finite step on fitting data', '',
                  'Exploratory Llama follow-up, reusing each 64-sequence score set. Starting at 0.1,',
                  'halve the predicted-reduction budget until actual fit improvement reaches 25% of',
                  'prediction and fit mean + 2 SE is negative; at most eight attempts, then NVFP4.',
                  'The chosen proposal receives one separate validation check. Validation/test loss',
                  'does not choose the step size. These are development follow-ups, not untouched-model tests.', '',
                  '| Seed | Fit attempts | Accepted budget | Tiles | Validation delta NLL | Validation supports improvement? | WikiText delta | C4 delta |',
                  '|---|---:|---:|---:|---:|---|---:|---:|']
        for seed, r in backtracked.items():
            if more:
                assert r['data_sha256'] == more[seed]['data_sha256']
            key = 'gradient_trust_backtracking'
            h = r['proposal_fit_calibration']
            budget = h[-1]['budget'] if h and h[-1]['accepted'] else 0.
            v = r['results'][key]['val']
            supported = v['mean'] + 2*v['se'] < 0
            lines.append(f"| {seed} | {len(h)} | {budget:.6f} | {r['results'][key]['tiles']:,} | "
                         f"{v['mean']:+.6f} | {'yes' if supported else 'no'} | "
                         f"{r['final'][key]['wikitext']['vs_baseline']['ppl_delta']:+.4f} | "
                         f"{r['final'][key]['c4']['vs_baseline']['ppl_delta']:+.4f} |")
    if confirmed:
        lines += ['', '## Third calibration seed, frozen finite-step procedure', '',
                  'Seed 20260908, fresh 64-sequence scores, with the identical backtracking constants.',
                  'These runs were declared before inspecting the preceding backtracking validation/test results.',
                  'They test calibration-seed stability on the development models; the test text is reused.', '',
                  '| Model | Fit attempts | Accepted budget | Tiles | Validation delta NLL | Validation supports improvement? | WikiText delta | C4 delta |',
                  '|---|---:|---:|---:|---:|---|---:|---:|']
        for m, r in confirmed.items():
            key = 'gradient_trust_backtracking'
            h = r['proposal_fit_calibration']
            budget = h[-1]['budget'] if h and h[-1]['accepted'] else 0.
            v = r['results'][key]['val']
            supported = v['mean'] + 2*v['se'] < 0
            lines.append(f"| {m} | {len(h)} | {budget:.6f} | {r['results'][key]['tiles']:,} | "
                         f"{v['mean']:+.6f} | {'yes' if supported else 'no'} | "
                         f"{r['final'][key]['wikitext']['vs_baseline']['ppl_delta']:+.4f} | "
                         f"{r['final'][key]['c4']['vs_baseline']['ppl_delta']:+.4f} |")
        lines += ['', '## Calibration command verification', '']
        for m in MODELS[:2]:
            folder = Path('results/task_sensitivity_gatecheck')/m
            gate = load(folder/'report.json')
            assert gate['export_decision'] == ('accepted' if m == 'qwen3-4b' else 'fallback_nvfp4')
            assert gate['data_sha256'] == repeats[m]['data_sha256']
            candidate = json.loads((folder/'candidate_type_map.json').read_text())
            reference = json.loads((Path('results/task_sensitivity_replication')/m/'type_map.json').read_text())
            assert candidate['modules'] == reference['modules']
            exported = json.loads((folder/'type_map.json').read_text())
            assert exported['modules'] == (candidate['modules'] if gate['export_decision'] == 'accepted' else {})
            lines.append(f"* {m}, seed 20260907: `{gate['export_decision']}`, "
                         f"{gate['selection']['tiles']:,} exported tiles; proposal matches the research map exactly.")
    proposal_runs = [(m+'/seed20260906/fit16', r, RULE) for m, r in reports.items()]
    proposal_runs += [(m+'/seed20260907/fit16', r, RULE) for m, r in repeats.items()]
    proposal_runs += [(f'llama-3.1-8b/seed{seed}/fit64', r, RULE) for seed, r in more.items()]
    proposal_runs += [(f'llama-3.1-8b/seed{seed}/fit64/backtracked', r, 'gradient_trust_backtracking')
                      for seed, r in backtracked.items()]
    proposal_runs += [(m+'/seed20260908/fit64/backtracked', r, 'gradient_trust_backtracking')
                      for m, r in confirmed.items()]
    accepted = [(name, r, key) for name, r, key in proposal_runs
                if r['results'][key]['val']['mean'] + 2*r['results'][key]['val']['se'] < 0]
    harmed = [name for name, r, key in accepted if r['final'][key]['wikitext']['vs_baseline']['mean'] > 0]
    c4_positive = [r['final'][key]['c4']['vs_baseline'] for _, r, key in accepted
                   if r['final'][key]['c4']['vs_baseline']['mean'] > 0]
    c4_clear = sum(d['mean'] - 2*d['se'] > 0 for d in c4_positive)
    errors = [abs(100*math.expm1(r['results'][key]['val']['mean']) -
                  100*math.expm1(r['final'][key]['wikitext']['vs_baseline']['mean']))
              for _, r, key in proposal_runs]
    proposal_rows = []
    for name, r, key in proposal_runs:
        p = r['results'][key]
        v = p['val']
        proposal_rows.append({'run': name, 'policy': key, 'seed': r['args']['seed'],
                              'fit_sequences': r['args']['fit'], 'tiles': p['tiles'],
                              'gradient_prediction_nll': p['predicted_delta_nll'],
                              'actual_fit_delta_nll': p['fit']['mean'],
                              'validation_delta_nll': v['mean'], 'validation_se': v['se'],
                              'validation_accepts': v['mean'] + 2*v['se'] < 0,
                              'forecast_relative_ppl_percent': 100*math.expm1(v['mean']),
                              'wikitext_relative_ppl_percent': 100*math.expm1(r['final'][key]['wikitext']['vs_baseline']['mean']),
                              'wikitext_delta_ppl': r['final'][key]['wikitext']['vs_baseline']['ppl_delta'],
                              'c4_delta_ppl': r['final'][key]['c4']['vs_baseline']['ppl_delta']})
    with open(root/'proposal_summary.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=list(proposal_rows[0]))
        writer.writeheader()
        writer.writerows(proposal_rows)
    lines += ['', '## Validation gate across reported proposal runs', '',
              f"The fixed check accepts {len(accepted)} of {len(proposal_runs)} proposals. "
              f"{len(harmed)} accepted proposals have a positive WikiText PPL delta in this record.",
              f"On C4, {'one accepted proposal has a positive PPL point estimate' if len(c4_positive) == 1 else str(len(c4_positive)) + ' accepted proposals have positive PPL point estimates'}; "
              f"{c4_clear} have paired delta NLL minus 2 SE above zero. Calibration targets WikiText, "
              'so cross-domain behavior needs a separate check.',
              f"The mean absolute error of the validation forecast is {sum(errors)/len(errors):.2f} "
              'percentage points of relative WikiText PPL change, including rejected proposals.',
              'This is a descriptive check over related models, repeated test text, and development',
              'follow-ups. It is not a bound on unseen-model failure probability. In particular,',
              'rejection means insufficient support on the calibration distribution, not proof',
              'that a map is harmful everywhere.', '',
              'All proposals, including failures, are tabulated in [proposal_summary.csv](proposal_summary.csv).', '',
              '![One-map validation forecast](forecast.png)']
    lines += ['', '## Calibration cost', '',
              'One H100 per model. Gradient-collection time excludes model/data loading and final evaluation.',
              'The first pair use their fresh-score replication timings; initial final-test runs reused scores.', '',
              '| Model | Gradient collection (seconds) | Peak allocated GPU memory (GiB, whole job) |',
              '|---|---:|---:|']
    for m, r in reports.items():
        measured = repeats.get(m, r)
        lines.append(f"| {m} | {measured['scores_seconds']:.1f} | {measured['gpu_peak_allocated_gb']:.1f} |")
    if confirmed:
        lines += ['', 'For the fresh 64-sequence confirmation runs:', '',
                  '| Model | Gradient collection (seconds) | Fit proposals evaluated |',
                  '|---|---:|---:|']
        for m, r in confirmed.items():
            lines.append(f"| {m} | {r['scores_seconds']:.1f} | {len(r['proposal_fit_calibration'])} |")
        lines += ['', 'Each fit proposal costs 64 forwards, plus one 64-window fit baseline.',
                  'Independent validation costs 16 forwards for the map and 16 for baseline.',
                  'The research runs additionally repeat fit evaluation and run full benchmark tests.']
    lines += ['', '## Cache-path check', '',
              'The repeated maps are also checked on their same validation windows with caching enabled.',
              'Primary final-test measurements above keep caching disabled.', '',
              '| Model | Cache-off validation delta NLL | Cache-on validation delta NLL |',
              '|---|---:|---:|']
    for m, r in repeats.items():
        lines.append(f"| {m} | {r['results'][RULE]['val']['mean']:+.6f} | "
                     f"{r['cache_robustness']['selected_vs_baseline']['mean']:+.6f} |")
    lines += ['', '## Scope and limitations', '',
              '* This predicts promising finite interventions, not exact final perplexity. The 0.1 budget is a heuristic, not a bound on actual loss.',
              '* Confidence margins are sequence-level stability checks, not simultaneous statistical guarantees over millions of tiles.',
              '* First-order scores cannot safely be summed over arbitrary numbers of switches: the initial broad policies failed.',
              '* WikiText and C4 evaluation windows may share documents; NLL standard errors are descriptive.',
              '* This is fake-quantization accuracy evidence on H100, not a benchmark of an E0M3 hardware kernel.',
              '* Token-by-token decoding and other evaluation domains are not established by these prefill results.',
              '* Six models from two families and limited calibration-seed repeats do not establish universal behavior.', '',
              '![Perplexity comparison](comparison.png)', '',
              'Whiskers transform paired NLL mean ± 1.96 SE; they are descriptive, not formal guarantees.', '',
              'See [PROTOCOL.md](PROTOCOL.md), [summary.csv](summary.csv), and [anatomy.json](anatomy.json).', '']
    (root/'REPORT.md').write_text('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    labels = [m.replace('-local','').replace('-ins','-instruct') for m in MODELS]
    colors = ['#1976b9', '#909090', '#d6a14a']
    for ax, ds in zip(axes, ['wikitext', 'c4']):
        for j, policy in enumerate([RULE, 'hess_impg16_h10', 'matched_random']):
            values, lower, upper = [], [], []
            for r in reports.values():
                d = r['final'][policy][ds]['vs_baseline']
                mu, se = d['mean'], d['se']
                v = 100*math.expm1(mu)
                values.append(v)
                lower.append(v-100*math.expm1(mu-1.96*se))
                upper.append(100*math.expm1(mu+1.96*se)-v)
            ax.bar(np.arange(len(MODELS))+(j-1)*.25, values, .24, color=colors[j],
                   yerr=[lower, upper], capsize=2, label=['Task-sensitive', 'Fixed impg16_h10', 'Matched random'][j])
        ax.scatter(np.arange(2)-.25,
                   [100*math.expm1(repeats[m]['final'][RULE][ds]['vs_baseline']['mean']) for m in MODELS[:2]],
                   marker='D', s=30, color='#b12828', zorder=5, label='Task rule, second seed (ungated)')
        ax.axhline(0, color='black', linewidth=.7)
        ax.set_xticks(np.arange(len(MODELS)), labels, rotation=35, ha='right')
        ax.set_title('WikiText test' if ds == 'wikitext' else 'C4 (64 windows)')
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Perplexity change vs paired NVFP4 baseline (%)\nNegative is better')
    axes[1].legend(frameon=False)
    fig.suptitle('16-sequence proposals, 8×64 weight type tiles, W4A4 simulation')
    fig.tight_layout()
    fig.savefig(root/'comparison.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    forecast_values, actual_values = [], []
    for name, r, key in proposal_runs:
        v = r['results'][key]['val']
        predicted = 100*math.expm1(v['mean'])
        actual = 100*math.expm1(r['final'][key]['wikitext']['vs_baseline']['mean'])
        supported = v['mean'] + 2*v['se'] < 0
        forecast_values.append(predicted)
        actual_values.append(actual)
        ax.scatter(predicted, actual, color='#1976b9' if supported else '#b12828',
                   marker='o' if supported else 'x', s=50)
    low = min(forecast_values+actual_values)-.5
    high = max(forecast_values+actual_values)+.5
    ax.plot([low, high], [low, high], '--', color='gray', linewidth=1)
    ax.axhline(0, color='black', linewidth=.6)
    ax.axvline(0, color='black', linewidth=.6)
    ax.scatter([], [], color='#1976b9', label='Validation accepts')
    ax.scatter([], [], color='#b12828', marker='x', label='Validation rejects')
    ax.set(xlabel='16-window validation forecast: relative PPL change (%)',
           ylabel='Full WikiText test: relative PPL change (%)',
           title='Forecasting one proposed 8×64 map before full evaluation')
    ax.legend(frameon=False)
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(root/'forecast.png', dpi=180)
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
