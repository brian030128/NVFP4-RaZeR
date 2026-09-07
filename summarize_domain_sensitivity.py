"""Generate paired cross-domain results and diagnostic tables; Slurm only."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Submit summaries through Slurm too.')
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def paired(a, b):
    assert len(a) == len(b) and len(a) >= 2
    d = [x-y for x, y in zip(a, b)]
    mu, se = statistics.mean(d), statistics.stdev(d)/len(d)**.5
    return mu, se


ap = argparse.ArgumentParser()
ap.add_argument('--partial', action='store_true')
args = ap.parse_args()
root = Path('results/task_sensitivity_domains')
paths = [root/f'seed{s}/report.json' for s in [20260912, 20260913]]
paths += [root/'panel'/m/'report.json' for m in ['qwen3-4b', 'llama-3.1-8b-local']]
paths += [root/'teacher'/m/'report.json' for m in ['qwen3-4b', 'llama-3.1-8b-local']]
reports = []
for p in paths:
    if p.exists():
        r = json.loads(p.read_text())
        if r['complete']:
            reports.append((p, r))
        elif not args.partial:
            raise AssertionError(f'Incomplete: {p}')
    elif not args.partial:
        raise AssertionError(f'Missing: {p}')
stress_path = root/'stress.json'
stress = json.loads(stress_path.read_text()) if stress_path.exists() else None
if not args.partial:
    assert stress and stress['complete'] and len(reports) == 6
rows = []
lines = ['# Cross-domain E0M3/E2M1 selection experiments', '',
         'All changes are relative to a matched FourOverSix W4A4 baseline. Negative '
         'paired NLL or relative perplexity change is better. All candidate maps were '
         'frozen before held-out evaluation. Rejected exports retain the baseline.', '',
         f'Completed experiment runs: {len(reports)}/6 (three models; teacher follow-ups reuse two panel models/seeds). '
         f'Target math/code stress complete: {bool(stress and stress["complete"])}.', '',
         'The rules share candidate formats, type tile 8x64, 64 fitting windows, '
         'a 0.1 predicted-loss budget in nats/token, fit backtracking and independent NLL validation. '
         'Mixed pools 32 WikiText + 32 C4 windows; consensus requires stable negative '
         'scores and measured joint improvement in each domain separately.', '',
         'Teacher follow-ups replace the scoring/fitting objective with full-precision '
         'teacher KL, while independent acceptance still checks actual NLL. '
         'Their fitting diagnostics motivated this prespecified follow-up; see TEACHER_PROTOCOL.md.', '',
         '## Candidate performance', '',
         'Absolute PPL change from mean example NLL. A reduction of 0.01 PPL is '
         'a worthwhile gain under the user-specified criterion. Brackets are paired '
         'mean ± two SE transformed with exp and scaled by baseline PPL; descriptive, '
         'not simultaneous confidence bounds. Practical magnitude and statistical '
         'uncertainty are reported separately; calibration gates remain as prespecified.', '',
         '| Model / seed | Rule | Tiles | Export accepted | WikiText ΔPPL | C4 ΔPPL | Math text ΔPPL | Code text ΔPPL |',
         '|---|---|---:|---|---:|---:|---:|---:|']
for path, r in reports:
    model = r.get('model', 'Qwen/Qwen3.8-27B')
    model = model.replace('Qwen/', '')
    baseline = r['final']['baseline']
    for rule, p in r['policies'].items():
        cells = []
        for d in ['wikitext', 'c4', 'math', 'code']:
            if d in r['final'][rule]:
                result, ref = r['final'][rule][d], baseline[d]
            elif stress and stress['complete'] and f'{r["seed"]}/{rule}' in stress['results']:
                result = stress['results'][f'{r["seed"]}/{rule}'][d]
                ref = stress['results']['baseline'][d]
            else:
                cells.append('pending')
                continue
            mu, se = paired(result['nll'], ref['nll'])
            assert abs(mu-result['vs_baseline']['mean']) < 1e-10
            rel = 100*math.expm1(mu)
            lo, hi = [100*math.expm1(mu+x*se) for x in [-2, 2]]
            dppl = result['ppl']-ref['ppl']
            plo, phi = [ref['ppl']*v/100 for v in [lo, hi]]
            cells.append(f'{dppl:+.4f} [{plo:+.4f}, {phi:+.4f}]')
            rows.append({'model': model, 'seed': r['seed'], 'rule': rule, 'domain': d,
                         'tiles': p['tiles'], 'accepted': p['accepted'], 'n': len(result['nll']),
                         'baseline_ppl': ref['ppl'], 'candidate_ppl': result['ppl'],
                         'delta_nll': mu, 'se': se, 'relative_ppl_percent': rel,
                         'lower_percent': lo, 'upper_percent': hi,
                         'delta_ppl': dppl, 'lower_ppl': plo, 'upper_ppl': phi,
                         'worthwhile_gain_0p01_ppl': dppl <= -.01,
                         'supported_gain_0p01_ppl': phi <= -.01,
                         'export_delta_ppl': dppl if p['accepted'] else 0.,
                         'export_relative_ppl_percent': rel if p['accepted'] else 0.,
                         'supported_improvement': mu+2*se < 0,
                         'supported_harm': mu-2*se > 0})
        lines.append(f'| {model} / {r["seed"]} | {rule} | {p["tiles"]:,} | {p["accepted"]} | '+' | '.join(cells)+' |')
lines += ['', '## Cross-domain rule checks', '',
          'Counts below describe measured model/seed/domain cells. Repeated seeds use '
          'the same held-out text and are not independent domain replications.', '',
          '| Rule | Candidate cells | Gains ≥0.01 PPL | Supported gains | Supported harms | Accepted exports | Worst accepted ΔPPL |',
          '|---|---:|---:|---:|---:|---:|---:|']
for rule in ['wiki', 'c4', 'mixed', 'consensus', 'teacher_mixed', 'teacher_consensus']:
    rs = [x for x in rows if x['rule'] == rule]
    accepted = [x for x in rs if x['accepted']]
    count = sum(r['policies'].get(rule, {}).get('accepted', False) for _, r in reports)
    total = sum(rule in r['policies'] for _, r in reports)
    worst = f'{max(x["delta_ppl"] for x in accepted):+.4f}' if accepted else 'none'
    lines.append(f'| {rule} | {len(rs)} | {sum(x["worthwhile_gain_0p01_ppl"] for x in rs)} | {sum(x["supported_improvement"] for x in rs)} | '
                 f'{sum(x["supported_harm"] for x in rs)} | {count}/{total} | {worst} |')
lines += ['', '## Fit and validation', '',
          '| Model / seed | Rule | Attempts | Last budget | Validation WikiText ΔNLL ± 2SE | Validation C4 ΔNLL ± 2SE |',
          '|---|---|---:|---:|---:|---:|']
for _, r in reports:
    for rule, p in r['policies'].items():
        checks = p['validation']
        history = p['history']
        budget = history[-1]['budget'] if history else 0
        c = [f'{checks[d]["mean"]:+.6f} ± {2*checks[d]["se"]:.6f}' for d in ['wiki', 'c4']]
        lines.append(f'| {r.get("model", "Qwen3.8-27B")} / {r["seed"]} | {rule} | {len(history)} | {budget:.6f} | '+' | '.join(c)+' |')
lines += ['', '## Isolated tile diagnostic', '',
          'Tiles were selected by prespecified score extremes/conflicts on fitting data. '
          'Measured effects use eight reserved training probes per domain. This selected '
          'sample cannot estimate all-tile error rates. “Stable wrong sign” requires '
          'predicted and measured two-SE intervals to exclude zero in opposite directions.', '',
          '| Seed | Domain | Tiles | Point-sign agreement | Stable wrong sign |',
          '|---|---|---:|---:|---:|']
probe_rows = []
for _, r in reports:
    probes = r.get('probes', [])
    if not probes:
        continue
    for d in ['wiki', 'c4']:
        agree, wrong = 0, 0
        for p in probes:
            pred, actual = p['predicted'][d], p['measured'][d]
            agree += pred['mean']*actual['mean'] > 0
            bad = ((pred['mean']+2*pred['se'] < 0 and actual['mean']-2*actual['se'] > 0)
                   or (pred['mean']-2*pred['se'] > 0 and actual['mean']+2*actual['se'] < 0))
            wrong += bad
            probe_rows.append({'seed': r['seed'], 'module': p['module'], 'tile': p['tile_flat_index'],
                               'domain': d, 'predicted': pred['mean'], 'predicted_se': pred['se'],
                               'measured': actual['mean'], 'measured_se': actual['se'], 'stable_wrong_sign': bad})
        lines.append(f'| {r["seed"]} | {d} | {len(probes)} | {agree}/{len(probes)} | {wrong} |')
contrast_rows = []
lines += ['', '## Direct paired policy contrasts', '',
          'Candidate A minus candidate B in NLL; negative favors A. These '
          'comparisons include rejected candidates and do not choose exports.', '',
          '| Model / seed | A minus B | WikiText ΔNLL ± 2SE | C4 ΔNLL ± 2SE |',
          '|---|---|---:|---:|']
for _, r in reports:
    for a, b in [('c4', 'wiki'), ('mixed', 'wiki'), ('consensus', 'mixed'), ('teacher_consensus', 'teacher_mixed')]:
        if a not in r['policies'] or b not in r['policies']:
            continue
        cells = []
        for d in ['wikitext', 'c4', 'math', 'code']:
            if d in r['final'][a]:
                va, vb = r['final'][a][d]['nll'], r['final'][b][d]['nll']
            elif stress and stress['complete'] and f'{r["seed"]}/{a}' in stress['results']:
                va, vb = [stress['results'][f'{r["seed"]}/{p}'][d]['nll'] for p in [a, b]]
            else:
                continue
            mu, se = paired(va, vb)
            contrast_rows.append({'model': r['model'], 'seed': r['seed'], 'candidate_a': a,
                                  'candidate_b': b, 'domain': d, 'delta_nll': mu, 'se': se,
                                  'relative_ppl_percent': 100*math.expm1(mu)})
            if d in ['wikitext', 'c4']:
                cells.append(f'{mu:+.6f} ± {2*se:.6f}')
        lines.append(f'| {r["model"]} / {r["seed"]} | {a} − {b} | '+' | '.join(cells)+' |')
    if 'parent_report' in r:
        parent = json.loads(Path(r['parent_report']).read_text())
        for a, b in [('teacher_mixed', 'mixed'), ('teacher_consensus', 'consensus')]:
            cells = []
            for d in ['wikitext', 'c4', 'math', 'code']:
                mu, se = paired(r['final'][a][d]['nll'], parent['final'][b][d]['nll'])
                contrast_rows.append({'model': r['model'], 'seed': r['seed'], 'candidate_a': a,
                                      'candidate_b': b, 'domain': d, 'delta_nll': mu, 'se': se,
                                      'relative_ppl_percent': 100*math.expm1(mu)})
                if d in ['wikitext', 'c4']:
                    cells.append(f'{mu:+.6f} ± {2*se:.6f}')
            lines.append(f'| {r["model"]} / {r["seed"]} | {a} − {b} | '+' | '.join(cells)+' |')
lines += ['', '## Map overlap', '',
          '| Model / seed | Maps | Shared tiles | Union tiles | Jaccard |',
          '|---|---|---:|---:|---:|']
for path, r in reports:
    sets = {}
    for rule in r['policies']:
        spec = json.loads(path.with_name(rule+'_candidate.json').read_text())
        sets[rule] = {(n, i) for n, entry in spec['modules'].items() for i in entry['e0m3_flat_indices']}
        assert len(sets[rule]) == r['policies'][rule]['tiles']
    for a, b in [('wiki', 'c4'), ('mixed', 'wiki'), ('consensus', 'mixed'), ('teacher_consensus', 'teacher_mixed')]:
        if a not in sets or b not in sets:
            continue
        shared, union = len(sets[a] & sets[b]), len(sets[a] | sets[b])
        jaccard = f'{shared/union:.4f}' if union else 'empty'
        lines.append(f'| {r["model"]} / {r["seed"]} | {a} / {b} | {shared:,} | {union:,} | {jaccard} |')
agreement_path = root/'score_agreement.json'
if agreement_path.exists():
    agreement = json.loads(agreement_path.read_text())
    lines += ['', '## Agreement of fitted domain scores', '',
              'CE rows use 64 windows per individual domain; teacher rows use 32. '
              'Consensus selection uses 32 per domain to keep its total budget at 64. '
              'The matching 32-window CE/teacher comparison is below. “Stable” means the descriptive '
              'two-SE sign check, without correction for millions of comparisons.', '',
              '| Run | N/domain | Cross-domain cosine | C4 split cosine | Wiki noise ratio | C4 noise ratio | Both stable negative |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, x in agreement.items():
        cosine = f'{x["score_cosine"]:.5f}' if x['score_cosine'] is not None else 'undefined'
        split = f'{x["c4_split_half_cosine"]:.5f}' if x.get('c4_split_half_cosine') is not None else 'undefined'
        noise = x['squared_se_over_squared_mean']
        lines.append(f'| {name} | {x["windows_per_domain"]} | {cosine} | {split} | {noise["wiki"]:.4f} | {noise["c4"]:.4f} | '
                     f'{x["both_stably_negative"]:,} |')
prefix_path = root/'score_agreement_prefix32.json'
if prefix_path.exists():
    prefix = json.loads(prefix_path.read_text())
    lines += ['', '### Matching 32-window objective comparison', '',
              'Noise ratio is sum(SE squared)/sum(mean squared), a descriptive estimate '
              'of noise relative to observed mean-score energy, not a guaranteed signal fraction.', '',
              '| Model | Objective | Cross-domain cosine | Wiki noise ratio | C4 noise ratio |',
              '|---|---|---:|---:|---:|']
    for name, x in prefix.items():
        if not name.startswith(('panel/', 'teacher/')):
            continue
        noise = x['squared_se_over_squared_mean']
        cosine = f'{x["score_cosine"]:.5f}' if x['score_cosine'] is not None else 'undefined'
        lines.append(f'| {name.split("/")[-1]} | {x["score_objective"]} | {cosine} | '
                     f'{noise["wiki"]:.4f} | {noise["c4"]:.4f} |')
lines += ['', '## Interpretation limits', '',
          '- A successful common calibration algorithm is different from a fixed '
          'domain-independent tile map. The tested maps use WikiText/C4 task gradients.',
          '- Worst-domain fit/validation is checked only on WikiText and C4. '
          'Math/code are uncalibrated stress tests, not guaranteed by those gates.',
          '- Math/code report full-reference-text language-model loss, not reasoning '
          'accuracy, answer-only loss, pass@k, or causal decoding performance. '
          'Variable-length examples are weighted equally in the main table; raw reports '
          'also contain token-weighted perplexity.',
          '- Windows and related models are not independent universal certification '
          'samples. Two-SE checks and sparse finite-step budgets are heuristics.',
          '- C4 train/test documents are hash-disjoint; WikiText splits are official '
          'train/validation. Dataset revisions and token hashes are in raw reports.',
          '- C4 length qualification and WikiText window boundaries depend on the '
          'tokenizer. Sampling seeds/splits are shared across models; exact C4 '
          'document selections can differ. Every within-model policy comparison '
          'uses identical examples.',
          '- Native Transformers panel baselines must not be compared as if identical '
          'to older copied-model implementations. All numbers here use matched native baselines.',
          '', 'See PROTOCOL.md, ROBUST_EXTENSION.md, STRESS_PROTOCOL.md, PANEL_PROTOCOL.md '
          'TEACHER_PROTOCOL.md and RUN_NOTES.md for the frozen design and execution changes.']
root.joinpath('REPORT.md').write_text('\n'.join(lines)+'\n')
for filename, values in [('summary.csv', rows), ('tile_probes.csv', probe_rows), ('contrasts.csv', contrast_rows)]:
    if values:
        with (root/filename).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
if rows:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = list(dict.fromkeys((x['model'], x['seed']) for x in rows))
    fig, axes = plt.subplots(1, 4, figsize=(17, 5), sharey=True)
    colors = {'wiki': '#3366cc', 'c4': '#d95f02', 'mixed': '#1b9e77', 'consensus': '#9944aa',
              'teacher_mixed': '#a6761d', 'teacher_consensus': '#e7298a'}
    for ax, d in zip(axes, ['wikitext', 'c4', 'math', 'code']):
        ax.axvline(0, color='0.5', linewidth=.8)
        ax.axvline(-.01, color='0.5', linewidth=.8, linestyle=':')
        for rule_index, (rule, color) in enumerate(colors.items()):
            for x in [v for v in rows if v['domain'] == d and v['rule'] == rule]:
                y = labels.index((x['model'], x['seed']))+(rule_index-2.5)*.12
                ax.errorbar(x['delta_ppl'], y,
                            xerr=[[x['delta_ppl']-x['lower_ppl']], [x['upper_ppl']-x['delta_ppl']]],
                            fmt='o' if x['accepted'] else 'x', color=color, markersize=4, capsize=2)
        ax.set_title(d)
        ax.set_xlabel('Candidate ΔPPL (dotted: −0.01)')
    axes[0].set_yticks(range(len(labels)), [f'{m}\n{s}' for m, s in labels])
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color=c, marker='o', label=k) for k, c in colors.items()],
               loc='lower center', ncol=6)
    fig.suptitle('Cross-domain type selection: paired ±2 SE; x = rejected export')
    fig.tight_layout(rect=(0, .08, 1, .94))
    fig.savefig(root/'comparison.png', dpi=180)
    plt.close(fig)
print('\n'.join(lines[:24]))
