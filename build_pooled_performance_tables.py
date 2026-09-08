"""Build the report's pooled-map tables and dataset means from recorded results.

Run through Slurm on this cluster. No models, new evaluations or checkpoints.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
START = '<!-- BEGIN POOLED PERFORMANCE TABLES -->'
END = '<!-- END POOLED PERFORMANCE TABLES -->'
MODELS = {
    'opt350m': 'OPT-350M', 'qwen06b': 'Qwen3-0.6B',
    'llama1b': 'Llama-3.2-1B-Instruct', 'olmo1b': 'OLMo-1B',
    'pythia14b': 'Pythia-1.4B', 'qwen4b': 'Qwen3-4B',
    'llama8b': 'Llama-3.1-8B', 'qwen27b': 'Qwen3.8-27B',
}
manifest = []


def read(relative):
    report = json.loads((ROOT / relative).read_text())
    assert report['status'] == 'complete', relative
    return report


def pair(report, policy, domain, path, stage):
    evaluation = report['evaluation']
    base = evaluation['four_over_six']
    selected = evaluation[policy]
    if domain != 'c4':
        base, selected = base[domain], selected[domain]
    b, p = base['ppl'], selected['ppl']
    assert math.isfinite(b) and math.isfinite(p) and b > 0 and p > 0
    assert len(base['nll']) == len(selected['nll'])
    manifest.append(dict(stage=stage, model=report['model'], domain=domain,
                         baseline_ppl=b, pooled_ppl=p, selected_policy=policy,
                         source_report=path, evaluated_sequences=len(base['nll'])))
    return b, p


def table(rows, columns):
    lines = ['| Model | ' + ' | '.join(label for _, label in columns) + ' | Mean PPL |',
             '|---|' + '---:|' * (len(columns) + 1)]
    for model, values in rows.items():
        cells, available = [], []
        for key, _ in columns:
            if key in values:
                b, p = values[key]
                cells.append(f'{b:.4f} → {p:.4f}')
                available.append((b, p))
            else:
                cells.append('—')
        assert available
        bmean = math.fsum(x[0] for x in available) / len(available)
        pmean = math.fsum(x[1] for x in available) / len(available)
        lines.append('| ' + MODELS[model] + ' | ' + ' | '.join(cells) + f' | {bmean:.4f} → {pmean:.4f} |')
    return lines


def main():
    causal = {}
    for model in MODELS:
        row = {}
        if model != 'qwen27b':
            path = (f'results/pooled_scale/model_332389_{model}/report.json'
                    if model in ('qwen4b', 'llama8b')
                    else f'results/causal_replay/model_332374_{model}/report.json')
            r = read(path)
            for d in ('literature', 'science', 'government'):
                row[d] = pair(r, 'pooled192', d, path, 'causal')
        path = ('results/pooled_qwen27b/model_332840/report.json' if model == 'qwen27b'
                else f'results/c4_frozen/model_332781_{model}/report.json')
        row['c4'] = pair(read(path), 'pooled192', 'c4', path, 'causal')
        causal[model] = row
    development = {}
    for model in ('opt350m', 'qwen06b', 'llama1b'):
        path = f'results/consensus_format/model_332332_{model}/report.json'
        r = read(path)
        development[model] = {d: pair(r, 'all_pooled', d, path, 'development_window')
                              for d in ('wiki', 'math', 'code')}
    confirmation = {}
    for model in ('opt350m', 'qwen06b', 'llama1b', 'olmo1b', 'pythia14b'):
        path = f'results/pooled_confirmation/model_332349_{model}/report.json'
        r = read(path)
        confirmation[model] = {d: pair(r, 'pooled192', d, path, 'confirmation_window')
                              for d in ('literature', 'science', 'government')}
    assert len(manifest) == 29 + 9 + 15
    assert set(causal['qwen27b']) == {'c4'}
    lines = [START, '## Pooled calibration: all evaluated models and datasets', '',
             'Each cell is **FourOverSix baseline PPL → pooled-map PPL**; lower is better. '
             'The pooled map uses all 192 calibration sequences with the common CE/KL two-SE '
             'rule and 256-tile cap (`pooled192`, named `all_pooled` in the initial study). '
             'These tables cover its held-out/reference-text evaluations across all eight models '
             'and all seven dataset families evaluated. Other selection methods and calibration '
             'fitting losses are not included in the dataset averages.', '',
             '**Mean PPL** is the unweighted arithmetic mean of the available dataset PPLs in '
             'that row, computed separately for baseline and pooled map from unrounded values. '
             'It is a descriptive dataset average, not pooled-corpus perplexity or a statistical '
             'significance test. **—** means not evaluated and is excluded from the mean. '
             'Different dataset coverage and model tokenizers limit comparisons between rows; '
             'the 27B mean below covers C4 only.', '',
             '### Current causal evaluation', '',
             'All policies use per-token activation factors. C4 has 256 validation documents '
             'per model; literature (PG19), science (arXiv articles), and government (GovReport '
             'reports) each have 64 test documents per model. Each document contributes a '
             '512-token crop. The same frozen map serves all available domains of a model.', '']
    domains = [('c4', 'C4'), ('literature', 'PG19 / literature'),
               ('science', 'arXiv / science'), ('government', 'GovReport / government')]
    lines += table(causal, domains)
    lines += ['', 'The 27B C4 difference is inconclusive (paired ΔNLL −0.000764 ±0.001538, '
              'descriptive two-SE); its small point gain must not be presented as a supported '
              'improvement. OLMo-1B literature is also inconclusive. The other 27 causal '
              'comparisons have supporting descriptive paired two-SE intervals. '
              '[Transfer results](results/transfer_rule/REPORT.md), '
              '[seven-model C4 results](results/c4_frozen/REPORT_332781.md), '
              '[27B C4 result](results/pooled_qwen27b/model_332840/REPORT.md).', '',
              '### Earlier development evaluations: window-wide activation factors', '',
              'These are the original three pooled maps, replayed unchanged in subsequent '
              'confirmation. WikiText-2 uses 32 held-out 512-token windows; GSM8K and MBPP use '
              '32 held-out reference-text examples each, truncated to at most 512 tokens. '
              'Their reported PPL is exp(mean example NLL); math and code scores are not '
              'answer accuracy or pass@k. The other five models were not evaluated on these '
              'datasets with the pooled rule.', '',
              '**Historical only:** window-wide activation factors can depend on future tokens. '
              'These numbers are retained for completeness and are not causal-likelihood evidence. '
              'Their averages are kept separate from the causal table.', '']
    lines += table(development, [('wiki', 'WikiText-2'), ('math', 'GSM8K reference text'), ('code', 'MBPP reference text')])
    lines += ['', '[Development study](results/consensus_format/REPORT_332332.md). '
              'The all-source pooled map was a secondary control; it does not rescue the failed '
              'leave-source-out consensus screen.', '',
              '### Original confirmation: window-wide activation factors', '',
              'These five-model measurements used the same maps and inputs later replayed in '
              'the causal table. They are shown to preserve every evaluation setting, and must '
              'not be counted again as independent confirmation. The full prespecified '
              'confirmation screen failed its C4-only comparison despite the baseline gains.', '']
    lines += table(confirmation, domains[1:])
    lines += ['', '[Original confirmation](results/pooled_confirmation/REPORT_332349.md). '
              'No window-wide confirmation run was performed for Qwen3-4B, Llama-3.1-8B, or '
              'Qwen3.8-27B under this pooled protocol.', '',
              'Source values and per-cell report paths: '
              '[table data](results/transfer_rule/pooled_performance_tables.json). '
              'Reproduce with `build_pooled_performance_tables.py` via '
              '`slurm/pooled_performance_tables.sbatch`.', END]
    replacement = '\n'.join(lines)
    target = ROOT / 'MIXFP4_REPORT.md'
    text = target.read_text()
    if START in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        text = before + replacement + after
    else:
        anchor = '## Current result: a frozen rule across seven models'
        assert text.count(anchor) == 1
        text = text.replace(anchor, replacement + '\n\n' + anchor, 1)
    target.write_text(text)
    artifact = dict(averaging='Unweighted arithmetic mean of available per-dataset PPL values, separately for baseline and pooled map',
                    causal_cells=29, development_window_cells=9, confirmation_window_cells=15, cells=manifest)
    (ROOT / 'results/transfer_rule/pooled_performance_tables.json').write_text(json.dumps(artifact, indent=2) + '\n')
    print('Inserted 29 causal, 9 development-window and 15 confirmation-window cells with per-model dataset means.')


if __name__ == '__main__': main()
