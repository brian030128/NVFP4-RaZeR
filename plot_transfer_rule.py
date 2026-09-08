import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path('results/transfer_rule')
    summary = json.loads((root/'summary.json').read_text())
    reports = [json.loads(Path(p).read_text()) for p in summary['source_reports']]
    labels = {'llama1b': 'Llama 1B', 'opt350m': 'OPT 350M', 'qwen06b': 'Qwen 0.6B',
              'pythia14b': 'Pythia 1.4B', 'olmo1b': 'OLMo 1B', 'qwen4b': 'Qwen 4B', 'llama8b': 'Llama 8B'}
    plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.7), sharey=True)
    for ax, policy, title in zip(axes, ('four_over_six', 'c4_64'),
                                ('Selected − matched FourOverSix', 'Pooled192 − C4-only64')):
        for i, (domain, color) in enumerate(zip(('literature', 'science', 'government'), ('#225f9c', '#d87919', '#32815b'))):
            cs = [r['contrasts'][domain][policy] for r in reports]
            ax.errorbar([c['mean_nll'] for c in cs], np.arange(len(reports))+(i-1)*.2,
                        xerr=[c['two_se'] for c in cs], fmt='o', ms=4, capsize=2, color=color, label=domain.title())
        ax.axvline(0, color='0.4', linewidth=.8); ax.axhline(4.5, color='0.65', linewidth=.7, linestyle='--')
        ax.set_title(title, fontsize=10); ax.set_xlabel('Δ mean NLL (lower is better)')
        ax.grid(axis='x', alpha=.2); ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(range(len(reports)), [labels[r['model']] for r in reports]); axes[0].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=8, loc='best')
    fig.suptitle('Causal evaluation of one fixed FP4 tile rule', fontsize=12)
    fig.text(.5, .035, 'Per-token activation factors;64 documents/domain; descriptive ±2 SE. Dashed line marks the size extension.', ha='center', fontsize=8)
    fig.text(.5, .01, 'The right panel uses different calibration-pool sizes; the equal-token diversity ablation is reported separately.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .075, 1, .95))
    for extension in ('pdf', 'svg', 'png'): fig.savefig(root/f'causal_transfer.{extension}', dpi=180)


if __name__ == '__main__': main()
