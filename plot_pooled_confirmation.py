"""Export a paired reference-text loss plot from completed frozen reports."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--job', type=int, required=True); args = ap.parse_args()
    root = Path('results/pooled_confirmation')
    models = ('llama1b', 'opt350m', 'qwen06b', 'pythia14b', 'olmo1b')
    labels = ('Llama 1B', 'OPT 350M', 'Qwen 0.6B', 'Pythia 1.4B', 'OLMo 1B')
    reports = [json.loads((root/f'model_{args.job}_{m}'/'report.json').read_text()) for m in models]
    assert all(r['status'] == 'complete' for r in reports)
    domains = ('literature', 'science', 'government'); colors = ('#225f9c', '#d87919', '#32815b')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, kind, title in zip(axes, ('baseline', 'diversity'),
                               ('Pooled192 − FourOverSix', 'Mixed64 − C4-only64 (equal tokens)')):
        for i, (domain, color) in enumerate(zip(domains, colors)):
            cs = [r['contrasts'][domain]['four_over_six'] if kind == 'baseline' else r['diversity_contrasts'][domain] for r in reports]
            y = np.arange(len(models))+(i-1)*.2
            ax.errorbar([c['mean_nll'] for c in cs], y, xerr=[c['two_se'] for c in cs],
                        fmt='o', ms=4, capsize=2, color=color, label=domain.title())
        ax.axvline(0, color='0.4', linewidth=.8)
        ax.set_title(title, fontsize=10); ax.set_xlabel('Δ mean NLL (lower is better)')
        ax.grid(axis='x', alpha=.2); ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(range(len(models)), labels); axes[0].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=8, loc='best')
    fig.suptitle('Frozen procedure: five models, three confirmation domains', fontsize=12)
    fig.text(.5, .015, '64 documents/domain; bars show descriptive ±2 SE. Fake-quantized reference-text scores.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .94))
    for extension in ('pdf', 'svg', 'png'): fig.savefig(root/f'confirmation_{args.job}.{extension}', dpi=180)


if __name__ == '__main__': main()
