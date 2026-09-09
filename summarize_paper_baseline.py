"""Render the paper-aligned NVFP4 / FourOverSix / MixFP4 panel.

All rows use the released 2048-token protocol under the corrected full-W4A4
scope: tensor-wide activation factors, SDPA, WikiText cached per window, C4
uncached, released float32 aggregation. MixFP4 maps come from the CE/KL
task-gradient calibration on OpenWebMath and CodeParrot only, so WikiText-2
and C4 are both untouched by calibration.
"""
import argparse
import json
import math
from pathlib import Path

REPRO = Path('/home/u4320956/NVFP4-RaZeR/results/released_reproduction/job_335297')
FIXED = Path('/home/u4320956/NVFP4-RaZeR/results/fixed256_paper_eval/job_335428')
MODELS = (('llama8b', 'llama-3.1-8b', 'Llama-3.1-8B'), ('qwen4b', 'qwen3-4b', 'Qwen3-4B'))
SETTINGS = ('math16', 'math32', 'math64', 'code16', 'code32', 'code64',
            'math_code16', 'math_code32', 'math_code64', 'math_code128')
PRIMARY = 'math_code128'


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    mean = sum(d) / len(d)
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1) / len(d))
    return mean, 2 * se


def ppl(losses):
    """Released FP32 aggregation, matching run_ppl.py."""
    import torch
    t = torch.tensor(losses, dtype=torch.float32) * 2048
    return float(torch.exp(t.sum() / (len(losses) * 2048)))


def released(model_slug, tag):
    r = json.loads((REPRO / f'{model_slug}_{tag}' / 'report.json').read_text())
    assert r['status'] == 'complete'
    n = r['wiki_windows']
    losses = [w['nll'] for w in r['windows']]
    return {'wiki': losses[:n], 'c4': losses[n:]}, r['ppl']


def evaluated(path):
    r = json.loads((path / 'report.json').read_text())
    assert r['status'] == 'complete' and r['source_weights_verified']
    return ({d: r['evaluation'][d]['nll'] for d in ('wiki', 'c4')},
            {'wikitext': r['evaluation']['wiki']['ppl'], 'c4': r['evaluation']['c4']['ppl']}, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    ap.add_argument('--out', default='results/paper_baseline')
    args = ap.parse_args()
    root = Path(args.out) / f'job_{args.job}'

    L = ['# Paper-aligned W4A4 panel: NVFP4, FourOverSix and calibrated MixFP4', '',
         'Every number uses the released 2048-token evaluation: WikiText-2 raw test in full',
         'nonoverlapping windows (141) and 256 seed-0 C4 validation crops from shard 00000 (256',
         'windows), tensor-wide activation factors, SDPA, WikiText cached per window with C4',
         'uncached, and the released float32 perplexity aggregation. The scope is corrected full',
         'W4A4: every targeted text linear weight and its input is quantized, including Qwen',
         "`o_proj`, which the released code omitted until the RaZeR author's own fix `abab3c6`.",
         '',
         'MixFP4 maps are 256 8x64 E0M3 type blocks over a FourOverSix E2M1 base, elected by the',
         'CE/KL two-SE task-gradient rule on OpenWebMath and CodeParrot only. Neither WikiText-2',
         'nor C4 supplies calibration data, gradients or any selection feedback.', '']

    for model, slug, label in MODELS:
        bf16_losses, bf16_ppl = released(slug, 'bf16')
        rows = []
        nv_losses, nv_ppl, nv_report = evaluated(root / f'{model}_nvfp4')
        fos_losses, fos_ppl, fos_report = evaluated(root / f'{model}_four_over_six')
        mix_losses, mix_ppl, mix_report = evaluated(root / f'{model}_fixed256_{PRIMARY}')
        assert nv_report['selected_e0m3_blocks'] == 0
        assert fos_report['selected_e0m3_blocks'] == 0
        assert mix_report['selected_e0m3_blocks'] == 256

        L += [f'## {label}', '',
              '| Policy | E0M3 blocks | WikiText-2 | C4 |', '|---|---:|---:|---:|',
              f"| BF16 (unquantized reference) | — | {bf16_ppl['wikitext']:.6f} | {bf16_ppl['c4']:.6f} |",
              f"| NVFP4 W4A4 | 0 | {nv_ppl['wikitext']:.6f} | {nv_ppl['c4']:.6f} |",
              f"| NVFP4 FourOverSix W4A4 | 0 | {fos_ppl['wikitext']:.6f} | {fos_ppl['c4']:.6f} |",
              f"| **MixFP4 ours ({PRIMARY})** | 256 | **{mix_ppl['wikitext']:.6f}** | **{mix_ppl['c4']:.6f}** |",
              '']
        L += ['Paired differences, negative favours the first policy. Two-SE intervals are',
              'descriptive over evaluation windows.', '',
              '| Comparison | WikiText ΔPPL | WikiText ΔNLL ±2SE | C4 ΔPPL | C4 ΔNLL ±2SE |',
              '|---|---:|---:|---:|---:|']
        for name, a_l, a_p, b_l, b_p in (
                ('MixFP4 ours − FourOverSix', mix_losses, mix_ppl, fos_losses, fos_ppl),
                ('MixFP4 ours − NVFP4', mix_losses, mix_ppl, nv_losses, nv_ppl),
                ('FourOverSix − NVFP4', fos_losses, fos_ppl, nv_losses, nv_ppl)):
            cells = []
            for dom, key in (('wiki', 'wikitext'), ('c4', 'c4')):
                m, se = paired(a_l[dom], b_l[dom])
                cells.append(f'{a_p[key] - b_p[key]:+.6f} | {m:+.6f} ±{se:.6f}')
            L.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        L += ['', 'Movement relative to the unquantized BF16 reference:', '']
        for dom, key in (('wiki', 'wikitext'), ('c4', 'c4')):
            gap = fos_ppl[key] - bf16_ppl[key]
            got = fos_ppl[key] - mix_ppl[key]
            note = (f'- {key}: FourOverSix is {gap:+.6f} above BF16; MixFP4 moves '
                    f'{got:.6f} toward it ({100 * got / gap:.1f}% of the gap)')
            if mix_ppl[key] < bf16_ppl[key]:
                note += ('. MixFP4 lands **below** the BF16 reference here, so the share '
                         'exceeds 100%; a quantized perplexity below an unquantized one is a '
                         'known effect on this model and is not evidence of a better model')
            L.append(note + '.')
        L.append('')

        L += [f'### {label}: all ten frozen calibration settings', '',
              'Every setting elects exactly 256 blocks with the same rule and differs only in the',
              'calibration corpus and sequence count. Reported so the primary setting is not read',
              'as a selected best.', '',
              '| Calibration | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |',
              '|---|---:|---:|---:|---:|']
        for s in SETTINGS:
            _, p, rep = evaluated(FIXED / f'{model}_fixed256_{s}')
            assert rep['selected_e0m3_blocks'] == 256
            mark = '**' if s == PRIMARY else ''
            L.append(f"| {mark}{s}{mark} | {p['wikitext']:.6f} | {p['c4']:.6f} | "
                     f"{p['wikitext'] - fos_ppl['wikitext']:+.6f} | {p['c4'] - fos_ppl['c4']:+.6f} |")
        L.append('')

    L += ['## Verification', '',
          "- For Llama-3.1-8B the NVFP4 and FourOverSix rows are asserted **exactly equal** to the",
          '  archived released-code reproduction (job 335297), which the evaluator checks at run',
          '  time. Llama is unaffected by the `o_proj` fix, so this anchors the whole path.',
          '- The FourOverSix and MixFP4 rows reproduce the published fixed-256 evaluation',
          '  (job 335428) to the last digit, which shows that adding the plain-NVFP4 policy left',
          '  the evaluator numerically inert.',
          '- Input token hashes are asserted identical to the released run for both models and',
          '  every policy; pristine weight hashes and frozen map hashes are verified.', '',
          '## Scope and limits', '',
          "- Qwen3-4B numbers are not comparable to RaZeR's published Qwen row. That row was",
          '  produced before commit `abab3c6`, when `o_proj` consumed unquantized attention',
          '  output. Our Qwen baseline is correspondingly harder, not easier.',
          '- Simulated W4A4 on text linear layers; no KV-cache quantization, generation accuracy,',
          '  or native FP4 kernel throughput is measured.',
          '- Tensor-wide activation factors span the whole teacher-forced window, so these are',
          '  reference-text perplexities and not causal generation likelihoods.',
          '- Two-SE intervals are descriptive evaluation-window intervals. They do not adjust for',
          '  multiple comparisons across the ten calibration settings, for WikiText article',
          '  dependence, or for calibration-draw variability (one draw per setting).']
    (Path(args.out) / 'REPORT.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
