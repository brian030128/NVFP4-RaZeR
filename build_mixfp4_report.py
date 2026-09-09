"""Rebuild MIXFP4_REPORT.md from paper-aligned results only.

Keeps the three already-verified 2048-token blocks verbatim (they are delimited
by HTML markers in the existing report), prepends the NVFP4/FourOverSix/MixFP4
panel, and replaces the remaining prose with method and limits sections. Every
512-token table is dropped from the report; the underlying records stay in
results/ and are listed as historical pointers.
"""
import argparse
import json
import math
from pathlib import Path

REPRO = Path('results/released_reproduction/job_335297')
MODELS = (('llama8b', 'llama-3.1-8b', 'Llama-3.1-8B'), ('qwen4b', 'qwen3-4b', 'Qwen3-4B'))
PRIMARY = 'math_code128'


def block(text, start, end):
    a = text.index(f'<!-- {start} -->')
    b = text.index(f'<!-- {end} -->') + len(f'<!-- {end} -->')
    return text[a:b]


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    mean = sum(d) / len(d)
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1) / len(d))
    return mean, 2 * se


def released(slug, tag):
    r = json.loads((REPRO / f'{slug}_{tag}' / 'report.json').read_text())
    assert r['status'] == 'complete'
    n = r['wiki_windows']
    losses = [w['nll'] for w in r['windows']]
    return {'wiki': losses[:n], 'c4': losses[n:]}, r['ppl']


def evaluated(path):
    r = json.loads((path / 'report.json').read_text())
    assert r['status'] == 'complete' and r['source_weights_verified']
    return ({d: r['evaluation'][d]['nll'] for d in ('wiki', 'c4')},
            {'wikitext': r['evaluation']['wiki']['ppl'], 'c4': r['evaluation']['c4']['ppl']}, r)


def panel(root):
    L = ['## 1. Paper-aligned W4A4 result', '',
         'Baselines are NVFP4 and NVFP4 FourOverSix. The method is MixFP4: the same FourOverSix',
         'E2M1 weights with 256 8x64 tiles switched to E0M3, elected by our CE/KL task-gradient',
         'calibration on OpenWebMath and CodeParrot only. WikiText-2 and C4 supply no calibration',
         'data, no gradients and no selection feedback, so both are held out for every row.', '']
    for model, slug, label in MODELS:
        _, bf16 = released(slug, 'bf16')
        nv_l, nv, nv_r = evaluated(root / f'{model}_nvfp4')
        fo_l, fo, fo_r = evaluated(root / f'{model}_four_over_six')
        mx_l, mx, mx_r = evaluated(root / f'{model}_fixed256_{PRIMARY}')
        assert nv_r['selected_e0m3_blocks'] == 0 and fo_r['selected_e0m3_blocks'] == 0
        assert mx_r['selected_e0m3_blocks'] == 256
        L += [f'### {label}', '',
              '| Policy | E0M3 blocks | WikiText-2 | C4 |', '|---|---:|---:|---:|',
              f"| BF16 reference | — | {bf16['wikitext']:.6f} | {bf16['c4']:.6f} |",
              f"| NVFP4 W4A4 | 0 | {nv['wikitext']:.6f} | {nv['c4']:.6f} |",
              f"| NVFP4 FourOverSix W4A4 | 0 | {fo['wikitext']:.6f} | {fo['c4']:.6f} |",
              f"| **MixFP4, ours** | 256 | **{mx['wikitext']:.6f}** | **{mx['c4']:.6f}** |", '',
              '| Comparison | Wiki ΔPPL | Wiki ΔNLL ±2SE | C4 ΔPPL | C4 ΔNLL ±2SE |',
              '|---|---:|---:|---:|---:|']
        for name, al, ap, bl, bp in (('MixFP4 − FourOverSix', mx_l, mx, fo_l, fo),
                                     ('MixFP4 − NVFP4', mx_l, mx, nv_l, nv),
                                     ('FourOverSix − NVFP4', fo_l, fo, nv_l, nv)):
            cells = []
            for dom, key in (('wiki', 'wikitext'), ('c4', 'c4')):
                m, se = paired(al[dom], bl[dom])
                cells.append(f'{ap[key] - bp[key]:+.6f} | {m:+.6f} ±{se:.6f}')
            L.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        L += ['', 'Share of the FourOverSix-to-BF16 gap that 256 switched tiles close:', '']
        for dom, key in (('wiki', 'wikitext'), ('c4', 'c4')):
            gap = fo[key] - bf16[key]
            got = fo[key] - mx[key]
            L.append(f'- {key}: FourOverSix sits {gap:+.6f} above BF16; MixFP4 recovers '
                     f'{got:.6f}, or {100 * got / gap:.1f}% of it.')
        L.append('')
        if 'released_comparison' in nv_r:
            c = nv_r['released_comparison']
            L += [f'Against the archived released NVFP4 run: WikiText {c["wiki_released"]:.6f} '
                  f'({c["wiki_delta"]:+.6f}), C4 {c["c4_released"]:.6f} ({c["c4_delta"]:+.6f}). '
                  'See section 3 for why these differ.', '']
        if fo_r.get('released_baseline_exact'):
            L += ['The FourOverSix row reproduces the archived released run exactly, asserted at '
                  'run time.', '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    ap.add_argument('--report', default='MIXFP4_REPORT.md')
    args = ap.parse_args()
    root = Path(f'results/paper_baseline/job_{args.job}')
    old = Path(args.report).read_text()

    fixed = block(old, 'FIXED256_PAPER_EVAL_START', 'FIXED256_PAPER_EVAL_END')
    audit = block(old, 'BASELINE_PROTOCOL_AUDIT_START', 'BASELINE_PROTOCOL_AUDIT_END')
    repro = block(old, 'RELEASED_REPRODUCTION_START', 'RELEASED_REPRODUCTION_END')

    head = f"""# MixFP4 with task-calibrated type selection

Every result in this report uses the released 2048-token evaluation protocol.
WikiText-2 raw test is read as 141 full nonoverlapping windows; C4 is 256 seed-0
crops from validation shard 00000. Activation factors are tensor-wide, attention
is SDPA, WikiText is cached per window and C4 uncached, and perplexity uses the
released float32 aggregation. Non-aligned 512-token studies have been removed
from this report; their records remain under `results/` and are listed in
section 8.

[Paper-aligned panel](results/paper_baseline/REPORT.md) (job {args.job}).

{panel(root)}
{fixed}

{audit}

{repro}

## 3. Two deliberate differences from the released code

Both make our baselines harder to beat, not easier.

**Qwen `o_proj` inputs are quantized.** At the archived release commit
`e230099`, `models/qmodule_qwen3.py` passed unquantized attention output into
`o_proj`, so Qwen "W4A4" left one projection's input in BF16. The RaZeR author
corrected this himself in commit `abab3c6`, after publication. We evaluate the
corrected behaviour, so our Qwen FourOverSix baseline is 14.269062 WikiText
where the pre-fix code gives 14.201942. Llama was never affected, and its rows
match the released run to the last digit; that is the control which shows the
difference comes only from that one line. Qwen rows here are therefore **not**
comparable with the published RaZeR Qwen row.

**NVFP4 saturation is clamped.** The archived `quant_nvfp4` used a rounding
path with no E2M1 saturation clamp, so an FP8 subnormal block scale that rounded
down could produce magnitude code 8, which is not a legal FP4 code. The current
`quant_nvfp4` clamps to [-6, 6]. Baseline and method both use the corrected
quantizer. Exact equality with the released NVFP4 run is therefore not expected
and not asserted; the released values are recorded as a comparison in each
report. The FourOverSix path is unchanged since the release and keeps a hard
exact-equality assertion.

## 4. The format

MixFP4 is NVFP4 plus a second, coarser block granularity that selects the FP4
element data type. The FP32 per-tensor global scale, the E4M3 block scale and
the 16-element scale block are inherited from NVFP4 unchanged.

A **scale block** is always 16 elements along K and owns one E4M3 scale. A
**type block** is a 2-D tile that owns one element data type and contains many
scale blocks. The two element types are **E2M1**, the standard FP4 grid with
maximum magnitude 6, and **E0M3**, the evenly spaced signed grid with maximum
magnitude 7. Both encode 15 distinct values in 16 codes and share the same
ue4m3 scale; only the spacing differs, so the better choice depends on the
distribution inside the tile.

The tile shape is not free. The public NVFP4 path issues

```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

and the same instruction can read either operand as E0M3. A single instruction
cannot subdivide its operand tile, so for weights in operand B the smallest
realizable type block is `n8 x k64`. All results here use exactly that 8x64
weight tile, with alpha fixed at 1 on the E0M3 branch.

## 5. Why a weight-error election is insufficient

The natural selector compares the squared weight error of the two candidates
per tile and takes the smaller. It measures how well a candidate approximates
the pristine weights, not which inputs reach the block or how its outputs move
the prediction. Even at one linear layer the output error is

$$
\\mathbb E\\|\\Delta W x\\|^2=\\operatorname{{tr}}(\\Delta W\\,\\mathbb E[xx^T]\\,\\Delta W^T),
$$

which depends on the input second moment, and that still omits the downstream
loss and interactions between quantized blocks. A weight-error gain of factor
$g$ certifies an output-error reduction only when $g > 1-1/\\kappa(S)$, and the
measured conditioning of $S$ puts that threshold far above the gains available
at a realizable tile size. This is why the election below is driven by a task
loss rather than a reconstruction loss.

MixFP4 can also appear to help merely because its implementation searches more
scales. We therefore hold the scale search fixed: the E2M1 baseline is
FourOverSix, the E0M3 alternative is fixed at alpha 1, and the experiment
changes only the tile's element type. No rotation, permutation, scale search or
weight training is added.

## 6. The selection method

Form the canonical FourOverSix Q0 and the E0M3-alpha1 Q1 once, so each 8x64
tile j has a fixed difference D_j. At the unchanged FourOverSix W4A4 student,
one forward per calibration sequence supports two backward passes, giving each
tile a directional derivative for next-token cross entropy and for KL from the
pristine BF16 teacher:

    g_CE[i,j] = <grad_Wj CE_i, D_j>,   g_KL[i,j] = <grad_Wj KL(teacher||student)_i, D_j>.

Activation quantization uses an identity straight-through derivative during
scoring. With n calibration sequences,

    u_j = max(mean(g_CE[:,j]) + 2 SE(g_CE[:,j]), mean(g_KL[:,j]) + 2 SE(g_KL[:,j])),

and the 256 most negative u_j are switched to E0M3. Requiring both objectives to
favour a switch is the point of the maximum: for each objective separately, the
sum of its estimated upper directional scores is bounded above by the sum of
per-tile maxima, so one selection bounds both. This is an approximate predictor
of a discrete switch, not a proof that a finite joint flip lowers the network
loss, and two SE does not control the many tile comparisons simultaneously. The
count 256 and the factor 2 are fixed empirical constants, never tuned per model
or per destination dataset.

Calibration uses OpenWebMath and CodeParrot only. Neither evaluation corpus
contributes gradients or any selection feedback.

## 7. Limits

- Simulated W4A4 on text linear weights and their inputs. No KV-cache
  quantization, generation accuracy, or native FP4 kernel throughput is
  measured, and no speedup is claimed.
- Tensor-wide activation factors span the whole teacher-forced window, so these
  are reference-text perplexities, not causal generation likelihoods.
- Two-SE intervals are descriptive evaluation-window intervals. They do not
  adjust for comparisons across the ten calibration settings, for WikiText
  article dependence, or for calibration-draw variability; one calibration draw
  per setting is used.
- The 256-tile count is a fixed constant here. Evidence that the count/gain
  curve has a model-dependent interior optimum exists only under the
  512-token protocol and is deliberately excluded from this report until it is
  re-measured under the aligned protocol.
- Gradient selection, distillation and sparse optimization are established
  tools. Their use here is not by itself a novelty claim.

## 8. Records not included in this report

These studies use protocols other than the aligned 2048-token one and are
excluded from the tables above. Their records are retained.

- [Pooled calibration transfer panel](results/transfer_rule/REPORT.md) and
  [held-out C4](results/c4_frozen/REPORT_332781.md): 512-token windows, causal
  per-token activation factors, C4 among the calibration sources.
- [Qwen3.8-27B pooled extension](results/pooled_qwen27b/model_332840/REPORT.md).
- [Tile-count sweep](results/cap_sweep/REPORT.md) and
  [calibration-chosen count](results/adaptive_count/REPORT.md): 512-token.
- [Adaptive-count study](results/math_code_adaptive/summary_333786_333788/REPORT.md):
  512-token evaluation of the curvature-penalised selector.
- [Earlier format exploration](results/MIXFP4_REPORT.md) and
  [decision-rule rounds](results/DECIDE_SUMMARY.md).
"""
    Path(args.report).write_text(head)
    print(f'wrote {args.report}: {len(head.splitlines())} lines')


if __name__ == '__main__':
    main()
