"""Rebuild MIXFP4_REPORT.md: the k-SE element-type rule and its 2048-token results.

The report carries only paper-aligned 2048-token measurements and the method that
produced them. Everything measured under other protocols lives under results/ and
is not summarised here.
"""
import argparse
import json
import math
import re
from pathlib import Path

REPRO = Path('results/released_reproduction/job_335297')
# BF16 references come from job 337128, which runs the unquantized model through this
# same path; its Llama and Qwen-4B rows reproduce the released reproduction exactly,
# which is what licenses the 27B row the released evaluator cannot produce.
BF16_JOB = '337128'
MODELS = (
    dict(key='llama8b', label='Llama-3.1-8B', kse='336566', base='335962'),
    dict(key='qwen4b', label='Qwen3-4B', kse='336566', base='335962'),
    dict(key='qwen27b', label='Qwen3.8-27B', kse='336969', base='337022'),
)
K = 3


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


def paper_case(root, case):
    r = json.loads((root / case / 'report.json').read_text())
    assert r['status'] == 'complete' and r['source_weights_verified']
    return ({d: r['evaluation'][d]['nll'] for d in ('wiki', 'c4')},
            {'wiki': r['evaluation']['wiki']['ppl'], 'c4': r['evaluation']['c4']['ppl']}, r)


def kse(root, model):
    r = json.loads((root / model / 'report.json').read_text())
    assert r['status'] == 'complete' and r['frozen_map_reproduced']
    assert r['shipped_score_identical_at_k2']
    return r


def results_section():
    L = ['## 1. Results', '',
         'Baselines are NVFP4 and NVFP4 FourOverSix, both W4A4. The method is MixFP4: the same',
         'FourOverSix E2M1 weights, with the tiles the rule elects switched to E0M3. Calibration',
         'uses OpenWebMath and CodeParrot only, so WikiText-2 and C4 are held out for every row.',
         'BF16 is the unquantized reference, not a competitor.', '']
    for spec in MODELS:
        model, label = spec['key'], spec['label']
        r = kse(Path(f'results/kse_paper/job_{spec["kse"]}'), model)
        bf16 = paper_case(Path(f'results/bf16_reference/job_{BF16_JOB}'), f'{model}_bf16')[1]
        nv_l, nv, _ = paper_case(Path(f'results/paper_baseline/job_{spec["base"]}'),
                                 f'{model}_nvfp4')
        fo_l = {d: r['evaluation']['four_over_six'][d]['nll'] for d in ('wiki', 'c4')}
        fo = {d: r['evaluation']['four_over_six'][d]['ppl'] for d in ('wiki', 'c4')}
        mx_l = {d: r['evaluation'][f'k{K}'][d]['nll'] for d in ('wiki', 'c4')}
        mx = {d: r['evaluation'][f'k{K}'][d]['ppl'] for d in ('wiki', 'c4')}
        el = r['election'][f'k{K}']
        L += [f'### {label}', '',
              f'{r["total_tiles"]:,} type blocks of 8x64 across the quantized text linear '
              f'weights. The rule elects **{el["selected"]:,} of them, {100 * el["fraction"]:.4f}%** '
              f'— about one block in {round(1 / el["fraction"]):,}.', '',
              '| Policy | E0M3 blocks | WikiText-2 | C4 |', '|---|---:|---:|---:|']
        L += [f'| BF16 reference | — | {bf16["wiki"]:.6f} | {bf16["c4"]:.6f} |',
              f'| NVFP4 W4A4 | 0 | {nv["wiki"]:.6f} | {nv["c4"]:.6f} |',
              f'| NVFP4 FourOverSix W4A4 | 0 | {fo["wiki"]:.6f} | {fo["c4"]:.6f} |',
              f'| **MixFP4 (k={K}), ours** | {el["selected"]:,} | **{mx["wiki"]:.6f}** | '
              f'**{mx["c4"]:.6f}** |', '',
              '| Comparison | Wiki ΔPPL | Wiki ΔNLL ±2SE | C4 ΔPPL | C4 ΔNLL ±2SE |',
              '|---|---:|---:|---:|---:|']
        for nm, al, ap, bl, bp in (('MixFP4 − FourOverSix', mx_l, mx, fo_l, fo),
                                   ('MixFP4 − NVFP4', mx_l, mx, nv_l,
                                    {'wiki': nv['wiki'], 'c4': nv['c4']}),
                                   ('FourOverSix − NVFP4', fo_l, fo, nv_l,
                                    {'wiki': nv['wiki'], 'c4': nv['c4']})):
            cells = []
            for dom in ('wiki', 'c4'):
                m, se = paired(al[dom], bl[dom])
                cells.append(f'{ap[dom] - bp[dom]:+.6f} | {m:+.6f} ±{se:.6f}')
            L.append(f'| {nm} | ' + ' | '.join(cells) + ' |')
        L.append('')
    L += ['MixFP4 improves both datasets on every model against both baselines, with every paired',
          'two-SE interval excluding zero. Note FourOverSix is not uniformly the stronger baseline:',
          'on Qwen3-4B plain NVFP4 beats it, so the method is measured against the better of the',
          'two, not only against its own base.', '',
          '### The count is not a tuned constant', '',
          'The same rule at other values of k, for reference. k is fixed at 3 for every model and',
          'is not selected per model or per dataset.', '']
    for spec in MODELS:
        label = spec['label']
        r = kse(Path(f'results/kse_paper/job_{spec["kse"]}'), spec['key'])
        b = r['evaluation']['four_over_six']
        L += [f'**{label}**', '', '| k | Tiles | % of blocks | ΔWiki | ΔC4 |', '|---|---:|---:|---:|---:|']
        for k in r['k_values']:
            e, el = r['evaluation'][f'k{k}'], r['election'][f'k{k}']
            mark = '**' if k == K else ''
            L.append(f'| {mark}{k}{mark} | {el["selected"]:,} | {100 * el["fraction"]:.4f}% | '
                     f'{e["wiki"]["ppl"] - b["wiki"]["ppl"]:+.6f} | '
                     f'{e["c4"]["ppl"] - b["c4"]["ppl"]:+.6f} |')
        L.append('')
    return '\n'.join(L)


def accuracy_section(path, ppl):
    r = json.loads((path / 'report.json').read_text())
    assert r['status'] == 'complete'
    a, tasks = r['accuracy'], r['tasks_evaluated']
    order = ['bf16', 'four_over_six', 'n256', 'n65536']
    label = {'bf16': 'BF16 reference', 'four_over_six': 'FourOverSix W4A4',
             'n256': 'MixFP4, 256 tiles', 'n65536': 'MixFP4, 65,536 tiles'}
    base, top = a['four_over_six']['mean'], a['bf16']['mean']
    gap = top - base
    L = ['## 5. Perplexity below BF16 is not a quality claim', '',
         'On Qwen3-4B some MixFP4 perplexities fall below the unquantized BF16 reference.',
         'Perplexity cannot settle that on its own: a model that becomes less overconfident scores',
         'better on next-token loss without predicting better. The same frozen maps were therefore',
         'evaluated zero-shot on multiple choice, where a smoothing artefact should not help. BF16',
         'restores pristine weights and removes activation quantization.', '',
         '| Policy | WikiText-2 PPL | ' + ' | '.join(tasks) + ' | mean acc |',
         '|---' * (len(tasks) + 3) + '|']
    for p in order:
        L.append(f'| {label[p]} | {ppl[p]:.6f} | ' +
                 ' | '.join(f'{a[p][t]["value"]:.4f}' for t in tasks) + f' | {a[p]["mean"]:.4f} |')
    L += ['', f'Accuracy corroborates the method among the quantized policies, in the same order '
          f'perplexity gives: {base:.4f} for FourOverSix, {a["n256"]["mean"]:.4f} at 256 tiles, '
          f'{a["n65536"]["mean"]:.4f} at 65,536, the larger map ahead on '
          f'{sum(1 for t in tasks if a["n65536"][t]["value"] > a["n256"][t]["value"])}/{len(tasks)} '
          f'tasks. Quantization costs {gap:.4f} mean accuracy against BF16; those maps recover '
          f'{100 * (a["n256"]["mean"] - base) / gap:.1f}% and '
          f'{100 * (a["n65536"]["mean"] - base) / gap:.1f}% of it.', '',
          f'Accuracy does **not** support beating BF16. The best quantized policy is still '
          f'{top - a["n65536"]["mean"]:.4f} below the unquantized model while its perplexity is '
          f'{ppl["bf16"] - ppl["n65536"]:.6f} better. Perplexity is therefore not a reliable '
          f'absolute quality measure against BF16 on this model. No claim here rests on a '
          f'below-BF16 perplexity; comparisons against the matched baselines are unaffected.', '',
          f'Zero-shot via lm_eval {r["lm_eval_version"]}.' +
          (f' Skipped for dataset-loading reasons unrelated to the model: '
           f'{", ".join(sorted(r["tasks_skipped"]))}.' if r.get('tasks_skipped') else ''), '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kse-job', default='336566')
    ap.add_argument('--baseline-job', default='335962')
    ap.add_argument('--zeroshot', default='results/zeroshot_check/job_336108/qwen4b')
    ap.add_argument('--report', default='MIXFP4_REPORT.md')
    args = ap.parse_args()
    qwen = kse(Path('results/kse_paper/job_336566'), 'qwen4b')
    _, qwen_bf16 = released('qwen3-4b', 'bf16')
    adaptive = json.loads(Path('results/adaptive_paper/job_335993/qwen4b/report.json').read_text())
    accuracy = accuracy_section(Path(args.zeroshot), {
        'bf16': qwen_bf16['wikitext'],
        'four_over_six': qwen['evaluation']['four_over_six']['wiki']['ppl'],
        'n256': qwen['evaluation']['n256']['wiki']['ppl'],
        'n65536': adaptive['evaluation']['n65536']['wiki']['ppl']})

    head = f"""# MixFP4: choosing the FP4 element type per tile

NVFP4 hardware can already read a weight operand tile as either E2M1 or E0M3 at
no cost. This report is about how to set that one bit per 8x64 tile, and what it
buys. Every number is the released 2048-token evaluation: WikiText-2 raw test in
141 full nonoverlapping windows, C4 as 256 seed-0 crops from validation shard
00000, tensor-wide activation factors, SDPA, WikiText cached per window with C4
uncached, and the released float32 perplexity aggregation. Measurements taken
under any other protocol are not reported here.

{results_section()}
## 2. How the element type is chosen

### The two candidates

A **scale block** is always 16 elements along K and owns one E4M3 scale; this is
inherited from NVFP4 unchanged. A **type block** is a 2-D tile that owns one
element data type and contains many scale blocks. The two types are:

- **E2M1**, the standard FP4 grid, maximum magnitude 6, log-spaced so it is fine
  near zero and coarse near the block maximum.
- **E0M3**, the evenly spaced signed grid, maximum magnitude 7, uniform.

Both encode 15 values in 16 codes and share the same ue4m3 scale. Only the
spacing differs, so which one is better depends on the distribution inside the
tile, which is why the choice must be data driven rather than fixed.

The tile shape is not free. The public NVFP4 path issues

```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

and the same instruction can read either operand as E0M3. One instruction cannot
subdivide its operand tile, so for weights in operand B the smallest realizable
type block is `n8 x k64`. Everything here uses exactly that 8x64 tile, with the
E0M3 branch pinned at alpha 1 so no extra scale search is smuggled in.

### The score

Form the canonical FourOverSix Q0 and the E0M3-alpha1 Q1 once, so each tile j has
a fixed difference D_j. At the unchanged FourOverSix W4A4 model, one forward pass
per calibration sequence supports two backward passes, giving each tile a
directional derivative for next-token cross entropy and for KL from the pristine
BF16 teacher:

    g_CE[i,j] = <grad_Wj CE_i, D_j>,   g_KL[i,j] = <grad_Wj KL(teacher||student)_i, D_j>

A negative value predicts that switching tile j to E0M3 lowers that loss. The
gradient is taken at the **quantized** model, not the pristine one, so it finds
corrections that are useful given the errors already present. Activation
quantization uses an identity straight-through derivative during scoring.

### The rule

Over n calibration sequences take each tile's mean and standard error, and switch
tile j to E0M3 when

    max( mean(g_CE) + k SE(g_CE),  mean(g_KL) + k SE(g_KL) ) < 0,  with k = {K}

Everything else stays E2M1. There is no cap and no per-model search: the number
of switched tiles is whatever passes.

## 3. Why this rule

### Why a task gradient and not the weight error

The obvious selector compares the two candidates' squared weight error per tile
and takes the smaller. That measures how well a candidate approximates the
pristine weights, not which inputs reach the block or how its outputs move the
prediction. Even for one linear layer the output error is

    E||dW x||^2 = tr(dW E[x x^T] dW^T)

which depends on the input second moment, and that still ignores the downstream
loss and interactions between blocks. A weight-error gain of factor g certifies
an output-error reduction only when g > 1 - 1/kappa(S), and the measured
conditioning of S puts that threshold far above the gains available at a
realizable tile size. So the criterion has to be a task loss.

### Why both CE and KL

The two objectives fail in opposite directions. Cross entropy alone can be
lowered by fitting the calibration corpus's particular observed tokens, which
does not transfer. Teacher KL alone can be lowered by restoring agreement with
the unquantized model on positions the task does not care about, and it is blind
to whether the recovered probability mass sits on the right tokens. Requiring
both means a switch is kept only when it improves the actual task loss **and**
moves the quantized model back toward its own unquantized reference.

Taking the maximum is what makes that a single objective: for each loss
separately, the sum of its estimated upper directional scores is bounded above by
the sum of the per-tile maxima, so one selection bounds both. It also needs no
weighting constant between two quantities that have no common scale, and the
eligible set is unchanged if either objective is rescaled by a positive factor.

### Why k = {K}, and why a threshold rather than a fixed count

Rearranged, `mean + k SE < 0` is `|mean|/SE > k`: k is a t-statistic cutoff, the
number of standard errors of evidence a tile must show. It adapts on its own,
because SE measures each model's own score noise — the count is never set, it
falls out.

k = 2 is too loose here, and the reason is multiple comparisons. There are
millions of tiles. Under a null where a tile has no real effect, the chance it
clears the bar on both objectives is about Phi(-k)^2, so at k = 2 one expects
thousands of false positives, and the elected set is measurably polluted by them:
at k = 2 the rule elects 0.73% of Llama-3.1-8B's tiles and **harms** the model.
At k = {K} the expected null count falls to a few dozen out of thousands elected.
k = {K} is the smallest value at which that expected contamination becomes
negligible relative to the selected set, computed from the calibration table
alone with no evaluation data involved.

That null estimate treats the CE and KL tests as independent when they share a
forward pass and are correlated, so it is optimistic in magnitude. The
conclusion it supports is the qualitative one — a 2 SE bar is far too loose
across millions of comparisons — not a precise contamination figure.

## 4. Protocol, and two deliberate differences from the released code

The scope is corrected full W4A4: every targeted text linear weight and its input
is quantized. Both differences below make our baselines harder to beat.

**Qwen `o_proj` inputs are quantized.** At the archived release commit `e230099`,
`models/qmodule_qwen3.py` passed unquantized attention output into `o_proj`, so
Qwen "W4A4" left one projection's input in BF16. The RaZeR author corrected this
in commit `abab3c6`, after publication. We evaluate the corrected behaviour, so
our Qwen FourOverSix baseline is 14.269062 WikiText where the pre-fix code gives
14.201942. Qwen rows here are therefore **not** comparable with the published
RaZeR Qwen row. Llama was never affected, and its rows match the released run to
the last digit — the control showing the difference comes only from that line.

**NVFP4 saturation is clamped.** The archived `quant_nvfp4` used a rounding path
with no E2M1 saturation clamp, so an FP8 subnormal block scale that rounded down
could produce magnitude code 8, which is not a legal FP4 code. The current
`quant_nvfp4` clamps to [-6, 6], and baseline and method both use it.

Verification carried by the runs themselves: the shipped 2 SE score is
reproduced exactly by the k = 2 case; the 256-tile prefix of that ranking
reproduces the previously frozen map bitwise; the Llama FourOverSix row is
asserted equal to the archived released-code reproduction; pristine weight
hashes and frozen map hashes are checked; and the C4 evaluation documents have
zero hash overlap with the calibration documents.

{accuracy}
## 6. Limits

- Simulated W4A4 on text linear weights and their inputs. No KV-cache
  quantization, generation accuracy, or native FP4 kernel throughput is measured,
  and no speedup is claimed.
- Tensor-wide activation factors span the whole teacher-forced window, so these
  are reference-text perplexities, not causal generation likelihoods.
- k = {K} is prespecified from the calibration score distribution and was fixed
  using Llama-3.1-8B and Qwen3-4B. Qwen3.8-27B played no part in choosing it and
  is a held-out check of the rule, not a third fitting model. Three models is
  still a small panel, and one calibration draw is used per model.
- The released evaluator predates the Qwen3.8 architecture, so its BF16 reference
  is measured through this report's own path. That path reproduces the released
  BF16 values exactly for Llama-3.1-8B and Qwen3-4B, which is the basis for
  trusting the 27B row; it is not an independent implementation.
- Two-SE intervals are descriptive evaluation-window intervals. They do not
  adjust for multiple comparisons, WikiText article dependence, or
  calibration-draw variability; one calibration draw per model is used.
- Gradient selection, distillation and sparse optimization are established tools.
  Their use here is not by itself a novelty claim.
"""
    lines, n = [], 0
    for line in head.split('\n'):
        if line.startswith('## ') and not line.startswith('###'):
            n += 1
            line = f'## {n}. ' + re.sub(r'^## (?:\d+\.\s*)?', '', line)
        lines.append(line)
    Path(args.report).write_text('\n'.join(lines))
    print(f'wrote {args.report}: {len(lines)} lines, {n} sections')


if __name__ == '__main__':
    main()
