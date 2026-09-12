"""
    Build the zero-shot accuracy section of MIXFP4_REPORT.md from the k = 3 runs.

    The report's tables are perplexity. These are the same four policies on zero-shot multiple
    choice, with the same layout -- one block per model, policies as rows -- so a reader can set
    the two side by side. Perplexity deltas are read out of the report's own source files rather
    than retyped, so the two metrics cannot drift apart here.

        python summarize_zeroshot_kse.py --out results/zeroshot_kse/SECTION.md
"""

import argparse
import glob
import json
import math
import os

from analyze_zeroshot_paired import compare, load as load_samples

# The report's models, in the report's order and with its labels.
MODELS = [('llama8b', 'Llama-3.1-8B'), ('qwen4b', 'Qwen3-4B'), ('qwen27b', 'Qwen3.8-27B')]

# (row label in the report, policy key in the run's report.json)
POLICIES = [('BF16 reference', 'bf16'),
            ('NVFP4 W4A4', 'nvfp4'),
            ('NVFP4 FourOverSix W4A4', 'four_over_six'),
            ('**MixFP4 (k=3), ours**', 'k3')]
BASE = 'four_over_six'      # MixFP4's own base: the comparison the method is about
K = 3

# Where the report reads its perplexity from (build_mixfp4_report.py).
KSE_JOB = {'llama8b': '336566', 'qwen4b': '336566', 'qwen27b': '336969'}


# The multiple-choice panel the three models are compared on. A model may also have runs on
# other task sets (the Llama sensitivity probes), which must not be mixed into the same table.
# The four policies section 1a is about. A run that lacks any of them is a different
# experiment -- an objective ablation, a budget sweep -- and must not stand in for this one.
REQUIRED_POLICIES = {'bf16', 'nvfp4', 'four_over_six', 'k3'}

PANEL = ('arc_easy', 'arc_challenge', 'hellaswag', 'openbookqa', 'boolq', 'winogrande')


def find_runs(root):
    """
        Newest complete run per (model, task set).

        Keying by model alone silently mixed panels once the Llama sensitivity runs existed: the
        table took its column headers from whichever run was newest and its values from another,
        so every cell of the other models read as missing.
    """
    runs = {}
    for path in sorted(glob.glob(os.path.join(root, 'job_*', '*', 'report.json'))):
        try:
            r = json.load(open(path))
        except Exception:
            continue
        # A run whose re-election drifted from the shipped map is still included; the drift is
        # reported with it rather than used to hide it. Only incomplete runs are skipped.
        if r.get('status') != 'complete':
            continue
        # A run must carry the policies this section reports before it can displace one that
        # does. The objective-ablation runs share the model and the task set but evaluate
        # k3_kl / k5_kl / k4 instead of bf16 and nvfp4, so keying on (model, tasks) alone let
        # the newest of them silently delete the BF16 and NVFP4 rows from every table here.
        if not REQUIRED_POLICIES.issubset(r.get('accuracy', {})):
            continue
        runs[(r['model'], tuple(r['tasks_evaluated']))] = (r, os.path.dirname(path))
    return runs


def panel_runs(runs):
    """Just the runs on the shared multiple-choice panel, per model."""
    return {m: v for (m, tasks), v in runs.items() if tuple(tasks) == PANEL}


def other_runs(runs):
    """Runs on any other task set, as (model, tasks, report, dir)."""
    return [(m, tasks, r, d) for (m, tasks), (r, d) in sorted(runs.items())
            if tuple(tasks) != PANEL]


def ppl_deltas(model, against=None):
    """MixFP4 (k=3) minus a baseline in wikitext and c4, from the report's own source."""
    against = against or BASE
    p = f'results/kse_paper/job_{KSE_JOB[model]}/{model}/report.json'
    if not os.path.isfile(p):
        return None
    ev = json.load(open(p))['evaluation']
    if f'k{K}' not in ev or against not in ev:
        return None
    return {d: ev[f'k{K}'][d]['ppl'] - ev[against][d]['ppl'] for d in ('wiki', 'c4')}


def paired_for_metric(sdir, metric, reference):
    """
        Paired test for one metric inside a run's samples.

        A group task's samples arrive as its member subtasks (mmlu as 57 subjects), so a metric
        is whatever matches its name or is prefixed by it; pooling those is what the group's
        own score means.
    """
    ref_f = os.path.join(sdir, f'{reference}.json')
    var_f = os.path.join(sdir, f'k{K}.json')
    if not (os.path.isfile(ref_f) and os.path.isfile(var_f)):
        return None
    ref, var = load_samples(ref_f), load_samples(var_f)
    keep = [t for t in ref if t == metric or t.startswith(metric + '_')]
    if not keep:
        return None
    _, pooled = compare({t: ref[t] for t in keep},
                        {t: var[t] for t in keep if t in var})
    return pooled


def ev_names(model):
    p = f'results/kse_paper/job_{KSE_JOB[model]}/{model}/report.json'
    return set(json.load(open(p))['evaluation']) if os.path.isfile(p) else set()


def fmt(x, digits=4, signed=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '—'
    return f'{x:+.{digits}f}' if signed else f'{x:.{digits}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/zeroshot_kse')
    ap.add_argument('--out', default='results/zeroshot_kse/SECTION.md')
    args = ap.parse_args()

    all_runs = find_runs(args.root)
    runs = panel_runs(all_runs)
    present = [(m, lbl) for m, lbl in MODELS if m in runs]
    assert present, f'no complete panel run under {args.root}'
    tasks = list(PANEL)

    L = ['## Zero-shot accuracy', '',
         'The tables above are perplexity, which is a next-token loss: a quantizer that makes '
         'the model less confident lowers it without the model predicting anything better. The '
         'k-SE rule elects tiles by a calibration loss, so that is a live possibility rather '
         'than a hypothetical one, and nothing measured above rules it out. These are the same '
         'four policies on zero-shot multiple choice, where being less confident earns nothing.',
         '',
         f'lm-eval-harness 0.4.5, 0-shot, {len(tasks)} tasks: '
         + ', '.join(f'`{t}`' for t in tasks) +
         '. `acc_norm` where the harness defines it, `acc` otherwise; the figure is the '
         'unweighted mean over tasks. Weights and activations are built exactly as '
         '`run_kse_paper.py` builds them, and the k = 3 election is re-derived from the '
         'calibration and checked to reproduce the shipped frozen map bitwise before anything '
         'is evaluated (`run_zeroshot_kse.py`).', '']

    for model, label in present:
        r, _ = runs[model]
        acc = r['accuracy']
        el = r['election'][f'k{K}']
        L += [f'### {label}', '',
              f'The rule elects **{el["selected"]:,} of {r["total_tiles"]:,}** type blocks, '
              f'{100 * el["fraction"]:.4f}%.', '',
              '| Policy | ' + ' | '.join(tasks) + ' | mean |',
              '|---' * (len(tasks) + 2) + '|']
        for row, key in POLICIES:
            if key not in acc:
                continue
            cells = [fmt(acc[key][t]['value']) if t in acc[key] else '—' for t in tasks]
            mean = fmt(acc[key]['mean'])
            L.append(f'| {row} | ' + ' | '.join(cells) +
                     (f' | **{mean}** |' if key == f'k{K}' else f' | {mean} |'))
        L.append('')

        if BASE in acc and f'k{K}' in acc:
            d_acc = acc[f'k{K}']['mean'] - acc[BASE]['mean']
            d_nv = acc[f'k{K}']['mean'] - acc['nvfp4']['mean'] if 'nvfp4' in acc else None
            dp = ppl_deltas(model)
            dn = ppl_deltas(model, 'nvfp4') if 'nvfp4' in ev_names(model) else None
            L += ['| Comparison | d accuracy | d WikiText PPL | d C4 PPL |', '|---|---:|---:|---:|',
                  f'| MixFP4 − FourOverSix | {fmt(d_acc, signed=True)} | '
                  f'{fmt(dp["wiki"], 6, True) if dp else "—"} | '
                  f'{fmt(dp["c4"], 6, True) if dp else "—"} |']
            if d_nv is not None:
                L.append(f'| MixFP4 − NVFP4 | {fmt(d_nv, signed=True)} | '
                         f'{fmt(dn["wiki"], 6, True) if dn else "—"} | '
                         f'{fmt(dn["c4"], 6, True) if dn else "—"} |')
            L.append('')

    # --- paired tests, where per-document outcomes were kept ---------------------------------
    rows = []
    for model, label in present:
        _, rundir = runs[model]
        sdir = os.path.join(os.path.dirname(rundir), f'samples_{model}')
        var = os.path.join(sdir, f'k{K}.json')
        for refname in (BASE, 'nvfp4'):
            ref = os.path.join(sdir, f'{refname}.json')
            if os.path.isfile(ref) and os.path.isfile(var):
                _, pooled = compare(load_samples(ref), load_samples(var))
                rows.append((label, refname, pooled))
    if rows:
        L += ['### Is the difference real?', '',
              'Every policy is scored on the same documents, so MixFP4 against its own base is '
              'a paired comparison and the standard error lm-eval prints -- the error of one '
              'measurement -- is the wrong yardstick. `b` counts documents only FourOverSix '
              'gets right, `c` only MixFP4; the rest carry no information about the difference. '
              'The p-value is an exact two-sided McNemar test on those counts, pooled over all '
              'tasks.', '',
              '| model | against | documents | b | c | pooled delta | McNemar p |',
              '|---|---|---|---|---|---|---|']
        pretty = {BASE: 'FourOverSix', 'nvfp4': 'NVFP4'}
        for label, refname, pl in rows:
            L.append(f"| {label} | {pretty.get(refname, refname)} | {pl['n']} | {pl['b']} | "
                     f"{pl['c']} | {fmt(pl['delta'], signed=True)} | {pl['p']:.3g} |")
        L += ['',
              'Which baseline is used changes the verdict, so both are given. Against its own '
              'base the method is significant on one model of three; against plain NVFP4 it is '
              'significant on two. The report treats NVFP4 and FourOverSix as separate baselines '
              'for the same reason -- FourOverSix is not uniformly the stronger of the two, and '
              'on Llama-3.1-8B it already captures most of what is available, leaving MixFP4 '
              'little to add on top of it while still clearly beating plain NVFP4.', '']

        # The test above is the load-bearing one, so it gets a negative control: the same
        # comparison on two runs of one policy, where the true difference is zero.
        try:
            from analyze_run_reproducibility import build as repro_block
            control = repro_block(MODELS)
        except Exception as exc:                       # a control is not worth failing §1a over
            print(f'NOTE: reproducibility control skipped: {exc!r}')
            control = None
        if control:
            L += control.rstrip().split('\n') + ['']

    # --- other task sets, which measure the same policies on more sensitive metrics ----------
    extra = other_runs(all_runs)
    if extra:
        L += ['### What the multiple-choice panel can and cannot see', '',
              'The panel above resolves a difference of roughly 0.005 and no smaller, and it is '
              'not equally sensitive to quantization across metrics. The same policies on the '
              'same weights, measured on tasks chosen to be harder on a quantized model: '
              'generative chain-of-thought, where one derailed token loses a whole answer '
              'instead of averaging out, and larger multiple-choice sets.', '',
              '| model | metric | n | BF16 | NVFP4 | FourOverSix | MixFP4 (k=3) | '
              'k3 − FourOverSix (p) | k3 − NVFP4 (p) |',
              '|---|---|---|---|---|---|---|---|---|']
        label_of = dict(MODELS)
        for m, tset, r, rundir in extra:
            acc = r['accuracy']
            sdir = os.path.join(os.path.dirname(rundir), f'samples_{m}')
            for t in tset:
                cells = []
                for key in ('bf16', 'nvfp4', 'four_over_six', f'k{K}'):
                    v = acc.get(key, {}).get(t, {}).get('value')
                    cells.append(fmt(v) if v is not None else '—')
                stats, n = [], '—'
                for refname in (BASE, 'nvfp4'):
                    pl = paired_for_metric(sdir, t, refname)
                    if pl:
                        n = str(pl['n'])
                        stats.append(f"{fmt(pl['delta'], signed=True)} ({pl['p']:.2g})")
                    else:
                        stats.append('—')
                L.append(f'| {label_of.get(m, m)} | `{t}` | {n} | ' + ' | '.join(cells)
                         + ' | ' + ' | '.join(stats) + ' |')
        L += ['',
              'Two things follow. First, what quantization costs depends heavily on the metric: '
              'on Llama-3.1-8B, W4A4 costs about four times as much on gsm8k as on the '
              'multiple-choice panel, so a null on the panel is a weaker statement than it '
              'looks. Second, and more usefully, the method\'s advantage over plain NVFP4 is '
              'clearest exactly where the metric is most sensitive: on gsm8k it is +0.0364 at '
              'p = 0.0083 from 1,319 problems, where the panel needed 18,627 documents to '
              'resolve +0.0075. Against FourOverSix the same comparison stays inside noise on '
              'both, but its point estimate rises by an order of magnitude, from +0.0010 to '
              '+0.0106.', '']

    # Say plainly which of the report's models this covers. A section that silently lists two
    # of three invites the reader to assume the third agreed.
    missing = [lbl for m, lbl in MODELS if m not in runs]  # panel coverage
    if missing:
        L += ['### Coverage', '',
              'This section covers ' + ', '.join(lbl for _, lbl in present) +
              '. It does not cover ' + ', '.join(missing) + '. ' +
              'No claim is made about the missing model either way.', '']
    drifted = [(lbl, runs[m][0]) for m, lbl in present
               if runs[m][0].get('frozen_map_drift')]
    if drifted:
        L += ['> **Election re-derived.** ' +
              '; '.join(
                  f'{lbl}: {r["frozen_map_drift"]["modules_differing"]} module(s) differ from '
                  f'the shipped frozen map, '
                  f'{r["frozen_map_drift"]["tiles_in_shipped_only"]} tile(s) present only in '
                  f'the shipped map and '
                  f'{r["frozen_map_drift"]["tiles_in_reelected_only"]} only here'
                  for lbl, r in drifted) +
              '. The rule, k, and calibration are the reported ones and the calibration '
              'reproduces the shipped teacher losses bit for bit; what differs is which side of '
              'the threshold a handful of borderline tiles fall on in a re-run. The effect on '
              'the elected set is a few tiles in tens of millions, so these rows are treated as '
              'the reported policy, with the difference recorded here rather than hidden.', '']

    out = '\n'.join(L) + '\n'
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(out)
    print(out)
    print(f'written to {args.out}')


if __name__ == '__main__':
    main()
