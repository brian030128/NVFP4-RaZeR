#!/usr/bin/env python3
"""Step 07: the paper tables from steps 03-06. CPU only, seconds. Missing results show as '—'.

    PAPER_PYTHON experiments/paper/07_tables.py [--models ...] [--smoke]

Writes <out>/tables/main.md (8x64 and 16x64), appendix.md (256x64, and the per-shape GEMM detail) and tables.json.

**Accuracy.**
- **Perplexity:** exp of the mean over windows of each window's mean NLL (run_ppl_deploy.py).
- **Paired ΔNLL:** the mean over windows of (policy − reference) window NLL, in nats per token, i.e. Δ log PPL. The
  windows must be the same (token hashes compared); ± is 2 SE of the paired differences. `*` marks |Δ| > 2 SE.
- **Downstream:** lm-eval's primary metric; for MMLU, the group aggregate (micro-average over all questions). Each
  accuracy is recomputed from the per-example correctness and must equal lm-eval's value.
- **Paired accuracy differences:** over the same examples, in percentage points, ± 2 SE. The mean over the five tasks
  has SE sqrt(sum of the tasks' SE²) / 5 (tasks independent).
- **Same samples:** every policy of a (model, task) must have the same `sample_digest` (lm-eval's doc / prompt /
  target hashes); a difference stops the step.

**Latency.**
- **Prefill:** the median over rounds of each process's median. The CUDA-graph numbers are the primary ones (main
  tables); eager is supplementary (appendix), where `†` marks a host-bound point (bench_prefill.py's flag in any
  round).
- **Ratios:** Ours / reference − 1 with the same activation quantizer, paired within rounds (the median over rounds of
  the per-round ratio, with its range), as Part R reported them:
  - `ours-<u>` against FourOverSix;
  - `ours-<u>-nvfp4act` (latency only) against NVFP4;
  - for 8x64, also against the weights-on-B references.
- **GEMM:** the per-forward sum over the quantized text Linears of each projection's kernel time times its module
  count. The activation quantizer's per-forward sum counts only its launches, net of the reuse measured in step 05
  (q/k/v and gate/up share one quantization). Without a step-05 record, it assumes no reuse, an upper bound, and says so.
"""
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as P  # noqa: E402

TASKS = ('mmlu', 'arc_challenge', 'arc_easy', 'hellaswag', 'piqa')
TASK_TITLES = dict(mmlu='MMLU (5-shot)', arc_challenge='ARC-C', arc_easy='ARC-E', hellaswag='HellaSwag', piqa='PIQA')
POLICY_TITLES = {'bf16': 'BF16', 'nvfp4': 'NVFP4', 'fo6': 'FourOverSix', 'ours-8x64': 'Ours 8x64',
                 'ours-16x64': 'Ours 16x64', 'ours-256x64': 'Ours 256x64', 'nvfp4-wB': 'NVFP4 (wB)', 'fo6-wB': 'FourOverSix (wB)'}
for _u in P.UNITS:
    POLICY_TITLES[f'ours-{_u}-nvfp4act'] = f'Ours {_u}, NVFP4 act. (latency only)'
GEMM_TITLES = dict(stock_wA='stock wA (NVFP4, FourOverSix)', stock_wB='stock wB', mixed_16x64='Ours 16x64 (n16k64_wA)',
                   mixed_256x64='Ours 256x64 (n16k64_wA)', n8k64_wB='Ours 8x64 (n8k64_wB)')
DASH = '—'
STRICT = True          # the registered checks must run; --smoke relaxes this for reports older than a check


def read(path):
    try:
        r = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return r if r.get('status') == 'complete' else None


def paired(a, b):
    """mean and 2 SE of the paired differences a - b."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    se = statistics.stdev(d) / math.sqrt(n) if n > 1 else float('nan')
    return dict(mean=mean, two_se=2 * se, n=n, significant=abs(mean) > 2 * se)


def fmt_diff(p, scale=1.0, digits=4):
    if p is None:
        return DASH
    return f"{p['mean'] * scale:+.{digits}f} ± {p['two_se'] * scale:.{digits}f}{' *' if p['significant'] else ''}"


def table(header, rows, lead=1):
    """A markdown table; the first `lead` columns are labels (left-aligned), the rest numbers."""
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' if i < lead else '---:' for i in range(len(header))) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


# ---------------------------------------------------------------------------------------------------------- accuracy

def ppl_results(out, model):
    res = {}
    for pol in P.ACCURACY_POLICIES:
        r = read(out / 'ppl' / model / pol / 'report.json')
        if r is not None:
            res[pol] = dict(evaluation=r['evaluations'][pol]['evaluation'],
                            tokens={d: r['data'][d]['token_sha256'] for d in ('wiki', 'c4_paper')},
                            limit_windows=r.get('limit_windows'))
    return res


def ppl_section(out, models, units):
    pols = ['bf16', 'nvfp4', 'fo6'] + [f'ours-{u}' for u in units]
    rows, drows, data = [], [], {}
    for model in models:
        res = ppl_results(out, model)
        data[model] = {}
        for corpus, title in (('wiki', 'WikiText-2'), ('c4', 'C4')):
            rows.append([P.TITLES[model], title] + [f"{res[p]['evaluation'][corpus]['ppl']:.4f}" if p in res else DASH for p in pols])
            n = {len(v['evaluation'][corpus]['nll']) for v in res.values()}
            drow = [P.TITLES[model], title, '/'.join(str(v) for v in sorted(n)) or DASH]
            for u in units:
                for ref in ('fo6', 'nvfp4'):
                    pol = f'ours-{u}'
                    p = None
                    if pol in res and ref in res:
                        dom = 'wiki' if corpus == 'wiki' else 'c4_paper'
                        assert res[pol]['tokens'][dom] == res[ref]['tokens'][dom], f'{model} {corpus}: windows differ'
                        p = paired(res[pol]['evaluation'][corpus]['nll'], res[ref]['evaluation'][corpus]['nll'])
                        p['delta_ppl'] = res[pol]['evaluation'][corpus]['ppl'] - res[ref]['evaluation'][corpus]['ppl']
                    data[model][f'{corpus}:{pol}-{ref}'] = p
                    drow.append(fmt_diff(p))
            drows.append(drow)
        data[model]['ppl'] = {p: {c: res[p]['evaluation'][c]['ppl'] for c in ('wiki', 'c4')} for p in res}
        data[model]['limit_windows'] = sorted({str(v['limit_windows']) for v in res.values()})
    head = ['model', 'corpus'] + [POLICY_TITLES[p] for p in pols]
    dhead = ['model', 'corpus', 'windows'] + [f'Ours {u} − {POLICY_TITLES[r]}' for u in units for r in ('fo6', 'nvfp4')]
    text = ('### Perplexity (NativeLinear (c); BF16 as loaded)\n\n' + table(head, rows, 2) +
            '\n\n### Paired ΔNLL, nats per token (= Δ log PPL), ± 2 SE over windows; * = |Δ| > 2 SE\n\n' + table(dhead, drows, 2))
    return text, data


def lmeval_results(out, model):
    res = {}
    for pol in P.ACCURACY_POLICIES:
        r = read(out / 'lmeval' / model / pol / 'report.json')
        if r is not None:
            res[pol] = dict(tasks=r['evaluations'][pol]['tasks'], limit=r.get('limit'), num_fewshot=r.get('num_fewshot'))
    return res


def accuracy(entry, task):
    """(accuracy recomputed from the examples, lm-eval's value); they must agree."""
    key = ('acc' if task == 'mmlu' else 'acc_norm') + ',none'
    ex = entry['examples']
    mine = sum(ex.values()) / len(ex)
    theirs = entry['metrics'][key]
    assert abs(mine - theirs) < 1e-9, (task, mine, theirs)
    return mine


def lmeval_section(out, models, units):
    pols = ['bf16', 'nvfp4', 'fo6'] + [f'ours-{u}' for u in units]
    rows, drows, tie_rows, data = [], [], [], {}
    for model in models:
        res = lmeval_results(out, model)
        data[model] = dict(accuracy={}, paired={})
        diffs = {}
        for task in TASKS:
            # registered check: every policy evaluated the same samples (lm-eval's doc / prompt / target hashes)
            digests = {p: res[p]['tasks'][task].get('sample_digest') for p in pols if p in res and task in res[p]['tasks']}
            if digests and all(digests.values()):
                if len(set(digests.values())) != 1:
                    raise SystemExit(f'{model} {task}: the policies evaluated different samples: {digests}')
                data[model].setdefault('samples_checked', {})[task] = 'equal sample_digest across ' + ', '.join(digests)
            elif STRICT and digests:
                raise SystemExit(f'{model} {task}: a report has no sample_digest (the same-samples check cannot run)')
            else:
                data[model].setdefault('samples_checked', {})[task] = 'NOT CHECKED: a report has no sample_digest'
            accs = {p: accuracy(res[p]['tasks'][task], task) for p in pols if p in res and task in res[p]['tasks']}
            data[model]['accuracy'][task] = accs
            rows.append([P.TITLES[model], TASK_TITLES[task]] + [f'{100 * accs[p]:.2f}' if p in accs else DASH for p in pols])
            n = {len(res[p]['tasks'][task]['examples']) for p in accs}
            drow = [P.TITLES[model], TASK_TITLES[task], '/'.join(str(v) for v in sorted(n)) or DASH]
            for u in units:
                for ref in ('fo6', 'nvfp4'):
                    pol, p = f'ours-{u}', None
                    if pol in accs and ref in accs:
                        a, b = res[pol]['tasks'][task]['examples'], res[ref]['tasks'][task]['examples']
                        assert set(a) == set(b), f'{model} {task}: example sets differ'
                        keys = sorted(a)
                        p = paired([a[k] for k in keys], [b[k] for k in keys])
                    diffs.setdefault((pol, ref), []).append(p)
                    data[model]['paired'][f'{task}:{pol}-{ref}'] = p
                    drow.append(fmt_diff(p, 100, 2))
            drows.append(drow)
        # the mean over tasks
        mean_row = [P.TITLES[model], 'mean of 5']
        for p in pols:
            vals = [data[model]['accuracy'][t].get(p) for t in TASKS]
            mean_row.append(f'{100 * sum(vals) / len(vals):.2f}' if all(v is not None for v in vals) else DASH)
        rows.append(mean_row)
        drow = [P.TITLES[model], 'mean of 5', '']
        for u in units:
            for ref in ('fo6', 'nvfp4'):
                ps = diffs.get((f'ours-{u}', ref), [])
                if ps and all(p is not None for p in ps):
                    mean = sum(p['mean'] for p in ps) / len(ps)
                    two_se = math.sqrt(sum(p['two_se'] ** 2 for p in ps)) / len(ps)
                    p = dict(mean=mean, two_se=two_se, significant=abs(mean) > two_se, n=len(ps))
                else:
                    p = None
                data[model]['paired'][f'mean:ours-{u}-{ref}'] = p
                drow.append(fmt_diff(p, 100, 2))
        drows.append(drow)
        data[model]['limit'] = sorted({str(v['limit']) for v in res.values()})
        # MMLU scores the raw per-choice log-likelihoods, which lm-eval computes from BF16 logits: the top two choices
        # can tie, and lm-eval's argmax then takes the earlier choice. The share of such questions, per policy:
        ties = {}
        for p in pols:
            lls = res.get(p, {}).get('tasks', {}).get('mmlu', {}).get('loglikelihoods')
            if lls:
                tops = [sorted(v['ll'])[-2:] for v in lls.values() if v['ll'] and len(v['ll']) > 1]
                ties[p] = sum(a == b for a, b in tops) / len(tops)
        data[model]['mmlu_top_ties'] = ties
        tie_rows.append([P.TITLES[model]] + [f'{100 * ties[p]:.1f}' if p in ties else DASH for p in pols])
    head = ['model', 'task'] + [POLICY_TITLES[p] for p in pols]
    dhead = ['model', 'task', 'examples'] + [f'Ours {u} − {POLICY_TITLES[r]}' for u in units for r in ('fo6', 'nvfp4')]
    text = ('### Downstream accuracy, % (lm-eval 0.4.11; MMLU 5-shot, the others 0-shot; acc_norm, MMLU acc)\n\n' +
            table(head, rows, 2) + '\n\n### Paired accuracy differences, percentage points, ± 2 SE; * = |Δ| > 2 SE\n\n' +
            table(dhead, drows, 2) +
            '\n\nMMLU questions whose top two choices tie in log-likelihood, % (lm-eval computes them from BF16 logits '
            'and its argmax takes the earlier choice; every policy is scored the same way):\n\n' +
            table(['model'] + [POLICY_TITLES[p] for p in pols], tie_rows))
    return text, data


# ----------------------------------------------------------------------------------------------------------- latency

def prefill_results(out, model):
    res = {}
    for d in sorted((out / 'latency' / model).glob('*')):
        runs = [r for r in (read(f) for f in sorted(d.glob('round*.json'))) if r is not None]
        if not runs:
            continue
        entry = {}
        for mode in ('eager', 'graph'):
            for spec in {s for r in runs for s in r[mode]}:
                ms = {r['round']: r[mode][spec]['ms'] for r in runs if 'ms' in r[mode].get(spec, {})}
                errors = sorted({r[mode][spec]['error'] for r in runs if 'error' in r[mode].get(spec, {})})
                entry.setdefault(mode, {})[spec] = dict(ms=statistics.median(ms.values()) if ms else None, rounds=ms,
                                                        errors=errors)
        flagged = {s for r in runs for s in r['host_bound']}
        entry['host_bound'] = {s: any(r['host_bound'].get(s, {}).get('flag') for r in runs) for s in flagged}
        entry['graph_equal_eager'] = all(g.get('logits_equal_eager', True) for r in runs for g in r['graph'].values())
        entry['quant_reuse'] = runs[0].get('quant_reuse')
        entry['rounds'] = len(runs)
        res[d.name] = entry
    return res


def latency_section(out, models, units, modes):
    """Prefill tables for `modes`: 'graph' (the primary numbers, main tables) and / or 'eager' (supplementary,
    appendix, with the host-bound mark)."""
    pols = ['bf16', 'nvfp4', 'fo6'] + (['nvfp4-wB', 'fo6-wB'] if '8x64' in units else [])
    for u in units:
        pols += [f'ours-{u}-nvfp4act', f'ours-{u}']
    ratios = []
    for u in units:
        ratios += [(f'ours-{u}', 'fo6'), (f'ours-{u}-nvfp4act', 'nvfp4')]
        if u == '8x64':
            ratios += [(f'ours-{u}', 'fo6-wB'), (f'ours-{u}-nvfp4act', 'nvfp4-wB')]
    parts, data = [], {}
    for model in models:
        res = prefill_results(out, model)
        data[model] = res
        if not res:
            continue
        for mode in modes:
            shapes = [s for s in P.PREFILL_SHAPES if any(s in res[p].get(mode, {}) for p in res)]
            if not shapes:
                continue
            rows, rrows = [], []
            for s in shapes:
                row = [s]
                for p in pols:
                    e = res.get(p, {}).get(mode, {}).get(s)
                    if e is None or e['ms'] is None:
                        row.append(DASH if e is None or not e['errors'] else 'failed')
                    else:
                        row.append(f"{e['ms']:.2f}{' †' if mode == 'eager' and res[p]['host_bound'].get(s) else ''}")
                rows.append(row)
                rrow = [s]
                for a, b in ratios:
                    ea, eb = res.get(a, {}).get(mode, {}).get(s), res.get(b, {}).get(mode, {}).get(s)
                    common = sorted(set(ea['rounds']) & set(eb['rounds'])) if ea and eb else []
                    if not common:
                        rrow.append(DASH)
                        continue
                    # paired within rounds (Part R's convention): the median over rounds of the per-round ratio
                    per = [100 * (ea['rounds'][r] / eb['rounds'][r] - 1) for r in common]
                    res[a].setdefault('ratios', {})[f'{mode}:{s}:vs:{b}'] = per
                    rrow.append(f'{statistics.median(per):+.1f} %' + (f' [{min(per):+.1f}, {max(per):+.1f}]' if len(per) > 1 else ''))
                rrows.append(rrow)
            rounds = sorted({res[p]['rounds'] for p in res})
            what = ('CUDA-graph prefill' if mode == 'graph' else
                    'eager prefill (supplementary; † host-bound: the graph is more than 5 % faster)')
            parts.append(f'#### {P.TITLES[model]}, {what}, ms, median of {rounds} round(s)\n\n' +
                         table(['batch x prompt'] + [POLICY_TITLES.get(p, p) for p in pols], rows) +
                         '\n\nOurs / reference − 1, same activation quantizer; paired within rounds: median [min, max] over rounds:\n\n' +
                         table(['batch x prompt'] + [f'{POLICY_TITLES[a]} vs {POLICY_TITLES[b]}' for a, b in ratios], rrows))
        errs = sorted({e for p in res.values() for s in p.get('graph', {}).values() for e in s['errors']})
        if errs:
            parts.append(f'CUDA-graph capture errors on {P.TITLES[model]}: ' + '; '.join(errs))
        if not all(p['graph_equal_eager'] for p in res.values()):
            parts.append(f'**{P.TITLES[model]}: a captured forward\'s logits differ from eager** (see tables.json)')
    title = ('### Prefill latency, CUDA graph (primary)' if modes == ('graph',) else
             '### Prefill latency, eager (supplementary)' if modes == ('eager',) else '### Prefill latency')
    return title + '\n\n' + ('\n\n'.join(parts) if parts else DASH), data


def gemm_section(out, models, units, prefill):
    cfgs = ['stock_wA', 'stock_wB'] + [c for u, c in (('16x64', 'mixed_16x64'), ('256x64', 'mixed_256x64'), ('8x64', 'n8k64_wB'))
                                       if u in units]
    ratios = [(c, 'stock_wA') for c in cfgs if c.startswith('mixed')] + \
             ([('n8k64_wB', 'stock_wB'), ('n8k64_wB', 'stock_wA')] if '8x64' in units else [])
    parts, detail, data = [], [], {}
    for model in models:
        r = read(out / 'gemm' / f'{model}.json')
        if r is None:
            continue
        projs = r['projections']
        reuse = next((v['quant_reuse'] for v in prefill.get(model, {}).values() if v.get('quant_reuse')), None)
        by = {}
        for row in r['rows']:
            by[(row['config'], row['proj'], row['tokens'])] = row
        tokens = sorted({row['tokens'] for row in r['rows']})
        timed = sorted({row['proj'] for row in r['rows']}, key=list(projs).index)
        complete = len(timed) == len(projs)
        sums, qsums = {}, {}
        for t in tokens:
            for c in cfgs + ['stock_wA_nvfp4']:
                if all((c, p, t) in by for p in timed):
                    sums[(c, t)] = sum(projs[p]['modules'] * by[(c, p, t)]['gemm_us'] for p in timed)
            for c, act in (('stock_wA', 'four_over_six_rows'), ('stock_wA_nvfp4', 'nvfp4_rows')):
                if all((c, p, t) in by for p in timed):
                    launches = {p: projs[p]['modules'] * (1 - (reuse[p]['quant_reused'] / reuse[p]['calls'] if reuse and p in reuse
                                                               and reuse[p]['calls'] else 0.0)) for p in timed}
                    qsums[(act, t)] = sum(launches[p] * by[(c, p, t)]['quant_us'] for p in timed)
        rows = [[t] + [f'{sums[(c, t)]:.0f}' if (c, t) in sums else DASH for c in cfgs] +
                [f'{100 * (sums[(a, t)] / sums[(b, t)] - 1):+.1f} %' if (a, t) in sums and (b, t) in sums else DASH for a, b in ratios]
                for t in tokens]
        qrows = [[t] + [f'{qsums[(a, t)]:.0f}' if (a, t) in qsums else DASH for a in ('nvfp4_rows', 'four_over_six_rows')] +
                 [f"{100 * (qsums[('four_over_six_rows', t)] / qsums[('nvfp4_rows', t)] - 1):+.1f} %"
                  if ('four_over_six_rows', t) in qsums and ('nvfp4_rows', t) in qsums else DASH] for t in tokens]
        scope = 'every quantized text Linear' if complete else f'ONLY {", ".join(timed)} (a subset)'
        parts.append(f'#### {P.TITLES[model]}: GEMM kernel time per forward, µs ({scope}; CUPTI median)\n\n' +
                     table(['T'] + [GEMM_TITLES[c] for c in cfgs] + [f'{GEMM_TITLES[a]} vs {GEMM_TITLES[b]}' for a, b in ratios], rows) +
                     '\n\nFourOverSix and NVFP4 run the same stock GEMM (stock wA); their activation quantizers differ '
                     '(supplementary):\n\n' + table(['T', 'NVFP4 quantizer, µs / forward', 'FourOverSix quantizer, µs / forward',
                                                     'FourOverSix vs NVFP4'], qrows) +
                     ('\n\nQuantizer launches net of the reuse measured in step 05.' if reuse else
                      '\n\nNo step-05 record: every Linear counted as quantizing its own input (an upper bound).'))
        for p in timed:
            for t in tokens:
                detail.append([P.TITLES[model], p, 'x'.join(str(v) for v in projs[p]['shape']), projs[p]['modules'], t] +
                              [f"{by[(c, p, t)]['gemm_us']:.1f}" + (f" (w{by[(c, p, t)]['width']})" if by[(c, p, t)]['width'] else '')
                               if (c, p, t) in by else DASH for c in cfgs] +
                              [f"{by[(c, p, t)]['quant_us']:.1f}" if (c, p, t) in by else DASH for c in ('stock_wA_nvfp4', 'stock_wA')])
        data[model] = dict(per_forward_gemm_us={f'{c}@{t}': v for (c, t), v in sums.items()},
                           per_forward_quant_us={f'{a}@{t}': v for (a, t), v in qsums.items()}, quant_reuse=reuse,
                           complete=complete, tags={p: projs[p]['weights'] for p in timed})
    text = '### GEMM latency\n\n' + ('\n\n'.join(parts) if parts else DASH)
    dtext = ('### GEMM kernel time per shape, µs (CUPTI median; wN = CTA tile width picked by the tile table)\n\n' +
             table(['model', 'projection', 'out x in', 'modules', 'T'] + [GEMM_TITLES[c] for c in cfgs] +
                   ['NVFP4 quantizer', 'FourOverSix quantizer'], detail, 3)) if detail else ''
    return text, dtext, data


def main():
    args = P.setup(P.parser(__doc__).parse_args())
    global STRICT
    STRICT = not args.smoke
    if args.smoke:
        args.models = ['llama8b']
    tdir = args.out / 'tables'
    tdir.mkdir(exist_ok=True)
    note = ('**SMOKE OUTPUT** (tiny subsets; plumbing only, not results)\n\n' if args.smoke else '')
    everything = {}
    docs = {}
    prefill_all = {m: prefill_results(args.out, m) for m in args.models}
    _, gemm_detail, _ = gemm_section(args.out, args.models, list(args.units), prefill_all)     # every unit's kernel
    for name, units in (('main', P.MAIN_UNITS), ('appendix', P.APPENDIX_UNITS)):
        units = [u for u in units if u in args.units]
        ppl_text, ppl_data = ppl_section(args.out, args.models, units)
        lm_text, lm_data = lmeval_section(args.out, args.models, units)
        lat_text, lat_data = latency_section(args.out, args.models, units, ('graph',))     # CUDA graph: the primary numbers
        eager_text = ''
        if name == 'appendix':        # eager prefill, every unit: supplementary, with the host-bound mark
            eager_text, _ = latency_section(args.out, args.models, list(args.units), ('eager',))
        gemm_text, _, gemm_data = gemm_section(args.out, args.models, units, prefill_all)
        title = ('Main tables: 8x64 and 16x64' if name == 'main' else
                 'Appendix: 256x64, eager prefill (every unit), and the per-shape GEMM detail')
        docs[name] = '\n\n'.join(x for x in (f'# {title}', note.strip(), ppl_text, lm_text, lat_text, gemm_text, eager_text,
                                             gemm_detail if name == 'appendix' else '') if x) + '\n'
        everything[name] = dict(units=units, ppl=ppl_data, downstream=lm_data, prefill=lat_data, gemm=gemm_data)
    for name, text in docs.items():
        (tdir / f'{name}.md').write_text(text)
    (tdir / 'tables.json').write_text(json.dumps(everything, indent=1, default=str) + '\n')
    print('wrote', tdir / 'main.md', tdir / 'appendix.md', tdir / 'tables.json')


if __name__ == '__main__':
    main()
