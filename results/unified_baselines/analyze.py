"""Unified comparison of TM-OPT+TC (OURS), QAT and scale-only (SCALE) (PROTOCOL.md). Reads the copied records in runs/.

Per model (Llama-3.1-8B, Mistral-7B-v0.3), from runs/<model>/eval/report.json (run_ppl_deploy.py, NativeLinear (c)):
- PPL of BF16, NVFP4, FourOverSix, OURS 8x64 / 16x64, SCALE, QAT (and QAT fake (c), the deployment check);
- paired ΔNLL ± 2 SE per window, 'better / n.s. / worse' by the sign of mean ± 2 SE:
  every arm − FourOverSix; OURS − SCALE; OURS − QAT; SCALE − QAT;
- the QAT deployment check: NativeLinear (c) − fake (c) of the QAT weights, B2's criterion |mean| ≤ 2 SE per corpus;
- the regression check: the reference policies equal the Parts 2-3 evaluation (and Llama's SCALE equals Task 2's)
  window for window.
Also: the learning-rate choices, the calibration cost (Task 1's definition), the run checks (the deployed run repeats
the chosen development run's per-step KL; nothing outside the scoped Linears changed; the trained parameters), the
artifact checks, and the phase-3 memory probes.

    python results/unified_baselines/analyze.py   -> unified.{json,md}
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tm_opt'))
from analyze_items import paired  # noqa: E402

RUNS = HERE / 'runs'
D = ('wiki', 'c4')
MODELS = {'llama8b': 'Llama-3.1-8B', 'mistral7b': 'Mistral-7B-v0.3'}
ARMS = ('ours-8x64', 'ours-16x64', 'scale', 'qat')
INCLUDED = ('model_load', 'data_load', 'teacher_precompute', 'candidate_packing', 'preparation', 'training', 'write_output',
            'save_state')
V = {'significantly better': 'better', 'significantly worse': 'worse', 'not significantly different': 'n.s.'}
# what an arm trains, for runs that predate the 'trained_parameters' record (OURS: Task 1; Llama's SCALE: Task 2)
FALLBACK_TRAINED = {'ours-8x64': 'tile logits (not model parameters)', 'ours-16x64': 'tile logits (not model parameters)',
                    'scale': 'block-scale factors (not model parameters; Task 2 run)', 'qat': '— (not recorded)'}


def load(p):
    return json.loads(Path(p).read_text())


def nll(ev, label):
    return {d: ev[label]['evaluation'][d]['nll'] for d in D}


def fmt(p):
    return f"{p['mean']:+.5f} ± {p['two_se']:.5f} ({V[p['verdict']]})"


def phases(report):
    res = report['resources']
    return res['by_phase'] if 'by_phase' in res else res['phases']['by_phase']


def cost(report):
    by = phases(report)
    inc = [v for k, v in by.items() if k in INCLUDED]
    res = report['resources']
    ru = res.get('cpu_peak_rss_gib') if 'cpu_peak_rss_gib' in res else res.get('phases', {}).get('cpu_peak_rss_gib')
    return dict(seconds=sum(v['seconds'] for v in inc), gpu_allocated_gib=max(v['gpu_peak_allocated_gib'][0] for v in inc),
                gpu_reserved_gib=max(v['gpu_peak_reserved_gib'][0] for v in inc), host_sampled_gib=max(v['host_peak_rss_gib'] for v in inc),
                host_ru_maxrss_gib=ru)


def run_paths(model):
    """The deployed (no-dev) calibration record of every arm."""
    nd = HERE.parent / 'nodev_cost' / 'runs'
    scale = HERE.parent / 'scale_additivity' / 'runs' / 'scale_nodev' if model == 'llama8b' else RUNS / model / 'scale_nodev'
    return {'ours-8x64': nd / f'nd_tc_{model}_8x64', 'ours-16x64': nd / f'nd_tc_{model}_16x64', 'scale': scale,
            'qat': RUNS / model / 'qat_nodev'}


def lr_choice(model, arm):
    if model == 'llama8b' and arm == 'scale':
        c = load(HERE.parent / 'scale_additivity' / 'runs' / 'lr_choice.json')      # Task 2's rule and runs
        return dict(chosen=c['chosen'], final_dev_kl=c['final_dev_kl'], extensions_run=[] if not c.get('extension') else [c['extension']],
                    source='results/scale_additivity (Task 2)')
    p = RUNS / model / f'lr_choice_{arm}.json'
    return dict(load(p), source=str(p.relative_to(HERE))) if p.exists() else None


def dev_run(model, arm, rate):
    if model == 'llama8b' and arm == 'scale':
        return HERE.parent / 'scale_additivity' / 'runs' / f'scale_dev_lr{rate}'
    return RUNS / model / f'{arm}_dev_lr{rate}'


def model_section(model, out, md):
    p = RUNS / model / 'eval' / 'report.json'
    if not p.exists():
        return
    ev = load(p)['evaluations']
    res = out[model] = dict(ppl={k: {d: v['evaluation'][d]['ppl'] for d in D} for k, v in ev.items()}, comparisons={})
    # regression: the reference policies reproduce the earlier evaluations window for window
    parts23 = load(HERE.parent / 'deploy_eval' / 'runs' / model / 'report.json')['evaluations']
    same = {k: all(ev[k]['evaluation'][d]['nll'] == parts23[p]['evaluation'][d]['nll'] for d in D)
            for k, p in (('BF16', 'BF16'), ('NVFP4', 'NVFP4'), ('FourOverSix', 'FourOverSix'), ('ours-8x64', 'tc-8x64'),
                         ('ours-16x64', 'tc-16x64')) if k in ev}
    if model == 'llama8b' and 'scale' in ev:
        t2 = load(HERE.parent / 'scale_additivity' / 'runs' / 'llama8b' / 'report.json')['evaluations']['scale']
        same['scale (Task 2)'] = all(ev['scale']['evaluation'][d]['nll'] == t2['evaluation'][d]['nll'] for d in D)
    res['reproduces_earlier_evaluations'] = same
    pairs = {f'{a} − FourOverSix': (a, 'FourOverSix') for a in ('NVFP4',) + ARMS}
    pairs.update({f'{o} − SCALE': (o, 'scale') for o in ('ours-8x64', 'ours-16x64')})
    pairs.update({f'{o} − QAT': (o, 'qat') for o in ('ours-8x64', 'ours-16x64')})
    pairs['SCALE − QAT'] = ('scale', 'qat')
    for name, (a, b) in pairs.items():
        if a in ev and b in ev:
            res['comparisons'][name] = {d: paired(nll(ev, a)[d], nll(ev, b)[d]) for d in D}
    if 'qat' in ev and 'qat-fake' in ev:
        chk = {}
        for d in D:
            x, y = nll(ev, 'qat')[d], nll(ev, 'qat-fake')[d]
            p_ = paired(x, y)
            chk[d] = dict(p_, max_abs=max(abs(a - b) for a, b in zip(x, y)),
                          ppl_difference=ev['qat']['evaluation'][d]['ppl'] - ev['qat-fake']['evaluation'][d]['ppl'],
                          negligible=abs(p_['mean']) <= p_['two_se'])
        res['qat_native_vs_fake'] = chk
    md += [f'## {MODELS[model]}', '', 'NativeLinear (c) PPL, WikiText-2 / C4 (released windows):', '',
           '| policy | WikiText-2 | C4 |', '|---|---:|---:|']
    for k in ('BF16', 'NVFP4', 'FourOverSix') + ARMS + ('qat-fake',):
        if k in res['ppl']:
            md.append(f"| {k}{' (fake (c), check only)' if k == 'qat-fake' else ''} | {res['ppl'][k]['wiki']:.4f} | {res['ppl'][k]['c4']:.4f} |")
    md += ['', 'Reference policies equal to the earlier evaluations, every window\'s NLL: ' +
           ', '.join(f"{k} {'yes' if v else '**no**'}" for k, v in same.items()) + '.', '',
           'Paired ΔNLL ± 2 SE per window (negative: the first is better):', '', '| comparison | WikiText-2 | C4 |', '|---|---|---|']
    for name, p_ in res['comparisons'].items():
        md.append(f"| {name} | {fmt(p_['wiki'])} | {fmt(p_['c4'])} |")
    if 'qat_native_vs_fake' in res:
        md += ['', 'QAT deployment check, NativeLinear (c) − fake (c) of the QAT weights (B2 criterion |mean| ≤ 2 SE):', '',
               '| corpus | mean ± 2 SE | max abs ΔNLL | ΔPPL | negligible |', '|---|---|---:|---:|---|']
        for d, c in res['qat_native_vs_fake'].items():
            md.append(f"| {d} | {c['mean']:+.6f} ± {c['two_se']:.6f} | {c['max_abs']:.5f} | {c['ppl_difference']:+.5f} | "
                      f"{'yes' if c['negligible'] else '**no**'} |")
    # learning rates
    md += ['', 'Learning rates (development set; not part of any calibration cost):', '',
           '| arm | rates run (final dev KL) | extensions | chosen |', '|---|---|---|---|']
    res['lr'] = {}
    for arm in ('scale', 'qat'):
        c = lr_choice(model, arm)
        if not c:
            continue
        res['lr'][arm] = c
        kls = ', '.join(f"{r}: {k:.5f}" for r, k in sorted(c['final_dev_kl'].items(), key=lambda x: float(x[0])))
        md.append(f"| {arm} | {kls} | {', '.join(c.get('extensions_run') or []) or 'none'} | {c.get('chosen') or '**none: ' + c.get('stop', '') + '**'} |")
    # runs, costs and checks
    md += ['', 'Calibration cost (Task 1 definition: model load, fit data and teacher, preparation, training, writing the '
           'output; no development set) and run checks:', '',
           '| arm | calibration | GPU allocated / reserved | host sampled / ru_maxrss | deployed run = chosen dev run (per-step KL) | '
           'outside the scoped Linears unchanged | trained |', '|---|---:|---:|---:|---|---|---|']
    res['runs'] = {}
    for arm, path in run_paths(model).items():
        rp = Path(path) / 'report.json'
        if not rp.exists():
            continue
        r = load(rp)
        c = cost(r)
        row = dict(cost=c, run=str(path))
        if arm in ('scale', 'qat'):
            ch = res['lr'].get(arm, {}).get('chosen')
            dv = dev_run(model, arm, ch) / 'report.json' if ch else None
            row['repeats_chosen_dev_run'] = bool(dv and dv.exists() and [x['kl'] for x in r['log']] == [x['kl'] for x in load(dv)['log']])
            row['outside_scope_unchanged'] = r.get('outside_scope_unchanged')
            row['trained_parameters'] = r.get('trained_parameters')
        res['runs'][arm] = row
        rep = {True: 'yes', False: '**no**', None: '— (not recorded)'}
        md.append(f"| {arm} | {c['seconds'] / 60:.1f} min | {c['gpu_allocated_gib']:.1f} / {c['gpu_reserved_gib']:.1f} GiB | "
                  f"{c['host_sampled_gib']:.1f} / {c['host_ru_maxrss_gib']:.1f} GiB | "
                  f"{rep[row['repeats_chosen_dev_run']] if 'repeats_chosen_dev_run' in row else '— (fixed lr; Task 1: map = committed)'} | "
                  f"{rep[row.get('outside_scope_unchanged')] if arm in ('scale', 'qat') else '— (no model parameter trained)'} | "
                  f"{(row.get('trained_parameters') or {}).get('group', FALLBACK_TRAINED[arm])} |")
    # artifacts
    md += ['', 'Artifacts (new in this study):', '', '| artifact | exporter checks | ownership (exact, format mismatches) | note |',
           '|---|---|---|---|']
    res['artifacts'] = {}
    for arm in ('scale', 'qat'):
        a = HERE / 'artifacts' / f'{model}_{arm}'
        if model == 'llama8b' and arm == 'scale':
            continue                                    # Task 2's artifact, checked there
        if not (a / 'artifact.json').exists():
            continue
        meta, own = load(a / 'artifact.json'), load(a / 'ownership.json')
        s = own.get('summary', own)
        rec = meta['note']['record']
        note = ''
        if arm == 'qat':
            tw = rec.get('trained_weights', {})
            note = f"trained weights: {tw.get('modules_changed')} of {tw.get('modules')} matrices changed"
        else:
            ls = rec.get('learned_scales', {})
            note = f"scales moved on {ls.get('blocks_scale_moved', 0):,} of {ls.get('blocks', 0):,} blocks"
        res['artifacts'][arm] = dict(ownership=s, record=dict(trained_weights=rec.get('trained_weights'), learned_scales=rec.get('learned_scales'),
                                                               candidates=rec.get('candidates')))
        packed = 'packed = fake quant of the trained weights, bitwise' if arm == 'qat' else \
            'packed = learned-scale fake-quant weight, value-equal (-0 packs as +0)'
        md.append(f"| {model}_{arm} | weights = calibration record, candidates equal ({rec['candidates']['modules']} modules), {packed} | "
                  f"{s.get('all_exact')}, {s.get('format_mismatches')} | {note} |")
    md.append('')


def probes_section(out, md):
    base = RUNS / 'probes'
    if not base.exists():
        return
    rows = []
    for d in sorted(base.iterdir()):
        rp = d / 'report.json'
        if not rp.exists():
            continue
        r = load(rp)
        c = r['config']
        oom = r.get('oom')
        by = phases(r) if 'resources' in r else {}
        gpu = oom['gpu_peak_allocated_gib'] if oom else max((v['gpu_peak_allocated_gib'][0] for v in by.values()), default=None)
        res = oom['gpu_peak_reserved_gib'] if oom else max((v['gpu_peak_reserved_gib'][0] for v in by.values()), default=None)
        rows.append(dict(name=d.name, model=c.get('model'), arm=c['arm'], optimizer=c['optimizer'], micro_batch=c['micro_batch'],
                         checkpointing=c['checkpointing'], status=r['status'], gpu_allocated_gib=gpu, gpu_reserved_gib=res,
                         steady_step_seconds=r.get('steady_step_seconds'), oom=oom and oom['message'][:160],
                         trainable_parameters=(r.get('memory_computed') or {}).get('trainable_parameters')))
    out['probes'] = rows
    md += ['## Memory probes (phase 3; 3 optimizer steps of 8 sequences, no deployed output)', '',
           '| probe | model | arm | optimizer | micro-batch | checkpointing | status | peak GPU allocated / reserved | s per step |',
           '|---|---|---|---|---:|---|---|---:|---:|']
    num = lambda v, f='{:.1f}': '—' if v is None else f.format(v)  # noqa: E731
    for x in rows:
        md.append(f"| {x['name']} | {x['model']} | {x['arm']} | {x['optimizer']} | {x['micro_batch']} | "
                  f"{'yes' if x['checkpointing'] else 'no'} | {x['status']} | {num(x['gpu_allocated_gib'])} / "
                  f"{num(x['gpu_reserved_gib'])} GiB | {num(x['steady_step_seconds'], '{:.2f}')} |")
    md.append('')


def main():
    out, md = {}, ['# Unified comparison: TM-OPT+TC, QAT and scale-only (generated by analyze.py)', '']
    for model in MODELS:
        model_section(model, out, md)
    probes_section(out, md)
    (HERE / 'unified.json').write_text(json.dumps(out, indent=1) + '\n')
    (HERE / 'unified.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
