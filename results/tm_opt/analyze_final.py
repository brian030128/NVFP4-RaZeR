"""Part R tables (PROTOCOL_QR.md): R1 calibration cost and R3 PPL over Llama-3.1-8B, Mistral-7B-v0.3, Phi-4 and
Qwen3.8-27B, for TM-OPT and TM-OPT+TC at 8x64, 16x64, 256x64, with NVFP4, FourOverSix, BF16 and (where it exists)
MR-OPT; and the Part Q tables (Qwen3.8-27B batch probes and runs). R2 (latency) is latency/analyze_latency.py.

python results/tm_opt/analyze_final.py qwen   -> final_qwen.{json,md}
python results/tm_opt/analyze_final.py cost   -> final_cost.{json,md}
python results/tm_opt/analyze_final.py ppl    -> final_ppl.{json,md}
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_items import D, M, R, UNITS, fmt, load, nll, paired  # noqa: E402

MODELS = (('llama8b', 'Llama-3.1-8B'), ('mistral7b', 'Mistral-7B-v0.3'), ('phi4', 'Phi-4'), ('qwen27b', 'Qwen3.8-27B'))
QAT_C1 = Path('results/cost_comparison/runs/grid_C1_lr1e-6/report.json')
SM120_REFERENCE = Path('/home/dev/n16k64_campaign/sm120_bench/sm120/eval/reference')


def tm_dir(model, method, unit):
    if model == 'qwen27b':
        return R / f'q_{method}_qwen27b_{unit}'
    if method == 'tmopt':
        return R / 'g2_tmopt' if (model == 'llama8b' and unit == '8x64') else R / f'tm_{model}_{unit}'
    return R / f'tc_{model}_{unit}'


SETUP = ('model_load', 'data_load', 'teacher_precompute', 'candidate_packing', 'initial_dev_eval')


def setup_text(c):
    sp = c['setup_parts']
    return f"{c['setup_seconds'] / 60:.1f} min ({' / '.join(f'{sp.get(k, 0):.0f}' for k in SETUP)} s)"


def cost_row(r, kind):
    res = r['resources']
    if kind == 'mropt':
        ph = res['phases']['by_phase']
        return dict(epoch_seconds=None, selection_seconds=res['optimization_seconds'], setup_seconds=res['setup_seconds'],
                    setup_parts={k: ph[k]['seconds'] for k in SETUP if k in ph},
                    gpu_peak_allocated_gib=res['gpu_peak_allocated_gib'][0], gpu_peak_reserved_gib=res['gpu_peak_reserved_gib'][0],
                    host_peak_rss_gib=res['cpu_peak_rss_gib'], rounds=res['scoring_passes'], dev_evaluations=res['development_evaluations'],
                    scoring_s_per_pass=ph['scoring']['seconds'] / ph['scoring']['count'])
    epochs = r['epochs']
    ph = res['phases']['by_phase']
    return dict(epoch_seconds=sum(e['epoch_seconds'] for e in epochs) / len(epochs), selection_seconds=r['optimization_seconds'],
                setup_seconds=r['setup_seconds'], setup_parts={k: ph[k]['seconds'] for k in SETUP if k in ph},
                gpu_peak_allocated_gib=res['gpu_peak_allocated_gib'][0], gpu_peak_reserved_gib=res['gpu_peak_reserved_gib'][0],
                host_peak_rss_gib=res['cpu_peak_rss_gib'], micro_batch=r['args']['batch'], accum=r['args']['accum'],
                e0m3_units=r['final_e0m3_units'])


def cost():
    out, lines = {}, ['| model | unit | method | per epoch | selection | setup (model / data / teacher / packing / initial dev) | peak GPU allocated / reserved | host RSS | micro-batch × accum |',
                      '|---|---|---|---:|---:|---|---:|---:|---|']
    for model, title in MODELS:
        for unit in UNITS:
            for method, name in (('tmopt', 'TM-OPT'), ('tc', 'TM-OPT+TC')):
                p = tm_dir(model, method, unit) / 'report.json'
                if not p.exists() or 'resources' not in load(p):
                    continue
                c = cost_row(load(p), method)
                out[f'{model} {unit} {name}'] = c
                lines.append(f"| {title} | {unit} | {name} | {c['epoch_seconds']:.1f} s | {c['selection_seconds'] / 60:.1f} min | "
                             f"{setup_text(c)} | {c['gpu_peak_allocated_gib']:.1f} / {c['gpu_peak_reserved_gib']:.1f} GiB | "
                             f"{c['host_peak_rss_gib']:.1f} GiB | {c['micro_batch']} × {c['accum']} |")
            p = M / model / f'mropt_{unit}' / 'report.json'
            if p.exists():
                c = cost_row(load(p), 'mropt')
                out[f'{model} {unit} MR-OPT'] = c
                lines.append(f"| {title} | {unit} | MR-OPT (reference) | {c['rounds']} rounds, {c['scoring_s_per_pass']:.0f} s/pass | "
                             f"{c['selection_seconds'] / 60:.1f} min | {setup_text(c)} | {c['gpu_peak_allocated_gib']:.1f} / "
                             f"{c['gpu_peak_reserved_gib']:.1f} GiB | {c['host_peak_rss_gib']:.1f} GiB | 16 / 8 |")
    q = load(QAT_C1)
    out['llama8b QAT C1'] = dict(training_seconds=q['training_seconds'], setup_seconds=q['setup_seconds'],
                                 gpu_peak_allocated_gib=q['resources']['gpu_peak_allocated_gib'][0],
                                 gpu_peak_reserved_gib=q['resources']['gpu_peak_reserved_gib'][0], host_peak_rss_gib=q['resources']['cpu_peak_rss_gib'])
    lines.append(f"| Llama-3.1-8B | — | QAT C1 (reference: full-weight QAT, 1 epoch, non-deterministic) | {q['training_seconds']:.1f} s | "
                 f"{q['training_seconds'] / 60:.1f} min (1 epoch) | {q['setup_seconds'] / 60:.1f} min | "
                 f"{out['llama8b QAT C1']['gpu_peak_allocated_gib']:.1f} / {out['llama8b QAT C1']['gpu_peak_reserved_gib']:.1f} GiB | "
                 f"{out['llama8b QAT C1']['host_peak_rss_gib']:.1f} GiB | 8 × 1 |")
    probe = R / 'tc_probe_nondet' / 'report.json'
    if probe.exists():
        e = [x['epoch_seconds'] for x in load(probe)['epochs']]
        out['llama8b 8x64 TM-OPT+TC non-deterministic'] = dict(epoch_seconds=e)
        lines.append(f"| Llama-3.1-8B | 8x64 | TM-OPT+TC, non-deterministic (3-epoch probe) | {sum(e) / len(e):.1f} s | — | — | — | — | 8 × 1 |")
    (HERE / 'final_cost.json').write_text(json.dumps(out, indent=1) + '\n')
    (HERE / 'final_cost.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def evaluations(model):
    """{backend: {label: nll}} and {label: ppl} for every map of the model, paired only where FourOverSix repeats."""
    if model == 'qwen27b':
        nat, fake, bf16 = (load(R / f'q_eval_{b}' / 'report.json')['evaluations'] for b in ('native', 'fake', 'bf16'))
        out = {'native': {k: v for k, v in nat.items()}, 'fake': {k: v for k, v in fake.items()}}
        out['fake']['BF16'] = bf16['BF16']
        return out, dict(single_process=True)
    sources = {'native': [R / f'items_{model}_eval_native', R / f'tc_{model}_eval_native', M / model / 'eval_native'],
               'fake': [R / f'items_{model}_eval_fake', R / f'tc_{model}_eval_fake', M / model / 'eval_fake', M / model / 'eval_bf16']}
    out, checks = {}, {}
    for backend, dirs in sources.items():
        merged, ref = {}, None
        for d in dirs:
            ev = load(d / 'report.json')['evaluations']
            if 'FourOverSix' in ev:
                if ref is None:
                    ref = nll(ev, 'FourOverSix')
                checks[f'{backend} {d.name}'] = nll(ev, 'FourOverSix') == ref
            for k, v in ev.items():
                merged.setdefault(k, v)
        out[backend] = merged
    return out, checks


def ppl():
    out, lines = {}, []
    for model, title in MODELS:
        try:
            ev, checks = evaluations(model)
        except FileNotFoundError:
            continue
        rows = {}
        for backend in ('native', 'fake'):
            e = ev[backend]
            labels = ['FourOverSix', 'NVFP4'] + [f'{m}-{u}' for m in ('tmopt', 'tc', 'mropt') for u in UNITS] + (['BF16'] if backend == 'fake' else [])
            for lab in labels:
                if lab not in e:
                    continue
                row = dict(ppl={d: e[lab]['evaluation'][d]['ppl'] for d in D})
                if lab != 'BF16':
                    row['vs_fourover6'] = {d: paired(nll(e, lab)[d], nll(e, 'FourOverSix')[d]) for d in D} if lab != 'FourOverSix' else None
                    row['vs_nvfp4'] = {d: paired(nll(e, lab)[d], nll(e, 'NVFP4')[d]) for d in D} if lab != 'NVFP4' else None
                rows[f'{backend} {lab}'] = row
            for u in UNITS:
                if f'tmopt-{u}' in e and f'tc-{u}' in e:
                    rows[f'{backend} tc-{u} minus tmopt-{u}'] = {d: paired(nll(e, f'tc-{u}')[d], nll(e, f'tmopt-{u}')[d]) for d in D}
        out[model] = dict(rows=rows, checks=checks)
        lines += [f'### {title}', '', '| backend | map | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | ΔWiki vs NVFP4 | ΔC4 vs NVFP4 |',
                  '|---|---|---:|---:|---|---|---|---|']
        for key, row in rows.items():
            if 'minus' in key:
                continue
            backend, lab = key.split(' ', 1)
            vf, vn = row.get('vs_fourover6'), row.get('vs_nvfp4')
            lines.append(f"| {backend} | {lab} | {row['ppl']['wiki']:.4f} | {row['ppl']['c4']:.4f} | "
                         f"{fmt(vf['wiki']) if vf else '—'} | {fmt(vf['c4']) if vf else '—'} | {fmt(vn['wiki']) if vn else '—'} | {fmt(vn['c4']) if vn else '—'} |")
        lines += ['', '| backend | TM-OPT+TC minus TM-OPT | ΔWiki | ΔC4 |', '|---|---|---|---|']
        for key, row in rows.items():
            if 'minus' in key:
                backend, rest = key.split(' ', 1)
                lines.append(f"| {backend} | {rest.split(' minus')[0].replace('tc-', '')} | {fmt(row['wiki'])} | {fmt(row['c4'])} |")
        lines += ['', f'Checks (FourOverSix window NLLs repeat across the merged processes): {json.dumps(checks)}', '']
    (HERE / 'final_ppl.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    (HERE / 'final_ppl.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def qwen_checks():
    """The Qwen evaluation processes: evaluation windows, map integrity, and the runs' own evaluations."""
    ref = {d: json.loads((SM120_REFERENCE / f'qwen27b_windows_{d}.json').read_text())['token_sha256'] for d in ('wiki', 'c4')}
    reports = {b: load(R / f'q_eval_{b}' / 'report.json') for b in ('native', 'fake', 'bf16')}
    windows = {b: dict(wiki=r['data']['wiki']['token_sha256'] == ref['wiki'], c4=r['data']['c4_paper']['token_sha256'] == ref['c4'])
               for b, r in reports.items()}
    maps, own = {}, {}
    for method in ('tmopt', 'tc'):
        for unit in UNITS:
            run = load(R / f'q_{method}_qwen27b_{unit}' / 'report.json')
            for b in ('native', 'fake'):
                e = reports[b]['evaluations'][f'{method}-{unit}']
                # a 16x64 or 256x64 map runs as whole 8x64 units in the evaluation process; its own count is recorded next to it
                tiles = e.get(f'e0m3_units_{unit}', e['e0m3_units'])
                maps[f'{b} {method}-{unit}'] = (e['sha256'] == run['map_sha256'] and tiles == run['final_e0m3_units']
                                                 and not e.get('lean_map_mismatches') and not e.get('native_map_mismatches'))
            nat = reports['native']['evaluations'][f'{method}-{unit}']['evaluation']
            own[f'{method}-{unit}'] = all(run['evaluation'][d]['nll'] == nat[d]['nll'] for d in D)
    return dict(windows_equal_sm120_reference=windows, maps_match_runs_and_no_mismatches=all(maps.values()),
                map_checks=maps, run_evaluation_repeats_in_native_process=own)


def qwen():
    """Part Q: the batch probes, and per run the cost, E0M3 tiles and the development-KL curve."""
    out, lines = dict(probes={}, runs={}), ['| probe | micro-batch × accum | status | training-phase peak GPU allocated / reserved | host RSS peak | step seconds |',
                                            '|---|---|---|---:|---:|---:|']
    for b in (8, 4, 2, 1):
        p = R / f'qwen_probe_b{b}' / 'report.json'
        if not p.exists():
            continue
        r = load(p)
        if r['status'] not in ('out_of_memory', 'probe_complete'):
            continue
        res = r['out_of_memory']['phases'] if r['status'] == 'out_of_memory' else r['resources']['phases']
        train = res['by_phase'].get('training', {})
        probe = r.get('probe', {})
        out['probes'][b] = dict(status=r['status'], accum=r['args']['accum'], training_peak_allocated_gib=train.get('gpu_peak_allocated_gib', [None])[0],
                                training_peak_reserved_gib=train.get('gpu_peak_reserved_gib', [None])[0],
                                host_peak_rss_gib=max(v['host_peak_rss_gib'] for v in res['by_phase'].values()),
                                step_seconds=probe.get('step_seconds'), message=r.get('out_of_memory', {}).get('message', '')[:160])
        x = out['probes'][b]
        status = {'probe_complete': 'FIT', 'out_of_memory': 'OOM'}.get(x['status'], x['status'])
        step = '—' if x['step_seconds'] is None else f"{x['step_seconds']:.1f}"
        lines.append(f"| qwen_probe_b{b} | {b} × {x['accum']} | {status} | {x['training_peak_allocated_gib']:.1f} / "
                     f"{x['training_peak_reserved_gib']:.1f} GiB | {x['host_peak_rss_gib']:.1f} GiB | {step} |")
    lines += ['', '| method | unit | micro-batch × accum | E0M3 tiles | per epoch | selection | setup (model / data / teacher / packing / initial dev) | peak GPU allocated / reserved | host RSS | dev KL initial → final |',
              '|---|---|---|---:|---:|---:|---|---:|---:|---|']
    for method, title in (('tmopt', 'TM-OPT'), ('tc', 'TM-OPT+TC')):
        for unit in UNITS:
            p = R / f'q_{method}_qwen27b_{unit}' / 'report.json'
            if not p.exists():
                continue
            r = load(p)
            if 'resources' not in r:
                continue
            c = cost_row(r, method)
            ph = r['resources']['phases']['by_phase']
            c.update(tiles=r['tiles'], initial_dev_kl=r['initial_dev']['kl'], final_dev_kl=r['final_dev']['kl'],
                     dev_kl_curve={e['epoch'] + 1: e['dev_kl'] for e in r['epochs'] if 'dev_kl' in e},
                     e0m3_curve={e['epoch'] + 1: e['e0m3_units'] for e in r['epochs'] if 'dev_kl' in e},
                     initial_dev_seconds=ph.get('initial_dev_eval', {}).get('seconds'))
            out['runs'][f'{method} {unit}'] = c
            lines.append(f"| {title} | {unit} | {c['micro_batch']} × {c['accum']} | {c['e0m3_units']:,} ({100 * c['e0m3_units'] / c['tiles']:.2f} %) | "
                         f"{c['epoch_seconds']:.1f} s | {c['selection_seconds'] / 60:.1f} min | {setup_text(c)} | "
                         f"{c['gpu_peak_allocated_gib']:.1f} / {c['gpu_peak_reserved_gib']:.1f} GiB | {c['host_peak_rss_gib']:.1f} GiB | "
                         f"{c['initial_dev_kl']:.5f} → {c['final_dev_kl']:.5f} |")
    lines += ['', '| method | unit | development KL (E0M3 tiles) by epoch |', '|---|---|---|']
    for key, c in out['runs'].items():
        method, unit = key.split()
        curve = ', '.join(f"{ep}: {kl:.5f} ({c['e0m3_curve'][ep]:,})" for ep, kl in c['dev_kl_curve'].items())
        lines.append(f"| {'TM-OPT' if method == 'tmopt' else 'TM-OPT+TC'} | {unit} | {curve} |")
    if (R / 'q_eval_bf16' / 'report.json').exists():
        out['checks'] = qwen_checks()
        lines += ['', f"Checks: {json.dumps(out['checks'])}"]
    (HERE / 'final_qwen.json').write_text(json.dumps(out, indent=1) + '\n')
    (HERE / 'final_qwen.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    {'cost': cost, 'ppl': ppl, 'qwen': qwen}[sys.argv[1]]()
