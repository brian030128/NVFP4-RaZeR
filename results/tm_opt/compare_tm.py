"""TM-OPT verification checks and tables (PROTOCOL.md). Each subcommand adds its result to summary.json here.

python results/tm_opt/compare_tm.py bitwise NAME RUN_A RUN_B   # group 1: every recorded value bitwise equal (exit 1 if not)
python results/tm_opt/compare_tm.py e2e RUNS                    # group 2 (deviation 2): the TM-OPT run, time and memory; writes tables.md
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / 'summary.json'
D = ('wiki', 'c4')
H200 = {'STE 8x64': dict(wiki=6.784682, c4=9.675402, e0m3=329837, source='results/mixfp4_potential/train_map/llama_8x64_ste'),
        'STE 256x64': dict(wiki=6.805528, c4=9.723484, e0m3=38176, source='results/mixfp4_potential/train_map/llama_256x64_ste')}


def load(p):
    return json.loads(Path(p).read_text())


def record(key, value):
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out[key] = value
    OUT.write_text(json.dumps(out, indent=1) + '\n')


def file_hash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bitwise(name, a, b):
    ra, rb = load(Path(a) / 'report.json'), load(Path(b) / 'report.json')
    checks = {}
    checks['theta_after_every_step'] = ra['theta_sha256'] == rb['theta_sha256']
    checks['steps_compared'] = len(ra['theta_sha256']) - 1
    for key in ('train_kl', 'e0m3_units', 'hard_flips', 'tau', 'dev_ce', 'dev_kl', 'dev_kl_values', 'dev_ce_values'):
        checks[f'epochs.{key}'] = [e.get(key) for e in ra['epochs']] == [e.get(key) for e in rb['epochs']]
    for key in ('initial_dev', 'final_dev'):
        checks[key] = all(ra[key][v] == rb[key][v] for v in ('ce_nll', 'kl_values'))
    checks['map'] = ra['map_sha256'] == rb['map_sha256']
    checks['theta_file'] = ra['theta_file_sha256'] == rb['theta_file_sha256']
    epoch_maps = sorted(p.name for p in Path(a).glob('map_epoch*.pt'))
    checks['epoch_maps'] = bool(epoch_maps) and epoch_maps == sorted(p.name for p in Path(b).glob('map_epoch*.pt')) and all(
        file_hash(Path(a) / m) == file_hash(Path(b) / m) for m in epoch_maps)
    checks['final_window_nll'] = all(ra['evaluation'][d]['nll'] == rb['evaluation'][d]['nll'] for d in D)
    passed = all(v for k, v in checks.items() if k != 'steps_compared')
    value = dict(runs=[str(a), str(b)], configurations=[ra['configuration'], rb['configuration']],
                 settings=[ra['settings'], rb['settings']], checks=checks, passed=passed,
                 final_e0m3_units=ra['final_e0m3_units'], flips_per_epoch=[e['hard_flips'] for e in ra['epochs']],
                 ppl={d: ra['evaluation'][d]['ppl'] for d in D})
    record(f'bitwise {name}', value)
    print(json.dumps(value, indent=1))
    sys.exit(0 if passed else 1)


def paired(x, y):
    d = [p - q for p, q in zip(x, y)]
    n = len(d)
    m = sum(d) / n
    two_se = 2 * math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
    return dict(mean=m, two_se=two_se, lower=m - two_se, upper=m + two_se,
                verdict='better' if m + two_se < 0 else ('worse' if m - two_se > 0 else 'inconclusive'))


def fmt(p):
    return f"{p['mean']:+.5f} ± {p['two_se']:.5f}" + ('' if p['verdict'] == 'inconclusive' else f" ({p['verdict']})")


def timing(r):
    epochs = r['epochs']
    per_epoch = [e['epoch_seconds'] - e.get('theta_hash_seconds', 0.0) for e in epochs]
    res = r['resources']
    return dict(configuration=r['configuration'], epochs=len(epochs), epoch_seconds_mean=sum(per_epoch) / len(per_epoch),
                epoch_seconds=per_epoch, setup_seconds=r['setup_seconds'], training_seconds=sum(per_epoch),
                monitor_seconds=r['monitor_seconds'], monitor_evaluations=sum('dev_kl' in e for e in epochs) + 2,
                selection_seconds=r['optimization_seconds'] - sum(e.get('theta_hash_seconds', 0.0) for e in epochs),
                gpu_peak_allocated_gib=res['gpu_peak_allocated_gib'][0], gpu_peak_reserved_gib=res['gpu_peak_reserved_gib'][0],
                host_peak_rss_gib=res['cpu_peak_rss_gib'],
                training_gpu_peak_allocated_gib=res['phases']['by_phase']['training']['gpu_peak_allocated_gib'][0])


def e2e(runs):
    """Group 2 after deviation 2: one full STE 8x64 run with the TM-OPT preset (no B1), evaluated with this branch's
    evaluator (native primary, fake secondary), against legacy time and memory from group 1's legacy short run."""
    runs = Path(runs)
    tm = load(runs / 'g2_tmopt' / 'report.json')
    nat, fake = load(runs / 'g2_eval_native' / 'report.json'), load(runs / 'g2_eval_fake' / 'report.json')
    ev, fev = nat['evaluations'], fake['evaluations']
    legacy, short = load(runs / 'g1_ste_legacy' / 'report.json'), load(runs / 'g1_ste_tmopt_nob1' / 'report.json')
    nll = {lab: {d: ev[lab]['evaluation'][d]['nll'] for d in D} for lab in ev}
    fnll = {lab: {d: fev[lab]['evaluation'][d]['nll'] for d in D} for lab in fev}
    t_tm, t_legacy, t_short = timing(tm), timing(legacy), timing(short)
    # legacy estimate for the full run: 20 epochs and 11 monitor evaluations (10 during training, the final one),
    # at group 1's legacy per-epoch and per-evaluation (fake monitor) times
    legacy_dev = [e['dev_seconds'] for e in legacy['epochs'] if 'dev_seconds' in e]
    epochs, monitor = tm['args']['epochs'], sum('dev_kl' in e for e in tm['epochs']) + 1
    legacy_estimate = dict(epoch_seconds=t_legacy['epoch_seconds_mean'], dev_evaluation_seconds=sum(legacy_dev) / len(legacy_dev),
                           epochs=epochs, monitor_evaluations=monitor)
    legacy_estimate['selection_seconds'] = epochs * legacy_estimate['epoch_seconds'] + monitor * legacy_estimate['dev_evaluation_seconds']
    value = dict(
        run=dict(configuration=tm['configuration'], settings=tm['settings'], final_e0m3_units=tm['final_e0m3_units'],
                 map_sha256=tm['map_sha256'], tiles=tm['tiles'],
                 per_epoch=[dict(epoch=e['epoch'] + 1, train_kl=e['train_kl'], e0m3_units=e['e0m3_units'], hard_flips=e['hard_flips'],
                                 dev_kl=e.get('dev_kl'), epoch_seconds=e['epoch_seconds']) for e in tm['epochs']],
                 initial_dev_kl=tm['initial_dev']['kl'], final_dev_kl=tm['final_dev']['kl'], monitor_backend=tm['settings']['dev_backend']),
        native_ppl={lab: {d: ev[lab]['evaluation'][d]['ppl'] for d in D} for lab in ev},
        fake_ppl={lab: {d: fev[lab]['evaluation'][d]['ppl'] for d in D} for lab in fev},
        native_vs_fourover6={d: paired(nll['tmopt'][d], nll['FourOverSix'][d]) for d in D},
        fake_vs_fourover6={d: paired(fnll['tmopt'][d], fnll['FourOverSix'][d]) for d in D},
        run_own_native_evaluation_repeated=all(tm['evaluation'][d]['nll'] == nll['tmopt'][d] for d in D),
        map_checks={lab: dict(native_map_mismatches=ev[lab].get('native_map_mismatches'), e0m3_units=ev[lab].get('e0m3_units'))
                    for lab in ev},
        timing=dict(tm_opt=t_tm, legacy_short=t_legacy, tm_opt_short_fake_monitor=t_short, legacy_full_run_estimate=legacy_estimate),
        h200_reference=H200)
    record('e2e STE 8x64 TM-OPT', value)
    lines = ['| run | configuration | epochs | training s / epoch | selection time | peak GPU allocated / reserved | '
             'training peak GPU allocated | host RSS |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for label, x in (('legacy (group 1 STE, 3 epochs)', t_legacy), ('TM-OPT, fake monitor (group 1 STE, 3 epochs)', t_short),
                     ('TM-OPT (group 2, 20 epochs, native monitor)', t_tm)):
        lines.append(f"| {label} | {x['configuration']} | {x['epochs']} | {x['epoch_seconds_mean']:.1f} | "
                     f"{x['selection_seconds'] / 60:.1f} min | {x['gpu_peak_allocated_gib']:.1f} / {x['gpu_peak_reserved_gib']:.1f} GiB | "
                     f"{x['training_gpu_peak_allocated_gib']:.1f} GiB | {x['host_peak_rss_gib']:.1f} GiB |")
    lines.append(f"| legacy, full-run estimate | legacy | {epochs} | {legacy_estimate['epoch_seconds']:.1f} | "
                 f"{legacy_estimate['selection_seconds'] / 60:.1f} min | | | |")
    lines += ['', '| map | E0M3 tiles | native WikiText-2 | native C4 | native ΔWiki vs FourOverSix | native ΔC4 vs FourOverSix | '
              'fake WikiText-2 | fake C4 | fake ΔWiki vs FourOverSix | fake ΔC4 vs FourOverSix |', '|---|---:|---:|---:|---|---|---:|---:|---|---|']
    for lab, name in (('FourOverSix', 'FourOverSix'), ('tmopt', 'TM-OPT STE 8x64')):
        nv = value['native_vs_fourover6'] if lab == 'tmopt' else None
        fv = value['fake_vs_fourover6'] if lab == 'tmopt' else None
        lines.append(f"| {name} | {tm['final_e0m3_units'] if lab == 'tmopt' else 0:,} | {value['native_ppl'][lab]['wiki']:.4f} | "
                     f"{value['native_ppl'][lab]['c4']:.4f} | {fmt(nv['wiki']) if nv else '—'} | {fmt(nv['c4']) if nv else '—'} | "
                     f"{value['fake_ppl'][lab]['wiki']:.4f} | {value['fake_ppl'][lab]['c4']:.4f} | {fmt(fv['wiki']) if fv else '—'} | "
                     f"{fmt(fv['c4']) if fv else '—'} |")
    h = H200['STE 8x64']
    lines += ['', f"H200 reference (main, fake evaluation, context only): STE 8x64 {h['wiki']:.4f} / {h['c4']:.4f}, {h['e0m3']:,} E0M3 tiles.",
              f"Run's own native final evaluation repeated in the joint process: {value['run_own_native_evaluation_repeated']}."]
    (HERE / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    if sys.argv[1] == 'bitwise':
        bitwise(*sys.argv[2:5])
    elif sys.argv[1] == 'e2e':
        e2e(sys.argv[2])
    else:
        raise SystemExit(__doc__)
