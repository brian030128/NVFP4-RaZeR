"""TM-OPT verification checks and tables (PROTOCOL.md). Each subcommand adds its result to summary.json here.

python results/tm_opt/compare_tm.py bitwise NAME RUN_A RUN_B   # group 1: every recorded value bitwise equal (exit 1 if not)
python results/tm_opt/compare_tm.py e2e RUNS                    # group 2: B1 on vs off, time and memory; writes tables.md
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
    runs = Path(runs)
    on, off = load(runs / 'g2_tmopt' / 'report.json'), load(runs / 'g2_tmopt_nob1' / 'report.json')
    nat, fake = load(runs / 'g2_eval_native' / 'report.json'), load(runs / 'g2_eval_fake' / 'report.json')
    ev, fev = nat['evaluations'], fake['evaluations']
    ma = torch.load(runs / 'g2_tmopt' / 'map.pt', weights_only=True)
    mb = torch.load(runs / 'g2_tmopt_nob1' / 'map.pt', weights_only=True)
    ta = torch.load(runs / 'g2_tmopt' / 'theta.pt', weights_only=True)
    tb = torch.load(runs / 'g2_tmopt_nob1' / 'theta.pt', weights_only=True)
    theta_diff = max(float((ta[n] - tb[n]).abs().max()) for n in ta)
    overlap = dict(shared=sum(int((ma[n] & mb[n]).sum()) for n in ma), only_b1=sum(int((ma[n] & ~mb[n]).sum()) for n in ma),
                   only_no_b1=sum(int((mb[n] & ~ma[n]).sum()) for n in ma))
    nll = {lab: {d: ev[lab]['evaluation'][d]['nll'] for d in D} for lab in ev}
    b1_vs_off = {d: paired(nll['tmopt'][d], nll['tmopt-nob1'][d]) for d in D}
    passed = all(b1_vs_off[d]['lower'] <= 0 for d in D)
    own = {lab: all(r['evaluation'][d]['nll'] == nll[lab][d] for d in D) for lab, r in (('tmopt', on), ('tmopt-nob1', off))}
    legacy = [load(runs / n / 'report.json') for n in ('g1_ste_legacy', 'g1_ste_tmopt_nob1')]
    value = dict(
        criterion='B1 map minus no-B1 map, native paired ΔNLL: mean - 2 SE <= 0 on WikiText-2 and C4', passed=passed,
        b1_minus_no_b1=b1_vs_off,
        vs_fourover6={lab: {d: paired(nll[lab][d], nll['FourOverSix'][d]) for d in D} for lab in ('tmopt', 'tmopt-nob1')},
        native_ppl={lab: {d: ev[lab]['evaluation'][d]['ppl'] for d in D} for lab in ev},
        fake_ppl={lab: {d: fev[lab]['evaluation'][d]['ppl'] for d in D} for lab in fev},
        runs_own_native_evaluation_repeated=own,
        final_e0m3_units=dict(b1=on['final_e0m3_units'], no_b1=off['final_e0m3_units']),
        flip_count_difference=on['final_e0m3_units'] - off['final_e0m3_units'], map_overlap=overlap,
        flips_per_epoch=dict(b1=[e['hard_flips'] for e in on['epochs']], no_b1=[e['hard_flips'] for e in off['epochs']]),
        e0m3_per_epoch=dict(b1=[e['e0m3_units'] for e in on['epochs']], no_b1=[e['e0m3_units'] for e in off['epochs']]),
        max_abs_theta_difference=theta_diff,
        monitor_dev_kl=dict(b1=[e.get('dev_kl') for e in on['epochs']], no_b1=[e.get('dev_kl') for e in off['epochs']]),
        timing=dict(tm_opt=timing(on), tm_opt_no_b1=timing(off), legacy_short=timing(legacy[0]),
                    tm_opt_no_b1_short=timing(legacy[1])),
        h200_reference=H200)
    record('e2e STE 8x64 B1', value)
    t = value['timing']
    lines = ['| run | configuration | epochs | training s / epoch | monitor evaluations | selection time | '
             'peak GPU allocated / reserved (run) | training peak GPU allocated | host RSS |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, key in (('legacy (group 1 STE, 3 epochs)', 'legacy_short'), ('TM-OPT without B1 (group 1 STE, fake eval)', 'tm_opt_no_b1_short'),
                       ('TM-OPT without B1 (full)', 'tm_opt_no_b1'), ('TM-OPT (full)', 'tm_opt')):
        x = t[key]
        lines.append(f"| {label} | {x['configuration']} | {x['epochs']} | {x['epoch_seconds_mean']:.1f} | {x['monitor_evaluations']} | "
                     f"{x['selection_seconds'] / 60:.1f} min | {x['gpu_peak_allocated_gib']:.1f} / {x['gpu_peak_reserved_gib']:.1f} GiB | "
                     f"{x['training_gpu_peak_allocated_gib']:.1f} GiB | {x['host_peak_rss_gib']:.1f} GiB |")
    lines += ['', '| map | E0M3 tiles | native WikiText-2 | native C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | fake WikiText-2 | fake C4 |',
              '|---|---:|---:|---:|---|---|---:|---:|']
    for lab, name in (('FourOverSix', 'FourOverSix'), ('tmopt', 'TM-OPT (B1)'), ('tmopt-nob1', 'TM-OPT without B1')):
        v = value['vs_fourover6'].get(lab)
        units = {'tmopt': on['final_e0m3_units'], 'tmopt-nob1': off['final_e0m3_units']}.get(lab, 0)
        lines.append(f"| {name} | {units:,} | {value['native_ppl'][lab]['wiki']:.4f} | {value['native_ppl'][lab]['c4']:.4f} | "
                     f"{fmt(v['wiki']) if v else '—'} | {fmt(v['c4']) if v else '—'} | {value['fake_ppl'][lab]['wiki']:.4f} | "
                     f"{value['fake_ppl'][lab]['c4']:.4f} |")
    lines += ['', f"B1 minus no-B1 (native): ΔWiki {fmt(b1_vs_off['wiki'])}, ΔC4 {fmt(b1_vs_off['c4'])} -> "
              f"{'PASS' if passed else 'FAIL'}; tiles shared / only B1 / only no-B1: {overlap['shared']:,} / "
              f"{overlap['only_b1']:,} / {overlap['only_no_b1']:,}; max |Δθ| {theta_diff:.3g}"]
    (HERE / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(json.dumps({k: value[k] for k in ('passed', 'b1_minus_no_b1', 'final_e0m3_units', 'map_overlap',
                                            'runs_own_native_evaluation_repeated')}, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'bitwise':
        bitwise(*sys.argv[2:5])
    elif sys.argv[1] == 'e2e':
        e2e(sys.argv[2])
    else:
        raise SystemExit(__doc__)
