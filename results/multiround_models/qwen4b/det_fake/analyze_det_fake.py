"""Tables and the per-unit SAFE verdict of the Qwen3-4B native-decision addendum (criteria: PROTOCOL.md).

python results/multiround_models/qwen4b/det_fake/analyze_det_fake.py   (writes summary.json and tables.md here)
Reads the DET-FAKE runs and their evaluations from /home/dev/n16k64_campaign/multimodel/runs/qwen4b_det_fake,
the DET-NATIVE runs and Part C's evaluations from .../runs/qwen4b. A unit whose evaluation is missing is skipped.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'native_decision'))
from analyze_decision import first_divergence, fmt, load, paired  # noqa: E402

RUNS = Path('/home/dev/n16k64_campaign/multimodel/runs')


def run_summary(r):
    res = r['resources']
    ph = res['phases']['by_phase']
    return dict(stopped=r.get('stopped'), tiles=r['final_e0m3_units'], map_sha256=r['map_sha256'],
                dev_kl=[r['initial_dev']['kl'], r['final_dev']['kl']], rounds=res['scoring_passes'],
                dev_evaluations=res['development_evaluations'], accepted=[x['accepted'] for x in r['rounds']],
                setup_seconds=res['setup_seconds'], optimization_seconds=res['optimization_seconds'],
                scoring_seconds_per_pass=ph['scoring']['seconds'] / ph['scoring']['count'],
                dev_eval_seconds_per_try=ph['dev_evaluation']['seconds'] / ph['dev_evaluation']['count'],
                gpu_peak_allocated_gib=res['gpu_peak_allocated_gib'][0], gpu_peak_reserved_gib=res['gpu_peak_reserved_gib'][0],
                host_peak_rss_gib=res['cpu_peak_rss_gib'], built_in_fake_ppl={k: v['ppl'] for k, v in r['evaluation'].items()})


def main():
    out = dict(units={})
    part_c = {b: load(RUNS / 'qwen4b' / f'eval_{b}' / 'report.json')['evaluations'] for b in ('native', 'fake')}
    lines = []
    for unit in ('256x64', '8x64'):
        fake_run = RUNS / 'qwen4b_det_fake' / f'det_fake_{unit}' / 'report.json'
        evals = {b: RUNS / 'qwen4b_det_fake' / f'eval_{unit}_{b}' / 'report.json' for b in ('native', 'fake')}
        if not fake_run.exists() or not all(p.exists() for p in evals.values()):
            continue
        fa, na = load(fake_run), load(RUNS / 'qwen4b' / f'calib_{unit}' / 'report.json')
        u = dict(runs={'DET-FAKE': run_summary(fa), 'DET-NATIVE': run_summary(na)},
                 first_divergence=first_divergence(fa, na), evaluation={})
        for backend, path in evals.items():
            ev = load(path)['evaluations']
            same = all(ev['FourOverSix']['evaluation'][d]['nll'] == part_c[backend]['FourOverSix']['evaluation'][d]['nll']
                       for d in ('wiki', 'c4'))
            base = ev['FourOverSix']['evaluation']
            table = dict(fourover6_bitwise_equal_to_part_c=same)
            for label in ('FourOverSix', f'DET-NATIVE-{unit}', f'DET-FAKE-{unit}'):
                e = ev[label]
                row = dict(tiles=e.get('e0m3_units'), wiki=e['evaluation']['wiki']['ppl'], c4=e['evaluation']['c4']['ppl'],
                           map_mismatches=e.get('native_map_mismatches', e.get('lean_map_mismatches')),
                           activation_checks=e.get('native_activation_checks'))
                if label != 'FourOverSix':
                    row['vs_fourover6'] = {d: paired(e['evaluation'][d]['nll'], base[d]['nll']) for d in ('wiki', 'c4')}
                table[label] = row
            dn, df = ev[f'DET-NATIVE-{unit}']['evaluation'], ev[f'DET-FAKE-{unit}']['evaluation']
            table['native_minus_fake_decided'] = {d: paired(dn[d]['nll'], df[d]['nll']) for d in ('wiki', 'c4')}
            # DET-NATIVE evaluated here must reproduce Part C's evaluation of the same map bitwise (256x64: Part C
            # evaluated it converted to 8x64 tiles, the same weights)
            part_c_label = f'MixFP4-{unit}'
            table['det_native_bitwise_equal_to_part_c'] = all(
                dn[d]['nll'] == part_c[backend][part_c_label]['evaluation'][d]['nll'] for d in ('wiki', 'c4'))
            u['evaluation'][backend] = table
        crit = u['evaluation']['native']['native_minus_fake_decided']
        worse = [d for d, p in crit.items() if p['verdict'] == 'worse']
        u['verdict'] = dict(safe=not worse, significantly_worse_on=worse)
        u['det_fake_worse_than_fourover6_on_c4_native'] = \
            u['evaluation']['native'][f'DET-FAKE-{unit}']['vs_fourover6']['c4']['verdict'] == 'worse'
        out['units'][unit] = u
        lines += [f'### {unit}', '', '| run | tiles | rounds / dev evaluations | setup | optimization | scoring per pass | '
                  'dev evaluation per try | peak GPU allocated / reserved | peak host RSS |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name, s in u['runs'].items():
            lines.append(f"| {name}-{unit} | {s['tiles']:,} | {s['rounds']} / {s['dev_evaluations']} | {s['setup_seconds'] / 60:.1f} min | "
                         f"{s['optimization_seconds'] / 60:.1f} min | {s['scoring_seconds_per_pass']:.1f} s | "
                         f"{s['dev_eval_seconds_per_try']:.1f} s | {s['gpu_peak_allocated_gib']:.1f} / {s['gpu_peak_reserved_gib']:.1f} GiB | "
                         f"{s['host_peak_rss_gib']:.1f} GiB |")
        lines += ['', '| backend | map | tiles | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |',
                  '|---|---|---:|---:|---:|---|---|']
        for backend in ('native', 'fake'):
            t = u['evaluation'][backend]
            for label in ('FourOverSix', f'DET-NATIVE-{unit}', f'DET-FAKE-{unit}'):
                r = t[label]
                v = r.get('vs_fourover6')
                lines.append(f"| {backend} | {label} | {r['tiles']:,} | {r['wiki']:.4f} | {r['c4']:.4f} | "
                             f"{fmt(v['wiki']) if v else '—'} | {fmt(v['c4']) if v else '—'} |")
        lines += ['', '| DET-NATIVE minus DET-FAKE | ΔWiki | ΔC4 |', '|---|---|---|']
        for backend in ('native', 'fake'):
            c = u['evaluation'][backend]['native_minus_fake_decided']
            lines.append(f"| {backend} evaluation{' (criterion)' if backend == 'native' else ''} | {fmt(c['wiki'])} | {fmt(c['c4'])} |")
        lines += ['', f"First divergence: {json.dumps(u['first_divergence'])}", '']
    (HERE / 'summary.json').write_text(json.dumps(out, indent=1) + '\n')
    (HERE / 'tables.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({unit: dict(verdict=u['verdict'], first_divergence=u['first_divergence'],
                                 det_fake_worse_on_c4=u['det_fake_worse_than_fourover6_on_c4_native'],
                                 fourover6_same={b: u['evaluation'][b]['fourover6_bitwise_equal_to_part_c'] for b in ('native', 'fake')},
                                 det_native_same={b: u['evaluation'][b]['det_native_bitwise_equal_to_part_c'] for b in ('native', 'fake')})
                      for unit, u in out['units'].items()}, indent=1))


if __name__ == '__main__':
    main()
