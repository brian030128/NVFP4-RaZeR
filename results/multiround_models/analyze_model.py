"""Tables for one model's Part C report (results/multiround_models/<model>/REPORT.md; PROTOCOL.md stages 5-6).

python results/multiround_models/analyze_model.py MODEL RUN_ROOT   (writes <model>/summary.json and tables.md)
RUN_ROOT: calib_8x64/, calib_256x64/, eval_native/, eval_fake/, eval_bf16/ (run_multiround.py report.json each).
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAPS = ('FourOverSix', 'NVFP4', 'MixFP4-8x64', 'MixFP4-256x64')


def load(path):
    return json.loads(Path(path).read_text())


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    mean = sum(d) / len(d)
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1))
    two_se = 2 * sd / math.sqrt(len(d))
    verdict = 'better' if mean + two_se < 0 else ('worse' if mean - two_se > 0 else 'inconclusive')
    return dict(mean=mean, two_se=two_se, verdict=verdict, windows=len(d))


def fmt(p):
    return f"{p['mean']:+.5f} ± {p['two_se']:.5f}" + ('' if p['verdict'] == 'inconclusive' else f" ({p['verdict']})")


def calibration(report):
    res = report['resources']
    phases = res['phases']['by_phase']
    # Every backtracking try runs in the 'dev_evaluation' phase (fake or native evaluator); the
    # 'native_dev_evaluation' phase is only the initial native evaluation of a native run.
    return dict(status=report['status'], stopped=report.get('stopped'), tiles=report['final_e0m3_units'],
                map_sha256=report['map_sha256'], eval_batch=report['eval_batch'], score_batch=report['score_batch'],
                dev_kl=[report['initial_dev']['kl'], report['final_dev']['kl']],
                rounds=res['scoring_passes'], dev_evaluations=res['development_evaluations'],
                setup_seconds=res['setup_seconds'], optimization_seconds=res['optimization_seconds'],
                scoring_seconds_per_pass=phases['scoring']['seconds'] / phases['scoring']['count'],
                dev_eval_seconds_per_try=phases['dev_evaluation']['seconds'] / phases['dev_evaluation']['count'],
                native_build_seconds_per_evaluation=(report['native_builds']['seconds'] /
                                                     max(1, report['native_builds']['count']))
                if 'native_builds' in report else None,
                gpu_peak_allocated_gib=res['gpu_peak_allocated_gib'][0], gpu_peak_reserved_gib=res['gpu_peak_reserved_gib'][0],
                host_peak_rss_gib=res['cpu_peak_rss_gib'],
                trajectory=[(r['round'], r['candidates'], r['accepted'], r.get('e0m3_units')) for r in report['rounds']])


def main():
    model, root = sys.argv[1], Path(sys.argv[2])
    out = dict(model=model, calibration={}, evaluation={})
    for unit in ('8x64', '256x64'):
        path = root / f'calib_{unit}' / 'report.json'
        if path.exists():
            out['calibration'][unit] = calibration(load(path))
    windows = {}
    for backend in ('native', 'fake'):
        path = root / f'eval_{backend}' / 'report.json'
        if not path.exists():
            continue
        report = load(path)
        windows[backend] = {d: report['data'][d]['token_sha256'] for d in ('wiki', 'c4_paper')}
        ev = report['evaluations']
        base = ev['FourOverSix']['evaluation']
        table = {}
        for label, e in ev.items():
            row = dict(tiles=e.get('e0m3_units'), tiles_256x64=e.get('e0m3_units_256x64'),
                       wiki=e['evaluation']['wiki']['ppl'], c4=e['evaluation']['c4']['ppl'],
                       map_mismatches=e.get('native_map_mismatches', e.get('lean_map_mismatches')),
                       activation_checks=e.get('native_activation_checks'))
            if label != 'FourOverSix':
                row['vs_fourover6'] = {d: paired(e['evaluation'][d]['nll'], base[d]['nll']) for d in ('wiki', 'c4')}
            table[label] = row
        out['evaluation'][backend] = table
    bf16 = root / 'eval_bf16' / 'report.json'
    if bf16.exists() and 'fake' in out['evaluation']:
        report = load(bf16)
        same = all(report['data'][d]['token_sha256'] == windows['fake'][d] for d in ('wiki', 'c4_paper'))
        e = report['evaluations']['BF16']['evaluation']
        fo6 = load(root / 'eval_fake' / 'report.json')['evaluations']['FourOverSix']['evaluation']
        out['evaluation']['bf16'] = dict(wiki=e['wiki']['ppl'], c4=e['c4']['ppl'], same_windows=same,
                                         vs_fourover6_fake={d: paired(e[d]['nll'], fo6[d]['nll']) for d in ('wiki', 'c4')}
                                         if same else None)
    out['same_windows_all_processes'] = len({json.dumps(w, sort_keys=True) for w in windows.values()}) <= 1
    target = HERE / model
    target.mkdir(exist_ok=True)
    (target / 'summary.json').write_text(json.dumps(out, indent=1) + '\n')
    lines = ['| unit | E0M3 tiles | rounds / dev evaluations | batch (eval / score) | setup | optimization | '
             'scoring per pass | dev evaluation per try | native build per evaluation | peak GPU allocated / reserved | '
             'peak host RSS | stop |', '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for unit, c in out['calibration'].items():
        build = c['native_build_seconds_per_evaluation']
        lines.append(f"| {unit} | {c['tiles']:,} | {c['rounds']} / {c['dev_evaluations']} | {c['eval_batch']} / {c['score_batch']} | "
                     f"{c['setup_seconds'] / 60:.1f} min | {c['optimization_seconds'] / 60:.1f} min | "
                     f"{c['scoring_seconds_per_pass']:.1f} s | {c['dev_eval_seconds_per_try']:.1f} s | "
                     f"{'—' if build is None else f'{build:.2f} s'} | {c['gpu_peak_allocated_gib']:.1f} / "
                     f"{c['gpu_peak_reserved_gib']:.1f} GiB | {c['host_peak_rss_gib']:.1f} GiB | {c['stopped']} |")
    lines += ['', '| backend | map | E0M3 tiles | WikiText-2 PPL | C4 PPL | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |',
              '|---|---|---:|---:|---:|---|---|']
    for backend in ('native', 'fake'):
        for label in MAPS:
            row = out['evaluation'].get(backend, {}).get(label)
            if row is None:
                continue
            v = row.get('vs_fourover6')
            tiles = '—' if row['tiles'] is None else f"{row['tiles']:,}"
            if row.get('tiles_256x64') is not None:
                tiles = f"{row['tiles_256x64']:,} (256x64)"
            lines.append(f"| {backend} | {label} | {tiles} | {row['wiki']:.4f} | {row['c4']:.4f} | "
                         f"{fmt(v['wiki']) if v else '—'} | {fmt(v['c4']) if v else '—'} |")
    b = out['evaluation'].get('bf16')
    if b:
        v = b['vs_fourover6_fake']
        lines.append(f"| fake | BF16 | — | {b['wiki']:.4f} | {b['c4']:.4f} | {fmt(v['wiki']) if v else '—'} | "
                     f"{fmt(v['c4']) if v else '—'} |")
    (target / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
