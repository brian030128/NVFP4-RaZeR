"""Part B verdict, time and memory tables for results/lean_memory/REPORT.md (criteria: PROTOCOL.md).

python results/lean_memory/analyze_partb.py RUN_ROOT   (writes summary.json and tables.md here)
RUN_ROOT holds legacy_<config>/ and lean_<config>/ run directories and checks/<run>.json from compare_runs.py.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGS = ('det_fake_256x64', 'det_native_256x64', 'det_fake_8x64', 'det_native_8x64')
LABEL = {'det_fake_256x64': 'DET-FAKE-256x64', 'det_native_256x64': 'DET-NATIVE-256x64',
         'det_fake_8x64': 'DET-FAKE-8x64', 'det_native_8x64': 'DET-NATIVE-8x64'}


def resources(report):
    res = report['resources']
    phases = res['phases']['by_phase']
    # Every backtracking try runs in the 'dev_evaluation' phase (fake or native evaluator); the
    # 'native_dev_evaluation' phase is only the initial native evaluation of a native run.
    tries = sum(len(r['tries']) for r in report['rounds'])
    out = dict(setup_seconds=res['setup_seconds'], optimization_seconds=res['optimization_seconds'],
               evaluation_seconds=res['evaluation_seconds'], total_seconds=res['total_seconds'],
               scoring_passes=res['scoring_passes'], development_evaluations=res['development_evaluations'],
               scoring_seconds_per_pass=phases['scoring']['seconds'] / phases['scoring']['count'],
               dev_eval_seconds_per_try=phases['dev_evaluation']['seconds'] / phases['dev_evaluation']['count'],
               initial_native_dev_eval_seconds=phases['native_dev_evaluation']['seconds']
               if 'native_dev_evaluation' in phases else None,
               tries=tries, gpu_peak_allocated_gib=res['gpu_peak_allocated_gib'][0],
               gpu_peak_reserved_gib=res['gpu_peak_reserved_gib'][0], host_peak_rss_gib=res['cpu_peak_rss_gib'],
               phase_gpu_peak_allocated_gib={k: v['gpu_peak_allocated_gib'][0] for k, v in phases.items()},
               phase_host_peak_rss_gib={k: v['host_peak_rss_gib'] for k, v in phases.items()},
               candidate_storage=report['candidate_storage'])
    if 'native_builds' in report:
        b = report['native_builds']
        out['native_build_seconds_per_evaluation'] = b['seconds'] / max(1, b['count'])
        out['native_builds'] = b['count']
    return out


def main():
    root = Path(sys.argv[1])
    summary = dict(configurations={}, verdict=None)
    passed = []
    for config in CONFIGS:
        entry = {}
        for mode in ('legacy', 'lean'):
            check = root / 'checks' / f'{mode}_{config}.json'
            run = root / f'{mode}_{config}' / 'report.json'
            entry[mode] = dict(check=json.loads(check.read_text()) if check.exists() else None,
                               resources=resources(json.loads(run.read_text())) if run.exists() and
                               json.loads(run.read_text()).get('status') == 'complete' else None)
        ok = all(entry[m]['check'] is not None and entry[m]['check']['passed'] for m in ('legacy', 'lean'))
        entry['passed'] = ok
        passed.append(ok)
        summary['configurations'][config] = entry
    summary['verdict'] = 'PASS' if all(passed) else ('FAIL' if any(
        e[m]['check'] is not None and not e[m]['check']['passed']
        for e in summary['configurations'].values() for m in ('legacy', 'lean')) else 'INCOMPLETE')
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')

    lines = ['| configuration | run | checks | rounds / tries | tiles | scoring per pass | dev evaluation per try | '
             'native build per evaluation | setup | optimization | peak GPU allocated / reserved | peak host RSS |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for config, entry in summary['configurations'].items():
        for mode in ('legacy', 'lean'):
            r, c = entry[mode]['resources'], entry[mode]['check']
            if r is None:
                continue
            checks = 'PASS' if c and c['passed'] else ('FAIL ' + ', '.join(c['failed_checks']) if c else '—')
            detail = c['vs_committed_detail'] if c else {}
            build = r.get('native_build_seconds_per_evaluation')
            lines.append(
                f"| {LABEL[config]} | {mode} | {checks} | {r['scoring_passes']} / {r['tries']} | "
                f"{json.loads((root / f'{mode}_{config}' / 'report.json').read_text())['final_e0m3_units']:,} | "
                f"{r['scoring_seconds_per_pass']:.1f} s | {r['dev_eval_seconds_per_try']:.1f} s | "
                f"{'—' if build is None else f'{build:.2f} s'} | {r['setup_seconds']:.0f} s | "
                f"{r['optimization_seconds'] / 60:.1f} min | {r['gpu_peak_allocated_gib']:.1f} / "
                f"{r['gpu_peak_reserved_gib']:.1f} GiB | {r['host_peak_rss_gib']:.1f} GiB |")
    lines += ['', '| configuration | phase | legacy peak GPU allocated | lean peak GPU allocated |', '|---|---|---:|---:|']
    for config, entry in summary['configurations'].items():
        a, b = entry['legacy']['resources'], entry['lean']['resources']
        if a is None or b is None:
            continue
        for phase in a['phase_gpu_peak_allocated_gib']:
            lines.append(f"| {LABEL[config]} | {phase} | {a['phase_gpu_peak_allocated_gib'][phase]:.1f} GiB | "
                         f"{b['phase_gpu_peak_allocated_gib'].get(phase, float('nan')):.1f} GiB |")
    (HERE / 'tables.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(verdict=summary['verdict'],
                          configurations={k: v['passed'] for k, v in summary['configurations'].items()}), indent=1))


if __name__ == '__main__':
    main()
