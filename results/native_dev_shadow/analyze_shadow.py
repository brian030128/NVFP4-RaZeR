"""Verdict and tables for results/native_dev_shadow/REPORT.md (criteria: PROTOCOL.md).

python results/native_dev_shadow/analyze_shadow.py RUN_ROOT   (writes summary.json and tables.md here)
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text())


def main():
    root = Path(sys.argv[1])
    out = dict(precheck={})
    for name in ('precheck_start_map', 'precheck_b256opt_map'):
        r = load(root / name / 'report.json')
        sn = r['shadow_native']
        out['precheck'][name] = dict(map=r.get('init_map'), fake_kl=r['initial_dev']['kl'], fake_ce=r['initial_dev']['ce'],
                                     native_kl=sn['initial']['kl'], native_ce=sn['initial']['ce'],
                                     difference=sn['initial']['difference'], verification=sn['verification'],
                                     fake_seconds=sn['initial']['fake_seconds'], native_seconds=sn['initial']['native_seconds'],
                                     activation_checks=sn['initial']['activation_checks'])
    r = load(root / 'shadow_256x64' / 'report.json')
    sn = r['shadow_native']
    tries = sn['tries']
    dis = [t for t in tries if not t['agree']]
    accepted = [t for t in tries if t['fake_accepts']]
    out['shadow'] = dict(
        status=r['status'], stopped=r.get('stopped'), tiles=r.get('final_e0m3_units'), map_sha256=r.get('map_sha256'),
        rounds=len(r['rounds']), tries=len(tries), accepted_tries=len(accepted), disagreements=len(dis),
        verdict='PASS' if not dis else 'FAIL', verification=sn['verification'],
        initial=dict(fake_kl=r['initial_dev']['kl'], native_kl=sn['initial']['kl'],
                     difference=sn['initial']['difference']),
        smallest_accepted_margin=min((t['margin'] for t in accepted), default=None),
        smallest_margin=min((t['margin'] for t in tries), default=None),
        largest_discrepancy=max((t['discrepancy'] for t in tries), default=None),
        tries_with_discrepancy_above_margin=sum(t['discrepancy'] > t['margin'] for t in tries),
        timing=dict(fake_seconds_mean=statistics.mean(t['fake_seconds'] for t in tries),
                    native_seconds_mean=statistics.mean(t['native_seconds'] for t in tries),
                    fake_seconds_median=statistics.median(t['fake_seconds'] for t in tries),
                    native_seconds_median=statistics.median(t['native_seconds'] for t in tries),
                    native_rebuild_seconds_mean=statistics.mean(t['native_rebuild_seconds'] for t in tries)),
        evaluation=r.get('evaluation') and {k: v['ppl'] for k, v in r['evaluation'].items()},
        resources={k: r['resources'][k] for k in ('setup_seconds', 'optimization_seconds', 'gpu_peak_allocated_gib',
                                                  'gpu_peak_reserved_gib', 'cpu_peak_rss_gib', 'scoring_passes',
                                                  'development_evaluations')} if r.get('resources') else None)
    out['tries'] = [{k: t[k] for k in ('round', 'try_index', 'size', 'fake_accepts', 'native_accepts', 'agree',
                                       'delta_fake', 'delta_native', 'discrepancy', 'margin', 'fake_seconds',
                                       'native_seconds', 'native_rebuild_seconds')} for t in tries]
    (HERE / 'summary.json').write_text(json.dumps(out, indent=2) + '\n')
    lines = ['| map | fake KL | native KL | native − fake (mean) | mean / max per-document abs ΔKL '
             '| fake CE | native CE | native − fake CE | mean / max per-document abs ΔCE |',
             '|---|---:|---:|---:|---|---:|---:|---:|---|']
    for name, p in out['precheck'].items():
        d = p['difference']
        lines.append(f"| {name} | {p['fake_kl']:.6f} | {p['native_kl']:.6f} | {d['kl']['mean_native_minus_fake']:+.6f} "
                     f"| {d['kl']['mean_abs_per_document']:.5f} / {d['kl']['max_abs_per_document']:.5f} | {p['fake_ce']:.6f} "
                     f"| {p['native_ce']:.6f} | {d['ce']['mean_native_minus_fake']:+.6f} "
                     f"| {d['ce']['mean_abs_per_document']:.5f} / {d['ce']['max_abs_per_document']:.5f} |")
    lines += ['', '| round | try | size | Δfake | Δnative | margin | discrepancy | fake | native | agree '
              '| fake s | native s (repack s) |', '|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|']
    for t in tries:
        lines.append(f"| {t['round']} | {t['try_index']} | {t['size']:,} | {t['delta_fake']:+.6f} | {t['delta_native']:+.6f} "
                     f"| {t['margin']:.6f} | {t['discrepancy']:.6f} | {'accept' if t['fake_accepts'] else 'reject'} "
                     f"| {'accept' if t['native_accepts'] else 'reject'} | {'yes' if t['agree'] else '**NO**'} "
                     f"| {t['fake_seconds']:.1f} | {t['native_seconds']:.1f} ({t['native_rebuild_seconds']:.1f}) |")
    (HERE / 'tables.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(out['shadow'], indent=1))


if __name__ == '__main__':
    main()
