"""Tables of the flipquant parity check (NOTE.md): flipquant's calibration.train_map_razer against NVFP4-RaZeR's
run_train_map.py, per pair of runs, with the pass criteria.

    python results/topk_cal/flipquant_parity/parity.py [--runs /home/dev/n16k64_campaign/fqparity]

Pairs (RaZeR -> flipquant):
- Llama-3.1-8B 16x64, A: topk-cal's run A (runs/A_full_e20.json) -> A' (fq_llama_A);
- Llama-3.1-8B 16x64, C: topk-cal's run C (runs/C_top256_e5.json) -> C' (fq_llama_C);
- Phi-4 16x64, full vocabulary, 20 epochs: rz_phi4 -> fq_phi4 (back to back).

The trainer's numbers come from each run's report.json (PhaseMonitor), the same metrics as results/topk_cal/analyze.py.
flipquant's wrapper numbers come from measure.py: the end-to-end wall time, the wrapper process's own RSS (top) and the
trainer subprocess's RSS (descendants).

Pass criteria:
- the same map sha256;
- per-epoch time and the trainer's total time within +-3 %;
- GPU peak allocated and reserved within 0.1 GiB;
- host RSS (sampled, and ru_maxrss) within 0.5 GiB.
"""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('analyze', HERE.parent / 'analyze.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
TIME_TOL, GPU_TOL, RSS_TOL = 0.03, 0.1, 0.5


def load(path):
    return json.loads(Path(path).read_text())


def pair(name, rz_report, fq_dir, runs, rz_measure=None):
    rz = A.metrics(load(rz_report))
    fq = A.metrics(load(runs / fq_dir / 'run' / 'report.json'))
    fm = load(runs / f'{fq_dir}.measure.json')
    rep = load(runs / fq_dir / 'reproduction.json')
    rm = load(runs / rz_measure) if rz_measure else None
    out = dict(model=name, razer=dict(report=str(rz_report), **{k: rz[k] for k in KEYS}),
               flipquant=dict(run=str(runs / fq_dir), **{k: fq[k] for k in KEYS}))
    out['flipquant']['wrapper'] = dict(
        end_to_end_seconds=fm['wall_seconds'], wrapper_reported_seconds=rep['wall_seconds'],
        overhead_seconds=fm['wall_seconds'] - fq['total_s'], wrapper_process_peak_rss_gib=fm['peak_rss_gib']['top'],
        trainer_subprocess_peak_rss_gib=fm['peak_rss_gib']['descendants_sum'], tree_peak_rss_gib=fm['peak_rss_gib']['tree'],
        largest_process_ru_maxrss_gib=fm['largest_process_ru_maxrss_gib'], gpu_used_mib_by_pid=fm['gpu_used_mib_by_pid'])
    if rm:
        out['razer']['process'] = dict(end_to_end_seconds=rm['wall_seconds'], peak_rss_gib=rm['peak_rss_gib']['top'],
                                       ru_maxrss_gib=rm['largest_process_ru_maxrss_gib'],
                                       gpu_used_mib_by_pid=rm['gpu_used_mib_by_pid'])
    ratio = {k: fq[k] / rz[k] for k in ('setup_s', 'per_epoch_s', 'training_s', 'total_s')}
    diff = {k: fq[k] - rz[k] for k in ('gpu_peak_allocated_gib', 'gpu_peak_reserved_gib', 'host_rss_sampled_gib',
                                        'host_ru_maxrss_gib', 'post_load_host_rss_gib')}
    rz_end = rm['wall_seconds'] if rm else rz['total_s']
    out['ratio'], out['diff'] = ratio, diff
    out['end_to_end_ratio'] = dict(value=fm['wall_seconds'] / rz_end,
                                   razer_basis='end-to-end wall (measure.py)' if rm else "the trainer's total (no wrapper)")
    out['criteria'] = {
        'same map sha256': fq['map_sha256'] == rz['map_sha256'],
        'per-epoch time within 3 %': abs(ratio['per_epoch_s'] - 1) <= TIME_TOL,
        "trainer's total time within 3 %": abs(ratio['total_s'] - 1) <= TIME_TOL,
        'GPU peak allocated within 0.1 GiB': abs(diff['gpu_peak_allocated_gib']) <= GPU_TOL,
        'GPU peak reserved within 0.1 GiB': abs(diff['gpu_peak_reserved_gib']) <= GPU_TOL,
        'host RSS (sampled) within 0.5 GiB': abs(diff['host_rss_sampled_gib']) <= RSS_TOL,
        'host RSS (ru_maxrss) within 0.5 GiB': abs(diff['host_ru_maxrss_gib']) <= RSS_TOL}
    out['passed'] = all(out['criteria'].values())
    return out


KEYS = ('model_load_s', 'data_load_s', 'teacher_precompute_s', 'candidate_packing_s', 'setup_s', 'epochs', 'per_epoch_s',
        'training_s', 'total_s', 'gpu_peak_allocated_gib', 'gpu_peak_reserved_gib', 'host_rss_sampled_gib',
        'host_ru_maxrss_gib', 'post_load_host_rss_gib', 'teacher_bytes', 'teacher_devices', 'final_e0m3', 'map_sha256')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=Path, default=Path('/home/dev/n16k64_campaign/fqparity'))
    args = ap.parse_args()
    topk = HERE.parent / 'runs'
    pairs = [pair('Llama-3.1-8B 16x64, A (full vocabulary, 20 epochs)', topk / 'A_full_e20.json', 'fq_llama_A', args.runs),
             pair('Llama-3.1-8B 16x64, C (top-256, 5 epochs)', topk / 'C_top256_e5.json', 'fq_llama_C', args.runs),
             pair('Phi-4 16x64 (full vocabulary, 20 epochs), back to back', args.runs / 'rz_phi4' / 'report.json', 'fq_phi4',
                  args.runs, rz_measure='rz_phi4.measure.json')]
    md = []
    for p in pairs:
        r, f, w = p['razer'], p['flipquant'], p['flipquant']['wrapper']
        md += [f"## {p['model']}\n", '| | NVFP4-RaZeR | flipquant | flipquant / RaZeR |', '|---|---:|---:|---:|']
        for label, k, nd in (('setup (s)', 'setup_s', 1), ('per epoch (s)', 'per_epoch_s', 2), ('training (s)', 'training_s', 1),
                             ("trainer's total (s)", 'total_s', 1)):
            md.append(f'| {label} | {r[k]:.{nd}f} | {f[k]:.{nd}f} | {f[k] / r[k]:.4f} |')
        rz_end = r.get('process', {}).get('end_to_end_seconds')
        md.append(f"| end to end (s) | {rz_end:.1f} | {w['end_to_end_seconds']:.1f} | {p['end_to_end_ratio']['value']:.4f} |"
                  if rz_end else f"| end to end (s) | — | {w['end_to_end_seconds']:.1f} (wrapper overhead "
                                 f"{w['overhead_seconds']:+.1f} s) | |")
        for label, k in (('peak GPU allocated (GiB)', 'gpu_peak_allocated_gib'), ('peak GPU reserved (GiB)', 'gpu_peak_reserved_gib'),
                         ('peak host RSS, sampled (GiB)', 'host_rss_sampled_gib'), ('peak host RSS, ru_maxrss (GiB)', 'host_ru_maxrss_gib'),
                         ('host RSS after the model load (GiB)', 'post_load_host_rss_gib')):
            md.append(f'| {label} | {r[k]:.2f} | {f[k]:.2f} | {f[k] - r[k]:+.2f} GiB |')
        md.append(f"| wrapper process peak RSS (GiB) | | {w['wrapper_process_peak_rss_gib']:.2f} | |")
        md.append(f"| map sha256 | {r['map_sha256'][:16]}… | {f['map_sha256'][:16]}… | "
                  f"{'equal' if r['map_sha256'] == f['map_sha256'] else 'DIFFERENT'} |")
        md.append('')
        md.append('Criteria: ' + '; '.join(f"{k}: {'pass' if v else 'FAIL'}" for k, v in p['criteria'].items()) + '.\n')
    result = dict(criteria=dict(time_tolerance=TIME_TOL, gpu_tolerance_gib=GPU_TOL, rss_tolerance_gib=RSS_TOL), pairs=pairs,
                  all_passed=all(p['passed'] for p in pairs))
    (HERE / 'parity.json').write_text(json.dumps(result, indent=1) + '\n')
    (HERE / 'parity.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    print('ALL PASSED' if result['all_passed'] else 'NOT ALL PASSED')


if __name__ == '__main__':
    main()
