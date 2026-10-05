"""Tables of the top-K calibration cost study (results/topk_cal/PROTOCOL.md) from the runs' report.json files.

    python results/topk_cal/analyze.py [--runs /home/dev/n16k64_campaign/topk/runs] [--out-dir results/topk_cal]

- Per run (A, B, C): the setup phases, setup, per-epoch time, training (the sum of the epochs), total; the run-wide
  peak GPU allocated / reserved; the host RSS peak, sampled and as ru_maxrss, in the model load, after it, and in the
  training phase; the teacher
  storage; and, as information, the tail mass, the final E0M3 tile count and the final epoch's training KL.
- C as ratios to A and to B (and B to A).
- Today's A against results/nodev_cost's deterministic Llama-3.1-8B 16x64 run (the same unit and settings), as a
  consistency check.
Writes topk_cal.json and topk_cal.md.
"""
import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = {'A': 'A_full_e20', 'B': 'B_full_e5', 'C': 'C_top256_e5'}
NODEV = Path('/home/dev/n16k64_campaign/nodev_cost/runs/nd_tc_llama8b_16x64/report.json')
SETUP = ('model_load', 'data_load', 'teacher_precompute', 'candidate_packing')
GIB = 2 ** 30


def metrics(r):
    s = r['resources']['phases']
    by = s['by_phase']

    def sec(n):
        return by.get(n, {}).get('seconds', 0.0)
    ep = [e['epoch_seconds'] for e in r['epochs']]
    storage = r.get('teacher_storage')
    if storage is None:      # results/nodev_cost's reports predate the field: the full teacher, 128 windows x 511 x V bf16
        storage = dict(k=0, bytes=128 * 511 * 128256 * 2, devices=['cpu'], derived=True)
    m = {f'{n}_s': sec(n) for n in SETUP}
    m.update(setup_s=sum(sec(n) for n in SETUP), epochs=len(ep), per_epoch_s=statistics.mean(ep),
             per_epoch_min_s=min(ep), per_epoch_max_s=max(ep), training_s=sum(ep), write_output_s=sec('write_output'),
             total_s=r['resources']['total_seconds'],
             gpu_peak_allocated_gib=max(s['gpu_peak_allocated_gib']), gpu_peak_reserved_gib=max(s['gpu_peak_reserved_gib']),
             training_gpu_peak_allocated_gib=max(by['training']['gpu_peak_allocated_gib']),
             teacher_gpu_peak_allocated_gib=max(by['teacher_precompute']['gpu_peak_allocated_gib']),
             host_rss_sampled_gib=s['host_peak_rss_sampled_gib'], host_ru_maxrss_gib=r['resources']['cpu_peak_rss_gib'],
             training_host_rss_gib=by['training']['host_peak_rss_gib'],
             model_load_host_rss_gib=by['model_load']['host_peak_rss_gib'],
             post_load_host_rss_gib=max(v['host_peak_rss_gib'] for n, v in by.items() if n != 'model_load'),
             teacher_bytes=storage['bytes'], teacher_devices=storage['devices'], teacher_k=storage['k'],
             mean_tail_mass=storage.get('mean_tail_mass'), max_tail_mass=storage.get('max_window_tail_mass'),
             final_e0m3=r['final_e0m3_units'], final_train_kl=r['epochs'][-1]['train_kl'], map_sha256=r['map_sha256'])
    return m


ROWS = [('model load (s)', 'model_load_s', 1), ('data load (s)', 'data_load_s', 1),
        ('teacher precompute (s)', 'teacher_precompute_s', 1), ('lean packing (s)', 'candidate_packing_s', 1),
        ('**setup** (s)', 'setup_s', 1), ('per epoch (s)', 'per_epoch_s', 2), ('epochs', 'epochs', 0),
        ('**training** (s)', 'training_s', 1), ('write output (s)', 'write_output_s', 1), ('**total** (s)', 'total_s', 1),
        ('**peak GPU allocated** (GiB)', 'gpu_peak_allocated_gib', 2), ('peak GPU reserved (GiB)', 'gpu_peak_reserved_gib', 2),
        ('peak GPU allocated, training (GiB)', 'training_gpu_peak_allocated_gib', 2),
        ('peak GPU allocated, teacher precompute (GiB)', 'teacher_gpu_peak_allocated_gib', 2),
        ('**peak host RSS**, sampled (GiB)', 'host_rss_sampled_gib', 2), ('peak host RSS, ru_maxrss (GiB)', 'host_ru_maxrss_gib', 2),
        ('peak host RSS, model load (GiB)', 'model_load_host_rss_gib', 2),
        ('peak host RSS after the model load (GiB)', 'post_load_host_rss_gib', 2),
        ('peak host RSS, training phase (GiB)', 'training_host_rss_gib', 2), ('**teacher storage** (MB)', 'teacher_mb', 1)]
RATIO_ROWS = [('setup', 'setup_s'), ('per epoch', 'per_epoch_s'), ('training', 'training_s'), ('total', 'total_s'),
              ('peak GPU allocated', 'gpu_peak_allocated_gib'), ('peak GPU reserved', 'gpu_peak_reserved_gib'),
              ('peak host RSS, sampled', 'host_rss_sampled_gib'), ('peak host RSS, ru_maxrss', 'host_ru_maxrss_gib'),
              ('peak host RSS after the model load', 'post_load_host_rss_gib'),
              ('peak host RSS, training phase', 'training_host_rss_gib'), ('teacher storage', 'teacher_bytes')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=Path, default=Path('/home/dev/n16k64_campaign/topk/runs'))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'results' / 'topk_cal')
    args = ap.parse_args()
    m = {k: metrics(json.loads((args.runs / d / 'report.json').read_text())) for k, d in RUNS.items()}
    m['nodev_cost A'] = metrics(json.loads(NODEV.read_text()))
    for v in m.values():
        v['teacher_mb'] = v['teacher_bytes'] / 1e6
    cols = ['A', 'B', 'C', 'nodev_cost A']
    md = ['## The runs (Llama-3.1-8B 16x64, TM-OPT+TC, deterministic, --no-dev --no-eval)\n',
          'A: full-vocabulary teacher, 20 epochs; B: full-vocabulary teacher, 5 epochs; C: top-256 teacher, 5 epochs. '
          '"nodev_cost A": results/nodev_cost\'s run of A\'s configuration (2026-09-28).\n',
          '| | ' + ' | '.join(cols) + ' |', '|---|' + '---:|' * len(cols)]
    for label, key, nd in ROWS:
        md.append(f'| {label} | ' + ' | '.join(f'{m[c][key]:,.{nd}f}' for c in cols) + ' |')
    md.append('| teacher device | ' + ' | '.join(', '.join(m[c]['teacher_devices']) for c in cols) + ' |')
    md.append('| tail mass, mean / max window | ' + ' | '.join(
        (f"{m[c]['mean_tail_mass']:.4f} / {m[c]['max_tail_mass']:.4f}" if m[c]['mean_tail_mass'] is not None else '—')
        for c in cols) + ' |')
    md.append('| final E0M3 tiles | ' + ' | '.join(f"{m[c]['final_e0m3']:,}" for c in cols) + ' |')
    md.append('| final epoch training KL | ' + ' | '.join(f"{m[c]['final_train_kl']:.5f}" for c in cols) + ' |')
    md.append('')
    md += ['## C as ratios to A and to B\n', '| | C / A | C / B | B / A |', '|---|---:|---:|---:|']
    ratios = {}
    for label, key in RATIO_ROWS:
        r = {p: m[a][key] / m[b][key] for p, (a, b) in {'C/A': ('C', 'A'), 'C/B': ('C', 'B'), 'B/A': ('B', 'A')}.items()}
        ratios[key] = r
        md.append(f"| {label} | {r['C/A']:.4f} | {r['C/B']:.4f} | {r['B/A']:.4f} |")
    md.append('')
    md += ['## Consistency: today\'s A against the results/nodev_cost run\n', '| | today | nodev_cost | ratio |',
           '|---|---:|---:|---:|']
    for label, key in (('total (min)', 'total_s'), ('per epoch (s)', 'per_epoch_s'), ('setup (s)', 'setup_s'),
                       ('peak GPU allocated (GiB)', 'gpu_peak_allocated_gib'), ('peak host RSS, ru_maxrss (GiB)', 'host_ru_maxrss_gib')):
        a, b = m['A'][key], m['nodev_cost A'][key]
        scale = 60 if key == 'total_s' else 1
        md.append(f'| {label} | {a / scale:.2f} | {b / scale:.2f} | {a / b:.4f} |')
    md.append(f"| map sha256 | {m['A']['map_sha256'][:16]}… | {m['nodev_cost A']['map_sha256'][:16]}… | "
              f"{'equal' if m['A']['map_sha256'] == m['nodev_cost A']['map_sha256'] else 'DIFFERENT'} |")
    md.append('')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'topk_cal.json').write_text(json.dumps(dict(runs=m, ratios=ratios), indent=1) + '\n')
    (args.out_dir / 'topk_cal.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
