"""Tables of the fit-windows cost runs (NOTE.md): E128 / E256 / E512 (top-1000, 5 epochs) next to topk-cal's A (full
vocabulary, 128 windows, 20 epochs) and C (top-256, 128 windows, 5 epochs), with ratios to A.

    python results/topk_cal/fit_windows/analyze_fit.py [--runs /home/dev/n16k64_campaign/topk/fit_windows]

The metrics are ../analyze.py's (the PhaseMonitor phases and peaks of each run's report.json), plus the optimizer
steps. Writes fit_windows.json and fit_windows.md here.
"""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('analyze', HERE.parent / 'analyze.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def run_metrics(report):
    m = A.metrics(report)
    m.update(steps=report['epochs'][-1]['step'], fit_windows=report['args'].get('fit_windows', 128),
             teacher_topk=report['args'].get('teacher_topk', 0), teacher_mb=m['teacher_bytes'] / 1e6)
    return m


ROWS = [('fit windows', 'fit_windows', 0), ('teacher top-K (0 = full vocabulary)', 'teacher_topk', 0), ('epochs', 'epochs', 0),
        ('optimizer steps', 'steps', 0), ('model load (s)', 'model_load_s', 1), ('data load (s)', 'data_load_s', 1),
        ('teacher precompute (s)', 'teacher_precompute_s', 1), ('lean packing (s)', 'candidate_packing_s', 1),
        ('**setup** (s)', 'setup_s', 1), ('**per epoch** (s)', 'per_epoch_s', 2), ('**training** (s)', 'training_s', 1),
        ('**total** (s)', 'total_s', 1), ('**peak GPU allocated** (GiB)', 'gpu_peak_allocated_gib', 2),
        ('peak GPU reserved (GiB)', 'gpu_peak_reserved_gib', 2), ('**peak host RSS**, ru_maxrss (GiB)', 'host_ru_maxrss_gib', 2),
        ('peak host RSS after the model load (GiB)', 'post_load_host_rss_gib', 2), ('**teacher storage** (MB)', 'teacher_mb', 1)]
RATIO = [('setup', 'setup_s'), ('per epoch', 'per_epoch_s'), ('training', 'training_s'), ('total', 'total_s'),
         ('optimizer steps', 'steps'), ('peak GPU allocated', 'gpu_peak_allocated_gib'), ('peak GPU reserved', 'gpu_peak_reserved_gib'),
         ('peak host RSS, ru_maxrss', 'host_ru_maxrss_gib'), ('host RSS after the model load', 'post_load_host_rss_gib'),
         ('teacher storage', 'teacher_bytes')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=Path, default=Path('/home/dev/n16k64_campaign/topk/fit_windows'))
    args = ap.parse_args()
    reports = {'A': json.loads((HERE.parent / 'runs' / 'A_full_e20.json').read_text()),
               'C': json.loads((HERE.parent / 'runs' / 'C_top256_e5.json').read_text())}
    for r in ('E128', 'E256', 'E512'):
        reports[r] = json.loads((args.runs / r / 'report.json').read_text())
    m = {k: run_metrics(v) for k, v in reports.items()}
    cols = list(m)
    md = ['## Llama-3.1-8B 16x64, TM-OPT+TC (deterministic, --no-dev --no-eval)\n',
          '| | ' + ' | '.join(cols) + ' |', '|---|' + '---:|' * len(cols)]
    for label, k, nd in ROWS:
        md.append(f'| {label} | ' + ' | '.join(f'{m[c][k]:,.{nd}f}' for c in cols) + ' |')
    md.append('| teacher device | ' + ' | '.join(', '.join(m[c]['teacher_devices']) for c in cols) + ' |')
    md.append('| tail mass, mean / max window | ' + ' | '.join(
        f"{m[c]['mean_tail_mass']:.4f} / {m[c]['max_tail_mass']:.4f}" if m[c]['mean_tail_mass'] is not None else '—'
        for c in cols) + ' |')
    md.append('| final E0M3 tiles | ' + ' | '.join(f"{m[c]['final_e0m3']:,}" for c in cols) + ' |')
    md.append('| final-epoch training KL | ' + ' | '.join(
        f"{m[c]['final_train_kl']:.5f}" + (' (top-K)' if m[c]['teacher_topk'] else ' (full)') for c in cols) + ' |')
    md += ['', '## Ratios to A (full vocabulary, 128 windows, 20 epochs)\n', '| | ' + ' | '.join(c for c in cols if c != 'A') + ' |',
           '|---|' + '---:|' * (len(cols) - 1)]
    ratios = {}
    for label, k in RATIO:
        ratios[k] = {c: m[c][k] / m['A'][k] for c in cols if c != 'A'}
        md.append(f'| {label} | ' + ' | '.join(f'{ratios[k][c]:.3f}' for c in cols if c != 'A') + ' |')
    ext = {r: reports[r].get('fit_extension') for r in ('E256', 'E512')}
    out = dict(runs=m, ratios_to_A=ratios,
               fit_extension={r: (None if e is None else {k: v for k, v in e.items() if k != 'records'} | dict(
                   documents=[x['document_sha256'] for x in e['records']])) for r, e in ext.items()},
               maps={c: m[c]['map_sha256'] for c in cols})
    (HERE / 'fit_windows.json').write_text(json.dumps(out, indent=1) + '\n')
    (HERE / 'fit_windows.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
