"""Tables and the SAFE verdict for results/native_decision_8x64/REPORT.md (criteria: PROTOCOL.md).

python results/native_decision_8x64/analyze_8x64.py RUN_ROOT   (writes summary.json and tables.md here)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'native_decision'))
from analyze_decision import first_divergence, fmt, load, paired, trajectory  # noqa: E402

TASK1_NATIVE = HERE.parent / 'native_decision' / 'runs' / 'eval_native' / 'report.json'


def run_summary(r):
    res = r['resources']
    phases = res['phases']['by_phase']
    dev_phase = 'native_dev_evaluation' if r.get('backends', {}).get('dev') == 'native' else 'dev_evaluation'
    return dict(status=r['status'], stopped=r.get('stopped'), tiles=r['final_e0m3_units'], map_sha256=r['map_sha256'],
                dev_initial=r['initial_dev']['kl'], dev_final=r['final_dev']['kl'],
                fake_initial_dev=r.get('fake_initial_dev', {}).get('kl'),
                setup_seconds=res['setup_seconds'], optimization_seconds=res['optimization_seconds'],
                scoring_passes=res['scoring_passes'], development_evaluations=res['development_evaluations'],
                scoring_seconds_per_pass=phases['scoring']['seconds'] / phases['scoring']['count'],
                dev_eval_seconds_per_try=phases[dev_phase]['seconds'] / max(1, phases[dev_phase]['count']),
                gpu_peak_allocated_gib=res['gpu_peak_allocated_gib'][0], gpu_peak_reserved_gib=res['gpu_peak_reserved_gib'][0],
                host_peak_rss_gib=res['cpu_peak_rss_gib'], built_in_evaluation={k: v['ppl'] for k, v in r['evaluation'].items()},
                trajectory=trajectory(r))


def main():
    root = Path(sys.argv[1])
    runs = {'DET-FAKE-8x64': load(root / 'det_fake_8x64' / 'report.json'),
            'DET-NATIVE-8x64': load(root / 'det_native_8x64' / 'report.json')}
    out = dict(runs={k: run_summary(r) for k, r in runs.items()},
               first_divergence=first_divergence(runs['DET-FAKE-8x64'], runs['DET-NATIVE-8x64']), evaluation={})
    f, n = out['runs']['DET-FAKE-8x64'], out['runs']['DET-NATIVE-8x64']
    out['speedup'] = dict(optimization=f['optimization_seconds'] / n['optimization_seconds'],
                          dev_eval_per_try=f['dev_eval_seconds_per_try'] / n['dev_eval_seconds_per_try'])
    evals = {b: load(root / f'eval_{b}' / 'report.json')['evaluations'] for b in ('native', 'fake')}
    # Verification: the fake map evaluation reproduces each DET run's built-in fake evaluation bitwise.
    out['fake_evaluate_map_reproduces_built_in'] = {
        label: all(evals['fake'][label]['evaluation'][d]['nll'] == runs[label]['evaluation'][d]['nll'] for d in ('wiki', 'c4'))
        for label in runs}
    for backend, ev in evals.items():
        base = ev['FourOverSix']['evaluation']
        table = {}
        for label, e in ev.items():
            x = e['evaluation']
            row = dict(tiles=e.get('e0m3_units'), wiki=x['wiki']['ppl'], c4=x['c4']['ppl'],
                       checks=dict(map_mismatches=e.get('native_map_mismatches'), activation_calls=e.get('native_activation_checks')))
            if label != 'FourOverSix':
                row['vs_fourover6'] = {d: paired(x[d]['nll'], base[d]['nll']) for d in ('wiki', 'c4')}
            table[label] = row
        dn, df = ev['DET-NATIVE-8x64']['evaluation'], ev['DET-FAKE-8x64']['evaluation']
        table['DET-NATIVE-8x64']['vs_det_fake_8x64'] = {d: paired(dn[d]['nll'], df[d]['nll']) for d in ('wiki', 'c4')}
        out['evaluation'][backend] = table
    crit = out['evaluation']['native']['DET-NATIVE-8x64']['vs_det_fake_8x64']
    worse = [d for d, p in crit.items() if p['verdict'] == 'worse']
    out['verdict'] = dict(safe=not worse, significantly_worse_on=worse)
    # Context: 8x64 maps vs Task 1's 256x64 DET maps (native), valid only if FourOverSix matches bitwise.
    t1 = load(TASK1_NATIVE)['evaluations']
    same_base = all(t1['FourOverSix']['evaluation'][d]['nll'] == evals['native']['FourOverSix']['evaluation'][d]['nll']
                    for d in ('wiki', 'c4'))
    out['context'] = dict(fourover6_native_identical_to_task1=same_base, pairs={})
    if same_base:
        for a in ('DET-FAKE-8x64', 'DET-NATIVE-8x64'):
            for b in ('DET-FAKE', 'DET-NATIVE'):
                xa, xb = evals['native'][a]['evaluation'], t1[b]['evaluation']
                out['context']['pairs'][f'{a} minus {b} (256x64)'] = {d: paired(xa[d]['nll'], xb[d]['nll']) for d in ('wiki', 'c4')}
    (HERE / 'summary.json').write_text(json.dumps(out, indent=2) + '\n')
    lines = ['| map | E0M3 tiles | WikiText | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |', '|---|---:|---:|---:|---|---|']
    for backend in ('native', 'fake'):
        lines.append(f'| **{backend} evaluation** | | | | | |')
        for label, row in out['evaluation'][backend].items():
            vf = row.get('vs_fourover6')
            lines.append(f"| {label} | {row['tiles']:,} | {row['wiki']:.6f} | {row['c4']:.6f} | {fmt(vf['wiki']) if vf else '—'} "
                         f"| {fmt(vf['c4']) if vf else '—'} |")
    lines += ['', '| DET-NATIVE-8x64 minus DET-FAKE-8x64 | ΔWiki | ΔC4 |', '|---|---|---|']
    for backend in ('native', 'fake'):
        v = out['evaluation'][backend]['DET-NATIVE-8x64']['vs_det_fake_8x64']
        lines.append(f"| {backend} evaluation | {fmt(v['wiki'])} | {fmt(v['c4'])} |")
    if out['context']['pairs']:
        lines += ['', '| context (native evaluation) | ΔWiki | ΔC4 |', '|---|---|---|']
        for k, v in out['context']['pairs'].items():
            lines.append(f"| {k} | {fmt(v['wiki'])} | {fmt(v['c4'])} |")
    (HERE / 'tables.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(verdict=out['verdict'], first_divergence=out['first_divergence'], speedup=out['speedup'],
                          reproduces=out['fake_evaluate_map_reproduces_built_in'], context_valid=same_base), indent=1))


if __name__ == '__main__':
    main()
