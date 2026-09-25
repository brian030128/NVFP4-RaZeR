"""Per-model tables of the MR-OPT variant study (PROTOCOL.md): calibration metrics, tile overlap, PPL, paired ΔNLL,
and acceptability under the official decision rule (deviation 2: mean - 2 SE <= 0 vs MR-OPT on both corpora) and,
as secondary, the task text's formula (mean + 2 SE <= 0).

python results/mr_variants/analyze_variants.py MODEL [OUT_DIR]   (writes <model>/summary.json and <model>/tables.md here)
"""
import json
import math
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get('MR_VARIANTS_RUNS', '/home/dev/n16k64_campaign/mr_variants/runs'))
VARIANTS = (('mropt', 'MR-OPT'), ('sig', 'MR-OPT+SIG'), ('ws', 'MR-OPT+WS'), ('sigws', 'MR-OPT+SIG+WS'))
UNITS = ('256x64', '8x64')


def load(p):
    return json.loads(Path(p).read_text())


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    two_se = 2 * math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) / math.sqrt(n) if n > 1 else 0.0
    return dict(mean=m, two_se=two_se, lower=m - two_se, upper=m + two_se,
                verdict='better' if m + two_se < 0 else ('worse' if m - two_se > 0 else 'inconclusive'))


def fmt(p):
    return f"{p['mean']:+.5f} ± {p['two_se']:.5f}" + ('' if p['verdict'] == 'inconclusive' else f" ({p['verdict']})")


def calibration(r):
    res = r['resources']
    ph = res['phases']['by_phase']
    return dict(rounds=res['scoring_passes'], dev_evaluations=res['development_evaluations'], tiles=r['final_e0m3_units'],
                dev_kl=[r['initial_dev']['kl'], r['final_dev']['kl']], setup_s=res['setup_seconds'],
                optimization_s=res['optimization_seconds'], evaluation_s=res['evaluation_seconds'],
                scoring_s_per_pass=ph['scoring']['seconds'] / ph['scoring']['count'],
                dev_s_per_try=ph['dev_evaluation']['seconds'] / ph['dev_evaluation']['count'],
                gpu_peak_allocated_gib=res['gpu_peak_allocated_gib'][0], gpu_peak_reserved_gib=res['gpu_peak_reserved_gib'][0],
                host_peak_rss_gib=res['cpu_peak_rss_gib'], stopped=r.get('stopped'), map_sha256=r['map_sha256'],
                accepted=[x['accepted'] for x in r['rounds']])


def main():
    model = sys.argv[1]
    root = RUNS / model
    out = dict(model=model, calibration={}, overlap={}, evaluation={}, acceptable={})
    for v, _ in VARIANTS:
        for u in UNITS:
            out['calibration'][f'{v}-{u}'] = calibration(load(root / f'{v}_{u}' / 'report.json'))
    for u in UNITS:
        ref = torch.load(root / f'mropt_{u}' / 'map.pt', weights_only=True)
        for v, _ in VARIANTS[1:]:
            m = torch.load(root / f'{v}_{u}' / 'map.pt', weights_only=True)
            out['overlap'][f'{v}-{u}'] = dict(shared=sum(int((m[n] & ref[n]).sum()) for n in ref),
                                             only_mropt=sum(int((ref[n] & ~m[n]).sum()) for n in ref),
                                             only_variant=sum(int((m[n] & ~ref[n]).sum()) for n in ref))
    windows = {}
    for backend in ('native', 'fake'):
        rep = load(root / f'eval_{backend}' / 'report.json')
        windows[backend] = {d: rep['data'][d]['token_sha256'] for d in ('wiki', 'c4_paper')}
        ev = rep['evaluations']
        fo6 = ev['FourOverSix']['evaluation']
        table = {}
        for label, e in ev.items():
            x = e['evaluation']
            row = dict(wiki=x['wiki']['ppl'], c4=x['c4']['ppl'], tiles=e.get('e0m3_units'),
                       tiles_256x64=e.get('e0m3_units_256x64'), map_mismatches=e.get('native_map_mismatches', e.get('lean_map_mismatches')),
                       activation_checks=e.get('native_activation_checks'))
            if label != 'FourOverSix':
                row['vs_fourover6'] = {d: paired(x[d]['nll'], fo6[d]['nll']) for d in ('wiki', 'c4')}
            unit = label.rsplit('-', 1)[-1]
            if label.split('-')[0] in dict(VARIANTS) and not label.startswith('mropt') and f'mropt-{unit}' in ev:
                row['vs_mropt'] = {d: paired(x[d]['nll'], ev[f'mropt-{unit}']['evaluation'][d]['nll']) for d in ('wiki', 'c4')}
            table[label] = row
        out['evaluation'][backend] = table
    bf16 = load(root / 'eval_bf16' / 'report.json')
    windows['bf16'] = {d: bf16['data'][d]['token_sha256'] for d in ('wiki', 'c4_paper')}
    out['evaluation']['bf16'] = {d: bf16['evaluations']['BF16']['evaluation'][d]['ppl'] for d in ('wiki', 'c4')}
    out['same_windows_all_processes'] = len({json.dumps(w, sort_keys=True) for w in windows.values()}) == 1
    for v, _ in VARIANTS[1:]:
        for u in UNITS:
            p = out['evaluation']['native'][f'{v}-{u}']['vs_mropt']
            out['acceptable'][f'{v}-{u}'] = dict(
                official_not_significantly_worse=all(p[d]['lower'] <= 0 for d in ('wiki', 'c4')),
                secondary_upper_not_above_zero=all(p[d]['upper'] <= 0 for d in ('wiki', 'c4')))
    out['total_optimization_minutes'] = {v: sum(out['calibration'][f'{v}-{u}']['optimization_s'] for u in UNITS) / 60 for v, _ in VARIANTS}
    if model == 'llama8b':
        out['sanity_gate'] = dict(mropt_256x64=out['calibration']['mropt-256x64']['map_sha256'].startswith('6e9704f5'),
                                  mropt_8x64=out['calibration']['mropt-8x64']['map_sha256'].startswith('471aa56a'))
        # Same maps and evaluation settings as Phase 2's evaluation of the committed maps: every window NLL should repeat.
        phase2 = Path(__file__).resolve().parents[1] / 'speedups' / 'runs_phase2'
        out['equal_to_phase2_committed_evaluation'] = {
            f'{backend} {u}': all(load(root / f'eval_{backend}' / 'report.json')['evaluations'][f'mropt-{u}']['evaluation'][d]['nll'] ==
                                  load(phase2 / f'eval_{backend}' / 'report.json')['evaluations'][f'COMMITTED-{u}']['evaluation'][d]['nll']
                                  for d in ('wiki', 'c4'))
            for backend in ('native', 'fake') for u in UNITS}
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / model
    target.mkdir(parents=True, exist_ok=True)
    (target / 'summary.json').write_text(json.dumps(out, indent=1) + '\n')
    lines = ['| unit | configuration | rounds / dev evaluations | E0M3 tiles | dev KL start → end | setup | optimization | '
             'scoring per pass | dev evaluation per try | peak GPU allocated / reserved | host RSS | stop |',
             '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for u in UNITS:
        for v, name in VARIANTS:
            c = out['calibration'][f'{v}-{u}']
            lines.append(f"| {u} | {name} | {c['rounds']} / {c['dev_evaluations']} | {c['tiles']:,} | {c['dev_kl'][0]:.5f} → {c['dev_kl'][1]:.5f} | "
                         f"{c['setup_s'] / 60:.1f} min | **{c['optimization_s'] / 60:.1f} min** | {c['scoring_s_per_pass']:.1f} s | "
                         f"{c['dev_s_per_try']:.1f} s | {c['gpu_peak_allocated_gib']:.1f} / {c['gpu_peak_reserved_gib']:.1f} GiB | "
                         f"{c['host_peak_rss_gib']:.1f} GiB | {c['stopped']} |")
    lines += ['', '| unit | configuration | tiles shared with MR-OPT | only MR-OPT | only variant | native WikiText-2 | native C4 | '
              'ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | ΔWiki vs MR-OPT | ΔC4 vs MR-OPT | acceptable: official / secondary |',
              '|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|']
    for u in UNITS:
        for v, name in VARIANTS:
            row = out['evaluation']['native'][f'{v}-{u}']
            ov = out['overlap'].get(f'{v}-{u}')
            vm = row.get('vs_mropt')
            acc = out['acceptable'].get(f'{v}-{u}')
            lines.append(f"| {u} | {name} | {ov['shared']:,} | {ov['only_mropt']:,} | {ov['only_variant']:,} | " if ov else f"| {u} | {name} | — | — | — | ")
            lines[-1] += (f"{row['wiki']:.4f} | {row['c4']:.4f} | {fmt(row['vs_fourover6']['wiki'])} | {fmt(row['vs_fourover6']['c4'])} | "
                          f"{fmt(vm['wiki']) if vm else '—'} | {fmt(vm['c4']) if vm else '—'} | "
                          f"{('**yes**' if acc['official_not_significantly_worse'] else '**NO**') + ' / ' + ('yes' if acc['secondary_upper_not_above_zero'] else 'no') if acc else '—'} |")
    lines += ['', '| backend | map | WikiText-2 | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |', '|---|---|---:|---:|---|---|']
    for backend in ('native', 'fake'):
        for label in ('FourOverSix', 'NVFP4', 'mropt-256x64', 'mropt-8x64'):
            r = out['evaluation'][backend][label]
            vf = r.get('vs_fourover6')
            lines.append(f"| {backend} | {label} | {r['wiki']:.4f} | {r['c4']:.4f} | {fmt(vf['wiki']) if vf else '—'} | {fmt(vf['c4']) if vf else '—'} |")
    lines.append(f"| fake | BF16 | {out['evaluation']['bf16']['wiki']:.4f} | {out['evaluation']['bf16']['c4']:.4f} | | |")
    (target / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(json.dumps(dict(same_windows=out['same_windows_all_processes'], acceptable=out['acceptable'],
                          total_optimization_minutes=out['total_optimization_minutes'], sanity_gate=out.get('sanity_gate'),
                          equal_to_phase2=out.get('equal_to_phase2_committed_evaluation')), indent=1))


if __name__ == '__main__':
    main()
