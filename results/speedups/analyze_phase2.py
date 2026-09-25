"""Phase 2 tables and the B1 adoption verdict (PROTOCOL_PHASE2.md).

python results/speedups/analyze_phase2.py   (writes summary_phase2.json and tables_phase2.md here)
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'native_decision'))
from analyze_decision import fmt, load, paired  # noqa: E402

P1 = Path('/home/dev/n16k64_campaign/speedups/phase1')
P2 = Path('/home/dev/n16k64_campaign/speedups/phase2')
COMMITTED = {'256x64': ('results/native_decision/runs/det_native/report.json', '/home/dev/n16k64_campaign/native_decision/det_native/map.pt'),
             '8x64': ('results/native_decision_8x64/runs/det_native_8x64/report.json',
                      '/home/dev/n16k64_campaign/native_decision_8x64/det_native_8x64/map.pt')}


def candidates(off, on):
    a, b = (torch.load(p / 'round0_scores.pt', weights_only=True) for p in (off, on))
    out = dict(tiles=0, legacy=0, b1=0, both=0, legacy_only=0, b1_only=0, mu_max_rel=0.0, se_max_rel=0.0,
               disagreeing_bound_over_se=[])
    for n in a:
        la, lb = a[n]['kl_mean'] + 2 * a[n]['kl_se'], b[n]['kl_mean'] + 2 * b[n]['kl_se']
        ca, cb = la < 0, lb < 0
        out['tiles'] += ca.numel(); out['legacy'] += int(ca.sum()); out['b1'] += int(cb.sum())
        out['both'] += int((ca & cb).sum()); out['legacy_only'] += int((ca & ~cb).sum()); out['b1_only'] += int((~ca & cb).sum())
        for key in ('kl_mean', 'kl_se'):
            ref = a[n][key]
            rel = float((b[n][key] - ref).abs().max() / ref.abs().max().clamp_min(1e-300))
            out['mu_max_rel' if key == 'kl_mean' else 'se_max_rel'] = max(out['mu_max_rel' if key == 'kl_mean' else 'se_max_rel'], rel)
        d = ca != cb
        if d.any():
            out['disagreeing_bound_over_se'] += (la[d].abs() / a[n]['kl_se'][d]).tolist()
    b_ = out.pop('disagreeing_bound_over_se')
    out['disagreeing_max_bound_over_se'] = max(b_) if b_ else 0.0
    return out


def resources(r):
    res = r['resources']
    ph = res['phases']['by_phase']
    return dict(optimization_min=res['optimization_seconds'] / 60, setup_s=res['setup_seconds'],
                scoring_s_per_pass=ph['scoring']['seconds'] / ph['scoring']['count'],
                dev_s_per_try=ph['dev_evaluation']['seconds'] / ph['dev_evaluation']['count'],
                gpu_peak_allocated_gib=res['gpu_peak_allocated_gib'][0], gpu_peak_reserved_gib=res['gpu_peak_reserved_gib'][0],
                host_peak_rss_gib=res['cpu_peak_rss_gib'], rounds=res['scoring_passes'],
                dev_evaluations=res['development_evaluations'], tiles=r['final_e0m3_units'])


def main():
    out = dict(unit_tests=json.loads((HERE / 'phase2_unit_tests.json').read_text())['summary'], candidates={}, e2e={}, evaluation={})
    out['candidates']['llama8b 256x64'] = candidates(P1 / 'llama_det_native_256x64', P2 / 'llama_det_native_256x64_b1')
    out['candidates']['llama8b 8x64'] = candidates(P1 / 'llama_det_native_8x64', P2 / 'llama_det_native_8x64_b1')
    out['candidates']['phi4 8x64'] = candidates(P1 / 'phi4_new_16x8', P2 / 'phi4_b1_16x8')
    for unit, (ref_report, ref_map) in COMMITTED.items():
        b1 = load(P2 / f'llama_det_native_{unit}_b1' / 'report.json')
        p1 = load(P1 / f'llama_det_native_{unit}' / 'report.json')
        mb, mc = (torch.load(p, weights_only=True) for p in (P2 / f'llama_det_native_{unit}_b1' / 'map.pt', ref_map))
        inter = sum(int((mb[n] & mc[n]).sum()) for n in mc)
        out['e2e'][unit] = dict(b1=resources(b1), phase1=resources(p1),
                                overlap=dict(committed=sum(int(m.sum()) for m in mc.values()), b1=sum(int(m.sum()) for m in mb.values()),
                                             both=inter, only_committed=sum(int((mc[n] & ~mb[n]).sum()) for n in mc),
                                             only_b1=sum(int((mb[n] & ~mc[n]).sum()) for n in mc)),
                                trajectory_b1=[(x['round'], x['candidates'], x['accepted']) for x in b1['rounds']],
                                trajectory_committed=[(x['round'], x['candidates'], x['accepted']) for x in load(ref_report)['rounds']])
    verdict = {}
    for backend in ('native', 'fake'):
        ev = load(P2 / f'eval_{backend}' / 'report.json')['evaluations']
        t = {}
        for label, e in ev.items():
            t[label] = dict(wiki=e['evaluation']['wiki']['ppl'], c4=e['evaluation']['c4']['ppl'], tiles=e.get('e0m3_units'))
        for unit in ('256x64', '8x64'):
            b, c = ev[f'B1-{unit}']['evaluation'], ev[f'COMMITTED-{unit}']['evaluation']
            t[f'B1-{unit} minus COMMITTED-{unit}'] = {d: paired(b[d]['nll'], c[d]['nll']) for d in ('wiki', 'c4')}
            if backend == 'native':
                verdict[unit] = [d for d, p in t[f'B1-{unit} minus COMMITTED-{unit}'].items() if p['verdict'] == 'worse']
        out['evaluation'][backend] = t
    out['adopted'] = not any(verdict.values())
    out['significantly_worse'] = verdict
    (HERE / 'summary_phase2.json').write_text(json.dumps(out, indent=1) + '\n')
    lines = ['| comparison (round 0) | tiles | legacy candidates | B1 candidates | both | legacy only | B1 only | '
             'max normwise rel. error mu / SE | max |bound|/SE of a disagreeing tile |', '|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for k, c in out['candidates'].items():
        lines.append(f"| {k} | {c['tiles']:,} | {c['legacy']:,} | {c['b1']:,} | {c['both']:,} | {c['legacy_only']} | {c['b1_only']} | "
                     f"{c['mu_max_rel']:.1e} / {c['se_max_rel']:.1e} | {c['disagreeing_max_bound_over_se']:.2e} |")
    lines += ['', '| unit | run | rounds / dev evaluations | tiles | optimization | scoring per pass | dev evaluation per try | '
              'peak GPU allocated / reserved | peak host RSS |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for unit, e in out['e2e'].items():
        for name in ('phase1', 'b1'):
            r = e[name]
            lines.append(f"| {unit} | {'items 1+3' if name == 'phase1' else 'items 1+3 + B1'} | {r['rounds']} / {r['dev_evaluations']} | "
                         f"{r['tiles']:,} | {r['optimization_min']:.1f} min | {r['scoring_s_per_pass']:.1f} s | {r['dev_s_per_try']:.2f} s | "
                         f"{r['gpu_peak_allocated_gib']:.1f} / {r['gpu_peak_reserved_gib']:.1f} GiB | {r['host_peak_rss_gib']:.1f} GiB |")
    lines += ['', '| unit | committed tiles | B1 tiles | both | only committed | only B1 |', '|---|---:|---:|---:|---:|---:|']
    for unit, e in out['e2e'].items():
        o = e['overlap']
        lines.append(f"| {unit} | {o['committed']:,} | {o['b1']:,} | {o['both']:,} | {o['only_committed']:,} | {o['only_b1']:,} |")
    lines += ['', '| backend | map | tiles | WikiText-2 | C4 |', '|---|---|---:|---:|---:|']
    for backend in ('native', 'fake'):
        for label, r in out['evaluation'][backend].items():
            if 'minus' not in label:
                lines.append(f"| {backend} | {label} | {r['tiles'] if r['tiles'] is not None else '—'} | {r['wiki']:.6f} | {r['c4']:.6f} |")
    lines += ['', '| backend | B1 minus committed | ΔWiki | ΔC4 |', '|---|---|---|---|']
    for backend in ('native', 'fake'):
        for unit in ('256x64', '8x64'):
            p = out['evaluation'][backend][f'B1-{unit} minus COMMITTED-{unit}']
            lines.append(f"| {backend}{' (criterion)' if backend == 'native' else ''} | {unit} | {fmt(p['wiki'])} | {fmt(p['c4'])} |")
    (HERE / 'tables_phase2.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(json.dumps(dict(adopted=out['adopted'], significantly_worse=verdict)))


if __name__ == '__main__':
    main()
