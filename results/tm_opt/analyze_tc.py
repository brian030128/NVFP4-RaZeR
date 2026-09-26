"""Tables of TM-OPT+TC (PROTOCOL_TC.md).

python results/tm_opt/analyze_tc.py profile   # the two profiles and the unit test  -> tc_profile.{json,md}
python results/tm_opt/analyze_tc.py runs      # the nine runs, evaluations, timing   -> tc_runs.{json,md}
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_items import D, M, R, TITLES, UNITS, fmt, load, nll, overlap, paired, run_record  # noqa: E402

SEED_SPREAD = dict(wiki=0.00109, c4=0.00038)      # #2, Llama 8x64: range of the three seeds' ΔNLL vs MR-OPT


def profile():
    out, lines = {}, []
    for label, name in (('TM-OPT', 'profile_tmopt_llama8b_8x64.json'), ('TM-OPT+TC', 'profile_tmopt_tc_llama8b_8x64.json')):
        p = load(R / name)
        out[label] = p['breakdown']
    phases = ['phase: forward', 'phase: loss', 'phase: backward', 'phase: optimizer']
    cats = sorted({c for lab in out for ph in phases for c in out[lab][ph]['gpu_ms']},
                  key=lambda c: -sum(out[lab][ph]['gpu_ms'].get(c, 0.0) for lab in out for ph in phases))
    lines += ['| phase | category | TM-OPT GPU s / epoch | TM-OPT+TC GPU s / epoch |', '|---|---|---:|---:|']
    for ph in phases:
        for c in cats:
            a, b = (out[lab][ph]['gpu_ms'].get(c, 0.0) / 1e3 for lab in ('TM-OPT', 'TM-OPT+TC'))
            if a > 0.005 or b > 0.005:
                lines.append(f'| {ph[7:]} | {c} | {a:.2f} | {b:.2f} |')
        lines.append(f"| {ph[7:]} | **total GPU / wall** | **{out['TM-OPT'][ph]['gpu_ms_total'] / 1e3:.2f} / {out['TM-OPT'][ph]['wall_ms'] / 1e3:.2f}** | "
                     f"**{out['TM-OPT+TC'][ph]['gpu_ms_total'] / 1e3:.2f} / {out['TM-OPT+TC'][ph]['wall_ms'] / 1e3:.2f}** |")
    for lab in out:
        out[lab]['epoch_gpu_seconds'] = sum(out[lab][ph]['gpu_ms_total'] for ph in phases) / 1e3
    lines += ['', f"Epoch under the profiler: TM-OPT {out['TM-OPT']['epoch_wall_seconds_with_profiler']:.1f} s wall "
              f"({out['TM-OPT']['epoch_gpu_seconds']:.1f} s GPU); TM-OPT+TC {out['TM-OPT+TC']['epoch_wall_seconds_with_profiler']:.1f} s wall "
              f"({out['TM-OPT+TC']['epoch_gpu_seconds']:.1f} s GPU)."]
    unit = load(R / 'unit_tests_tc.json') if (R / 'unit_tests_tc.json').exists() else None
    if unit:
        out['unit_test'] = unit['summary']
        lines += ['', '| case | FP32 error vs FP64 | TC error vs FP64 | threshold | result |', '|---|---:|---:|---:|---|']
        for k, c in unit['cases'].items():
            lines.append(f"| {k} | {c['error_vs_fp64']['fp32']:.2e} | {c['error_vs_fp64']['tc']:.2e} | {c['threshold']:.2e} | "
                         f"{'pass' if c['passed'] else 'FAIL'} |")
        lines += ['', f"Unit test: {'PASS' if unit['summary']['passed'] else 'FAIL'} ({len(unit['summary']['failures'])} of "
                  f"{unit['summary']['cases']} cases fail); max error FP32 {unit['summary']['max_error_fp32']:.2e}, "
                  f"TC {unit['summary']['max_error_tc']:.2e}; GEMM time over the shapes FP32 {unit['summary']['gemm_ms_total']['fp32']:.1f} ms, "
                  f"TC {unit['summary']['gemm_ms_total']['tc']:.1f} ms."]
    (HERE / 'tc_profile.json').write_text(json.dumps(out, indent=1) + '\n')
    (HERE / 'tc_profile.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def runs():
    out, lines = {}, []
    for model, title in TITLES.items():
        ev = load(R / f'tc_{model}_eval_native' / 'report.json')['evaluations']
        fev = load(R / f'tc_{model}_eval_fake' / 'report.json')['evaluations']
        items_nat = load(R / f'items_{model}_eval_native' / 'report.json')['evaluations']
        items_fake = load(R / f'items_{model}_eval_fake' / 'report.json')['evaluations']
        checks = dict(fourover6_native_repeats=nll(ev, 'FourOverSix') == nll(items_nat, 'FourOverSix'),
                      tmopt_native_repeats={u: nll(ev, f'tmopt-{u}') == nll(items_nat, f'tmopt-{u}') for u in UNITS},
                      fourover6_fake_repeats=nll(fev, 'FourOverSix') == nll(items_fake, 'FourOverSix'))
        rows = {}
        for u in UNITS:
            tm_dir = R / 'g2_tmopt' if (model == 'llama8b' and u == '8x64') else R / f'tm_{model}_{u}'
            tc_dir = R / f'tc_{model}_{u}'
            tm, tc = load(tm_dir / 'report.json'), load(tc_dir / 'report.json')
            tm_map, tc_map = (torch.load(d / 'map.pt', weights_only=True) for d in (tm_dir, tc_dir))
            old_nat = {**load(M / model / 'eval_native' / 'report.json')['evaluations'],
                       **load(M / model / 'eval16_native' / 'report.json')['evaluations']}
            row = dict(tc=dict(run_record(tc), native_ppl={d: ev[f'tc-{u}']['evaluation'][d]['ppl'] for d in D},
                               fake_ppl={d: fev[f'tc-{u}']['evaluation'][d]['ppl'] for d in D}),
                       tmopt=dict(run_record(tm), native_ppl={d: ev[f'tmopt-{u}']['evaluation'][d]['ppl'] for d in D}),
                       tc_minus_tmopt_native={d: paired(nll(ev, f'tc-{u}')[d], nll(ev, f'tmopt-{u}')[d]) for d in D},
                       tc_vs_fourover6_native={d: paired(nll(ev, f'tc-{u}')[d], nll(ev, 'FourOverSix')[d]) for d in D},
                       overlap_with_tmopt=overlap(tc_map, tm_map))
            if checks['fourover6_native_repeats']:
                row['tc_minus_mropt_native'] = {d: paired(nll(ev, f'tc-{u}')[d], nll(old_nat, f'mropt-{u}')[d]) for d in D}
            if checks['fourover6_fake_repeats']:
                row['tc_minus_tmopt_fake'] = {d: paired(nll(fev, f'tc-{u}')[d], nll(items_fake, f'tmopt-{u}')[d]) for d in D}
            p = row['tc_minus_tmopt_native']
            row['acceptable'] = all(p[d]['lower'] <= 0 for d in D)
            row['abs_difference_within_seed_spread'] = {d: abs(p[d]['mean']) <= SEED_SPREAD[d] for d in D}
            rows[u] = row
        out[model] = dict(checks=checks, units=rows)
        lines += [f'### {title}', '',
                  '| unit | TC tiles / TM-OPT tiles | Jaccard | TC native Wiki / C4 | TC − TM-OPT ΔWiki | ΔC4 | acceptable | fake ΔWiki | fake ΔC4 | TC − MR-OPT ΔWiki | ΔC4 | epoch s TM-OPT → TC | selection TM-OPT → TC | peak GPU TM-OPT → TC |',
                  '|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|']
        for u, r in rows.items():
            p, f, m = r['tc_minus_tmopt_native'], r.get('tc_minus_tmopt_fake'), r.get('tc_minus_mropt_native')
            lines.append(f"| {u} | {r['tc']['e0m3_units']:,} / {r['tmopt']['e0m3_units']:,} | {r['overlap_with_tmopt']['jaccard']:.3f} | "
                         f"{r['tc']['native_ppl']['wiki']:.4f} / {r['tc']['native_ppl']['c4']:.4f} | {fmt(p['wiki'])} | {fmt(p['c4'])} | "
                         f"{'**yes**' if r['acceptable'] else '**NO**'} | {fmt(f['wiki']) if f else '—'} | {fmt(f['c4']) if f else '—'} | "
                         f"{fmt(m['wiki']) if m else '—'} | {fmt(m['c4']) if m else '—'} | "
                         f"{r['tmopt']['epoch_seconds_mean']:.1f} → {r['tc']['epoch_seconds_mean']:.1f} | "
                         f"{r['tmopt']['selection_seconds'] / 60:.1f} → {r['tc']['selection_seconds'] / 60:.1f} min | "
                         f"{r['tmopt']['gpu_peak_allocated_gib']:.1f} → {r['tc']['gpu_peak_allocated_gib']:.1f} GiB |")
        lines += ['', f'Checks: {json.dumps(checks)}', '']
    probe = R / 'tc_probe_nondet' / 'report.json'
    if probe.exists():
        pr = load(probe)
        out['nondeterministic_probe'] = dict(epoch_seconds=[e['epoch_seconds'] for e in pr['epochs']], settings=pr['settings'])
        lines += [f"Non-deterministic TM-OPT+TC probe (Llama 8x64, 3 epochs): epoch seconds "
                  f"{' / '.join(f'{e:.1f}' for e in out['nondeterministic_probe']['epoch_seconds'])} (QAT C1 non-deterministic: 26.9 s)."]
    (HERE / 'tc_runs.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    (HERE / 'tc_runs.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    {'profile': profile, 'runs': runs}[sys.argv[1]]()
