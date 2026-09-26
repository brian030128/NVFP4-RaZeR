"""Tables of the 16x64 addendum (ADDENDUM_16x64.md).

python results/mr_variants/analyze_16x64.py mropt      # MR-OPT 16x64 vs 8x64 and 256x64, per model   -> mropt_16x64.{json,md}
python results/mr_variants/analyze_16x64.py variants   # the variants at 16x64, per model             -> variants_16x64.{json,md}
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_variants import RUNS, calibration, fmt, load, paired  # noqa: E402

MODELS = (('llama8b', 'Llama-3.1-8B'), ('mistral7b', 'Mistral-7B-v0.3'), ('phi4', 'Phi-4'))
VARIANTS = (('sig', 'MR-OPT+SIG'), ('ws', 'MR-OPT+WS'), ('sigws', 'MR-OPT+SIG+WS'))
D = ('wiki', 'c4')


def nll(ev, label):
    return {d: ev[label]['evaluation'][d]['nll'] for d in D}


def same(a, b):
    return all(a[d] == b[d] for d in D)


def granules(root):
    """All 8x64 granules of the model's quantized matrices (the 8x64 map's shape)."""
    m = torch.load(root / 'mropt_8x64' / 'map.pt', weights_only=True)
    return sum(t.numel() for t in m.values())


def mropt():
    out, lines = {}, []
    for model, title in MODELS:
        root = RUNS / model
        nat, fake = (load(root / f'eval16_{b}' / 'report.json') for b in ('native', 'fake'))
        old_nat, old_fake = (load(root / f'eval_{b}' / 'report.json') for b in ('native', 'fake'))
        ev, fev = nat['evaluations'], fake['evaluations']
        total = granules(root)
        cal = {u: calibration(load(root / f'mropt_{u}' / 'report.json')) for u in ('8x64', '16x64', '256x64')}
        rows = {}
        for u in ('8x64', '16x64', '256x64'):
            e = ev[f'mropt-{u}']
            rows[u] = dict(calibration=cal[u], tiles=e.get(f'e0m3_units_{u}', e['e0m3_units']), granules_8x64=e['e0m3_units'],
                           e0m3_share=e['e0m3_units'] / total, wiki=e['evaluation']['wiki']['ppl'], c4=e['evaluation']['c4']['ppl'],
                           vs_fourover6={d: paired(nll(ev, f'mropt-{u}')[d], nll(ev, 'FourOverSix')[d]) for d in D})
        m16 = nll(ev, 'mropt-16x64')
        comparisons = {f'16x64 - {u}': {d: paired(m16[d], nll(ev, f'mropt-{u}')[d]) for d in D} for u in ('8x64', '256x64')}
        f16 = fev['mropt-16x64']['evaluation']
        fake_row = dict(fourover6={d: fev['FourOverSix']['evaluation'][d]['ppl'] for d in D}, wiki=f16['wiki']['ppl'], c4=f16['c4']['ppl'],
                        vs_fourover6={d: paired(f16[d]['nll'], fev['FourOverSix']['evaluation'][d]['nll']) for d in D})
        checks = dict(
            same_windows=len({json.dumps({d: r['data'][d]['token_sha256'] for d in ('wiki', 'c4_paper')}, sort_keys=True)
                               for r in (nat, fake, old_nat, old_fake)}) == 1,
            native_repeats_earlier_evaluation={lab: same(nll(ev, lab), nll(old_nat['evaluations'], lab))
                                               for lab in ('FourOverSix', 'mropt-8x64', 'mropt-256x64')},
            fake_fourover6_repeats_earlier=same(nll(fev, 'FourOverSix'), nll(old_fake['evaluations'], 'FourOverSix')),
            map_mismatches=[lab for lab, e in list(ev.items()) + list(fev.items()) if e.get('native_map_mismatches') or e.get('lean_map_mismatches')],
            converted=ev['mropt-16x64'].get('converted_from'))
        out[model] = dict(fourover6={d: ev['FourOverSix']['evaluation'][d]['ppl'] for d in D}, units=rows,
                          mropt_16x64_vs=comparisons, fake=fake_row, checks=checks)
        lines += [f'### {title}', '',
                  '| unit | E0M3 tiles (own unit) | E0M3 share of weights | rounds / dev evaluations | optimization | '
                  'native WikiText-2 | native C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |',
                  '|---|---:|---:|---:|---:|---:|---:|---|---|',
                  f"| FourOverSix | 0 | 0 % | — | — | {out[model]['fourover6']['wiki']:.4f} | {out[model]['fourover6']['c4']:.4f} | — | — |"]
        for u in ('8x64', '16x64', '256x64'):
            r = rows[u]; c = r['calibration']
            lines.append(f"| {u} | {r['tiles']:,} | {100 * r['e0m3_share']:.2f} % | {c['rounds']} / {c['dev_evaluations']} | "
                         f"{c['optimization_s'] / 60:.1f} min | {r['wiki']:.4f} | {r['c4']:.4f} | {fmt(r['vs_fourover6']['wiki'])} | "
                         f"{fmt(r['vs_fourover6']['c4'])} |")
        lines += ['', '| MR-OPT 16x64 minus | ΔWiki | ΔC4 |', '|---|---|---|']
        for k, v in comparisons.items():
            lines.append(f"| MR-OPT {k.split(' - ')[1]} | {fmt(v['wiki'])} | {fmt(v['c4'])} |")
        lines += ['', f"Fake: FourOverSix {fake_row['fourover6']['wiki']:.4f} / {fake_row['fourover6']['c4']:.4f}; MR-OPT 16x64 "
                  f"{fake_row['wiki']:.4f} / {fake_row['c4']:.4f}, ΔWiki {fmt(fake_row['vs_fourover6']['wiki'])}, "
                  f"ΔC4 {fmt(fake_row['vs_fourover6']['c4'])} vs FourOverSix.", '',
                  f"Checks: {json.dumps(checks)}", '']
        c = cal['16x64']
        lines += [f"MR-OPT 16x64 calibration: setup {c['setup_s'] / 60:.1f} min, scoring {c['scoring_s_per_pass']:.1f} s per pass, "
                  f"dev evaluation {c['dev_s_per_try']:.1f} s per try, dev KL {c['dev_kl'][0]:.5f} → {c['dev_kl'][1]:.5f}, "
                  f"peak GPU {c['gpu_peak_allocated_gib']:.1f} / {c['gpu_peak_reserved_gib']:.1f} GiB, host {c['host_peak_rss_gib']:.1f} GiB, "
                  f"stop: {c['stopped']}.", '']
    (HERE / 'mropt_16x64.json').write_text(json.dumps(out, indent=1) + '\n')
    (HERE / 'mropt_16x64.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def variants():
    out, lines = {}, []
    for model, title in MODELS:
        root = RUNS / model
        if not (root / 'eval16v_native' / 'report.json').exists():
            continue
        nat = load(root / 'eval16v_native' / 'report.json')
        first = load(root / 'eval16_native' / 'report.json')
        ev = nat['evaluations']
        ref = torch.load(root / 'mropt_16x64' / 'map.pt', weights_only=True)
        rows = {}
        for v, name in (('mropt', 'MR-OPT'),) + VARIANTS:
            c = calibration(load(root / f'{v}_16x64' / 'report.json'))
            e = ev[f'{v}-16x64']
            row = dict(name=name, calibration=c, wiki=e['evaluation']['wiki']['ppl'], c4=e['evaluation']['c4']['ppl'],
                       vs_fourover6={d: paired(nll(ev, f'{v}-16x64')[d], nll(ev, 'FourOverSix')[d]) for d in D})
            if v != 'mropt':
                m = torch.load(root / f'{v}_16x64' / 'map.pt', weights_only=True)
                row['overlap'] = dict(shared=sum(int((m[n] & ref[n]).sum()) for n in ref),
                                      only_mropt=sum(int((ref[n] & ~m[n]).sum()) for n in ref),
                                      only_variant=sum(int((m[n] & ~ref[n]).sum()) for n in ref))
                p = {d: paired(nll(ev, f'{v}-16x64')[d], nll(ev, 'mropt-16x64')[d]) for d in D}
                row['vs_mropt'] = p
                row['acceptable'] = dict(official_not_significantly_worse=all(p[d]['lower'] <= 0 for d in D),
                                         secondary_upper_not_above_zero=all(p[d]['upper'] <= 0 for d in D))
            rows[v] = row
        checks = dict(same_windows=all(nat['data'][d]['token_sha256'] == first['data'][d]['token_sha256'] for d in ('wiki', 'c4_paper')),
                      repeats_first_evaluation={lab: same(nll(ev, lab), nll(first['evaluations'], lab)) for lab in ('FourOverSix', 'mropt-16x64')},
                      map_mismatches=[lab for lab, e in ev.items() if e.get('native_map_mismatches') or e.get('lean_map_mismatches')])
        out[model] = dict(rows=rows, checks=checks,
                          total_optimization_minutes={v: rows[v]['calibration']['optimization_s'] / 60 for v in rows})
        lines += [f'### {title}', '',
                  '| configuration | rounds / dev evaluations | E0M3 tiles | dev KL start → end | setup | optimization | '
                  'peak GPU allocated / reserved | host RSS |', '|---|---:|---:|---|---:|---:|---:|---:|']
        for v, r in rows.items():
            c = r['calibration']
            lines.append(f"| {r['name']} | {c['rounds']} / {c['dev_evaluations']} | {c['tiles']:,} | {c['dev_kl'][0]:.5f} → {c['dev_kl'][1]:.5f} | "
                         f"{c['setup_s'] / 60:.1f} min | **{c['optimization_s'] / 60:.1f} min** | {c['gpu_peak_allocated_gib']:.1f} / "
                         f"{c['gpu_peak_reserved_gib']:.1f} GiB | {c['host_peak_rss_gib']:.1f} GiB |")
        lines += ['', '| configuration | tiles shared with MR-OPT / only MR-OPT / only variant | WikiText-2 | C4 | ΔWiki vs FourOverSix | '
                  'ΔC4 vs FourOverSix | ΔWiki vs MR-OPT | ΔC4 vs MR-OPT | acceptable: official / secondary |',
                  '|---|---|---:|---:|---|---|---|---|---|']
        for v, r in rows.items():
            ov, vm, acc = r.get('overlap'), r.get('vs_mropt'), r.get('acceptable')
            overlap = '—' if not ov else '{:,} / {:,} / {:,}'.format(ov['shared'], ov['only_mropt'], ov['only_variant'])
            lines.append(f"| {r['name']} | {overlap} | "
                         f"{r['wiki']:.4f} | {r['c4']:.4f} | {fmt(r['vs_fourover6']['wiki'])} | {fmt(r['vs_fourover6']['c4'])} | "
                         f"{fmt(vm['wiki']) if vm else '—'} | {fmt(vm['c4']) if vm else '—'} | "
                         + (('**yes**' if acc['official_not_significantly_worse'] else '**NO**') + ' / '
                            + ('yes' if acc['secondary_upper_not_above_zero'] else 'no') if acc else '—') + ' |')
        lines += ['', f'Checks: {json.dumps(checks)}', '']
    (HERE / 'variants_16x64.json').write_text(json.dumps(out, indent=1) + '\n')
    (HERE / 'variants_16x64.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    {'mropt': mropt, 'variants': variants}[sys.argv[1]]()
