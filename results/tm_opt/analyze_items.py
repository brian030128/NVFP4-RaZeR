"""Tables of items #2 and #1 (PROTOCOL_ITEMS.md).

python results/tm_opt/analyze_items.py seeds    # #2: seed variance, Llama 8x64       -> items_seeds.{json,md}
python results/tm_opt/analyze_items.py models   # #1: TM-OPT vs MR-OPT, three models  -> items_models.{json,md}
"""
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
R = Path('/home/dev/n16k64_campaign/tm_opt/runs')
M = Path('/home/dev/n16k64_campaign/mr_variants/runs')
D = ('wiki', 'c4')
UNITS = ('8x64', '16x64', '256x64')
TITLES = {'llama8b': 'Llama-3.1-8B', 'mistral7b': 'Mistral-7B-v0.3', 'phi4': 'Phi-4'}
H200 = Path('results/mixfp4_potential/train_map/llama_8x64_ste/report.json')


def load(p):
    return json.loads(Path(p).read_text())


def paired(x, y):
    d = [a - b for a, b in zip(x, y)]
    n = len(d)
    m = sum(d) / n
    two_se = 2 * math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
    verdict = 'significantly better' if m + two_se < 0 else ('significantly worse' if m - two_se > 0 else 'not significantly different')
    return dict(mean=m, two_se=two_se, lower=m - two_se, upper=m + two_se, verdict=verdict)


def fmt(p):
    short = {'significantly better': 'better', 'significantly worse': 'worse', 'not significantly different': 'n.s.'}[p['verdict']]
    return f"{p['mean']:+.5f} ± {p['two_se']:.5f} ({short})"


def nll(ev, label):
    return {d: ev[label]['evaluation'][d]['nll'] for d in D}


def overlap(a, b):
    """a, b: {module: bool map}; shared, only a, only b, Jaccard."""
    shared = sum(int((a[n] & b[n]).sum()) for n in a)
    only_a = sum(int((a[n] & ~b[n]).sum()) for n in a)
    only_b = sum(int((b[n] & ~a[n]).sum()) for n in a)
    union = shared + only_a + only_b
    return dict(shared=shared, only_first=only_a, only_second=only_b, jaccard=shared / union if union else 1.0)


def run_record(r):
    res = r['resources']
    epochs = r['epochs']
    return dict(e0m3_units=r['final_e0m3_units'], setup_seconds=r['setup_seconds'], selection_seconds=r['optimization_seconds'],
                epoch_seconds_mean=sum(e['epoch_seconds'] for e in epochs) / len(epochs),
                gpu_peak_allocated_gib=res['gpu_peak_allocated_gib'][0], gpu_peak_reserved_gib=res['gpu_peak_reserved_gib'][0],
                host_peak_rss_gib=res['cpu_peak_rss_gib'], initial_dev_kl=r['initial_dev']['kl'], final_dev_kl=r['final_dev']['kl'],
                dev_kl_curve={e['epoch'] + 1: e['dev_kl'] for e in epochs if 'dev_kl' in e},
                e0m3_curve={e['epoch'] + 1: e['e0m3_units'] for e in epochs if 'dev_kl' in e})


def seeds():
    runs = {0: R / 'g2_tmopt', 1: R / 'tm_llama8b_8x64_seed1', 2: R / 'tm_llama8b_8x64_seed2'}
    reports = {s: load(p / 'report.json') for s, p in runs.items()}
    maps = {s: torch.load(p / 'map.pt', weights_only=True) for s, p in runs.items()}
    mr_map = torch.load(M / 'llama8b' / 'mropt_8x64' / 'map.pt', weights_only=True)
    nat, fake = load(R / 'seeds_eval_native' / 'report.json'), load(R / 'seeds_eval_fake' / 'report.json')
    ev, fev = nat['evaluations'], fake['evaluations']
    old_nat, old_fake = load(M / 'llama8b' / 'eval_native' / 'report.json')['evaluations'], load(M / 'llama8b' / 'eval_fake' / 'report.json')['evaluations']
    g2 = load(R / 'g2_eval_native' / 'report.json')['evaluations']
    checks = dict(fourover6_native_repeats=nll(ev, 'FourOverSix') == nll(old_nat, 'FourOverSix'),
                  mropt_8x64_native_repeats=nll(ev, 'mropt-8x64') == nll(old_nat, 'mropt-8x64'),
                  seed0_native_repeats_group2=nll(ev, 'tm-seed0') == nll(g2, 'tmopt'),
                  fourover6_fake_repeats=nll(fev, 'FourOverSix') == nll(old_fake, 'FourOverSix'))
    out = dict(checks=checks, seeds={}, pairs={}, vs_mropt={}, fake_vs_mropt={})
    for s in runs:
        lab = f'tm-seed{s}'
        out['seeds'][s] = dict(run_record(reports[s]), native_ppl={d: ev[lab]['evaluation'][d]['ppl'] for d in D},
                               fake_ppl={d: fev[lab]['evaluation'][d]['ppl'] for d in D},
                               native_vs_fourover6={d: paired(nll(ev, lab)[d], nll(ev, 'FourOverSix')[d]) for d in D})
        out['vs_mropt'][s] = dict({d: paired(nll(ev, lab)[d], nll(ev, 'mropt-8x64')[d]) for d in D},
                                  overlap=overlap(maps[s], mr_map))
        if checks['fourover6_fake_repeats']:
            out['fake_vs_mropt'][s] = {d: paired(nll(fev, lab)[d], nll(old_fake, 'mropt-8x64')[d]) for d in D}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        out['pairs'][f'{a}-{b}'] = dict({d: paired(nll(ev, f'tm-seed{a}')[d], nll(ev, f'tm-seed{b}')[d]) for d in D},
                                        overlap=overlap(maps[a], maps[b]))
    rule = {}
    for d in D:
        means = [out['vs_mropt'][s][d]['mean'] for s in runs]
        gap, spread = sum(means) / len(means), max(means) - min(means)
        rule[d] = dict(gap=gap, spread=spread, small=spread < abs(gap) / 2,
                       every_seed_significantly_better=all(out['vs_mropt'][s][d]['verdict'] == 'significantly better' for s in runs))
    out['spread_vs_gap'] = dict(per_corpus=rule, small_on_both=all(rule[d]['small'] for d in D))
    h = load(H200)
    h_curve = {e['epoch'] + 1: e['e0m3_units'] for e in h['epochs'] if 'dev_kl' in e}
    h200 = {}
    for ep, count in h_curve.items():
        local = [out['seeds'][s]['e0m3_curve'][ep] for s in runs]
        lo, hi, mean = min(local), max(local), sum(local) / len(local)
        h200[ep] = dict(h200=count, local=local, within_seed_range=lo <= count <= hi,
                        relative_seed_spread=(hi - lo) / mean if mean else None,
                        relative_difference_seed0=(reports[0]['epochs'][ep - 1]['e0m3_units'] - count) / count if count else None)
    out['h200_tile_counts'] = h200
    (HERE / 'items_seeds.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    lines = ['| seed | E0M3 tiles | final dev KL | native WikiText-2 | native C4 | fake WikiText-2 | fake C4 | ΔWiki vs MR-OPT 8x64 | ΔC4 vs MR-OPT 8x64 | Jaccard with MR-OPT | selection |',
             '|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|']
    for s in runs:
        x, v = out['seeds'][s], out['vs_mropt'][s]
        lines.append(f"| {s} | {x['e0m3_units']:,} | {x['final_dev_kl']:.5f} | {x['native_ppl']['wiki']:.4f} | {x['native_ppl']['c4']:.4f} | "
                     f"{x['fake_ppl']['wiki']:.4f} | {x['fake_ppl']['c4']:.4f} | {fmt(v['wiki'])} | {fmt(v['c4'])} | "
                     f"{v['overlap']['jaccard']:.3f} | {x['selection_seconds'] / 60:.1f} min |")
    lines += ['', '| seeds | ΔWiki | ΔC4 | Jaccard | shared / only first / only second |', '|---|---|---|---:|---|']
    for k, v in out['pairs'].items():
        o = v['overlap']
        lines.append(f"| {k} | {fmt(v['wiki'])} | {fmt(v['c4'])} | {o['jaccard']:.3f} | {o['shared']:,} / {o['only_first']:,} / {o['only_second']:,} |")
    lines += ['', '| corpus | gap (mean ΔNLL vs MR-OPT) | seed spread (range) | small (< gap/2) | every seed significantly better |', '|---|---:|---:|---|---|']
    for d in D:
        r = rule[d]
        lines.append(f"| {d} | {r['gap']:+.5f} | {r['spread']:.5f} | {r['small']} | {r['every_seed_significantly_better']} |")
    lines += ['', '| epoch | H200 E0M3 | local seeds 0 / 1 / 2 | H200 within the seed range | relative seed spread | seed 0 vs H200 |', '|---:|---:|---|---|---:|---:|']
    for ep, v in h200.items():
        spread = '' if v['relative_seed_spread'] is None else f"{100 * v['relative_seed_spread']:.1f} %"
        diff = '' if v['relative_difference_seed0'] is None else f"{100 * v['relative_difference_seed0']:+.1f} %"
        local = ' / '.join(f'{c:,}' for c in v['local'])
        lines.append(f"| {ep} | {v['h200']:,} | {local} | {v['within_seed_range']} | {spread} | {diff} |")
    lines += ['', f'Checks: {json.dumps(checks)}']
    (HERE / 'items_seeds.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def models():
    out, lines = {}, []
    for model, title in TITLES.items():
        nat_p, fake_p = R / f'items_{model}_eval_native' / 'report.json', R / f'items_{model}_eval_fake' / 'report.json'
        if not nat_p.exists():
            continue
        ev, fev = load(nat_p)['evaluations'], load(fake_p)['evaluations']
        old_nat = {**load(M / model / 'eval_native' / 'report.json')['evaluations'], **load(M / model / 'eval16_native' / 'report.json')['evaluations']}
        old_fake = {**load(M / model / 'eval_fake' / 'report.json')['evaluations'], **load(M / model / 'eval16_fake' / 'report.json')['evaluations']}
        checks = dict(fourover6_native_repeats=nll(ev, 'FourOverSix') == nll(old_nat, 'FourOverSix'),
                      mropt_native_repeats={u: nll(ev, f'mropt-{u}') == nll(old_nat, f'mropt-{u}') for u in UNITS},
                      fourover6_fake_repeats=nll(fev, 'FourOverSix') == nll(old_fake, 'FourOverSix'))
        rows = {}
        for u in UNITS:
            tm_dir = R / 'g2_tmopt' if (model == 'llama8b' and u == '8x64') else R / f'tm_{model}_{u}'
            tm, mr = load(tm_dir / 'report.json'), load(M / model / f'mropt_{u}' / 'report.json')
            tm_map, mr_map = torch.load(tm_dir / 'map.pt', weights_only=True), torch.load(M / model / f'mropt_{u}' / 'map.pt', weights_only=True)
            lab_t, lab_m = f'tmopt-{u}', f'mropt-{u}'
            row = dict(tmopt=dict(run_record(tm), run=str(tm_dir), native_ppl={d: ev[lab_t]['evaluation'][d]['ppl'] for d in D},
                                  fake_ppl={d: fev[lab_t]['evaluation'][d]['ppl'] for d in D},
                                  native_vs_fourover6={d: paired(nll(ev, lab_t)[d], nll(ev, 'FourOverSix')[d]) for d in D},
                                  fake_vs_fourover6={d: paired(nll(fev, lab_t)[d], nll(fev, 'FourOverSix')[d]) for d in D}),
                       mropt=dict(e0m3_units=mr['final_e0m3_units'], selection_seconds=mr['resources']['optimization_seconds'],
                                  setup_seconds=mr['resources']['setup_seconds'],
                                  gpu_peak_allocated_gib=mr['resources']['gpu_peak_allocated_gib'][0],
                                  gpu_peak_reserved_gib=mr['resources']['gpu_peak_reserved_gib'][0],
                                  host_peak_rss_gib=mr['resources']['cpu_peak_rss_gib'], final_dev_kl=mr['final_dev']['kl'],
                                  native_ppl={d: ev[lab_m]['evaluation'][d]['ppl'] for d in D},
                                  native_vs_fourover6={d: paired(nll(ev, lab_m)[d], nll(ev, 'FourOverSix')[d]) for d in D}),
                       tm_minus_mr_native={d: paired(nll(ev, lab_t)[d], nll(ev, lab_m)[d]) for d in D},
                       overlap=overlap(tm_map, mr_map))
            if checks['fourover6_fake_repeats']:
                row['tm_minus_mr_fake'] = {d: paired(nll(fev, lab_t)[d], nll(old_fake, lab_m)[d]) for d in D}
            rows[u] = row
        out[model] = dict(checks=checks, units=rows)
        lines += [f'### {title}', '',
                  '| unit | method | E0M3 tiles | native WikiText-2 | native C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | final dev KL | selection | setup | peak GPU allocated / reserved | host RSS |',
                  '|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|']
        for u, row in rows.items():
            for key, name in (('mropt', 'MR-OPT'), ('tmopt', 'TM-OPT')):
                x = row[key]
                lines.append(f"| {u} | {name} | {x['e0m3_units']:,} | {x['native_ppl']['wiki']:.4f} | {x['native_ppl']['c4']:.4f} | "
                             f"{fmt(x['native_vs_fourover6']['wiki'])} | {fmt(x['native_vs_fourover6']['c4'])} | {x['final_dev_kl']:.5f} | "
                             f"{x['selection_seconds'] / 60:.1f} min | {x['setup_seconds'] / 60:.1f} min | "
                             f"{x['gpu_peak_allocated_gib']:.1f} / {x['gpu_peak_reserved_gib']:.1f} GiB | {x['host_peak_rss_gib']:.1f} GiB |")
        lines += ['', '| unit | TM-OPT − MR-OPT native ΔWiki | native ΔC4 | fake ΔWiki | fake ΔC4 | shared / only TM-OPT / only MR-OPT | Jaccard |',
                  '|---|---|---|---|---|---|---:|']
        for u, row in rows.items():
            p, o, f = row['tm_minus_mr_native'], row['overlap'], row.get('tm_minus_mr_fake')
            lines.append(f"| {u} | {fmt(p['wiki'])} | {fmt(p['c4'])} | {fmt(f['wiki']) if f else '—'} | {fmt(f['c4']) if f else '—'} | "
                         f"{o['shared']:,} / {o['only_first']:,} / {o['only_second']:,} | {o['jaccard']:.3f} |")
        lines += ['', f'Checks: {json.dumps(checks)}', '']
    (HERE / 'items_models.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    (HERE / 'items_models.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    {'seeds': seeds, 'models': models}[sys.argv[1]]()
