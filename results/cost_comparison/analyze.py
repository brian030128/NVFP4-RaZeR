"""Tables and statistics for results/cost_comparison/REPORT.md from the run directories.

python results/cost_comparison/analyze.py RUNS_DIR   (writes summary.json and tables.md next to this file)
"""
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GIB = 2 ** 30
PUBLISHED = {
    'A': REPO / 'results/fixed256_paper_eval/job_335428/llama8b_four_over_six/report.json',
    'B-256-ref': REPO / 'results/mixfp4_potential/multiround_256x64_kl/report.json',
    'B-8x64-opt': REPO / 'results/mixfp4_potential/multiround_8x64_kl/report.json',
}
PUBLISHED_NUMBERS = {  # MIXFP4_REPORT.md section 4/5 (H200)
    'B-256-opt': dict(tiles=8393, wiki=6.835411, c4=9.771621, passes=4, dev_evals=33, minutes=15.3, gpu=64.5, cpu=42.9),
    'B-256-ref': dict(tiles=8405, wiki=6.841998, c4=9.774137, passes=5, dev_evals=42, minutes=43.9, gpu=46.1, cpu=41.2),
    'B-8x64-opt': dict(tiles=3654, wiki=6.819751, c4=9.750492, passes=10, dev_evals=133, minutes=135.0, gpu=47.3, cpu=41.3),
    'A': dict(tiles=0, wiki=6.875525, c4=9.823733),
}
UNITS = {'256x64': 425984, '8x64': 13631488}  # selectable tiles over the 224 matrices
BRUNS = {'B-256-opt': 'B_256x64_opt', "B'-256-opt": 'Bprime_256x64_opt', 'B-256-ref': 'B_256x64_ref',
         'B-8x64-opt': 'B_8x64_opt'}


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def paired(a, b):
    """mean and 2 SE of per-window a - b."""
    d = [x - y for x, y in zip(a, b)]
    assert len(d) == len(a) == len(b)
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return dict(mean=m, two_se=2 * sd / math.sqrt(n), n=n,
                verdict='lower' if m + 2 * sd / math.sqrt(n) < 0 else ('higher' if m - 2 * sd / math.sqrt(n) > 0
                                                                        else 'inconclusive'))


def fmt_pair(p, digits=5):
    return f'{p["mean"]:+.{digits}f} ± {p["two_se"]:.{digits}f}'


def map_overlap(a, b):
    ma = torch.load(a, map_location='cpu', weights_only=True)
    mb = torch.load(b, map_location='cpu', weights_only=True)
    assert list(ma) == list(mb)
    both = sum(int((ma[n] & mb[n]).sum()) for n in ma)
    only_a = sum(int((ma[n] & ~mb[n]).sum()) for n in ma)
    only_b = sum(int((~ma[n] & mb[n]).sum()) for n in ma)
    return dict(both=both, only_first=only_a, only_second=only_b,
                jaccard=both / max(1, both + only_a + only_b))


def phase_table(resources):
    # run_multiround.py nests the monitor summary under 'phases'; run_cost_distill.py stores it directly.
    by_phase = resources['by_phase'] if 'by_phase' in resources else resources['phases']['by_phase']
    rows = []
    for name, a in by_phase.items():
        rows.append(dict(phase=name, count=a['count'], seconds=a['seconds'],
                         gpu_alloc=max(a['gpu_peak_allocated_gib']), gpu_res=max(a['gpu_peak_reserved_gib']),
                         host=a['host_peak_rss_gib']))
    return rows


def multiround_row(name, r, a_eval):
    res = r['resources']
    passes = res['scoring_passes']
    dev_evals = res['development_evaluations']
    tiles = r['final_e0m3_units']
    bwd = 1 if r.get('skip_ce_backward') else 2
    scored = sum(1 for x in r['rounds'] if 'score_seconds' in x)
    row = dict(
        arm=name, unit=r['unit'], gpu=res['gpus'][0],
        calibration_seconds=res['setup_seconds'] + res['optimization_seconds'],
        selection_seconds=res['optimization_seconds'], setup_seconds=res['setup_seconds'],
        evaluation_seconds=res['evaluation_seconds'],
        teacher_forward_sequences=192 + 128,
        scoring_forward_sequences=scored * 128, scoring_backward_passes=scored * 128 * bwd,
        development_forward_sequences=dev_evals * 192, scoring_passes=passes, development_evaluations=dev_evals,
        gpu_peak_allocated_gib=res['gpu_peak_allocated_gib'][0], gpu_peak_reserved_gib=res['gpu_peak_reserved_gib'][0],
        host_peak_rss_gib=res['cpu_peak_rss_gib'], host_peak_rss_sampled_gib=res['phases']['host_peak_rss_sampled_gib'],
        trainable='map: ' + f'{tiles:,} E0M3 tiles', optimizer_state_bytes=0, tiles=tiles,
        wiki=r['evaluation']['wiki']['ppl'], c4=r['evaluation']['c4']['ppl'],
        dev_before=dict(ce=r['initial_dev']['ce'], kl=r['initial_dev']['kl']),
        dev_after=dict(ce=r['final_dev']['ce'], kl=r['final_dev']['kl']),
        map_sha256=r.get('map_sha256'), phases=phase_table(res),
        units=UNITS[r['unit']])
    if a_eval is not None:
        row['d_wiki'] = paired(r['evaluation']['wiki']['nll'], a_eval['wiki']['nll'])
        row['d_c4'] = paired(r['evaluation']['c4']['nll'], a_eval['c4']['nll'])
    return row


def main():
    runs = Path(sys.argv[1])
    out = dict()
    a = load(runs / 'A_fourover6/report.json')
    a_eval = a['evaluation'] if a and a.get('status') == 'complete' else None
    pub_a = load(PUBLISHED['A'])['evaluation']
    if a_eval:
        out['A'] = dict(wiki=a_eval['wiki']['ppl'], c4=a_eval['c4']['ppl'],
                        vs_published=dict(wiki=paired(a_eval['wiki']['nll'], pub_a['wiki']['nll']),
                                          c4=paired(a_eval['c4']['nll'], pub_a['c4']['nll'])),
                        dev=dict(ce=a['initial_dev']['ce'], kl=a['initial_dev']['kl']),
                        resources=phase_table(a['resources']), setup_seconds=a['resources']['setup_seconds'],
                        evaluation_seconds=a['resources']['evaluation_seconds'],
                        gpu_peak_allocated_gib=a['resources']['gpu_peak_allocated_gib'][0],
                        gpu_peak_reserved_gib=a['resources']['gpu_peak_reserved_gib'][0],
                        host_peak_rss_gib=a['resources']['cpu_peak_rss_gib'])
    eq = load(runs / 'eval_equivalence/report.json')
    if eq and a_eval:
        out['equivalence'] = dict(
            dev_ce=eq['initial_dev']['ce_nll'] == a['initial_dev']['ce_nll'],
            dev_kl=eq['initial_dev']['kl_values'] == a['initial_dev']['kl_values'],
            wiki=eq['evaluation']['wiki']['nll'] == a_eval['wiki']['nll'],
            c4=eq['evaluation']['c4']['nll'] == a_eval['c4']['nll'])
    out['B'] = {}
    for name, d in BRUNS.items():
        r = load(runs / d / 'report.json')
        if not r or r.get('status') != 'complete':
            continue
        row = multiround_row(name, r, a_eval)
        pub = load(PUBLISHED[name]) if name in PUBLISHED else None
        if pub:
            row['published'] = dict(map_sha256=pub['map_sha256'], tiles=pub['final_e0m3_units'],
                                    map_identical=pub['map_sha256'] == r.get('map_sha256'),
                                    wiki=pub['evaluation']['wiki']['ppl'], c4=pub['evaluation']['c4']['ppl'],
                                    d_wiki=paired(r['evaluation']['wiki']['nll'], pub['evaluation']['wiki']['nll']),
                                    d_c4=paired(r['evaluation']['c4']['nll'], pub['evaluation']['c4']['nll']),
                                    rounds=[(x['round'], x['candidates'], x['accepted'], x['e0m3_units'], len(x['tries']))
                                            for x in pub['rounds']],
                                    initial_dev_identical=pub['initial_dev']['kl_values'] == r['initial_dev']['kl_values'])
        row['rounds'] = [(x['round'], x['candidates'], x['accepted'], x.get('e0m3_units'), len(x['tries']))
                         for x in r['rounds']]
        row['published_numbers'] = PUBLISHED_NUMBERS.get(name.replace("B'", 'B'))
        out['B'][name] = row
    overlaps = {}
    for x, y in (('B_256x64_opt', 'Bprime_256x64_opt'), ('B_256x64_opt', 'B_256x64_ref'),
                 ('Bprime_256x64_opt', 'B_256x64_ref')):
        if (runs / x / 'map.pt').exists() and (runs / y / 'map.pt').exists():
            overlaps[f'{x} vs {y}'] = map_overlap(runs / x / 'map.pt', runs / y / 'map.pt')
    out['map_overlaps'] = overlaps
    out['probes'] = {}
    for p in sorted((runs / 'probes').glob('*/report.json')):
        r = load(p)
        c = r['config']
        entry = dict(arm=c['arm'], optimizer=c['optimizer'], micro_batch=c['micro_batch'],
                     checkpointing=c['checkpointing'], live_teacher=c['live_teacher'], status=r['status'])
        if r['status'] == 'oom':
            entry.update(r['oom'])
        res = r.get('resources')
        if res:
            train = [ph for ph in res['phases'] if ph['name'] == 'training']
            entry['training_gpu_peak_allocated_gib'] = max(ph['gpu_peak_allocated'][0] for ph in train) if train else None
            entry['training_gpu_peak_reserved_gib'] = max(ph['gpu_peak_reserved'][0] for ph in train) if train else None
            entry['run_gpu_peak_allocated_gib'] = res['gpu_peak_allocated_gib'][0]
            entry['host_peak_rss_gib'] = res['cpu_peak_rss_gib']
        entry['steady_step_seconds'] = r.get('steady_step_seconds')
        entry['memory_computed'] = r.get('memory_computed')
        out['probes'][p.parent.name] = entry
    out['grids'] = {}
    for p in sorted(runs.glob('grid_*/report.json')):
        r = load(p)
        c = r['config']
        key = p.parent.name.rsplit('_lr', 1)[0]
        e = dict(run=p.parent.name, lr=c['lr'], status=r['status'], steps=r.get('counts', {}).get('optimizer_steps'),
                 dev_before=dict(ce=r['initial_dev']['ce'], kl=r['initial_dev']['kl']) if r.get('initial_dev') else None,
                 dev_after=dict(ce=r['final_dev']['ce'], kl=r['final_dev']['kl']) if r.get('final_dev') else None,
                 seconds=(r['setup_seconds'] + r['training_seconds'] + r['final_dev_seconds'])
                 if r.get('final_dev_seconds') is not None else None, setup_seconds=r.get('setup_seconds'),
                 training_seconds=r.get('training_seconds'), final_dev_seconds=r.get('final_dev_seconds'),
                 counts=r.get('counts'), memory_computed=r.get('memory_computed'),
                 gpu_peak_allocated_gib=r['resources']['gpu_peak_allocated_gib'][0] if r.get('resources') else None,
                 gpu_peak_reserved_gib=r['resources']['gpu_peak_reserved_gib'][0] if r.get('resources') else None,
                 host_peak_rss_gib=r['resources']['cpu_peak_rss_gib'] if r.get('resources') else None,
                 phases=phase_table(r['resources']) if r.get('resources') else None)
        out['grids'].setdefault(key, []).append(e)
    for key, entries in out['grids'].items():
        ok = [e for e in entries if e['status'] == 'complete' and e['dev_after'] and math.isfinite(e['dev_after']['kl'])]
        if ok:
            best = min(ok, key=lambda e: (e['dev_after']['kl'], e['lr']))
            for e in entries:
                e['selected'] = e is best
            ev = load(runs / (best['run'] + '_eval') / 'report.json')
            if ev and ev.get('status') == 'complete' and a_eval:
                best['evaluation'] = dict(wiki=ev['evaluation']['wiki']['ppl'], c4=ev['evaluation']['c4']['ppl'],
                                          d_wiki=paired(ev['evaluation']['wiki']['nll'], a_eval['wiki']['nll']),
                                          d_c4=paired(ev['evaluation']['c4']['nll'], a_eval['c4']['nll']),
                                          final_dev_matches_training_run=ev.get('final_dev_matches_training_run'),
                                          evaluation_seconds=ev['resources']['by_phase'].get('final_evaluation', {}).get('seconds')
                                          if 'by_phase' in ev['resources'] else None)
                for bname in ('B-256-opt', 'B-8x64-opt'):
                    b = load(runs / BRUNS[bname] / 'report.json')
                    if b and b.get('status') == 'complete':
                        best['evaluation'][f'vs_{bname}'] = dict(
                            wiki=paired(ev['evaluation']['wiki']['nll'], b['evaluation']['wiki']['nll']),
                            c4=paired(ev['evaluation']['c4']['nll'], b['evaluation']['c4']['nll']))
    (HERE / 'summary.json').write_text(json.dumps(out, indent=2) + '\n')
    (HERE / 'tables.md').write_text(tables(out))
    print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in out.items()}, indent=1)[:3000])


def tables(out):
    """The REPORT.md cost-vs-accuracy table (PROTOCOL.md section 6 definitions)."""
    fit_tokens = 128 * 512

    def mtok(sequences):
        return f'{sequences * 512 / 1e6:.2f}M'
    head = ('| arm | unit | GPU | GPU-hours | tokens seen (fwd+bwd / dev fwd) | fwd / bwd passes (sequences) '
            '| peak GPU alloc / reserved (GiB) | peak host RAM (GiB) | trainable params / optimizer state '
            '| WikiText | ΔWiki ± 2SE | C4 | ΔC4 ± 2SE |')
    lines = [head, '|' + '---|' * 13]
    a = out.get('A')
    if a:
        lines.append(f"| A: FourOverSix RTN | — | RTX PRO 6000 | 0 (RTN) | 0 | 0 / 0 | {a['gpu_peak_allocated_gib']:.1f} / "
                     f"{a['gpu_peak_reserved_gib']:.1f} (eval) | {a['host_peak_rss_gib']:.1f} (eval) | 0 / 0 "
                     f"| {a['wiki']:.4f} | — | {a['c4']:.4f} | — |")
    for name, r in out.get('B', {}).items():
        fwd = r['scoring_forward_sequences'] + r['development_forward_sequences'] + r['teacher_forward_sequences']
        lines.append(
            f"| {name} (ours) | {r['unit']} | RTX PRO 6000 | {r['calibration_seconds'] / 3600:.3f} "
            f"| {mtok(r['scoring_forward_sequences'])} / {mtok(r['development_forward_sequences'])} "
            f"| {fwd:,} / {r['scoring_backward_passes']:,} | {r['gpu_peak_allocated_gib']:.1f} / {r['gpu_peak_reserved_gib']:.1f} "
            f"| {r['host_peak_rss_gib']:.1f} | {r['units']:,} binary tiles / 0 "
            f"| {r['wiki']:.4f} | {fmt_pair(r['d_wiki'])} | {r['c4']:.4f} | {fmt_pair(r['d_c4'])} |")
    for key, entries in out.get('grids', {}).items():
        for e in entries:
            if not e.get('selected'):
                continue
            c = e['counts'] or {}
            fwd = c.get('student_forward_sequences', 0) + c.get('recompute_forward_sequences', 0) + \
                c.get('teacher_forward_sequences', 0) + c.get('development_forward_sequences', 0)
            grid_hours = sum((x['seconds'] or 0) for x in entries) / 3600
            ev = e.get('evaluation') or {}
            mem = e['memory_computed'] or {}
            opt = sum((mem.get('optimizer_state_bytes') or {}).values())
            lines.append(
                f"| {key.replace('grid_', '')} (lr {e['lr']:g}) | — | RTX PRO 6000 | {(e['seconds'] or 0) / 3600:.3f} "
                f"(grid {grid_hours:.3f}) | {mtok(c.get('student_forward_sequences', 0))} / "
                f"{mtok(c.get('development_forward_sequences', 0))} | {fwd:,} / {c.get('student_backward_sequences', 0):,} "
                f"| {e['gpu_peak_allocated_gib']:.1f} / {e['gpu_peak_reserved_gib']:.1f} | {e['host_peak_rss_gib']:.1f} "
                f"| {mem.get('trainable_parameters', 0):,} / {opt / 1e9:.2f} GB "
                f"| {ev.get('wiki', float('nan')):.4f} | {fmt_pair(ev['d_wiki']) if ev else '—'} "
                f"| {ev.get('c4', float('nan')):.4f} | {fmt_pair(ev['d_c4']) if ev else '—'} |")
    lines.append('')
    lines.append('| probe | status | steady step (s) | training-phase peak GPU alloc / reserved (GiB) | peak host RAM (GiB) '
                 '| trainable / optimizer state |')
    lines.append('|---|---|---:|---|---:|---|')
    for name, p in out.get('probes', {}).items():
        mem = p.get('memory_computed') or {}
        opt = sum((mem.get('optimizer_state_bytes') or {}).values()) if mem else 0
        status = p['status'] if p['status'] != 'oom' else (f"OOM after {p['step']} step(s) at "
                                                           f"{p['gpu_peak_allocated_gib']:.2f} / {p['gpu_peak_reserved_gib']:.2f}")
        peak = (f"{p['training_gpu_peak_allocated_gib']:.2f} / {p['training_gpu_peak_reserved_gib']:.2f}"
                if p.get('training_gpu_peak_allocated_gib') else '—')
        lines.append(f"| {name} | {status} | {p['steady_step_seconds'] or 0:.2f} | {peak} | {p.get('host_peak_rss_gib', 0):.1f} "
                     f"| {mem.get('trainable_parameters', 0):,} / {opt / 1e9:.2f} GB |")
    lines.append('')
    lines.append('| grid run | lr | steps | dev KL before → after | dev CE before → after | GPU-hours | selected |')
    lines.append('|---|---:|---:|---|---|---:|---|')
    for key, entries in out.get('grids', {}).items():
        for e in sorted(entries, key=lambda x: x['lr']):
            if not e['dev_after'] or not e['dev_before']:
                continue
            lines.append(f"| {e['run']} | {e['lr']:g} | {e['steps']} | {e['dev_before']['kl']:.6f} → {e['dev_after']['kl']:.6f} "
                         f"| {e['dev_before']['ce']:.6f} → {e['dev_after']['ce']:.6f} | {(e['seconds'] or 0) / 3600:.3f} "
                         f"| {'**yes**' if e.get('selected') else ''} |")
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
