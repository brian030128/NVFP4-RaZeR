"""Task 1 tables (results/nodev_cost/PROTOCOL.md): calibration cost without the development set.

Per run: the included phases' wall times and their sum (the calibration cost), the process's total, and peak GPU
allocated / reserved and peak host RSS over the included phases. TM-OPT+TC runs are also checked:
- deterministic maps: sha256 equal to the committed map;
- non-deterministic maps: tile count and Jaccard overlap with the deterministic map.
The with-dev costs of results/tm_opt/final_cost.json are set alongside. Host memory is compared like for like: the
with-dev records hold ru_maxrss (the process's lifetime peak), so the comparison uses ru_maxrss on both sides.

python results/nodev_cost/analyze.py [RUNS_DIR]   -> cost.{json,md}

The maps are not committed (the deterministic ones are bitwise equal to the committed runs'); they are read from MAPS for
the hash cross-check and the Jaccard overlaps.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'runs'
TM = Path('/home/dev/n16k64_campaign/tm_opt/runs')
MAPS = Path('/home/dev/n16k64_campaign/nodev_cost/runs')
INCLUDED = ('model_load', 'data_load', 'teacher_precompute', 'candidate_packing', 'preparation', 'training', 'write_output',
            'save_state')
TITLES = {'llama8b': 'Llama-3.1-8B', 'mistral7b': 'Mistral-7B-v0.3', 'phi4': 'Phi-4', 'qwen27b': 'Qwen3.8-27B'}


def load(p):
    return json.loads(Path(p).read_text())


def phases(report):
    # run_cost_distill.py stores the monitor summary as 'resources'; run_train_map.py under resources['phases']
    res = report['resources']
    return dict(res['by_phase'] if 'by_phase' in res else res['phases']['by_phase'])


def cost(report):
    by = phases(report)
    unknown = [k for k in by if k not in INCLUDED]
    assert not unknown, f'phases outside the calibration-cost definition ran: {unknown}'
    inc = {k: by[k] for k in INCLUDED if k in by}
    return dict(phase_seconds={k: v['seconds'] for k, v in inc.items()}, calibration_seconds=sum(v['seconds'] for v in inc.values()),
                gpu_peak_allocated_gib=max(v['gpu_peak_allocated_gib'][0] for v in inc.values()),
                gpu_peak_reserved_gib=max(v['gpu_peak_reserved_gib'][0] for v in inc.values()),
                host_peak_rss_gib=max(v['host_peak_rss_gib'] for v in inc.values()),
                host_ru_maxrss_gib=report['resources'].get('cpu_peak_rss_gib'),
                process_seconds=report['resources'].get('total_seconds') or report.get('total_seconds'))


def committed_map(model, unit):
    return TM / (f'q_tc_qwen27b_{unit}' if model == 'qwen27b' else f'tc_{model}_{unit}')


def jaccard(a, b):
    shared = sum(int((a[n] & b[n]).sum()) for n in a)
    union = sum(int((a[n] | b[n]).sum()) for n in a)
    return shared / union if union else 1.0


def main():
    out, rows = {}, []
    for d in sorted(RUNS.iterdir()):
        rp = d / 'report.json'
        if not d.is_dir() or not rp.exists():
            continue
        r = load(rp)
        if r.get('status') != 'complete':
            out[d.name] = dict(status=r.get('status'))
            continue
        c = cost(r)
        assert abs(c['process_seconds'] - c['calibration_seconds']) < 1.0, (d.name, c['process_seconds'], c['calibration_seconds'])
        name = d.name
        if name.startswith('nd_tc_'):
            rest = name[len('nd_tc_'):]
            deterministic = not rest.endswith('_detoff')
            model, unit = rest.replace('_detoff', '').rsplit('_', 1)
            method = 'TM-OPT+TC'
            ref = committed_map(model, unit)
            new_sha = r['map_sha256']       # the trainer's hash of the map.pt it wrote
            if (MAPS / name / 'map.pt').exists():
                assert hashlib.sha256((MAPS / name / 'map.pt').read_bytes()).hexdigest() == new_sha, name
            c.update(e0m3_tiles=r['final_e0m3_units'], epochs=len(r['epochs']),
                     epoch_seconds=sum(e['epoch_seconds'] for e in r['epochs']) / len(r['epochs']),
                     micro_batch=r['args']['batch'], accum=r['args']['accum'])
            if deterministic:
                c['map_equals_committed'] = new_sha == load(ref / 'report.json')['map_sha256']
            else:
                det = RUNS / f'nd_tc_{model}_{unit}'
                a = torch.load(MAPS / name / 'map.pt', weights_only=True)
                b = torch.load(MAPS / det.name / 'map.pt', weights_only=True)
                c['jaccard_with_deterministic'] = jaccard(a, b)
                c['deterministic_e0m3_tiles'] = sum(int(v.sum()) for v in b.values())
        else:
            arm = r['config']['arm']
            model, unit, method = 'llama8b', '—', {'qat': 'QAT C1', 'scale': 'scale-only D1'}[arm]
            deterministic = bool(r['config'].get('deterministic'))
            c.update(lr=r['config']['lr'], optimizer_steps=r['counts']['optimizer_steps'], state_sha256=r.get('state_sha256'))
        c.update(model=model, unit=unit, method=method, deterministic=deterministic)
        out[name] = c
        rows.append(c)
    order = {'TM-OPT+TC': 0, 'QAT C1': 1, 'scale-only D1': 2}
    rows.sort(key=lambda c: (list(TITLES).index(c['model']), order[c['method']], not c['deterministic'], c['unit']))
    md = ['# Calibration cost without the development set (generated by analyze.py)', '',
          'Included phases (PROTOCOL.md): model load, fit data, fit teacher, preparation (TM: candidate packing), training, '
          'writing the output. Memory: maximum over the included phases; host RSS sampled every 0.1 s, and ru_maxrss '
          '(the process\'s lifetime peak).', '',
          '| model | unit | method | deterministic | model / data / teacher / prep. / training / write (s) | calibration | '
          'peak GPU allocated / reserved | peak host RSS sampled / ru_maxrss | check |', '|---|---|---|---|---|---:|---:|---:|---|']
    for c in rows:
        ps = c['phase_seconds']
        prep = ps.get('candidate_packing', ps.get('preparation', 0.0))
        write = ps.get('write_output', ps.get('save_state', 0.0))
        parts = ' / '.join(f'{x:.0f}' for x in (ps.get('model_load', 0), ps.get('data_load', 0), ps.get('teacher_precompute', 0), prep,
                                                   ps.get('training', 0), write))
        if 'map_equals_committed' in c:
            check = 'map = committed' if c['map_equals_committed'] else '**map differs**'
        elif 'jaccard_with_deterministic' in c:
            check = f"{c['e0m3_tiles']:,} tiles (det {c['deterministic_e0m3_tiles']:,}), Jaccard {c['jaccard_with_deterministic']:.3f}"
        else:
            check = f"lr {c['lr']:g}, {c['optimizer_steps']} steps"
        md.append(f"| {TITLES[c['model']]} | {c['unit']} | {c['method']} | {'yes' if c['deterministic'] else 'no'} | {parts} | "
                  f"{c['calibration_seconds'] / 60:.1f} min | {c['gpu_peak_allocated_gib']:.1f} / {c['gpu_peak_reserved_gib']:.1f} GiB | "
                  f"{c['host_peak_rss_gib']:.1f} / {c['host_ru_maxrss_gib']:.1f} GiB | {check} |")
    # with-dev comparison (results/tm_opt/final_cost.json; QAT/D from results/cost_comparison)
    prior = json.loads((HERE.parent / 'tm_opt' / 'final_cost.json').read_text())
    md += ['', 'Compared with the earlier with-dev records (TM: setup + selection; QAT/D: every phase of the recorded run). '
           'Host memory: ru_maxrss on both sides (the with-dev records hold only ru_maxrss).', '',
           '| model | unit | method | with dev | without dev | change | host ru_maxrss with / without dev |',
           '|---|---|---|---:|---:|---:|---|']
    for c in rows:
        if c['method'] == 'TM-OPT+TC' and c['deterministic']:
            old = prior.get(f"{c['model']} {c['unit']} TM-OPT+TC")
            if old:
                before = old['setup_seconds'] + old['selection_seconds']
                md.append(f"| {TITLES[c['model']]} | {c['unit']} | TM-OPT+TC | {before / 60:.1f} min | {c['calibration_seconds'] / 60:.1f} min | "
                          f"{100 * (c['calibration_seconds'] / before - 1):+.0f} % | {old['host_peak_rss_gib']:.1f} / {c['host_ru_maxrss_gib']:.1f} GiB |")
    for key, name in (('C1', 'QAT C1'), ('D1', 'scale-only D1')):
        rp = Path(f"results/cost_comparison/runs/{'grid_C1_lr1e-6' if key == 'C1' else 'grid_D1_lr1e-3'}/report.json")
        c = next((x for x in rows if x['method'] == name and not x['deterministic']), None)
        if rp.exists() and c:
            r = load(rp)
            before = sum(v['seconds'] for v in phases(r).values())
            md.append(f"| Llama-3.1-8B | — | {name} | {before / 60:.1f} min | {c['calibration_seconds'] / 60:.1f} min | "
                      f"{100 * (c['calibration_seconds'] / before - 1):+.0f} % | {r['resources']['cpu_peak_rss_gib']:.1f} / {c['host_ru_maxrss_gib']:.1f} GiB |")
    (HERE / 'cost.json').write_text(json.dumps(out, indent=1) + '\n')
    (HERE / 'cost.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
