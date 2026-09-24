"""Tables and the SAFE verdict for results/native_decision/REPORT.md (criteria: PROTOCOL.md).

python results/native_decision/analyze_decision.py RUN_ROOT   (writes summary.json and tables.md here)
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAKE_DECIDED = ('DET-FAKE', 'B-256-opt', 'Bprime-256-opt', 'SHADOW', 'B-256-ref')


def load(path):
    return json.loads(Path(path).read_text())


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    assert len(d) == len(a) == len(b)
    n = len(d)
    m = sum(d) / n
    se2 = 2 * math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) / math.sqrt(n)
    return dict(mean=m, two_se=se2, verdict='better' if m + se2 < 0 else ('worse' if m - se2 > 0 else 'inconclusive'))


def fmt(p):
    return f"{p['mean']:+.5f} ± {p['two_se']:.5f} ({p['verdict']})"


def trajectory(r):
    return [dict(round=x['round'], candidates=x['candidates'], accepted=x['accepted'], tiles=x.get('e0m3_units'),
                 dev_kl=x.get('dev_kl'), tries=[(t['size'], t['dev_delta_kl']) for t in x['tries']],
                 seconds=x.get('round_seconds'), score_seconds=x.get('score_seconds')) for x in r['rounds']]


def first_divergence(fa, na):
    for rf, rn in zip(fa['rounds'], na['rounds']):
        if rf['candidates'] != rn['candidates']:
            return dict(round=rf['round'], what='candidate count', fake=rf['candidates'], native=rn['candidates'])
        for i, (tf, tn) in enumerate(zip(rf['tries'], rn['tries'])):
            if tf['size'] != tn['size']:
                return dict(round=rf['round'], try_index=i, what='try size', fake=tf['size'], native=tn['size'])
            accept_f = rf['accepted'] == tf['size'] and i == len(rf['tries']) - 1
            accept_n = rn['accepted'] == tn['size'] and i == len(rn['tries']) - 1
            if accept_f != accept_n:
                return dict(round=rf['round'], try_index=i, size=tf['size'], what='decision',
                            fake=dict(accepts=accept_f, dev_delta_kl=tf['dev_delta_kl']),
                            native=dict(accepts=accept_n, dev_delta_kl=tn['dev_delta_kl']))
        if len(rf['tries']) != len(rn['tries']) or rf['accepted'] != rn['accepted']:
            return dict(round=rf['round'], what='round outcome', fake=rf['accepted'], native=rn['accepted'])
    if len(fa['rounds']) != len(na['rounds']):
        return dict(what='number of rounds', fake=len(fa['rounds']), native=len(na['rounds']))
    return None


def main():
    root = Path(sys.argv[1])
    out = dict(runs={}, evaluation={})
    runs = {'DET-FAKE': load(root / 'det_fake' / 'report.json'), 'DET-NATIVE': load(root / 'det_native' / 'report.json')}
    for name, r in runs.items():
        res = r['resources']
        phases = res['phases']['by_phase']
        dev_phase = 'native_dev_evaluation' if r.get('backends', {}).get('dev') == 'native' else 'dev_evaluation'
        out['runs'][name] = dict(
            status=r['status'], stopped=r.get('stopped'), tiles=r['final_e0m3_units'], map_sha256=r['map_sha256'],
            dev_initial=r['initial_dev']['kl'], dev_final=r['final_dev']['kl'], fake_initial_dev=r.get('fake_initial_dev', {}).get('kl'),
            setup_seconds=res['setup_seconds'], optimization_seconds=res['optimization_seconds'],
            scoring_passes=res['scoring_passes'], development_evaluations=res['development_evaluations'],
            scoring_seconds=phases['scoring']['seconds'],
            dev_eval_seconds_per_try=phases[dev_phase]['seconds'] / max(1, phases[dev_phase]['count']),
            gpu_peak_allocated_gib=res['gpu_peak_allocated_gib'][0], gpu_peak_reserved_gib=res['gpu_peak_reserved_gib'][0],
            host_peak_rss_gib=res['cpu_peak_rss_gib'], trajectory=trajectory(r))
    out['first_divergence'] = first_divergence(runs['DET-FAKE'], runs['DET-NATIVE'])
    f, n = out['runs']['DET-FAKE'], out['runs']['DET-NATIVE']
    out['speedup'] = dict(optimization=f['optimization_seconds'] / n['optimization_seconds'],
                          dev_eval_per_try=f['dev_eval_seconds_per_try'] / n['dev_eval_seconds_per_try'])
    for backend in ('native', 'fake'):
        ev = load(root / f'eval_{backend}' / 'report.json')['evaluations']
        base = ev['FourOverSix']['evaluation']
        det = ev['DET-FAKE']['evaluation']
        table = {}
        for label, e in ev.items():
            x = e['evaluation']
            row = dict(tiles=e.get('e0m3_units'), wiki=x['wiki']['ppl'], c4=x['c4']['ppl'])
            if label not in ('FourOverSix', 'BF16'):
                row['vs_fourover6'] = {d: paired(x[d]['nll'], base[d]['nll']) for d in ('wiki', 'c4')}
            if label not in ('DET-FAKE', 'BF16', 'FourOverSix'):
                row['vs_det_fake'] = {d: paired(x[d]['nll'], det[d]['nll']) for d in ('wiki', 'c4')}
            table[label] = row
        if 'DET-NATIVE' in ev:
            dn = ev['DET-NATIVE']['evaluation']
            table['DET-NATIVE']['vs_fake_decided'] = {m: {d: paired(dn[d]['nll'], ev[m]['evaluation'][d]['nll'])
                                                          for d in ('wiki', 'c4')} for m in FAKE_DECIDED}
        out['evaluation'][backend] = table
    native = out['evaluation']['native']['DET-NATIVE']['vs_fake_decided']
    worse = [(m, d) for m, v in native.items() for d, p in v.items() if p['verdict'] == 'worse']
    out['verdict'] = dict(safe=not worse, significantly_worse_against=worse)
    (HERE / 'summary.json').write_text(json.dumps(out, indent=2) + '\n')
    lines = ['| map | E0M3 tiles | WikiText | C4 | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix | ΔWiki vs DET-FAKE | ΔC4 vs DET-FAKE |',
             '|---|---:|---:|---:|---|---|---|---|']
    for backend in ('native', 'fake'):
        lines.append(f'| **{backend} evaluation** | | | | | | | |')
        for label, row in out['evaluation'][backend].items():
            vf = row.get('vs_fourover6')
            vd = row.get('vs_det_fake')
            lines.append(f"| {label} | {row['tiles'] if row['tiles'] is not None else '—'} | {row['wiki']:.6f} | {row['c4']:.6f} "
                         f"| {fmt(vf['wiki']) if vf else '—'} | {fmt(vf['c4']) if vf else '—'} "
                         f"| {fmt(vd['wiki']) if vd else '—'} | {fmt(vd['c4']) if vd else '—'} |")
    lines += ['', '| DET-NATIVE minus | ΔWiki (native eval) | ΔC4 (native eval) |', '|---|---|---|']
    for m, v in native.items():
        lines.append(f"| {m} | {fmt(v['wiki'])} | {fmt(v['c4'])} |")
    (HERE / 'tables.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(verdict=out['verdict'], first_divergence=out['first_divergence'], speedup=out['speedup']), indent=1))


if __name__ == '__main__':
    main()
