"""Tables for the Qwen3-4B one-shot k=3 map evaluation (PROTOCOL.md here).

python results/multiround_models/qwen4b/oneshot_k3_eval/analyze_oneshot.py   (writes summary.json and tables.md here)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'native_decision'))
from analyze_decision import fmt, load, paired  # noqa: E402

RUNS = Path('/home/dev/n16k64_campaign/multimodel/runs')
ONESHOT = ('N16K64-n8-k3', 'N16K64-n16-k3')
DET = ('DET-NATIVE-8x64', 'DET-NATIVE-256x64', 'DET-FAKE-8x64', 'DET-FAKE-256x64')
# The campaign's own records, per-token activation scales (convention (c)): a different protocol.
CAMPAIGN = {'N16K64-n8-k3': dict(fake=(11.7066, 15.7092), native=(11.6925, 15.6958)),
            'N16K64-n16-k3': dict(fake=(12.1095, 15.9714), native=(12.0888, 15.9623), native_after_fix=(12.0933, 15.9515)),
            'FourOverSix': dict(fake=(14.2183, 17.3175), native=(14.2106, 17.3062))}


def main():
    out = dict(evaluation={}, fourover6_bitwise_equal_to_part_c={})
    lines = ['| backend | map | E0M3 8x64 tiles | WikiText-2 PPL | C4 PPL | ΔWiki vs FourOverSix | ΔC4 vs FourOverSix |',
             '|---|---|---:|---:|---:|---|---|']
    pair_lines = ['| backend | one-shot map | minus | ΔWiki | ΔC4 |', '|---|---|---|---|---|']
    for backend in ('native', 'fake'):
        ev = load(RUNS / 'qwen4b_oneshot' / f'eval_{backend}' / 'report.json')['evaluations']
        part_c = load(RUNS / 'qwen4b' / f'eval_{backend}' / 'report.json')['evaluations']['FourOverSix']['evaluation']
        out['fourover6_bitwise_equal_to_part_c'][backend] = all(
            ev['FourOverSix']['evaluation'][d]['nll'] == part_c[d]['nll'] for d in ('wiki', 'c4'))
        base = ev['FourOverSix']['evaluation']
        table = {}
        for label, e in ev.items():
            x = e['evaluation']
            row = dict(tiles=e.get('e0m3_units'), wiki=x['wiki']['ppl'], c4=x['c4']['ppl'],
                       map_mismatches=e.get('native_map_mismatches', e.get('lean_map_mismatches')),
                       activation_checks=e.get('native_activation_checks'), converted_from=e.get('converted_from'))
            if label != 'FourOverSix':
                row['vs_fourover6'] = {d: paired(x[d]['nll'], base[d]['nll']) for d in ('wiki', 'c4')}
            if label in ONESHOT:
                row['vs_det'] = {det: {d: paired(x[d]['nll'], ev[det]['evaluation'][d]['nll']) for d in ('wiki', 'c4')}
                                 for det in DET}
            table[label] = row
            v = row.get('vs_fourover6')
            lines.append(f"| {backend} | {label} | {row['tiles']:,} | {row['wiki']:.4f} | {row['c4']:.4f} | "
                         f"{fmt(v['wiki']) if v else '—'} | {fmt(v['c4']) if v else '—'} |")
        for label in ONESHOT:
            for det, p in table[label]['vs_det'].items():
                pair_lines.append(f"| {backend} | {label} | {det} | {fmt(p['wiki'])} | {fmt(p['c4'])} |")
        out['evaluation'][backend] = table
    ref = ['| map (campaign record, per-token activations: a different protocol) | fake WikiText-2 / C4 | native WikiText-2 / C4 |',
           '|---|---|---|']
    for label, r in CAMPAIGN.items():
        ref.append(f"| {label} | {r['fake'][0]:.4f} / {r['fake'][1]:.4f} | {r['native'][0]:.4f} / {r['native'][1]:.4f}"
                   + (f" (after the weights-on-A fix: {r['native_after_fix'][0]:.4f} / {r['native_after_fix'][1]:.4f})"
                      if 'native_after_fix' in r else '') + ' |')
    out['campaign_records_per_token'] = CAMPAIGN
    (HERE / 'summary.json').write_text(json.dumps(out, indent=1) + '\n')
    (HERE / 'tables.md').write_text('\n'.join(lines + [''] + pair_lines + [''] + ref) + '\n')
    print('\n'.join(lines + [''] + pair_lines + [''] + ref))
    print(json.dumps(out['fourover6_bitwise_equal_to_part_c']))


if __name__ == '__main__':
    main()
