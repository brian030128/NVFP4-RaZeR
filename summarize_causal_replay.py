import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--job', type=int, required=True); args = ap.parse_args()
    root = Path('results/causal_replay'); models = ('llama1b', 'opt350m', 'qwen06b', 'pythia14b', 'olmo1b')
    reports = [json.loads((root/f'model_{args.job}_{m}'/'report.json').read_text()) for m in models]
    assert all(r['status'] == 'complete' and r['exact_input_replay'] for r in reports)
    cells = [(r, d, c) for r in reports for d, c in r['contrasts'].items()]
    retention = []
    for r, d, c in cells:
        old = r['prior_contrasts'][d]['four_over_six']
        assert old['mean_nll']+old['two_se'] < 0
        retention.append(dict(model=r['model'], domain=d, ratio=c['four_over_six']['mean_nll']/old['mean_nll']))
    s = dict(cells=len(cells), gains=sum(c['four_over_six']['ppl_delta'] <= -.01 for _, _, c in cells),
             supported_gains=sum(c['four_over_six']['mean_nll']+c['four_over_six']['two_se'] < 0 for _, _, c in cells),
             supported_harms=sum(c['four_over_six']['mean_nll']-c['four_over_six']['two_se'] > 0 for _, _, c in cells),
             beats_mse=sum(c['weight_mse']['mean_nll'] < 0 for _, _, c in cells),
             beats_c4=sum(c['c4_64']['mean_nll'] < 0 for _, _, c in cells),
             retains_half=sum(c['ratio'] >= .5 for c in retention), retention=retention,
             prefix_independent_models=sum(all(r['suffix_intervention'][p]['equal'] for p in ('four_over_six_row', 'pooled192_row')) for r in reports),
             window_suffix_dependent_models=sum(not r['suffix_intervention']['four_over_six_window']['equal'] for r in reports),
             previous_full_screen_passes=False)
    s['passes_causal_audit'] = s['gains'] >= 12 and s['supported_harms'] == 0 and s['retains_half'] >= 12 and s['prefix_independent_models'] == 5
    lines = ['# Frozen-map causal activation audit', '', '```json', json.dumps(s, indent=2), '```', '',
             'All policies use per-token FP32 activation factors. Maps and inputs are unchanged from332349.', '',
             '| Model / domain | FourOverSix | Pooled192 | ΔPPL | ΔNLL ±2SE | Prior NLL gain retained |',
             '|---|---:|---:|---:|---:|---:|']
    for (r, d, c), retained in zip(cells, retention):
        b = c['four_over_six']; e = r['evaluation']
        lines.append(f'| {r["model"]} / {d} | {e["four_over_six"][d]["ppl"]:.6f} | {e["pooled192"][d]["ppl"]:.6f} | {b["ppl_delta"]:+.6f} | {b["mean_nll"]:+.6f} ±{b["two_se"]:.6f} | {retained["ratio"]:.3f} |')
    lines += ['', '## Fixed-prefix, changed-suffix intervention', '',
              '| Model | Window factor max logit change | Row factor baseline max change | Row factor selected max change |',
              '|---|---:|---:|---:|']
    for r in reports:
        a = r['suffix_intervention']
        lines.append('| '+r['model']+' | '+' | '.join(f'{a[p]["max_logit_difference"]:.6f}' for p in ('four_over_six_window', 'four_over_six_row', 'pooled192_row'))+' |')
    lines += ['', 'This audit does not alter the failed C4-only comparison in the original confirmation. '
              'It tests causal scaling and retention on the same already-inspected inputs. '
              'Per-row factors change the activation representation; no unchanged native-kernel interface or speed claim is made.']
    (root/f'summary_{args.job}.json').write_text(json.dumps(s, indent=2)+'\n')
    (root/f'REPORT_{args.job}.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(s, indent=2))


if __name__ == '__main__': main()
