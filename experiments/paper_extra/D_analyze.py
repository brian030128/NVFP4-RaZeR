#!/usr/bin/env python3
"""Experiment D tables: results/paper_extra/D/{D.csv, D.md, D.json} from the D_decode.py records. CPU, seconds.

    PAPER_PYTHON experiments/paper_extra/D_analyze.py [--models llama8b,mistral7b,phi4] [--out D_OUT] [--dest ...]

- Tokens per second, the median over rounds, per (model, policy, batch x prompt).
- The change against FourOverSix in %, paired within rounds (the median over rounds of the per-round ratio − 1):
  every policy against FourOverSix on weights-on-A; Ours 8x64 also against FourOverSix on weights-on-B (the same
  placement).
- The CTA widths decode used (T = batch): from the tile table for the width-selecting sets.
- The token checks per setting: graph = eager StaticCache (registered), and the DynamicCache agreement (recorded).
CSV: model, policy, batch, prompt, tokens_per_s, ms_per_token, change_vs_fo6_pct, change_vs_fo6_wB_pct.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'paper'))
import paper_common as P  # noqa: E402
from D_decode import OUT, POLICIES, SETTINGS  # noqa: E402

TITLES = {'bf16': 'BF16', 'nvfp4': 'NVFP4', 'fo6': 'FourOverSix', 'ours-16x64': 'FlipQuant (ours) 16x64',
          'ours-8x64': 'FlipQuant (ours) 8x64',
          'nvfp4-wB': 'NVFP4 (wB)', 'fo6-wB': 'FourOverSix (wB)'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default='llama8b,mistral7b,phi4')
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--dest', type=Path, default=P.REPO / 'results' / 'paper_extra' / 'D')
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    csv = ['model,policy,batch,prompt,tokens_per_s,ms_per_token,change_vs_fo6_pct,change_vs_fo6_wB_pct']
    md, data = ['# Experiment D: decode latency (CUDA graph, StaticCache, 64 generated tokens)\n'], {}
    settings = SETTINGS.split(',')
    for model in args.models.split(','):
        rec = {}
        for pol in POLICIES:
            runs = {}
            for f in sorted((args.out / 'decode' / model / pol).glob('round*.json')):
                r = json.loads(f.read_text())
                if r.get('status') == 'complete':
                    runs[r['round']] = r
            if runs:
                rec[pol] = runs
        if 'fo6' not in rec:
            continue
        d = data[model] = {}
        md += [f'## {P.TITLES[model]}\n', 'Tokens per second (median over rounds); change vs FourOverSix in % (paired within '
               'rounds; for FlipQuant (ours) 8x64 also vs FourOverSix on wB).\n',
               '| policy | ' + ' | '.join(settings) + ' | decode widths |', '|---|' + '---:|' * len(settings) + '---|']
        for pol, runs in rec.items():
            cells = []
            for s in settings:
                tps = {k: r['decode'][s]['tokens_per_s'] for k, r in runs.items() if 'tokens_per_s' in r['decode'].get(s, {})}
                if not tps:
                    cells.append('—')
                    continue
                med = statistics.median(tps.values())
                ms = statistics.median(r['decode'][s]['ms_per_token'] for r in runs.values() if 'ms_per_token' in r['decode'].get(s, {}))

                def change(ref):
                    if ref not in rec or pol == ref:
                        return None
                    per = [100 * (tps[k] / rec[ref][k]['decode'][s]['tokens_per_s'] - 1) for k in tps
                           if k in rec[ref] and 'tokens_per_s' in rec[ref][k]['decode'].get(s, {})]
                    return statistics.median(per) if per else None
                c, cw = change('fo6'), (change('fo6-wB') if pol == 'ours-8x64' else None)
                b_, p_ = s.split('x')
                d.setdefault(pol, {})[s] = dict(tokens_per_s=med, ms_per_token=ms, change_vs_fo6_pct=c, change_vs_fo6_wB_pct=cw,
                                                rounds=len(tps),
                                                token_match_dynamic=[r['decode'][s].get('token_match_dynamic_eager') for r in runs.values()])
                csv.append(f"{model},{pol},{b_},{p_},{med:.3f},{ms:.4f},{'' if c is None else f'{c:.3f}'},"
                           f"{'' if cw is None else f'{cw:.3f}'}")
                cells.append(f'{med:.1f}' + ('' if c is None else f' ({c:+.1f} %)') + ('' if cw is None else f' [wB {cw:+.1f} %]'))
            first = next(iter(runs.values()))['decode']
            widths = '; '.join(f"b{b_}: {first[s_]['decode_widths']}" for b_, s_ in
                               {s_.split('x')[0]: s_ for s_ in settings if 'decode_widths' in first.get(s_, {})}.items()) or '—'
            md.append(f'| {TITLES[pol]} | ' + ' | '.join(cells) + f' | {widths} |')
        checks = [r['decode'][s].get('tokens_equal_static_eager') is True for runs in rec.values() for r in runs.values()
                  for s in settings if s in r['decode']]
        assert all(checks), f'{model}: a record failed the registered token check (graph vs eager StaticCache)'
        dyn = {pol: min((min(v for v in x['token_match_dynamic'] if v is not None) for x in d[pol].values()
                         if any(v is not None for v in x['token_match_dynamic'])), default=None) for pol in d}
        md.append(f'\nGraph = eager StaticCache greedy tokens (33 per setting): passed in all {len(checks)} (policy, round, '
                  f'setting) records (registered check). Lowest agreement with an eager DynamicCache decode, per policy '
                  f'(recorded, not a check): {dyn}.\n')
    (args.dest / 'D.csv').write_text('\n'.join(csv) + '\n')
    (args.dest / 'D.md').write_text('\n'.join(md) + '\n')
    (args.dest / 'D.json').write_text(json.dumps(data, indent=1) + '\n')
    print('wrote', args.dest / 'D.csv', args.dest / 'D.md')


if __name__ == '__main__':
    main()
