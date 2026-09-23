"""Paired per-window NLL contrasts for the map-conditioned GPTQ arms."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '/work/u4320956/mapped_gptq/llama8b_20260923')
ARMS = ['rtn_four_over_six', 'rtn_raw256', 'gptq_four_over_six', 'gptq_raw256', 'gptq_fine8x64']
CONTRASTS = [('gptq_four_over_six', 'rtn_four_over_six'), ('rtn_raw256', 'rtn_four_over_six'),
             ('gptq_raw256', 'gptq_four_over_six'), ('gptq_fine8x64', 'gptq_four_over_six'),
             ('gptq_raw256', 'gptq_fine8x64'), ('gptq_raw256', 'rtn_raw256')]


def pair(a, b):
    x = [i - j for i, j in zip(a, b)]
    n = len(x); m = sum(x) / n
    se = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1) / n)
    return dict(delta_nll=m, two_se=2 * se, t=m / se)


def main():
    r = {k: json.loads((ROOT / k / 'report.json').read_text()) for k in ARMS if (ROOT / k / 'report.json').exists()}
    out = dict(ppl={k: {d: v['evaluation'][d]['ppl'] for d in ('wiki', 'c4')} for k, v in r.items()},
               tiles={k: v['elected_tiles'] for k, v in r.items()},
               rtn_limit_agreement={k: v['rtn_limit_agreement'] for k, v in r.items() if 'rtn_limit_agreement' in v},
               contrasts={})
    for a, b in CONTRASTS:
        if a in r and b in r:
            out['contrasts'][f'{a} - {b}'] = {d: pair(r[a]['evaluation'][d]['nll'], r[b]['evaluation'][d]['nll'])
                                             for d in ('wiki', 'c4')}
    (ROOT / 'summary.json').write_text(json.dumps(out, indent=2) + '\n')
    for k, v in out['ppl'].items():
        print(f'{k:22s} tiles={out["tiles"][k]:5d} wiki={v["wiki"]:.6f} c4={v["c4"]:.6f}')
    for k, v in out['contrasts'].items():
        print(k, ' | '.join(f'{d} {x["delta_nll"]:+.6f} ± {x["two_se"]:.6f} (t={x["t"]:+.2f})' for d, x in v.items()))
    print('rtn limit agreement', out['rtn_limit_agreement'])


if __name__ == '__main__':
    main()
