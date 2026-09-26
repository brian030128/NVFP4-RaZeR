"""Aggregate the 12-shard native GB200 GSM8K runs of Qwen3.8-27B and compare policies per problem."""
import json, math, sys
from pathlib import Path

ROOT = Path('/work/u4320956/mixfp4_potential/qwen_native_gsm8k')
POLICIES = ('bf16', 'four_over_six', 'mix256x64')
N_SHARDS, N_DOCS = 12, 1319
FILTERS = ('strict_match', 'flexible_extract')

scores = {p: {f: {} for f in FILTERS} for p in POLICIES}
info = {p: [] for p in POLICIES}
for p in POLICIES:
    for i in range(N_SHARDS):
        r = json.loads((ROOT / f'{p}_s{i}' / 'report.json').read_text())
        assert r['status'] == 'complete' and r['shard'] == f'{i}/{N_SHARDS}' and r['policy'] == p, (p, i)
        assert r['hflm'].get('enable_thinking') is False
        s = r['samples']['gsm8k_llama']
        half = len(s) // 2
        expect = list(range(i, N_DOCS, N_SHARDS))
        # Records are strict-match for every document, then flexible-extract for every document.
        for f, part in zip(FILTERS, (s[:half], s[half:])):
            assert sorted(x['doc_id'] for x in part) == expect, (p, i, f)
            got = sum(x['exact_match'] for x in part) / len(part)
            assert abs(got - r['metrics']['gsm8k_llama'][f'exact_match,{f}']) < 1e-12, (p, i, f)
            for x in part:
                scores[p][f][x['doc_id']] = x['exact_match']
        info[p].append(dict(job=r['job'], eval_s=r['eval_seconds'], pack_s=r.get('pack_seconds'),
                            padded=r.get('row_padded_matrices'), tiles=r.get('e0m3_tiles'), bitwise=r.get('packed_bitwise')))
    for f in FILTERS:
        assert sorted(scores[p][f]) == list(range(N_DOCS)), (p, f)

out = {'accuracy': {}, 'paired': {}, 'runs': {}}
for p in POLICIES:
    out['accuracy'][p] = {f: sum(scores[p][f].values()) / N_DOCS for f in FILTERS}
    out['runs'][p] = dict(gpu_hours=sum(x['eval_s'] for x in info[p]) / 3600,
                          e0m3_tiles=info[p][0]['tiles'], row_padded=info[p][0]['padded'],
                          all_bitwise=all(x['bitwise'] for x in info[p]))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


for a, b_ in (('mix256x64', 'four_over_six'), ('mix256x64', 'bf16'), ('four_over_six', 'bf16')):
    for f in FILTERS:
        d = [scores[a][f][i] - scores[b_][f][i] for i in range(N_DOCS)]
        mean = sum(d) / N_DOCS
        sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (N_DOCS - 1))
        wins, losses = sum(x > 0 for x in d), sum(x < 0 for x in d)
        out['paired'][f'{a} - {b_} ({f})'] = dict(diff=mean, se=sd / math.sqrt(N_DOCS), a_only=wins, b_only=losses,
                                                   mcnemar_p=mcnemar(wins, losses))
json.dump(out, sys.stdout, indent=1)
