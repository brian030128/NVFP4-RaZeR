"""Stage 3 of PROTOCOL.md: is the model's batched forward bitwise identical to one document at a time?

python results/multiround_models/compare_batching.py BATCHED_RUN SINGLE_RUN > batching.json

Both runs are run_multiround.py --stop-after-scoring --dump-round0-scores --record-dev-values
--dev-backend native at the same unit, differing only in --eval-batch / --score-batch. Compared
bitwise: the fake and the native initial development per-document CE and KL, and round 0's
per-unit score mean and SE of every module. Prints the decision (the batch sizes calibration
uses) and exits 0 either way; a failed comparison is a result, not an error.
"""
import json
import sys
from pathlib import Path

import torch


def same(a, b):
    view = {torch.float64: torch.int64, torch.float32: torch.int32}
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(view[a.dtype]), b.view(view[b.dtype]))


def main():
    batched, single = Path(sys.argv[1]), Path(sys.argv[2])
    ra, rb = (json.loads((d / 'report.json').read_text()) for d in (batched, single))
    assert ra['status'] == rb['status'] == 'scores_complete'
    va, vb = (torch.load(d / 'dev_values.pt', weights_only=True) for d in (batched, single))
    sa, sb = (torch.load(d / 'round0_scores.pt', weights_only=True) for d in (batched, single))
    out = dict(batched=dict(run=str(batched), eval_batch=ra['eval_batch'], score_batch=ra['score_batch']),
               single=dict(run=str(single), eval_batch=rb['eval_batch'], score_batch=rb['score_batch']))
    checks = {}
    for key in ('fake_initial', 'initial'):
        for loss in ('ce', 'kl'):
            a, b = va[key][loss], vb[key][loss]
            checks[f'{key}_{loss}'] = same(a, b)
            out[f'{key}_{loss}_max_abs_difference'] = float((a - b).abs().max())
    differing = [n for n in sa if not all(same(sa[n][k], sb[n][k]) for k in ('ce_mean', 'ce_se', 'kl_mean', 'kl_se'))]
    checks['round0_scores'] = not differing
    out['round0_modules_differing'] = len(differing)
    out['round0_kl_mean_max_abs_difference'] = max(float((sa[n]['kl_mean'] - sb[n]['kl_mean']).abs().max()) for n in sa)
    out['checks'] = checks
    out['identical'] = all(checks.values())
    out['decision'] = (dict(eval_batch=ra['eval_batch'], score_batch=ra['score_batch']) if out['identical']
                       else dict(eval_batch=1, score_batch=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
