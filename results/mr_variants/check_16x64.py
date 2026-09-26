"""Pre-run checks of the 16x64 addendum (ADDENDUM_16x64.md). Each subcommand adds its result to checks_16x64.json
here and exits 1 if its registered criterion fails (the queue then stops before any calibration run).

python results/mr_variants/check_16x64.py unit JSON                  # B1 unit tests: normwise rel. error <= 1e-6
python results/mr_variants/check_16x64.py round0 NAME LEGACY_DIR B1_DIR   # round-0 candidate sets, legacy vs B1
python results/mr_variants/check_16x64.py regression RUN_DIR         # the 8x64 code path is unchanged
"""
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('CHECKS_16X64_OUT', HERE / 'checks_16x64.json'))
TOLERANCE = 1e-6            # PROTOCOL_PHASE2.md: normwise relative error of g, mu and SE
NOISE = 1e-5                # a disagreeing tile's |mu + 2 SE| / max|mu| of its matrix; above it the flip is not noise
PHASE2_8X64 = HERE.parent / 'speedups' / 'runs_phase2' / 'llama_det_native_8x64_b1' / 'report.json'
MROPT_8X64 = HERE / 'runs' / 'llama8b' / 'mropt_8x64' / 'report.json'


def record(key, value, ok):
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out[key] = dict(value, passed=ok)
    OUT.write_text(json.dumps(out, indent=1) + '\n')
    print(key, 'PASS' if ok else 'FAIL', json.dumps(value))
    sys.exit(0 if ok else 1)


def unit(path):
    result = json.loads(Path(path).read_text())
    summary = result['summary']
    worst = {k: max(r[k]['normwise_rel'] for r in result['layers'].values()) for k in ('g', 'mu', 'se')}
    record('unit_tests', dict(file=str(path), units=result['units'], models=result['models'], cases=len(result['layers']),
                              max_normwise_rel=worst, max_elementwise_rel=summary['max_elementwise_rel'],
                              candidate_disagreements=summary['candidate_disagreements'],
                              candidate_tiles=summary['candidate_tiles'], time_ms=summary['time_ms']),
           all(v <= TOLERANCE for v in worst.values()))


def round0(name, off, on):
    a, b = (torch.load(Path(p) / 'round0_scores.pt', weights_only=True) for p in (off, on))
    ra, rb = (json.loads((Path(p) / 'report.json').read_text()) for p in (off, on))
    assert not ra['speedups']['tile_score_kernel'] and rb['speedups']['tile_score_kernel'], name
    assert ra['unit'] == rb['unit'] == '16x64', name
    out = dict(tiles=0, legacy=0, b1=0, both=0, legacy_only=0, b1_only=0, mu_max_rel=0.0, se_max_rel=0.0,
               disagreeing_max_bound_over_se=0.0, disagreeing_max_bound_over_max_mu=0.0,
               native_verification=[r['native']['verification'] for r in (ra, rb)])
    for n in a:
        la, lb = a[n]['kl_mean'] + 2 * a[n]['kl_se'], b[n]['kl_mean'] + 2 * b[n]['kl_se']
        ca, cb = la < 0, lb < 0
        out['tiles'] += ca.numel(); out['legacy'] += int(ca.sum()); out['b1'] += int(cb.sum())
        out['both'] += int((ca & cb).sum()); out['legacy_only'] += int((ca & ~cb).sum()); out['b1_only'] += int((~ca & cb).sum())
        scale = a[n]['kl_mean'].abs().max().clamp_min(1e-300)
        out['mu_max_rel'] = max(out['mu_max_rel'], float((b[n]['kl_mean'] - a[n]['kl_mean']).abs().max() / scale))
        out['se_max_rel'] = max(out['se_max_rel'], float((b[n]['kl_se'] - a[n]['kl_se']).abs().max()
                                                         / a[n]['kl_se'].abs().max().clamp_min(1e-300)))
        d = ca != cb
        if d.any():
            out['disagreeing_max_bound_over_se'] = max(out['disagreeing_max_bound_over_se'],
                                                       float((la[d].abs() / a[n]['kl_se'][d]).max()))
            out['disagreeing_max_bound_over_max_mu'] = max(out['disagreeing_max_bound_over_max_mu'],
                                                           float(la[d].abs().max() / scale))
    record(f'round0 {name}', out, out['disagreeing_max_bound_over_max_mu'] <= NOISE)


def regression(run):
    r = json.loads((Path(run) / 'report.json').read_text())
    p2, mr = (json.loads(p.read_text()) for p in (PHASE2_8X64, MROPT_8X64))
    same_dev = all(r[key][v] == mr[key][v] for key in ('initial_dev', 'fake_initial_dev') for v in ('ce_nll', 'kl_values'))
    out = dict(unit=r['unit'], round0_scores_sha256=r['round0_scores_sha256'],
               phase2_round0_scores_sha256=p2['round0_scores_sha256'],
               equal_round0_scores=r['round0_scores_sha256'] == p2['round0_scores_sha256'],
               equal_initial_dev_values_to_mropt_8x64=same_dev, native_verification=r['native']['verification'])
    record('regression llama8b 8x64', out, r['unit'] == '8x64' and out['equal_round0_scores'] and same_dev)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'unit':
        unit(sys.argv[2])
    elif cmd == 'round0':
        round0(*sys.argv[2:5])
    elif cmd == 'regression':
        regression(sys.argv[2])
    else:
        raise SystemExit(__doc__)
