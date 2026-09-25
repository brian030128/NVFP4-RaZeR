"""Part B criteria (PROTOCOL.md): one configuration's runs against each other, bitwise.

python results/lean_memory/compare_runs.py COMMITTED_REPORT COMMITTED_MAP RUN_DIR [LEGACY_DIR]

Always: RUN_DIR (a legacy rerun or a lean run) against the committed run, on every field the
committed record holds -- per-round candidate count, every try's step size, predicted and measured
development change, every acceptance decision, the per-round tile count and development loss, the
initial (and fake initial) and final per-document development CE and KL, the final map, and the
final per-window WikiText-2 / C4 NLL.
With LEGACY_DIR (RUN_DIR is then the lean run): additionally every try's per-document development
CE and KL (dev_values.pt) and the round-0 per-unit CE/KL score mean and SE (round0_scores.pt).
Floats are compared as bit patterns (IEEE-754 binary64), so -0.0 vs +0.0 counts as a mismatch.
Writes a JSON summary to stdout; exit status 1 on any mismatch.
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

import torch


def bits(x):
    return struct.pack('<d', x) if isinstance(x, float) else x


def same_value(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    return type(a) is type(b) and bits(a) == bits(b)


def same_tensor(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.is_floating_point:
        view = {torch.float64: torch.int64, torch.float32: torch.int32, torch.bfloat16: torch.int16}[a.dtype]
        return torch.equal(a.view(view), b.view(view))
    return torch.equal(a, b)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


DEV_FIELDS = ('ce', 'kl', 'ce_nll', 'kl_values')
TRY_FIELDS = ('size', 'predicted_ce', 'predicted_kl', 'dev_delta_ce', 'dev_delta_kl')
ROUND_FIELDS = ('round', 'candidates', 'accepted', 'e0m3_units', 'dev_ce', 'dev_kl')


def against_record(ref, run, ref_map, run_map):
    """Every field of the committed record that a rerun must reproduce."""
    checks = {}
    checks['round_count'] = len(ref['rounds']) == len(run['rounds'])
    bad_rounds, bad_tries, tries = [], [], 0
    for a, b in zip(ref['rounds'], run['rounds']):
        if not all(same_value(a.get(k), b.get(k)) for k in ROUND_FIELDS):
            bad_rounds.append(a['round'])
        if len(a['tries']) != len(b['tries']):
            bad_tries.append((a['round'], 'count'))
        for i, (x, y) in enumerate(zip(a['tries'], b['tries'])):
            tries += 1
            if not all(same_value(x[k], y[k]) for k in TRY_FIELDS):
                bad_tries.append((a['round'], i))
    checks['rounds'] = not bad_rounds
    checks['tries'] = not bad_tries
    for key in ('initial_dev', 'final_dev', 'fake_initial_dev'):
        if key in ref or key in run:
            checks[key] = key in ref and key in run and all(same_value(ref[key][k], run[key][k]) for k in DEV_FIELDS)
    checks['stopped'] = ref.get('stopped') == run.get('stopped')
    checks['final_e0m3_units'] = ref['final_e0m3_units'] == run['final_e0m3_units']
    a, b = torch.load(ref_map, weights_only=True), torch.load(run_map, weights_only=True)
    checks['final_map'] = list(a) == list(b) and all(same_tensor(a[n], b[n]) for n in a)
    checks['evaluation'] = all(same_value(ref['evaluation'][d]['nll'], run['evaluation'][d]['nll'])
                               and same_value(ref['evaluation'][d]['ppl'], run['evaluation'][d]['ppl'])
                               for d in ('wiki', 'c4'))
    detail = dict(rounds=len(run['rounds']), tries=tries, mismatched_rounds=bad_rounds, mismatched_tries=bad_tries,
                  map_file_sha256_equal=digest(ref_map) == digest(run_map),
                  windows={d: len(run['evaluation'][d]['nll']) for d in ('wiki', 'c4')})
    return checks, detail


def against_legacy(legacy_dir, lean_dir):
    """The per-document values of every try and the round-0 scores, which only the reruns record."""
    checks = {}
    a = torch.load(legacy_dir / 'dev_values.pt', weights_only=True)
    b = torch.load(lean_dir / 'dev_values.pt', weights_only=True)
    checks['dev_values_initial'] = all(same_tensor(a['initial'][k], b['initial'][k]) for k in ('ce', 'kl'))
    if a['fake_initial'] is not None or b['fake_initial'] is not None:
        checks['dev_values_fake_initial'] = (a['fake_initial'] is not None and b['fake_initial'] is not None and
                                             all(same_tensor(a['fake_initial'][k], b['fake_initial'][k])
                                                 for k in ('ce', 'kl')))
    bad = [i for i, (x, y) in enumerate(zip(a['tries'], b['tries']))
           if (x['round'], x['try_index'], x['size']) != (y['round'], y['try_index'], y['size'])
           or not same_tensor(x['ce'], y['ce']) or not same_tensor(x['kl'], y['kl'])]
    checks['dev_values_tries'] = len(a['tries']) == len(b['tries']) and not bad
    sa = torch.load(legacy_dir / 'round0_scores.pt', weights_only=True)
    sb = torch.load(lean_dir / 'round0_scores.pt', weights_only=True)
    bad_scores = [n for n in sa if n not in sb or not all(same_tensor(sa[n][k], sb[n][k])
                                                         for k in ('ce_mean', 'ce_se', 'kl_mean', 'kl_se'))]
    checks['round0_scores'] = list(sa) == list(sb) and not bad_scores
    units = sum(t['kl_mean'].numel() for t in sa.values())
    detail = dict(tries_with_per_document_values=len(a['tries']), documents=int(a['initial']['kl'].numel()),
                  mismatched_try_indices=bad, round0_score_units=units, round0_mismatched_modules=bad_scores,
                  round0_scores_file_sha256_equal=digest(legacy_dir / 'round0_scores.pt') ==
                  digest(lean_dir / 'round0_scores.pt'))
    return checks, detail


def main():
    ref_path, ref_map, run_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    legacy_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    ref = json.loads(ref_path.read_text())
    assert digest(ref_map) == ref['map_sha256'], 'the committed map file is not the one the record names'
    run = json.loads((run_dir / 'report.json').read_text())
    assert run['status'] == 'complete', run_dir
    out = dict(committed=str(ref_path), run=str(run_dir), memory_mode=run.get('memory_mode'))
    out['vs_committed'], out['vs_committed_detail'] = against_record(ref, run, ref_map, run_dir / 'map.pt')
    if legacy_dir is not None:
        out['vs_legacy'], out['vs_legacy_detail'] = against_legacy(legacy_dir, run_dir)
    checks = dict(out['vs_committed'], **{f'legacy_{k}': v for k, v in out.get('vs_legacy', {}).items()})
    out['passed'] = all(checks.values())
    out['failed_checks'] = sorted(k for k, v in checks.items() if not v)
    print(json.dumps(out, indent=1))
    sys.exit(0 if out['passed'] else 1)


if __name__ == '__main__':
    main()
