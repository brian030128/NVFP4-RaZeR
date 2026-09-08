"""Validate the completed 27B extension and show it beside the seven-model C4 run."""
import argparse
import json
import math
from pathlib import Path
from run_c4_frozen import digest_file
from run_conditional_model import paired
from verify_c4_disjointness import audit_report


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--job', type=int, required=True); args = ap.parse_args()
    folder = Path(f'results/pooled_qwen27b/model_{args.job}')
    r = json.loads((folder / 'report.json').read_text())
    assert r['status'] == 'complete' and r['maps_frozen'] and r['frozen_map_unchanged']
    assert r['map_file_sha256'] == digest_file(folder / 'maps.pt')
    assert len(r['matrices']) == 496 and r['selected_tiles']['pooled192'] <= 256
    fit = {d['document_sha256'] for meta in r['fit'].values() for d in meta['documents']}
    test = {d['document_sha256'] for d in r['c4_data']['documents']}
    assert len(fit) == 192 and len(test) == 256 and not fit & test
    assert all(x['equal'] and x['max_logit_difference'] == 0 for x in r['suffix_intervention'].values())
    for p, e in r['evaluation'].items():
        assert len(e['nll']) == 256 and e['scored_tokens'] == 256 * 511
        assert math.isclose(e['ppl'], math.exp(sum(e['nll'])/256), rel_tol=1e-12)
        if p != 'pooled192':
            assert paired(r['evaluation']['pooled192']['nll'], e['nll']) == r['contrasts'][p]
    source = Path('results/c4_frozen/summary_332781.json')
    previous = json.loads(source.read_text())
    audited_paths = previous['source_reports'] + [str(folder / 'report.json')]
    audits = [audit_report(p) for p in audited_paths]
    (folder.parent / f'disjointness_{args.job}.json').write_text(json.dumps(audits, indent=2) + '\n')
    rows = [json.loads(Path(p).read_text()) for p in previous['source_reports']] + [r]
    cs = [x['contrasts']['four_over_six'] for x in rows]
    summary = dict(models=8, point_gains=sum(c['mean_nll'] < 0 for c in cs),
                   gains_at_least_0p01_ppl=sum(c['ppl_delta'] <= -.01 for c in cs),
                   supported_gains=sum(c['mean_nll'] + c['two_se'] < 0 for c in cs),
                   supported_harms=sum(c['mean_nll'] - c['two_se'] > 0 for c in cs),
                   qwen27b_selected_tiles=r['selected_tiles']['pooled192'],
                   qwen27b_gain_at_least_0p01_ppl=r['contrasts']['four_over_six']['ppl_delta'] <= -.01,
                   qwen27b_contrasts=r['contrasts'], qwen27b_report=str(folder/'report.json'),
                   previous_summary=str(source), calibration_hash_overlap=0)
    lines = ['# Frozen-rule C4 results including Qwen3.8-27B', '',
             'Same 192-sequence calibration recipe, CE/KL two-SE score, and 256-tile cap. '
             'One map per model; no candidate-loss backtracking or evaluation-based selection. '
             'All rows use 256 held-out C4 validation documents with 512-token crops and causal '
             'per-token activation factors. C4 is a calibration source; these are within-source results.', '',
             '| Model | FourOverSix C4 PPL | Selected C4 PPL | ΔPPL | ΔNLL ±2SE |',
             '|---|---:|---:|---:|---:|']
    for x, c in zip(rows, cs):
        e = x['evaluation']
        lines.append(f'| {x["model"]} | {e["four_over_six"]["ppl"]:.6f} | '
                     f'{e["pooled192"]["ppl"]:.6f} | {c["ppl_delta"]:+.6f} | '
                     f'{c["mean_nll"]:+.6f} ±{c["two_se"]:.6f} |')
    lines += ['', f'The 27B map selects {r["selected_tiles"]["pooled192"]} E0M3 tiles. '
              'It uses native Transformers 5.16.1 for the hybrid Qwen3.5-family text architecture; '
              'the preceding seven models used 4.57.3. Scope is nonhead text linear weights and inputs; '
              'vision, recurrent state, convolution and normalization remain native. '
              'These are simulated W4A4 results, not native throughput or generation accuracy.', '',
              f'[27B controls and detailed result](model_{args.job}/REPORT.md). '
              f'[Recomputed eight-model calibration overlap audit](disjointness_{args.job}.json). '
              'All 27B model-loading, map, label-count, hash-exclusion and prefix-independence checks pass. '
              'One calibration pool per model and descriptive two-SE intervals do not establish a '
              'universal guarantee. The earlier source-diversity screen remains failed.', '',
              f'Of the eight point comparisons, {summary["gains_at_least_0p01_ppl"]} improve by at least '
              '0.01 PPL, the practical reference threshold used in the preceding record. '
              f'The 27B result meets that reference: {summary["qwen27b_gain_at_least_0p01_ppl"]}. '
              'The follow-up protocol specifies measurement without a new pass/fail screen; '
              'the reference threshold is reported separately from the sign and paired uncertainty.', '',
              '```json', json.dumps(summary, indent=2), '```']
    root = folder.parent
    (root/f'C4_COMPARISON_{args.job}.md').write_text('\n'.join(lines)+'\n')
    (root/f'summary_{args.job}.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
