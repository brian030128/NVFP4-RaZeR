"""Join disjoint dataset jobs without combining or selecting calibration maps."""
import argparse
import copy
import json
import os
from pathlib import Path
from run_c4_frozen import digest_file

DOMAINS = ('c4', 'wiki', 'literature', 'science', 'government')


def merge(reports):
    first = reports[0]
    result = copy.deepcopy(first)
    result['evaluation'] = {p: {} for p in first['evaluation']}
    result['contrasts'] = {p: {} for p in first['contrasts']}
    covered = set()
    for r in reports:
        assert r['status'] == 'complete' and r['maps_unchanged'] and r['source_weights_verified']
        for key in ('model', 'source', 'revision', 'torch_version', 'transformers_version', 'prepared',
                    'prepared_sha256', 'source_sha256', 'activation_convention', 'block_statistics', 'data'):
            assert r[key] == first[key], key
        assert set(r['evaluation']) == set(first['evaluation'])
        domains = set(r['evaluated_domains'])
        assert domains and not (covered & domains)
        covered.update(domains)
        for policy, values in r['evaluation'].items():
            assert set(values) == domains
            assert r['suffix_intervention'][policy]['equal']
            assert r['suffix_intervention'][policy]['max_logit_difference'] == 0
            result['evaluation'][policy].update(values)
        assert set(r['contrasts']) == set(first['contrasts'])
        for policy, values in r['contrasts'].items():
            assert set(values) == domains
            result['contrasts'][policy].update(values)
    assert covered == set(DOMAINS)
    result['evaluated_domains'] = list(DOMAINS)
    result['evaluation_job_ids'] = [r['job_id'] for r in reports]
    result['all_part_prefix_checks_passed'] = True
    return result


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs=3, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    reports = [json.loads(Path(p).read_text()) for p in args.parts]
    result = merge(reports)
    result['part_reports_sha256'] = {p: digest_file(p) for p in args.parts}
    result['merge_job_id'] = os.environ['SLURM_JOB_ID']
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Merged all five datasets with identical frozen maps, data, and quantizer sources', flush=True)


if __name__ == '__main__':
    main()
