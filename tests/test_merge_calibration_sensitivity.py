import copy
from merge_calibration_sensitivity import merge, DOMAINS


def main():
    first = dict(status='complete', maps_unchanged=True, source_weights_verified=True,
        model='test', source='test', revision='test', torch_version='test', transformers_version='test',
        prepared='maps.json', prepared_sha256='abc', source_sha256={}, activation_convention='causal',
        block_statistics={}, data={}, evaluation={}, contrasts={}, job_id='test',
        suffix_intervention={'baseline': dict(equal=True, max_logit_difference=0)})
    parts = []
    for domains in (['c4'], ['wiki'], list(DOMAINS[2:])):
        r = copy.deepcopy(first)
        r['evaluated_domains'] = domains
        r['evaluation'] = {'baseline': {d: {'ppl': i+1} for i,d in enumerate(domains)}}
        parts.append(r)
    result = merge(parts)
    assert set(result['evaluation']['baseline']) == set(DOMAINS)
    assert result['evaluation']['baseline']['government']['ppl'] == 3
    for invalid in ('duplicate', 'identity', 'prefix', 'missing'):
        wrong = copy.deepcopy(parts)
        if invalid == 'duplicate': wrong.append(copy.deepcopy(parts[0]))
        if invalid == 'identity': wrong[1]['prepared_sha256'] = 'different'
        if invalid == 'prefix': wrong[1]['suffix_intervention']['baseline']['equal'] = False
        if invalid == 'missing': wrong.pop()
        try:
            merge(wrong)
        except AssertionError:
            pass
        else:
            raise AssertionError(invalid)
    print('Dataset merge rejects duplicates, missing domains, mismatched maps, and failed causal checks')


if __name__ == '__main__':
    main()
