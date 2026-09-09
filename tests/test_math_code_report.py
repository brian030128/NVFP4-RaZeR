import copy
from summarize_math_code_adaptive import merge_parts, performance_table
from quantize.adaptive_prefix import source_subsets


def main():
    r=dict(block_statistics={},evaluation={})
    for setting in source_subsets():
        for mode,count,base in [('adaptive',17,3.),('fixed256',256,4.)]:
            p=f'{mode}_{setting}'; r['block_statistics'][p]=dict(selected_blocks=count)
            r['evaluation'][p]={'wiki':dict(ppl=base+.1234567),'c4':dict(ppl=base+2.1234567)}
    rows=[s for s in performance_table('qwen4b',r) if s.startswith('| ')][1:]
    assert len(rows)==10
    for row in rows:
        values=[x.strip() for x in row.split('|')[1:-1]]
        assert values[2:]==['17','3.123457','5.123457','4.123457','256','4.123457','6.123457','5.123457']
    template=dict(status='complete',maps_unchanged=True,source_weights_verified=True,
        model='test',source='test',revision='test',torch_version='test',transformers_version='test',
        calibration='test',calibration_report_sha256='same',map_sha256='same',source_sha256={},block_statistics={},
        uses_c4_calibration=False,uses_wiki_calibration=False,evaluation={},contrasts={},data={},
        suffix_intervention={},job_id='test')
    parts=[]
    for domain in ('wiki','c4'):
        part=copy.deepcopy(template)
        part['data']={domain:dict(windows=2)}
        part['evaluation']={'adaptive_math16':{domain:dict(nll=[1.,2.],ppl=4.)}}
        part['contrasts']={'adaptive_math16':{domain:{}}}
        part['suffix_intervention']={'adaptive_math16':dict(equal=True,max_logit_difference=0)}
        parts.append(part)
    assert set(merge_parts(parts)['evaluation']['adaptive_math16'])=={'wiki','c4'}
    for wrong in (parts[:1],parts+[parts[0]]):
        try: merge_parts(wrong)
        except AssertionError: pass
        else: raise AssertionError('Missing/duplicate adaptive cells accepted')
    bad=copy.deepcopy(parts); bad[1]['map_sha256']='different'
    try: merge_parts(bad)
    except AssertionError: pass
    else: raise AssertionError('Mismatched maps accepted')
    print('Separate Wiki/C4 values, actual counts, averages, and shard coverage verified')


if __name__=='__main__': main()
