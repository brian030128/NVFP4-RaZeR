"""Validate the math/code-only study and report adaptive counts and both PPLs."""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
from quantize.adaptive_prefix import source_subsets
from run_c4_frozen import digest_file
from run_conditional_model import paired

MODELS=('qwen4b','llama8b','qwen27b')
LABELS={'qwen4b':'Qwen3-4B','llama8b':'Llama-3.1-8B','qwen27b':'Qwen3.8-27B'}
DOMAINS=('wiki','c4')
START='<!-- MATH_CODE_ADAPTIVE_START -->'
END='<!-- MATH_CODE_ADAPTIVE_END -->'


def merge_parts(parts):
    first=parts[0]; result=copy.deepcopy(first)
    result['evaluation']={}; result['contrasts']={}; result['data']={}; result['suffix_intervention']={}
    for r in parts:
        assert r['status']=='complete' and r['maps_unchanged'] and r['source_weights_verified']
        for key in ('model','source','revision','torch_version','transformers_version','calibration',
                    'calibration_report_sha256','map_sha256','source_sha256','block_statistics',
                    'uses_c4_calibration','uses_wiki_calibration'):
            assert r[key]==first[key],key
        for d,metadata in r['data'].items():
            if d in result['data']: assert metadata==result['data'][d]
            result['data'][d]=metadata
        for p,domains in r['evaluation'].items():
            assert r['suffix_intervention'][p]['equal'] and r['suffix_intervention'][p]['max_logit_difference']==0
            result['suffix_intervention'][p]=r['suffix_intervention'][p]
            dest=result['evaluation'].setdefault(p,{})
            for d,v in domains.items():
                if d in dest:
                    assert p in ('four_over_six','weight_mse'), 'Only controls may repeat across shards'
                    assert v['nll']==dest[d]['nll'] and v['ppl']==dest[d]['ppl']
                else: dest[d]=v
        for p,domains in r['contrasts'].items(): result['contrasts'].setdefault(p,{}).update(domains)
    assert set(result['data'])==set(DOMAINS)
    assert all(set(v)==set(DOMAINS) for v in result['evaluation'].values())
    result['policies']=list(result['evaluation']); result['domains']=list(DOMAINS); result['group']='merged'
    result['evaluation_job_ids']=[r['job_id'] for r in parts]
    return result


def performance_table(model,r):
    lines=[f'### {LABELS[model]}','',
        '| Calibration | Sequences | Adaptive E0M3 blocks | Adaptive Wiki PPL | Adaptive C4 PPL | Adaptive average | Fixed E0M3 blocks | Fixed Wiki PPL | Fixed C4 PPL | Fixed average |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for setting,ids in source_subsets().items():
        fields=[setting,str(len(ids))]
        for mode in ('adaptive','fixed256'):
            policy=f'{mode}_{setting}'
            if policy not in r['evaluation']:
                fields+=['—']*4; continue
            values=[r['evaluation'][policy][d]['ppl'] for d in DOMAINS]
            fields += [str(r['block_statistics'][policy]['selected_blocks']),
                       *(f'{v:.6f}' for v in values),f'{sum(values)/2:.6f}']
        lines.append('| '+' | '.join(fields)+' |')
    return lines


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap=argparse.ArgumentParser(); ap.add_argument('--small-job',required=True); ap.add_argument('--large-job',required=True)
    ap.add_argument('--out',required=True); ap.add_argument('--update-report',action='store_true'); args=ap.parse_args()
    root=Path('results/math_code_adaptive'); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    paths={m:root/f'evaluation_{args.small_job}_{m}/report.json' for m in MODELS[:2]}
    large_paths=[root/f'evaluation_{args.large_job}_qwen27b_{group}/report.json'
                 for group in ('c4','wiki_adaptive','wiki_fixed')]
    merged=merge_parts([json.loads(p.read_text()) for p in large_paths])
    merged['part_report_sha256']={str(p):digest_file(p) for p in large_paths}
    merged_path=out/'qwen27b_merged.json'; merged_path.write_text(json.dumps(merged,indent=2)+'\n')
    paths['qwen27b']=merged_path
    reports={}; calibrations={}; maps={}; cells=[]
    for model,path in paths.items():
        r=json.loads(path.read_text()); reports[model]=r
        assert r['model']==model and r['status']=='complete' and r['maps_unchanged'] and r['source_weights_verified']
        assert not r['uses_c4_calibration'] and not r['uses_wiki_calibration']
        calibration=Path(r['calibration']); assert digest_file(calibration/'report.json')==r['calibration_report_sha256']
        c=json.loads((calibration/'report.json').read_text()); calibrations[model]=c
        assert c['status']=='complete' and c['maps_frozen'] and set(c['fit'])=={'math','code'}
        assert c['source']==r['source'] and c['revision']==r['revision']
        assert c['fit']['math']['repo']=='open-web-math/open-web-math'
        assert c['fit']['code']['repo']=='codeparrot/codeparrot-clean'
        assert c['subsets']==source_subsets() and len(c['initial_fit_losses'])==128
        assert c['block_statistics']==r['block_statistics']
        excluded={v['document_sha256'] for meta in c['fit'].values() for v in meta['documents']}
        assert len(excluded)==128
        assert not excluded.intersection(v['document_sha256'] for v in r['data']['c4']['documents'])
        assert not excluded.intersection(r['data']['wiki']['nonempty_row_sha256'])
        assert r['data']['c4']['calibration_hash_overlap']==0 and r['data']['c4']['windows']==256
        wm=r['data']['wiki']; assert wm['calibration_exact_row_overlap']==[]
        assert wm['windows']*512+wm['omitted_tail_tokens']==wm['total_tokens']
        assert wm['offsets']==[i*512 for i in range(wm['windows'])]
        assert digest_file(calibration/'maps.json')==r['map_sha256']
        maps[model]=json.loads((calibration/'maps.json').read_text())['maps']
        assert set(r['evaluation'])=={'four_over_six','weight_mse',*maps[model]}
        for p,e in r['evaluation'].items():
            assert set(e)==set(DOMAINS) and r['suffix_intervention'][p]['equal']
            assert r['suffix_intervention'][p]['max_logit_difference']==0
            s=r['block_statistics'][p]
            if p in maps[model]:
                assert sum(len(v) for v in maps[model][p].values())==s['selected_blocks']
            if p.startswith('adaptive_'):
                assert s['count_cap'] is None
                assert s['predicted_upper_objective']<=0
            for d,v in e.items():
                n=r['data'][d]['windows']; assert len(v['nll'])==n and v['scored_tokens']==n*511
                assert len(r['data'][d]['token_sha256'])==n and all(math.isfinite(x) for x in v['nll'])
                assert abs(math.exp(sum(v['nll'])/n)-v['ppl'])<1e-12
                contrast=paired(v['nll'],r['evaluation']['four_over_six'][d]['nll'])
                if p!='four_over_six': assert contrast==r['contrasts'][p][d]
                if v['origin']['reused']:
                    assert digest_file(v['origin']['report'])==v['origin']['report_sha256']
                    assert v['origin']['first_window_absolute_error']<=1e-6
                cells.append(dict(model=model,policy=p,dataset=d,ppl=v['ppl'],
                    baseline_ppl=r['evaluation']['four_over_six'][d]['ppl'],**contrast,
                    selected_e0m3_type_blocks=s['selected_blocks'],selected_fraction=s['selected_fraction'],
                    calibration_sequences=s.get('calibration_sequences',0),reused=v['origin']['reused'],report=str(path)))
    with (out/'cells.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cells[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(cells)
    lines=['## Math/code-only calibration: adaptive E0M3 count','',
        'Calibration uses **OpenWebMath and CodeParrot only**. Neither C4 nor WikiText supplies calibration '
        'examples, gradients, or count-selection feedback. All measurements below evaluate the frozen maps '
        'only on **WikiText-2 test and held-out C4**. No seed replication or best-setting election is performed.','',
        'One shared 128-sequence causal scoring pass per model supplies all ten source/sample-count settings. '
        'The adaptive method minimizes a directional-benefit plus estimated-curvature penalty over all '
        'negative-score prefixes, including zero switches. It has **no count cap**. The fixed-256 comparison '
        'uses the same fresh causal CE/KL derivatives and math/code subsets; it is not the older C4-containing pooled map.','',
        'Counts are **8x64 E0M3 type blocks**, each containing 512 weights and 32 distinct 16-element scale '
        'blocks. Each average is the unweighted arithmetic mean of the two displayed dataset PPLs. '
        'PPL is computed from per-token loss over identical scored windows within each model.','',
        '| Model | Baseline Wiki PPL | Baseline C4 PPL | Baseline average | Weight-MSE E0M3 blocks | Weight-MSE Wiki PPL | Weight-MSE C4 PPL | Weight-MSE average |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model,r in reports.items():
        base=[r['evaluation']['four_over_six'][d]['ppl'] for d in DOMAINS]
        mse=[r['evaluation']['weight_mse'][d]['ppl'] for d in DOMAINS]
        lines.append(f'| {LABELS[model]} | '+ ' | '.join(f'{v:.6f}' for v in (*base,sum(base)/2))+
                     f' | {r["block_statistics"]["weight_mse"]["selected_blocks"]:,} | '+
                     ' | '.join(f'{v:.6f}' for v in (*mse,sum(mse)/2))+' |')
    lines+=['']
    for model,r in reports.items(): lines+=performance_table(model,r)+['']
    compact=list(lines)
    lines+=['### Adaptive objective, counts, and fractions','',
        '| Model | Setting | Eligible blocks | Selected blocks | Selected % of all type blocks | Predicted linear term | Curvature penalty | Predicted total |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    comparisons={}; overlaps={}
    for model,r in reports.items():
        comparisons[model]={}; overlaps[model]={}
        for setting in source_subsets():
            p='adaptive_'+setting; s=r['block_statistics'][p]
            lines.append(f'| {LABELS[model]} | {setting} | {s["eligible_blocks"]:,} | {s["selected_blocks"]:,} | '
                f'{100*s["selected_fraction"]:.6f}% | {s["predicted_linear"]:+.6g} | '
                f'{s["predicted_curvature_penalty"]:.6g} | {s["predicted_upper_objective"]:+.6g} |')
            fixed='fixed256_'+setting
            if fixed in r['evaluation']:
                comparisons[model][setting]={d:paired(r['evaluation'][p][d]['nll'],r['evaluation'][fixed][d]['nll']) for d in DOMAINS}
        for p,ms in maps[model].items():
            a={(n,i) for n,v in ms.items() for i in v}; overlaps[model][p]={}
            for q,ns in maps[model].items():
                b={(n,i) for n,v in ns.items() for i in v}; union=a|b
                overlaps[model][p][q]=dict(intersection=len(a&b),union=len(union),jaccard=len(a&b)/len(union) if union else 1.)
    lines+=['','### Baseline-relative paired NLL differences','',
        '| Model | Policy | Wiki ΔNLL ±2SE | C4 ΔNLL ±2SE |','|---|---|---:|---:|']
    for model,r in reports.items():
        for p,ds in r['contrasts'].items():
            lines.append(f'| {LABELS[model]} | {p} | '+' | '.join(f'{ds[d]["mean_nll"]:+.6f} ±{ds[d]["two_se"]:.6f}' for d in DOMAINS)+' |')
    limits=('The curvature inequality is exact for the estimated PSD matrix, but its sampled GGN diagonal '
        'and straight-through derivatives do not bound the true finite-switch network loss. This is an '
        'adaptive surrogate method, not a universal guarantee or a claim of methodological novelty. '
        'Two-SE intervals describe evaluation-window variation; they do not adjust for multiple comparisons, '
        'WikiText article dependence, or unmeasured calibration-draw variability. Both target families were '
        'previously inspected. Source exclusion is not a near-duplicate or pretraining-overlap audit.')
    lines+=['','### Limits','',limits,'',
        'The cancelled three-source/five-dataset experiment is historical and is not combined with these results. '
        'Simulated nonhead-text-linear W4A4; no generation-accuracy or native-throughput claim.']
    summary=dict(models=3,settings_per_model=10,evaluation_datasets=list(DOMAINS),
        calibration_sources=['OpenWebMath','CodeParrot'],seed_replication=False,cells=len(cells),
        adaptive_block_counts={m:{p:s['selected_blocks'] for p,s in r['block_statistics'].items() if p.startswith('adaptive_')}
                               for m,r in reports.items()},adaptive_minus_fixed=comparisons,
        source_reports={m:str(p) for m,p in paths.items()})
    outcomes={}
    for model,r in reports.items():
        outcomes[model]={}
        for mode in ('adaptive','fixed256'):
            cs=[r['contrasts'][p][d] for p in r['contrasts'] if p.startswith(mode+'_') for d in DOMAINS]
            outcomes[model][mode]=dict(cells=len(cs),point_gains=sum(v['mean_nll']<0 for v in cs),
                supported_gains=sum(v['mean_nll']+v['two_se']<0 for v in cs),
                supported_harms=sum(v['mean_nll']-v['two_se']>0 for v in cs))
        outcomes[model]['adaptive_beats_fixed_point']=sum(v['mean_nll']<0 for ds in comparisons[model].values() for v in ds.values())
    summary['descriptive_outcomes']=outcomes
    count_values=[v for counts in summary['adaptive_block_counts'].values() for v in counts.values()]
    comparison_cells=sum(o['adaptive']['cells'] for o in outcomes.values())
    wins=sum(o['adaptive_beats_fixed_point'] for o in outcomes.values())
    fixed_gains=sum(o['fixed256']['point_gains'] for o in outcomes.values())
    fixed_supported=sum(o['fixed256']['supported_gains'] for o in outcomes.values())
    fixed_harms=sum(o['fixed256']['supported_harms'] for o in outcomes.values())
    interpretation=(f'**Result: this adaptive rule is not a competitive replacement for fixed-256.** '
        f'It selected {min(count_values)}–{max(count_values)} blocks and achieved lower PPL than '
        f'fixed-256 in only {wins}/{comparison_cells} paired model/setting/dataset comparisons. '
        f'Fixed-256 improved point PPL over FourOverSix in {fixed_gains}/{comparison_cells} cells; '
        f'{fixed_supported} had baseline-relative ΔNLL + 2SE < 0 and {fixed_harms} had ΔNLL − 2SE > 0. '
        'These correlated-cell counts are descriptive, not independent replications. '
        'The post hoc curvature audit documents conservatism in the interaction penalty; it does not '
        'establish that a less conservative selector would generalize. The study supports a negative '
        'finding for this particular adaptive surrogate, not an impossibility claim about adaptive selection.')
    lines[4:4]=[interpretation,'','[Frozen protocol and selection equations](../PROTOCOL.md).','']
    compact[4:4]=[interpretation,'','[Frozen protocol and selection equations](results/math_code_adaptive/PROTOCOL.md).','']
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'selection_overlap.json').write_text(json.dumps(overlaps,indent=2)+'\n')
    from diagnose_math_code_curvature import audit
    audit([r['calibration'] for r in reports.values()],out)
    from plot_math_code_adaptive import plot
    plot(out/'summary.json',out)
    lines+=['','[Count and transfer-sensitivity figure](sensitivity.pdf). Colors identify math, code, '
            'and balanced math+code calibration; solid/dashed lines identify adaptive/fixed-256. '
            'Error bars are descriptive evaluation-window two-SE intervals, not calibration-seed intervals.']
    lines+=['','[Post hoc curvature audit](CURVATURE_AUDIT.md) compares the conservative interaction '
            'penalty with the same sampled GGN quadratic form on the already-frozen maps. '
            'It changes no maps and does not measure true finite-switch losses.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    compact+= [limits,'',f'[Full adaptive study and paired results]({out.as_posix()}/REPORT.md) · '
        f'[All separate-dataset values]({out.as_posix()}/cells.csv) · '
        f'[Selection overlap]({out.as_posix()}/selection_overlap.json) · '
        f'[Sensitivity figure]({out.as_posix()}/sensitivity.pdf) · '
        f'[Post hoc curvature audit]({out.as_posix()}/CURVATURE_AUDIT.md)','']
    if args.update_report:
        path=Path('MIXFP4_REPORT.md'); text=path.read_text(); block=START+'\n'+'\n'.join(compact)+'\n'+END
        if START in text:
            assert text.count(START)==text.count(END)==1
            text=text[:text.index(START)]+block+text[text.index(END)+len(END):]
        else:
            anchor='<!-- END POOLED PERFORMANCE TABLES -->'; pos=text.index(anchor)+len(anchor)
            text=text[:pos]+'\n\n'+block+'\n'+text[pos:]
        path.write_text(text)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
