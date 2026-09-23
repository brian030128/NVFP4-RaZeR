"""Existing composition/budget cells; prefix sharing is not independent draws."""
from common import *
from mechanism_review import sealed

def main():
    start=now();rows=[]
    for model in ['qwen4b','mistral7b']:
        prefix='V31' if model=='qwen4b' else 'V62'
        paths=[PRIMARY/f'runs/{prefix}_ppl_primary_{model}_attempt1/ppl/ppl_report.json',PRIMARY/f'runs/V51_ppl_sizedomain_{model}_attempt1/ppl/ppl_report.json']
        for p in paths:
            sealed(p);r=load(p);assert r['status']=='complete'
            lr=load(p.parent.parent/'launch_record.json');assert lr['status']=='complete' and not lr.get('invalid_gpu_cotenancy')
            for ins in r['installs']:
                if ins.get('type_block') not in [[8,64],[16,64]]:continue
                pol=ins['map_header_policy'];assert pol['rule']=='ce_kl' and pol['k']==3
                if 'ppl_primary' in str(p) and ins['name'] not in ['n8_k3','n16_k3']:continue
                mp=Path(ins['map_path']);assert sha(mp)==ins['map_sha256'];head,masks=mapread(mp)
                parts=pol.get('subset_sequences',dict(math=64,code=64));n=sum(parts.values())
                for corpus in ['wiki','c4']:
                    a=r['evaluation'][ins['name']][corpus];b=r['evaluation']['four_over_six'][corpus]
                    assert a['tokens']==b['tokens'] if 'tokens' in a else sum(x['tokens'] for x in a['windows'])==sum(x['tokens'] for x in b['windows'])
                    rows.append(dict(model=model,N=ins['type_block'][0],policy=ins['name'],draw=pol['draw'],corpus=corpus,
                        recipe=pol.get('subset','mc128' if pol['draw']=='seed0' else 'heldout_mc128'),math_sequences=parts.get('math',0),code_sequences=parts.get('code',0),
                        calibration_sequences=n,input_tokens=n*512,valid_shifted_target_tokens=n*511,available_replications_for_this_recipe=1,
                        sampling='nested prefixes of seed0' if 'subset' in pol else ('full seed0' if pol['draw']=='seed0' else 'separate heldout draw'),
                        ppl=a['ppl'],delta_nll=a['mean_nll']-b['mean_nll'],relative_ppl=float(np.expm1(a['mean_nll']-b['mean_nll'])),
                        selected_tiles=head['totals']['selected_tiles'],map_sha256=ins['map_sha256'],source=str(p.relative_to(REPO)),source_sha256=sha(p),
                        inference='descriptive existing cells; no across-composition population variance or five-draw factorial inferred'))
    csvout(OUT/'results/composition_existing.csv',rows)
    log('T4-composition','python scripts/composition_context.py',start,[OUT/'results/composition_existing.csv'])
    print('Existing composition/budget cells',len(rows))

if __name__=='__main__':main()
