"""Frozen math/code-only maps evaluated exclusively on WikiText and C4."""
import argparse
import json
import math
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer
from quantize.causal_four_over_six import quantize_rows
from quantize.interacting_format import apply_mask
from quantize.basis import BASES, build_pair
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_c4_frozen import digest_file, heldout_data
from run_conditional_format import save, sha
from run_conditional_model import paired
from run_math_code_calibration import load_model
from run_wiki_frozen import wiki_data


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap=argparse.ArgumentParser()
    ap.add_argument('--calibration',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--group',choices=('all','c4','wiki_adaptive','wiki_fixed'),default='all')
    args=ap.parse_args()
    old=Path(args.calibration); prior=json.loads((old/'report.json').read_text())
    assert prior['status']=='complete' and prior['maps_frozen'] and prior['source_weights_verified']
    assert not prior['uses_c4_calibration'] and not prior['uses_wiki_calibration']
    assert set(prior['fit']) == {'math','code'}
    assert transformers.__version__ == prior['transformers_version']
    assert digest_file(old/'maps.json') == prior['map_sha256']
    assert digest_file(old/'weight_mse.pt') == prior['weight_mse_sha256']
    basis=prior.get('basis','e0m3')
    assert basis in BASES, basis
    bundle=json.loads((old/'maps.json').read_text())
    assert bundle['source']==prior['source'] and bundle['revision']==prior['revision']
    assert bundle['type_block']==[8,64]
    assert bundle['baseline']==BASES[basis]['baseline'] and bundle['alternative']==BASES[basis]['alternative']
    assert bundle.get('basis',basis)==basis
    policies=['four_over_six',*bundle['maps'],'weight_mse']
    domains=('c4','wiki') if args.group=='all' else (('c4',) if args.group=='c4' else ('wiki',))
    if args.group.startswith('wiki_'):
        prefix='adaptive_' if args.group=='wiki_adaptive' else 'fixed256_'
        policies=[p for p in policies if p in ('four_over_six','weight_mse') or p.startswith(prefix)]
    target=prior['model']=='qwen27b'
    torch.set_num_threads(12 if target else 4); torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    files=('run_math_code_evaluation.py','run_math_code_calibration.py','quantize/adaptive_prefix.py',
           'quantize/causal_four_over_six.py','quantize/quantizer.py','quantize/interacting_format.py',
           'quantize/basis.py',
           'run_c4_frozen.py','run_wiki_frozen.py','results/math_code_adaptive/PROTOCOL.md')
    r=dict(status='running',model=prior['model'],source=prior['source'],revision=prior['revision'],
        torch_version=torch.__version__,transformers_version=transformers.__version__,job_id=os.environ['SLURM_JOB_ID'],
        calibration=str(old),calibration_report_sha256=digest_file(old/'report.json'),map_sha256=prior['map_sha256'],
        source_sha256={f:digest_file(f) for f in files},block_statistics=prior['block_statistics'],
        calibration_sources=prior['calibration_sources'],uses_c4_calibration=False,uses_wiki_calibration=False,
        group=args.group,policies=policies,domains=list(domains),evaluation={},data={},suffix_intervention={},
        basis=basis,basis_definition=BASES[basis])
    for f in ('quantize/causal_four_over_six.py','quantize/quantizer.py'):
        assert r['source_sha256'][f] == prior['source_sha256'][f]
    save(out,r)
    model,modules=load_model(prior,target)
    tok=AutoTokenizer.from_pretrained(r['source'],revision=r['revision'])
    device=model.get_input_embeddings().weight.device
    maps={}
    for policy in policies:
        if policy in ('four_over_six','weight_mse'): continue
        sparse=bundle['maps'][policy]; assert set(sparse)==set(modules)
        maps[policy]={}
        for n,m in modules.items():
            shape=(m.weight.shape[0]//8,m.weight.shape[1]//64)
            mask=torch.zeros(shape[0]*shape[1],dtype=torch.bool); indices=sparse[n]
            assert len(indices)==len(set(indices)) and all(0<=i<mask.numel() for i in indices)
            mask[indices]=True; maps[policy][n]=mask.reshape(shape)
        assert sum(int(m.sum()) for m in maps[policy].values()) == r['block_statistics'][policy]['selected_blocks']
        if policy.startswith('fixed256_'): assert r['block_statistics'][policy]['selected_blocks']<=256
    mse=torch.load(old/'weight_mse.pt',map_location='cpu',weights_only=True)
    assert mse['source']==r['source'] and mse['revision']==r['revision']
    maps['weight_mse']=mse['weight_mse']
    base,alt={},{}
    with torch.no_grad():
        for n,m in modules.items():
            assert sha(m.weight)==prior['matrices'][n]['source_sha256'],n
            b,a=build_pair(m.weight,basis)
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            base[n],alt[n]=(b.cpu().pin_memory(),a.cpu().pin_memory()) if target else (b,a)
        del a,b
    r['source_weights_verified']=True
    excluded={d['document_sha256'] for meta in prior['fit'].values() for d in meta['documents']}
    assert len(excluded)==128
    batches={}
    for domain in domains:
        batches[domain],r['data'][domain]=(heldout_data if domain=='c4' else wiki_data)(tok,excluded)
        r['data'][domain]['windows']=len(batches[domain])
        assert all(b.shape==(1,512) for b in batches[domain])
        assert [sha(b) for b in batches[domain]]==r['data'][domain]['token_sha256']
    # Controls only: reuse previous losses when all relevant identities match.
    history=Path(prior['origin'])
    old_maps=torch.load(history/'maps.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(maps['weight_mse'][n],old_maps['maps']['weight_mse'][n]) for n in modules)
    refs={}
    for domain in domains:
        path=(history/'report.json' if target else Path(f'results/c4_frozen/model_332781_{r["model"]}/report.json')) if domain=='c4' else Path(
            f'results/wiki_frozen/model_{"332976" if target else "332974"}_{r["model"]}/report.json')
        ref=json.loads(path.read_text())
        assert ref['status']=='complete' and ref['source']==r['source'] and ref['revision']==r['revision']
        assert ref['transformers_version']==r['transformers_version'] and ref['torch_version']==r['torch_version']
        assert all(ref['source_sha256'][f]==r['source_sha256'][f] for f in ('quantize/causal_four_over_six.py','quantize/quantizer.py'))
        if ref[f'{domain}_data']['token_sha256']==r['data'][domain]['token_sha256']:
            refs[domain]=(ref,path)
    save(out,r)
    print('DATA '+json.dumps({d:len(bs) for d,bs in batches.items()}),flush=True)

    def act(module,inputs): return (quantize_rows(inputs[0]),*inputs[1:])
    handles=[m.register_forward_pre_hook(act) for m in modules.values()]

    def install(policy):
        for n,m in modules.items():
            b=base[n].to(m.weight.device,non_blocking=True)
            m.weight.copy_(b if policy=='four_over_six' else apply_mask(b,
                alt[n].to(m.weight.device,non_blocking=True),maps[policy][n].to(m.weight.device)))

    def nll(batch):
        ids=batch.to(device); logits=model(input_ids=ids,use_cache=False).logits
        value=float(F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1).to(logits.device)))
        assert math.isfinite(value)
        return value

    with torch.no_grad():
        for policy in policies:
            install(policy)
            ids=next(iter(batches.values()))[0].to(device); changed=ids.clone()
            changed[:,128:]=tok.eos_token_id if tok.eos_token_id is not None else 0
            a=model(input_ids=ids,use_cache=False).logits[:,:128].clone()
            b=model(input_ids=changed,use_cache=False).logits[:,:128].clone()
            assert torch.isfinite(a).all() and torch.isfinite(b).all() and torch.equal(a,b),policy
            r['suffix_intervention'][policy]=dict(equal=True,prefix_tokens=128,max_logit_difference=0.)
            del a,b
            r['evaluation'][policy]={}
            for domain,bs in batches.items():
                if policy in ('four_over_six','weight_mse') and domain in refs:
                    ref,path=refs[domain]; values=ref['evaluation'][policy]['nll']; first=nll(bs[0])
                    assert abs(first-values[0])<=1e-6
                    origin=dict(reused=True,report=str(path),report_sha256=digest_file(path),first_window_absolute_error=abs(first-values[0]))
                else:
                    values=[]
                    for i,batch in enumerate(bs):
                        values.append(nll(batch))
                        if (i+1)%128==0: print(f'EVAL {policy} {domain} {i+1}/{len(bs)}',flush=True)
                    origin=dict(reused=False)
                assert len(values)==len(bs) and all(math.isfinite(v) for v in values)
                ppl=math.exp(sum(values)/len(values))
                r['evaluation'][policy][domain]=dict(nll=values,ppl=ppl,scored_tokens=len(values)*511,origin=origin)
                save(out,r); print(f'PPL {policy} {domain} {ppl:.6f} blocks={r["block_statistics"][policy]["selected_blocks"]}',flush=True)
    for h in handles: h.remove()
    r['contrasts']={p:{d:paired(e[d]['nll'],r['evaluation']['four_over_six'][d]['nll']) for d in domains}
                    for p,e in r['evaluation'].items() if p!='four_over_six'}
    assert digest_file(old/'maps.json')==r['map_sha256']
    assert digest_file(old/'report.json')==r['calibration_report_sha256']
    r['maps_unchanged']=True; r['status']='complete'; save(out,r)
    print('COMPLETE '+r['model']+' '+args.group,flush=True)


if __name__=='__main__': main()
