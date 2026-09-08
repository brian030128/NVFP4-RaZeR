"""Read-only audit: do frozen curvature-selected maps improve their fit objective?"""
import argparse
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from run_directed_format import fit_data
from run_conditional_format import sha
from run_conditional_model import paired
from quantize.quantizer import quant_nvfp4_4over6,quant_mix_4_6
from quantize.interacting_format import apply_mask


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=Path('results/directed_format')/f'model_332305_{args.model}'
    r=json.loads((root/'report.json').read_text());assert r['status']=='complete'
    artifact=torch.load(root/'maps.pt',map_location='cpu',weights_only=True)
    source=r['source'];revision=r['revision']
    model=AutoModelForCausalLM.from_pretrained(source,revision=revision,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval()
    teacher=AutoModelForCausalLM.from_pretrained(source,revision=revision,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager').eval()
    tok=AutoTokenizer.from_pretrained(source,revision=revision)
    modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and 'lm_head' not in n}
    base={};alt={}
    with torch.no_grad():
        for n,m in modules.items():
            assert sha(m.weight)==r['matrices'][n]['source_sha256']
            base[n]=quant_nvfp4_4over6(m.weight,4,16)
            alt[n]=quant_mix_4_6(m.weight,4,16,type_block=(8,64),clip='a1',elect='always')
    batches,metadata=fit_data(tok);assert metadata==r['fit']
    def act(module,inputs):return (quant_nvfp4_4over6(inputs[0],4,16),*inputs[1:])
    handles=[m.register_forward_pre_hook(act) for m in modules.values()]
    result={}
    for policy in ('four_over_six','fixed_budget','standard_fisher','directed'):
        values=[];nll=[]
        with torch.no_grad():
            for n,m in modules.items():m.weight.copy_(base[n] if policy=='four_over_six' else apply_mask(base[n],alt[n],artifact['maps'][policy][n].cuda()))
            for batch in batches:
                ids=batch.cuda();tl=teacher(input_ids=ids,use_cache=False).logits[:,:-1].float().log_softmax(-1)
                logits=model(input_ids=ids,use_cache=False).logits[:,:-1].float()
                lp=logits.log_softmax(-1)
                values.append(float(F.kl_div(lp.reshape(-1,lp.shape[-1]),tl.reshape(-1,tl.shape[-1]),log_target=True,reduction='batchmean')))
                nll.append(float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1))))
        result[policy]=dict(kl=values,nll=nll,mean_kl=sum(values)/len(values),ppl=math.exp(sum(nll)/len(nll)))
        print(policy,result[policy]['mean_kl'],result[policy]['ppl'],flush=True)
    max_replay_delta=max(abs(x-y) for x,y in zip(result['four_over_six']['kl'],r['fit_baseline_kl']))
    assert max_replay_delta<1e-5
    contrasts={p:dict(kl=paired(v['kl'],result['four_over_six']['kl']),nll=paired(v['nll'],result['four_over_six']['nll'])) for p,v in result.items() if p!='four_over_six'}
    output=dict(status='complete',model=args.model,original_job=332305,baseline_replay_max_delta=max_replay_delta,
                evaluation=result,contrasts=contrasts,predicted_optimizer=r['optimizer'])
    (root/'FIT_AUDIT.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=[f'# Frozen-map fit audit: {args.model}','',
        'Read-only evaluation on the original64 fitting examples. No map, setting or export changed.','',
        '| Policy | Fit mean teacher KL | Fit PPL | Actual ΔKL |','|---|---:|---:|---:|']
    for p,v in result.items():lines.append(f'| {p} | {v["mean_kl"]:.6f} | {v["ppl"]:.6f} | {v["mean_kl"]-result["four_over_six"]["mean_kl"]:+.6f} |')
    (root/'FIT_AUDIT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
