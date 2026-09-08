"""Frozen type-only coordinate descent transfer panel."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import HfApi
from run_conditional_format import CHECKPOINT,REVISIONS,sha,save,wiki
from run_conditional_model import group_name,paired
from quantize.quantizer import quant_nvfp4_4over6
from quantize.interacting_format import select_formats


def data(tok):
    ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',revision=REVISIONS['wiki'],split='test')
    ids=tok('\n\n'.join(ds['text']),return_tensors='pt').input_ids
    result={'wiki':[ids[:,i*512:(i+1)*512] for i in range(96,128)]}
    assert all(b.numel()==512 for b in result['wiki'])
    for domain,repo,config in [('math','openai/gsm8k','main'),('code','google-research-datasets/mbpp','full')]:
        ds=load_dataset(repo,config,revision=REVISIONS[domain],split='test')
        texts=[('Question: '+r['question']+'\nAnswer: '+r['answer']) if domain=='math'
               else ('Problem: '+r['text']+'\nCode:\n'+r['code']) for r in ds.select(range(112,144))]
        result[domain]=[tok(t,return_tensors='pt').input_ids[:,:512] for t in texts]
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['llama1b','opt350m','qwen06b'],required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parent
    source={'llama1b':CHECKPOINT,'opt350m':'facebook/opt-350m','qwen06b':'Qwen/Qwen3-0.6B'}[args.model]
    revision=Path(source).name if Path(source).is_dir() else HfApi().model_info(source).sha
    files=['run_interacting_format.py','quantize/interacting_format.py','quantize/conditional_format.py',
           'quantize/quantizer.py','tests/test_interacting_format.py','results/interacting_format/PROTOCOL.md']
    r=dict(status='running',model=args.model,source=source,revision=revision,job_id=os.environ.get('SLURM_JOB_ID'),
        torch_version=torch.__version__,transformers_version=transformers.__version__,
        source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
        matrices={},evaluation={},dataset_revisions=REVISIONS)
    save(out,r)
    model=AutoModelForCausalLM.from_pretrained(source,revision=revision,torch_dtype=torch.bfloat16,
          device_map='cuda',attn_implementation='eager').eval()
    tok=AutoTokenizer.from_pretrained(source,revision=revision)
    modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and 'lm_head' not in n}
    groups={}
    for n in modules:groups.setdefault(group_name(n),[]).append(n)
    moments={};counts={}
    def make_hook(key):
        def hook(module,inputs):
            x=quant_nvfp4_4over6(inputs[0],4,16).reshape(-1,inputs[0].shape[-1]).double()
            moments[key]=moments.get(key,0)+x.T@x;counts[key]=counts.get(key,0)+len(x)
        return hook
    handles=[modules[ns[0]].register_forward_pre_hook(make_hook(k)) for k,ns in groups.items()]
    fit=wiki(tok,'train',32);r['fit_token_sha256']=[sha(b) for b in fit]
    start=time.perf_counter()
    with torch.no_grad():
        for batch in fit:model(input_ids=batch.cuda(),use_cache=False)
    for handle in handles:handle.remove()
    r['calibration_seconds']=time.perf_counter()-start
    policies=('four_over_six','weight_mse','independent','interacting')
    all_weights={p:{} for p in policies};all_masks={p:{} for p in policies[1:]}
    for gi,(key,names) in enumerate(groups.items()):
        h=moments.pop(key)/counts[key];hhash=sha(h)
        for n in names:
            start=time.perf_counter();w=modules[n].weight.detach();qs,masks,stats=select_formats(w,h)
            r['matrices'][n]=dict(shape=list(w.shape),source_sha256=sha(w),h_sha256=hhash,
                seconds=time.perf_counter()-start,stats=stats,
                selected_tiles={p:int(m.sum()) for p,m in masks.items()},tile_count=masks['interacting'].numel(),
                quantized_sha256={p:sha(q) for p,q in qs.items()})
            for p,q in qs.items():all_weights[p][n]=q.cpu()
            for p,m in masks.items():all_masks[p][n]=m.cpu()
        save(out,r)
        print(f'QUANTIZED GROUP {gi+1}/{len(groups)} {key}',flush=True)
        del h,qs,masks
    # Candidate type maps are replayable from the original weights; save metadata.
    torch.save(dict(source=source,revision=revision,masks=all_masks,
        baseline='canonical FourOverSix',alternative='E0M3 alpha1',type_block=(8,64)),out/'type_maps.pt')
    r['maps_frozen']=True;save(out,r)
    batches=data(tok);r['eval_token_sha256']={d:[sha(b) for b in bs] for d,bs in batches.items()}
    def act(module,inputs):return (quant_nvfp4_4over6(inputs[0],4,16),*inputs[1:])
    handles=[m.register_forward_pre_hook(act) for m in modules.values()]
    for p in policies:
        with torch.no_grad():
            for n,m in modules.items():m.weight.copy_(all_weights[p][n])
            result={}
            for domain,bs in batches.items():
                values=[]
                for batch in bs:
                    ids=batch.cuda();logits=model(input_ids=ids,use_cache=False).logits
                    values.append(float(F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1))))
                result[domain]=dict(nll=values,ppl=math.exp(sum(values)/len(values)))
        r['evaluation'][p]=result;save(out,r)
        print(f'EVAL {p} '+json.dumps({d:result[d]['ppl'] for d in batches}),flush=True)
    for handle in handles:handle.remove()
    r['contrasts']={d:{p:paired(r['evaluation']['interacting'][d]['nll'],r['evaluation'][p][d]['nll'])
        for p in policies[:-1]} for d in batches}
    r['status']='complete';save(out,r)
    lines=[f'# Interaction-aware tile formats: {args.model}','',
        'All candidates fixed at original weights. One shared fit pass; fresh evaluation.',
        'Reference-text loss only; descriptive paired intervals.','',
        '| Domain | FourOverSix PPL | Weight-MSE PPL | Independent PPL | Interacting PPL | Interacting − baseline ΔNLL ± 2SE |',
        '|---|---:|---:|---:|---:|---:|']
    for d in batches:
        c=r['contrasts'][d]['four_over_six'];e=r['evaluation']
        lines.append('| '+d+' | '+' | '.join(f'{e[p][d]["ppl"]:.6f}' for p in policies)+f' | {c["mean_nll"]:+.6f} ± {c["two_se"]:.6f} |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
