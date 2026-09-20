"""Full-model transfer of the frozen conditional format algorithm."""
import argparse
import json
import math
import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import HfApi
from transformers import AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset
from run_conditional_format import CHECKPOINT,REVISIONS,sha,save,wiki
from quantize.conditional_format import compensation_plan,quantize_compensated,asymmetric_transport
from quantize.quantizer import quant_nvfp4_4over6

POLICIES=('e2m1','dynamic_mse','conditional')


def group_name(name):
    for suffix in ('q_proj','k_proj','v_proj'):
        if name.endswith('.'+suffix):return name.rsplit('.',1)[0]+'.qkv_shared'
    for suffix in ('gate_proj','up_proj'):
        if name.endswith('.'+suffix):return name.rsplit('.',1)[0]+'.gate_up_shared'
    return name


def evaluation_data(tok,asymmetric=False):
    ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',revision=REVISIONS['wiki'],split='test')
    ids=tok('\n\n'.join(ds['text']),return_tensors='pt').input_ids
    start=32 if asymmetric else 0
    result={'wiki':[ids[:,i*512:(i+1)*512].clone() for i in range(start,start+32)]}
    assert all(b.numel()==512 for b in result['wiki'])
    for domain,repo,config in [('math','openai/gsm8k','main'),('code','google-research-datasets/mbpp','full')]:
        ds=load_dataset(repo,config,revision=REVISIONS[domain],split='test')
        texts=[('Question: '+r['question']+'\nAnswer: '+r['answer']) if domain=='math'
               else ('Problem: '+r['text']+'\nCode:\n'+r['code']) for r in ds.select(range(48,80) if asymmetric else range(16,48))]
        result[domain]=[tok(t,return_tensors='pt').input_ids[:,:512] for t in texts]
    return result


def paired(a,b):
    d=[x-y for x,y in zip(a,b)];mean=sum(d)/len(d)
    se=math.sqrt(sum((x-mean)**2 for x in d)/(len(d)-1)/len(d))
    return dict(mean_nll=mean,two_se=2*se,ppl_delta=math.exp(sum(a)/len(a))-math.exp(sum(b)/len(b)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['llama1b','opt350m'],required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--asymmetric',action='store_true');args=ap.parse_args()
    policies=(*POLICIES,'quantized_input_conditional') if args.asymmetric else POLICIES
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    source=CHECKPOINT if args.model=='llama1b' else 'facebook/opt-350m'
    revision=Path(CHECKPOINT).name if args.model=='llama1b' else HfApi().model_info(source).sha
    root=Path(__file__).resolve().parent
    files=['run_conditional_model.py','run_conditional_format.py','quantize/conditional_format.py',
           'quantize/quantizer.py','results/asymmetric_format/PROTOCOL.md' if args.asymmetric else 'results/conditional_format/MODEL_PROTOCOL.md']
    import hashlib
    r=dict(status='running',model=args.model,source=source,revision=revision,asymmetric=args.asymmetric,job_id=os.environ.get('SLURM_JOB_ID'),
        torch_version=torch.__version__,transformers_version=transformers.__version__,dataset_revisions=REVISIONS,
        source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},matrices={},evaluation={})
    save(out,r)
    tok=AutoTokenizer.from_pretrained(source,revision=revision)
    model=AutoModelForCausalLM.from_pretrained(source,revision=revision,torch_dtype=torch.bfloat16,
          device_map='cuda',attn_implementation='eager').eval()
    modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and 'lm_head' not in n}
    assert all(m.weight.shape[0]%8==0 and m.weight.shape[1]%64==0 for m in modules.values())
    original={n:m.weight.detach().cpu().clone() for n,m in modules.items()}
    groups={}
    for n in modules:groups.setdefault(group_name(n),[]).append(n)
    moments={};crosses={};counts={};handles=[]
    def hook(key):
        def collect(module,inputs):
            x=inputs[0].detach().reshape(-1,inputs[0].shape[-1]).double()
            if args.asymmetric:
                xq=quant_nvfp4_4over6(inputs[0],4,16).reshape_as(x).double()
                moments[key]=moments.get(key,0)+xq.T@xq
                crosses[key]=crosses.get(key,0)+x.T@xq
            else:
                moments[key]=moments.get(key,0)+x.T@x
            counts[key]=counts.get(key,0)+len(x)
        return collect
    for key,names in groups.items():handles.append(modules[names[0]].register_forward_pre_hook(hook(key)))
    fit=wiki(tok,'train',32);r['fit_token_sha256']=[sha(b) for b in fit]
    start=time.perf_counter()
    with torch.no_grad():
        for batch in fit:model(input_ids=batch.cuda(),use_cache=False)
    torch.cuda.synchronize();r['calibration_seconds']=time.perf_counter()-start
    for handle in handles:handle.remove()
    assert set(moments)==set(groups)
    all_quantized={p:{} for p in ('four_over_six',*policies)}
    for group_index,(key,names) in enumerate(groups.items()):
        h=moments.pop(key)/counts[key]
        torch.cuda.synchronize();start=time.perf_counter();plan,regularized=compensation_plan(h)
        torch.cuda.synchronize();plan_seconds=time.perf_counter()-start
        transport=asymmetric_transport(crosses.pop(key)/counts[key],regularized,.01*float(h.diag().mean())) if args.asymmetric else None
        hhash=sha(h)
        for n in names:
            w=modules[n].weight.detach();entry=dict(shape=list(w.shape),weight_sha256=sha(w),
                 second_moment_sha256=hhash,shared_group=key,group_plan_seconds=plan_seconds,policies={})
            all_quantized['four_over_six'][n]=quant_nvfp4_4over6(w,4,16).cpu()
            target=w.double()@transport if args.asymmetric else w
            for p in policies:
                center=w if p=='quantized_input_conditional' else target
                rule='conditional' if p=='quantized_input_conditional' else p
                gs=(w.float().abs().amax()/(6*448)).clamp_min(torch.finfo(torch.float32).tiny)
                torch.cuda.synchronize();start=time.perf_counter();q,mask,trace=quantize_compensated(center,plan,rule,global_scale=gs)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                error=q.double()-center.double();loss=float(((error@regularized)*error).sum())
                assert abs(loss-trace['conditional_cost_sum'])<1e-5*max(loss,1e-30)
                all_quantized[p][n]=q.cpu()
                entry['policies'][p]=dict(quantized_sha256=sha(q),selected_tiles=int(mask.sum()),
                    tile_count=mask.numel(),seconds=elapsed,regularized_error=loss,
                    mask_sha256=sha(mask),trace=trace)
            r['matrices'][n]=entry
        save(out,r)
        print(f'QUANTIZED GROUP {group_index+1}/{len(groups)} {key} plan_seconds={plan_seconds:.2f}',flush=True)
        del h,plan,regularized,q,error,mask,transport,target,center
    # Reproducible compensated weights; ordinary fixed-candidate type-map loaders do not apply.
    for p,weights in all_quantized.items():
        torch.save(dict(policy=p,source=source,revision=revision,weights=weights),out/f'{p}_weights.pt')
    r['weights_frozen']=True;save(out,r)
    batches=evaluation_data(tok,args.asymmetric);r['eval_token_sha256']={d:[sha(b) for b in bs] for d,bs in batches.items()}
    def activation_hook(module,inputs):
        return (quant_nvfp4_4over6(inputs[0],4,16),*inputs[1:])
    # Read-only comparison: same weight maps under A16 and fixed A4, no re-selection.
    for activation in (('a4',) if args.asymmetric else ('a16','a4')):
        ah=[m.register_forward_pre_hook(activation_hook) for m in modules.values()] if activation=='a4' else []
        r['evaluation'][activation]={}
        for policy,weights in [('bf16',original),*all_quantized.items()]:
            if policy=='bf16' and activation=='a4':continue
            with torch.no_grad():
                for n,m in modules.items():m.weight.copy_(weights[n])
            result={}
            with torch.no_grad():
                for domain,bs in batches.items():
                    values=[]
                    for batch in bs:
                        ids=batch.cuda();logits=model(input_ids=ids,use_cache=False).logits
                        value=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1))
                        values.append(float(value))
                    result[domain]=dict(nll=values,ppl=math.exp(sum(values)/len(values)))
            r['evaluation'][activation][policy]=result;save(out,r)
            print(f'EVAL {activation} {policy} '+json.dumps({d:v['ppl'] for d,v in result.items()}),flush=True)
        for handle in ah:handle.remove()
    r['contrasts']={}
    for activation,results in r['evaluation'].items():
        r['contrasts'][activation]={d:{p:paired(results['conditional'][d]['nll'],results[p][d]['nll'])
             for p in ('dynamic_mse','e2m1','four_over_six',*(['quantized_input_conditional'] if args.asymmetric else []))} for d in batches}
    r['status']='complete';save(out,r)
    lines=[f'# Frozen full-model conditional selection: {args.model}','',
        'One calibration pass shared by all quantizers. No per-domain tuning or acceptance gate.',
        'Math/code are reference-text NLL, not generated task accuracy. Intervals are descriptive.','',
        '| Activations / domain | Conditional PPL | Dynamic MSE PPL | E2M1 compensated PPL | FourOverSix PPL | Conditional − dynamic ΔNLL ± 2SE |',
        '|---|---:|---:|---:|---:|---:|']
    for a,results in r['evaluation'].items():
        for d in batches:
            c=r['contrasts'][a][d]['dynamic_mse']
            lines.append(f'| {a} / {d} | {results["conditional"][d]["ppl"]:.6f} | {results["dynamic_mse"][d]["ppl"]:.6f} | {results["e2m1"][d]["ppl"]:.6f} | {results["four_over_six"][d]["ppl"]:.6f} | {c["mean_nll"]:+.6f} ± {c["two_se"]:.6f} |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
