"""One shared score pass, one whole-network discrete objective, held-out transfer."""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import HfApi
from run_conditional_format import CHECKPOINT,REVISIONS,sha,save
from run_conditional_model import paired
from quantize.quantizer import quant_nvfp4_4over6,quant_mix_4_6
from quantize.predictive_format import predictive_election
from quantize.interacting_format import apply_mask

C4_REVISION='1588ec454efa1a09f29cd18ddd04fe05fc8653a2'


def fit_data(tok):
    path='en/c4-train.00003-of-01024.json.gz'
    stream=load_dataset('allenai/c4',revision=C4_REVISION,data_files={'train':path},split='train',streaming=True)
    batches=[];docs=[];seen=set();rng=random.Random(20260920)
    for row in stream:
        digest=hashlib.sha256(row['text'].encode()).hexdigest()
        if digest in seen:continue
        seen.add(digest);ids=tok(row['text'],return_tensors='pt').input_ids
        if ids.shape[1]<512:continue
        start=rng.randrange(ids.shape[1]-512+1)
        batches.append(ids[:,start:start+512].clone());docs.append(dict(document_sha256=digest,offset=start))
        if len(batches)==64:return batches,dict(revision=C4_REVISION,path=path,documents=docs,token_sha256=[sha(b) for b in batches])
    raise RuntimeError('Insufficient calibration documents')


def data(tok):
    ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',revision=REVISIONS['wiki'],split='test')
    ids=tok('\n\n'.join(ds['text']),return_tensors='pt').input_ids
    result={'wiki':[ids[:,i*512:(i+1)*512] for i in range(288,320)]}
    assert all(b.numel()==512 for b in result['wiki'])
    for domain,repo,config in [('math','openai/gsm8k','main'),('code','google-research-datasets/mbpp','full')]:
        ds=load_dataset(repo,config,revision=REVISIONS[domain],split='test')
        texts=[('Question: '+r['question']+'\nAnswer: '+r['answer']) if domain=='math'
               else ('Problem: '+r['text']+'\nCode:\n'+r['code']) for r in ds.select(range(304,336))]
        result[domain]=[tok(t,return_tensors='pt').input_ids[:,:512] for t in texts]
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['llama1b','opt350m','qwen06b'],required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parent
    source={'llama1b':CHECKPOINT,'opt350m':'facebook/opt-350m','qwen06b':'Qwen/Qwen3-0.6B'}[args.model]
    revision=Path(source).name if Path(source).is_dir() else HfApi().model_info(source).sha
    files=['run_predictive_format.py','quantize/predictive_format.py','quantize/quantizer.py',
           'tests/test_predictive_format.py','results/predictive_format/PROTOCOL.md']
    r=dict(status='running',model=args.model,source=source,revision=revision,job_id=os.environ.get('SLURM_JOB_ID'),
        torch_version=torch.__version__,transformers_version=transformers.__version__,
        source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
        matrices={},evaluation={},dataset_revisions=REVISIONS)
    save(out,r)
    model=AutoModelForCausalLM.from_pretrained(source,revision=revision,torch_dtype=torch.bfloat16,
          device_map='cuda',attn_implementation='eager').eval().requires_grad_(False)
    teacher=AutoModelForCausalLM.from_pretrained(source,revision=revision,torch_dtype=torch.bfloat16,
          device_map='cuda',attn_implementation='eager').eval().requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(source,revision=revision)
    modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and 'lm_head' not in n}
    base={};alt={}
    with torch.no_grad():
        for n,m in modules.items():
            w=m.weight.detach();r['matrices'][n]=dict(shape=list(w.shape),source_sha256=sha(w))
            base[n]=quant_nvfp4_4over6(w,4,16)
            alt[n]=quant_mix_4_6(w,4,16,type_block=(8,64),clip='a1',elect='always')
            m.weight.copy_(base[n])
    scoring=True
    def act(module,inputs):
        x=inputs[0];q=quant_nvfp4_4over6(x.detach(),4,16)
        return (q+(x-x.detach()) if scoring else q,*inputs[1:])
    ah=[m.register_forward_pre_hook(act) for m in modules.values()]
    scores={n:[] for n in modules};fisher_scores={n:[] for n in modules};phase='kl'
    def make_hook(n):
        def forward(module,inputs,output):
            x=inputs[0].detach().reshape(-1,inputs[0].shape[-1])
            def backward(dy):
                grad=dy.detach().reshape(-1,dy.shape[-1]).float().T@x.float()
                score=(grad*(alt[n].float()-base[n].float())).reshape(grad.shape[0]//8,8,grad.shape[1]//64,64).sum((1,3))
                assert torch.isfinite(score).all()
                (scores if phase=='kl' else fisher_scores)[n].append(score.flatten().cpu())
            output.register_hook(backward)
        return forward
    handles=[m.register_forward_hook(make_hook(n)) for n,m in modules.items()]
    fit,r['fit']=fit_data(tok);fit_losses=[];start=time.perf_counter()
    generator=torch.Generator(device='cuda').manual_seed(20260922)
    for i,batch in enumerate(fit):
        ids=batch.cuda();embeds=model.get_input_embeddings()(ids).detach().requires_grad_()
        with torch.no_grad():
            teacher_log=teacher(input_ids=ids,use_cache=False).logits[:,:-1].float().log_softmax(-1).reshape(-1,teacher.config.vocab_size)
        logits=model(inputs_embeds=embeds,use_cache=False).logits
        logprob=logits[:,:-1].float().reshape(-1,logits.shape[-1]).log_softmax(-1)
        loss=F.kl_div(logprob,teacher_log,reduction='batchmean',log_target=True)
        fit_losses.append(float(loss));phase='kl';loss.backward(retain_graph=True)
        with torch.no_grad():labels=torch.multinomial(logprob.detach().exp(),1,generator=generator).squeeze(-1)
        phase='fisher';floss=F.nll_loss(logprob,labels);floss.backward()
        del embeds,logits,loss,floss,labels,teacher_log,logprob
        print(f'SCORE PASS {i+1}/64',flush=True)
    for handle in handles:handle.remove()
    scoring=False;r['fit_seconds']=time.perf_counter()-start;r['fit_baseline_kl']=fit_losses
    del teacher
    names=list(modules);assert all(len(scores[n])==64 for n in names)
    matrix=torch.cat([torch.stack(scores[n]) for n in names],dim=1)
    fmatrix=torch.cat([torch.stack(fisher_scores[n]) for n in names],dim=1)
    del scores,fisher_scores
    torch.save(dict(scores=matrix,fisher_scores=fmatrix,names=names,shapes={n:list(modules[n].weight.shape) for n in names}),out/'scores.pt')
    masks,stats=predictive_election(matrix.cuda(),fmatrix.cuda(),511);r['optimizer']=stats
    del matrix,fmatrix
    policies=('four_over_six','fixed_budget','diagonal_fisher','unshrunk','predictive')
    maps={p:{} for p in masks};offset=0
    for n in names:
        o,k=modules[n].weight.shape;count=o//8*(k//64)
        for p,flat in masks.items():maps[p][n]=flat[offset:offset+count].reshape(o//8,k//64).cpu()
        offset+=count
    r['selected_tiles']={p:sum(int(m.sum()) for m in ms.values()) for p,ms in maps.items()}
    torch.save(dict(source=source,revision=revision,maps=maps,baseline='FourOverSix',alternative='E0M3 alpha1',type_block=(8,64)),out/'maps.pt')
    r['maps_frozen']=True;save(out,r)
    print('FROZEN '+json.dumps(dict(selected=r['selected_tiles'],rho=stats['predictive']['rho'],objective=stats['predictive']['objective'],converged=stats['predictive']['converged'])),flush=True)
    batches=data(tok);r['eval_token_sha256']={d:[sha(b) for b in bs] for d,bs in batches.items()}
    for p in policies:
        with torch.no_grad():
            for n,m in modules.items():m.weight.copy_(base[n] if p=='four_over_six' else apply_mask(base[n],alt[n],maps[p][n].cuda()))
            result={}
            for d,bs in batches.items():
                values=[]
                for batch in bs:
                    ids=batch.cuda();logits=model(input_ids=ids,use_cache=False).logits
                    values.append(float(F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1))))
                result[d]=dict(nll=values,ppl=math.exp(sum(values)/len(values)))
        r['evaluation'][p]=result;save(out,r)
        print(f'EVAL {p} '+json.dumps({d:v['ppl'] for d,v in result.items()}),flush=True)
    for handle in ah:handle.remove()
    r['contrasts']={d:{p:paired(r['evaluation']['predictive'][d]['nll'],r['evaluation'][p][d]['nll'])
        for p in policies[:-1]} for d in batches}
    r['status']='complete';save(out,r)
    lines=[f'# Predictive-curvature teacher tile election: {args.model}','',
        'One shared score pass; frozen type-only maps. No candidate-loss backtracking.','',
        '| Domain | FourOverSix | Fixed budget | Diagonal Fisher | Unshrunk | Predictive | Predictive − baseline ΔNLL ±2SE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for d in batches:
        c=r['contrasts'][d]['four_over_six'];e=r['evaluation']
        lines.append('| '+d+' | '+' | '.join(f'{e[p][d]["ppl"]:.6f}' for p in policies)+f' | {c["mean_nll"]:+.6f} ±{c["two_se"]:.6f} |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
