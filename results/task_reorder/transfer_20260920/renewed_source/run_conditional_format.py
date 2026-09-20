"""Frozen single-calibration-pass comparison of conditional format costs."""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
from quantize.conditional_format import compensation_plan,quantize_compensated
from quantize.quantizer import quant_nvfp4_4over6

CHECKPOINT='/work/u4320956/hf/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6'
POLICIES=('e2m1','static_mse','dynamic_mse','conditional')
REVISIONS={'wiki':'b08601e04326c79dfdd32d625aee71d232d685c3',
           'math':'740312add88f781978c0658806c59bc2815b9866',
           'code':'4bb6404fdc6cacfda99d4ac4205087b89d32030c'}


def sha(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def save(out,r):
    p=out/'report.tmp';p.write_text(json.dumps(r,indent=2)+'\n');p.replace(out/'report.json')


def wiki(tok,split,count):
    ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',revision=REVISIONS['wiki'],split=split)
    ids=tok('\n\n'.join(ds['text']),return_tensors='pt').input_ids
    assert ids.shape[1]>=count*512
    return [ids[:,i*512:(i+1)*512].clone() for i in range(count)]


def stress(tok,kind):
    if kind=='math':
        ds=load_dataset('openai/gsm8k','main',revision=REVISIONS[kind],split='test')
        texts=['Question: '+r['question']+'\nAnswer: '+r['answer'] for r in ds.select(range(16))]
    else:
        ds=load_dataset('google-research-datasets/mbpp','full',revision=REVISIONS[kind],split='test')
        texts=['Problem: '+r['text']+'\nCode:\n'+r['code'] for r in ds.select(range(16))]
    return [tok(t,return_tensors='pt').input_ids[:,:512] for t in texts]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parent
    sources=['run_conditional_format.py','quantize/conditional_format.py','tests/test_conditional_format.py',
             'results/conditional_format/PROTOCOL.md']
    r=dict(status='running',job_id=os.environ.get('SLURM_JOB_ID'),checkpoint=CHECKPOINT,
        torch_version=torch.__version__,transformers_version=transformers.__version__,
        source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in sources},
        revisions=REVISIONS,matrices={},data={})
    save(out,r)
    tok=AutoTokenizer.from_pretrained(CHECKPOINT,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,torch_dtype=torch.bfloat16,
         device_map='cuda',local_files_only=True,attn_implementation='eager').eval()
    fit=wiki(tok,'train',32)
    r['data']['fit']=[sha(b) for b in fit]
    names=[f'model.layers.{layer}.self_attn.{p}_proj' for layer in (0,8,15) for p in ('q','k','v')]
    modules=dict(model.named_modules());moments={};counts={};handles=[]
    def fit_hook(layer):
        def hook(module,inputs):
            x=inputs[0].detach().reshape(-1,inputs[0].shape[-1]).double()
            moments[layer]=moments.get(layer,0)+x.T@x
            counts[layer]=counts.get(layer,0)+len(x)
        return hook
    for layer in (0,8,15):handles.append(modules[f'model.layers.{layer}.self_attn.q_proj'].register_forward_pre_hook(fit_hook(layer)))
    print('ONE CALIBRATION PASS: 32 Wiki training windows',flush=True)
    with torch.no_grad():
        for batch in fit:model(input_ids=batch.cuda(),use_cache=False)
    for handle in handles:handle.remove()
    policies={}
    for layer in (0,8,15):
        h=moments[layer]/counts[layer]
        torch.cuda.synchronize();start=time.perf_counter();plan,regularized=compensation_plan(h)
        torch.cuda.synchronize();plan_seconds=time.perf_counter()-start
        for proj in ('q','k','v'):
            name=f'model.layers.{layer}.self_attn.{proj}_proj';w=modules[name].weight.detach()
            entry=dict(weight_sha256=sha(w),shape=list(w.shape),second_moment_sha256=sha(h),
                       plan_seconds=plan_seconds,policies={})
            policies[name]={}
            for policy in POLICIES:
                torch.cuda.synchronize();start=time.perf_counter()
                q,mask,trace=quantize_compensated(w,plan,policy)
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                error=q.double()-w.double();loss=float(((error@h)*error).sum())
                regloss=float(((error@regularized)*error).sum())
                assert abs(regloss-trace['conditional_cost_sum'])<1e-6*max(regloss,1e-30)
                entry['policies'][policy]=dict(fit_output_error=loss,regularized_error=regloss,
                     quantization_seconds=seconds,selected_tiles=int(mask.sum()),tile_count=mask.numel(),
                     quantized_sha256=sha(q),trace=trace)
                torch.save(dict(weight=q.cpu(),mask=mask.cpu(),baseline='FourOverSix',policy=policy),
                           out/f'{name}_{policy}.pt')
                policies[name][policy]=q
            r['matrices'][name]=entry;save(out,r)
            print(f'QUANTIZED {name} fit conditional/dynamic={entry["policies"]["conditional"]["fit_output_error"]/entry["policies"]["dynamic_mse"]["fit_output_error"]:.6f}',flush=True)
        del plan,regularized,h
    # All quantized weights and formats frozen before any held-out data is loaded.
    captured={}
    def eval_hook(name):
        def hook(module,inputs):captured[name]=inputs[0].detach().reshape(-1,inputs[0].shape[-1])
        return hook
    handles=[modules[n].register_forward_pre_hook(eval_hook(n)) for n in names]
    r['evaluation']={}
    for domain in ('wiki','math','code'):
        batches=wiki(tok,'validation',16) if domain=='wiki' else stress(tok,domain)
        r['data'][domain]=[sha(b) for b in batches]
        result={n:{p:[] for p in POLICIES} for n in names}
        with torch.no_grad():
            for batch in batches:
                model(input_ids=batch.cuda(),use_cache=False)
                for n in names:
                    x=captured[n].double();w=modules[n].weight.detach().double()
                    for p,q in policies[n].items():
                        # Relative layer output error is evaluated on identical pristine inputs.
                        error=x@(q.double()-w).T
                        result[n][p].append(float(error.square().sum()/len(x)))
        r['evaluation'][domain]=result;save(out,r)
    for handle in handles:handle.remove()
    summary={};ratios=[]
    for domain,result in r['evaluation'].items():
        rs=[sum(v['conditional'])/sum(v['dynamic_mse']) for v in result.values()]
        ratios.extend(rs)
        summary[domain]=dict(geometric_mean_conditional_over_dynamic=math.exp(sum(map(math.log,rs))/len(rs)),
                            matrices_improved=sum(a<1 for a in rs),matrix_ratios=rs)
    r['summary']=summary
    r['close_enough_for_next_stage']=all(s['geometric_mean_conditional_over_dynamic']<=.995 for s in summary.values()) and max(ratios)<=1.01
    r['status']='complete';save(out,r)
    lines=['# Conditional format-selection mechanism panel','',
        'One Wiki calibration pass shared by all comparators. Nine matrices; three held-out input domains.',
        'Layer reconstruction only, on pristine model trajectories. No downstream accuracy claim.','',
        '```json',json.dumps(dict(summary=summary,close_enough=r['close_enough_for_next_stage']),indent=2),'```','',
        '| Matrix | Fit conditional / dynamic MSE | Wiki | Math | Code |',
        '|---|---:|---:|---:|---:|']
    for n,m in r['matrices'].items():
        f=m['policies'];cols=[f['conditional']['fit_output_error']/f['dynamic_mse']['fit_output_error']]
        cols +=[sum(r['evaluation'][d][n]['conditional'])/sum(r['evaluation'][d][n]['dynamic_mse']) for d in ('wiki','math','code')]
        lines.append('| '+n+' | '+' | '.join(f'{v:.6f}' for v in cols)+' |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summary=summary,close_enough=r['close_enough_for_next_stage']),indent=2),flush=True)


if __name__=='__main__':main()
