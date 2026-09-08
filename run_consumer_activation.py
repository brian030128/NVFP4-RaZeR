"""Frozen runtime format rule across models and fresh text, with no calibration."""
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
from run_conditional_format import CHECKPOINT,REVISIONS,sha,save
from run_conditional_model import paired
from quantize.quantizer import quant_nvfp4_4over6
from quantize.consumer_activation import consumer_energy,quantize_input


def data(tok):
    ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',revision=REVISIONS['wiki'],split='test')
    ids=tok('\n\n'.join(ds['text']),return_tensors='pt').input_ids
    result={'wiki':[ids[:,i*512:(i+1)*512] for i in range(64,96)]}
    assert all(b.numel()==512 for b in result['wiki'])
    for domain,repo,config in [('math','openai/gsm8k','main'),('code','google-research-datasets/mbpp','full')]:
        ds=load_dataset(repo,config,revision=REVISIONS[domain],split='test')
        texts=[('Question: '+r['question']+'\nAnswer: '+r['answer']) if domain=='math'
               else ('Problem: '+r['text']+'\nCode:\n'+r['code']) for r in ds.select(range(80,112))]
        result[domain]=[tok(t,return_tensors='pt').input_ids[:,:512] for t in texts]
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['llama1b','opt350m','qwen06b'],required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args()
    torch.set_num_threads(4)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parent
    source={'llama1b':CHECKPOINT,'opt350m':'facebook/opt-350m','qwen06b':'Qwen/Qwen3-0.6B'}[args.model]
    revision=Path(source).name if Path(source).is_dir() else HfApi().model_info(source).sha
    files=['run_consumer_activation.py','quantize/consumer_activation.py','quantize/quantizer.py',
           'results/consumer_activation/PROTOCOL.md']
    r=dict(status='running',model=args.model,source=source,revision=revision,job_id=os.environ.get('SLURM_JOB_ID'),
        torch_version=torch.__version__,transformers_version=transformers.__version__,
        source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
        metadata={},evaluation={},dataset_revisions=REVISIONS)
    save(out,r)
    model=AutoModelForCausalLM.from_pretrained(source,revision=revision,torch_dtype=torch.bfloat16,
          device_map='cuda',attn_implementation='eager').eval()
    tok=AutoTokenizer.from_pretrained(source,revision=revision)
    modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and 'lm_head' not in n}
    energies={}
    with torch.no_grad():
        for n,m in modules.items():
            m.weight.copy_(quant_nvfp4_4over6(m.weight,4,16))
            energies[n]=consumer_energy(m.weight)
            r['metadata'][n]=dict(weight_sha256=sha(m.weight),energy_sha256=sha(energies[n]),
                                  energy_bytes=energies[n].numel()*4)
    r['metadata_frozen']=True;save(out,r)
    batches=data(tok);r['token_sha256']={d:[sha(b) for b in bs] for d,bs in batches.items()}
    for policy in ('four_over_six','activation_mse','consumer_energy'):
        stats={'tiles':0,'e0m3':0};handles=[]
        def make_hook(name):
            def hook(module,inputs):
                if policy=='four_over_six':q=quant_nvfp4_4over6(inputs[0],4,16)
                else:
                    q,s=quantize_input(inputs[0],energies[name] if policy=='consumer_energy' else None)
                    for key in stats:stats[key]+=s[key]
                return (q,*inputs[1:])
            return hook
        for n,m in modules.items():handles.append(m.register_forward_pre_hook(make_hook(n)))
        result={};start=time.perf_counter()
        with torch.no_grad():
            for domain,bs in batches.items():
                values=[]
                for batch in bs:
                    ids=batch.cuda();logits=model(input_ids=ids,use_cache=False).logits
                    values.append(float(F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1))))
                result[domain]=dict(nll=values,ppl=math.exp(sum(values)/len(values)))
        for handle in handles:handle.remove()
        result['runtime_diagnostics']=dict(**stats,seconds=time.perf_counter()-start)
        r['evaluation'][policy]=result;save(out,r)
        print(f'EVAL {policy} '+json.dumps({d:result[d]['ppl'] for d in batches}),flush=True)
    r['contrasts']={d:{p:paired(r['evaluation']['consumer_energy'][d]['nll'],r['evaluation'][p][d]['nll'])
        for p in ('four_over_six','activation_mse')} for d in batches}
    r['status']='complete';save(out,r)
    lines=[f'# Consumer-aware activation formats: {args.model}','',
        'All weights fixed at FourOverSix. No calibration, labels or test losses enter format decisions.',
        'Reference-text losses only; runtime timings use unfused fake quantization.','',
        '| Domain | Baseline PPL | Activation-MSE PPL | Consumer PPL | Consumer − baseline ΔNLL ± 2SE |',
        '|---|---:|---:|---:|---:|']
    for d in batches:
        c=r['contrasts'][d]['four_over_six'];e=r['evaluation']
        lines.append(f'| {d} | {e["four_over_six"][d]["ppl"]:.6f} | {e["activation_mse"][d]["ppl"]:.6f} | {e["consumer_energy"][d]["ppl"]:.6f} | {c["mean_nll"]:+.6f} ± {c["two_se"]:.6f} |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
