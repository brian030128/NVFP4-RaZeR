"""Replay frozen fixed-256 maps under the paper's 2048-token PPL protocol."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantize import QuantConfig
from quantize.quantizer import quant_nvfp4_4over6, quant_mix_4_6
from quantize.interacting_format import apply_mask
from run_baseline_protocol_audit import data
from run_c4_frozen import digest_file
from run_conditional_format import sha


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--case',type=int,required=True)
    ap.add_argument('--repair-wiki',action='store_true'); args=ap.parse_args()
    case=json.loads((args.root/'cases.json').read_text())[args.case]
    old=Path(case['calibration']); prior=json.loads((old/'report.json').read_text())
    bundle=json.loads((old/'maps.json').read_text()); policy=case['policy']
    assert prior['status']=='complete' and prior['maps_frozen'] and prior['source_weights_verified']
    assert not prior['uses_c4_calibration'] and not prior['uses_wiki_calibration']
    assert bundle['baseline']=='FourOverSix' and bundle['alternative']=='E0M3 alpha1'
    assert bundle['type_block']==[8,64]
    assert digest_file(old/'maps.json')==prior['map_sha256']
    assert policy=='four_over_six' or policy.startswith('fixed256_')
    assert transformers.__version__==prior['transformers_version']
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(0)
    out=args.root/case['id']; previous=None
    if args.repair_wiki:
        previous=json.loads((out/'report.json').read_text())
        assert not previous.get('wiki_use_cache',False)
        assert previous['source_weights_verified'] and 'c4' in previous['evaluation']
        assert previous['map_sha256']==prior['map_sha256']
        (out/'report_cache_disabled.json').write_text(json.dumps(previous,indent=2)+'\n')
    else:
        out.mkdir(exist_ok=False)
    report=dict(status='running',case=case,source=prior['source'],revision=prior['revision'],
        calibration_report_sha256=digest_file(old/'report.json'),map_sha256=prior['map_sha256'],
        job_id=os.environ['SLURM_JOB_ID'],step_id=os.environ.get('SLURM_STEP_ID'),
        account=os.environ.get('SLURM_JOB_ACCOUNT'),gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__,transformers_version=transformers.__version__,
        evaluation={},length=2048,activation='FourOverSix tensor-wide factor',
        wiki_use_cache=True,c4_use_cache=False,
        qwen_o_proj_quantized=True,kv_quantization=False,recalibration=False,
        calibration_sources=['OpenWebMath','CodeParrot'],uses_c4_calibration=False,uses_wiki_calibration=False,
        source_sha256={f:digest_file(f) for f in ('run_fixed256_paper_eval.py','run_baseline_protocol_audit.py',
            'quantize/quantizer.py','quantize/interacting_format.py')})
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        target=prior['model']=='qwen27b'
        if target:
            from transformers import Qwen3_5ForConditionalGeneration
            model,loading=Qwen3_5ForConditionalGeneration.from_pretrained(case['model_path'],
                dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda',output_loading_info=True)
            assert not loading['missing_keys'] and not loading.get('mismatched_keys') and not loading.get('error_msgs')
            modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)
                     and 'language_model' in n and 'head' not in n}
        else:
            model=AutoModelForCausalLM.from_pretrained(case['model_path'],torch_dtype=torch.bfloat16,
                attn_implementation='sdpa',device_map='cuda')
            modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)
                     and m is not model.get_output_embeddings()}
        model.eval().requires_grad_(False)
        assert list(modules)==list(prior['matrices'])
        report['attention_backend']=model.config._attn_implementation
        report['model_class']=type(model).__name__
        report['quantized_weight_sha256']={}; selected=0
        for n,m in modules.items():
            assert sha(m.weight)==prior['matrices'][n]['source_sha256'],n
            b=quant_nvfp4_4over6(m.weight,4,16)
            indices=[] if policy=='four_over_six' else bundle['maps'][policy][n]
            if indices:
                shape=(m.weight.shape[0]//8,m.weight.shape[1]//64)
                mask=torch.zeros(shape[0]*shape[1],dtype=torch.bool,device=m.weight.device)
                assert len(indices)==len(set(indices)) and all(0<=i<mask.numel() for i in indices)
                mask[indices]=True
                a=quant_mix_4_6(m.weight,4,16,type_block=(8,64),clip='a1',elect='always')
                b=apply_mask(b,a,mask.reshape(shape)); selected+=len(indices)
                del a,mask
            m.weight.copy_(b); report['quantized_weight_sha256'][n]=sha(m.weight)
            del b
        assert selected==(0 if policy=='four_over_six' else 256)
        report['selected_e0m3_blocks']=selected; report['source_weights_verified']=True
        tok=AutoTokenizer.from_pretrained(case['model_path'])
        batches,report['data']=data(tok,prior,2048)
        excluded={d['document_sha256'] for meta in prior['fit'].values() for d in meta['documents']}
        assert excluded.isdisjoint(d['document_sha256'] for d in report['data']['c4_paper']['documents'])
        if not target:
            model_name={'qwen4b':'qwen3-4b','llama8b':'llama-3.1-8b'}[prior['model']]
            reference=json.loads(Path(f'results/released_reproduction/job_335297/{model_name}_four_over_six_w4a4/report.json').read_text())
            assert report['data']['wiki']['token_sha256']+report['data']['c4_paper']['token_sha256']==[w['input_sha256'] for w in reference['windows']]
            report['released_inputs_identical']=True
        def act(module,inputs): return (quant_nvfp4_4over6(inputs[0],4,16),*inputs[1:])
        handles=[m.register_forward_pre_hook(act) for m in modules.values()]
        report['activation_quantized_modules']=list(modules)
        assert not target or any(n.endswith('self_attn.o_proj') for n in modules)
        if not target and policy=='four_over_six':
            if prior['model']=='llama8b':
                from models.qmodule_llama import QuantLlamaForCausalLM as Wrapper
            else:
                from models.qmodule_qwen3 import QuantQwen3ForCausalLM as Wrapper
            qc=QuantConfig(a_bits=4,a_dtype='nvfp4_4over6',a_groupsize=16)
            with torch.device('meta'): wrapper=Wrapper(copy.deepcopy(model.config),qc)
            wrapper.load_state_dict(model.state_dict(),assign=True)
            for name,buffer in model.named_buffers():
                parent,_,leaf=name.rpartition('.'); wrapper.get_submodule(parent)._buffers[leaf]=buffer
            wrapper.eval().requires_grad_(False)
            ids=batches['wiki'][0].cuda()
            a=model(ids,use_cache=True).logits; b=wrapper(ids,use_cache=True).logits
            report['corrected_wrapper_equivalence']=dict(equal=torch.equal(a,b),max_abs_error=float((a-b).abs().max()))
            assert torch.equal(a,b),report['corrected_wrapper_equivalence']
            del a,b,wrapper
        save()
        for domain,bs in batches.items():
            if previous is not None and domain=='c4_paper':
                assert previous['data'][domain]['token_sha256']==report['data'][domain]['token_sha256']
                assert previous['quantized_weight_sha256']==report['quantized_weight_sha256']
                assert previous['activation_quantized_modules']==report['activation_quantized_modules']
                assert previous['torch_version']==report['torch_version']
                assert previous['transformers_version']==report['transformers_version']
                check_ids=bs[0].cuda(); check_logits=model(input_ids=check_ids,use_cache=False).logits
                check_nll=float(F.cross_entropy(check_logits[:,:-1].float().reshape(-1,check_logits.shape[-1]),check_ids[:,1:].reshape(-1)))
                first_error=abs(check_nll-previous['evaluation']['c4']['nll'][0])
                assert first_error<=1e-6,first_error
                del check_ids,check_logits
                report['evaluation']['c4']=previous['evaluation']['c4']
                report['c4_reused_after_cache_fix']=dict(source_report_sha256=digest_file(out/'report_cache_disabled.json'),
                    inputs_weights_and_scope_identical=True,first_window_nll_absolute_error=first_error)
                save(); continue
            losses=[]
            for i,ids in enumerate(bs):
                ids=ids.cuda(); logits=model(input_ids=ids,use_cache=domain=='wiki').logits
                value=float(F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1)))
                assert math.isfinite(value); losses.append(value); del logits
                if (i+1)%128==0: print(f'EVAL {case["id"]} {domain} {i+1}/{len(bs)}',flush=True)
            # Same FP32 aggregation as released run_ppl.py, not Python float64 exp.
            ppl=float(torch.exp((torch.tensor(losses,dtype=torch.float32)*2048).sum()/(len(losses)*2048)))
            report['evaluation']['c4' if domain=='c4_paper' else domain]=dict(ppl=ppl,nll=losses,
                windows=len(losses),scored_tokens=len(losses)*2047)
            save(); print(f'PPL {case["id"]} {domain} {ppl:.6f}',flush=True)
        if prior['model']=='llama8b' and policy=='four_over_six':
            assert report['evaluation']['wiki']['ppl']==reference['ppl']['wikitext']
            assert report['evaluation']['c4']['ppl']==reference['ppl']['c4']
            report['released_baseline_exact']=True
        for h in handles: h.remove()
        assert digest_file(old/'maps.json')==report['map_sha256']
        assert digest_file(old/'report.json')==report['calibration_report_sha256']
        report['maps_unchanged']=True; report['status']='complete'; save()
    except BaseException as exc:
        report['status']='failed'; report['error']=repr(exc); save(); raise


if __name__=='__main__': main()
