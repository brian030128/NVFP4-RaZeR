"""Fixed-input activation-bound feasibility experiment, with no calibration."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from quantize.activation_guard import (activation_candidates,apply_map,build_geometry,
    guarded_map,proxy,upper_change)
from quantize.quantizer import quant_nvfp4_4over6

CHECKPOINT='/work/u4320956/hf/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6'
PROMPTS={
    'prose':'A village library opened beside the railway station. Every morning the librarian arranged the returned books, checked the weather, and prepared a reading table for visitors. Describe how a shared library can support a small community.',
    'code':'def merge_sorted(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    return result + left[i:] + right[j:]\n',
    'math':'Let a_n = 3n + 2. Find the sum of the first twenty terms. Explain why the average of the first and last terms multiplied by the number of terms gives the sum. Then solve 2x + 7 = 31 and verify the result by substitution.',
    'multilingual':'今天早上，我們走到河邊觀察水位，並記錄天氣的變化。La biblioteca abre cada mañana y recibe a estudiantes de distintas edades. Les visiteurs peuvent lire, discuter et apprendre ensemble. Translate the descriptions and compare their subjects.',
}


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def save(out,report):
    temp=out/'report.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out/'report.json')


def exact_loss(x,q,w):
    error=q.double()-x.double()
    per_token=(error@w.double().T).square().sum(-1)
    return float(per_token.sum()),per_token


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    torch.set_num_threads(4)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent
    sources=['quantize/activation_guard.py','run_activation_guard.py','tests/test_activation_guard.py',
             'quantize/quantizer.py','results/activation_guard/PROTOCOL.md']
    report=dict(status='running',job_id=os.environ.get('SLURM_JOB_ID'),checkpoint=CHECKPOINT,
        torch_version=torch.__version__,transformers_version=transformers.__version__,
        source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in sources},
        metadata={},prompts={},packets=[])
    save(out,report)
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,torch_dtype=torch.bfloat16,
          device_map='cuda',local_files_only=True,attn_implementation='eager').eval()
    tok=AutoTokenizer.from_pretrained(CHECKPOINT,local_files_only=True)
    with torch.no_grad():
        for name,module in model.named_modules():
            if isinstance(module,torch.nn.Linear) and 'head' not in name:
                module.weight.copy_(quant_nvfp4_4over6(module.weight,4,16))
    names=[f'model.layers.{i}.self_attn.q_proj' for i in (0,8,15)]
    modules=dict(model.named_modules());geometries={};weights={}
    # Construct and hash all metadata BEFORE processing any prompt.
    for name in names:
        print(f'METADATA {name}',flush=True)
        weight=modules[name].weight.detach()
        torch.cuda.synchronize();start=time.perf_counter()
        geo,info=build_geometry(weight,rank=16)
        torch.cuda.synchronize();info['build_seconds']=time.perf_counter()-start
        info['weight_sha256']=sha(weight);info['scale_sha256']=sha(geo.scale);info['factor_sha256']=sha(geo.factor)
        info['nvfp4_weight_bytes_excluding_tensor_scale']=weight.numel()*.5+weight.numel()/16
        geometries[name]=geo;weights[name]=weight
        report['metadata'][name]=info;save(out,report)
    captured={}
    handles=[]
    def make_hook(name):
        def hook(module,inputs):
            captured[name]=inputs[0].detach().reshape(-1,inputs[0].shape[-1])[-16:].clone()
        return hook
    for name in names:handles.append(modules[name].register_forward_pre_hook(make_hook(name)))
    for label,text in PROMPTS.items():
        ids=tok((text+'\n')*32,return_tensors='pt',add_special_tokens=True).input_ids[:,:256].cuda()
        assert ids.shape[1]==256
        report['prompts'][label]=dict(text=text,tokens=256,token_sha256=sha(ids),packet_positions=[240,256])
        with torch.no_grad():model(input_ids=ids,use_cache=False)
        for name in names:
            x=captured[name];w=weights[name];geo=geometries[name]
            start=time.perf_counter();base,alt=activation_candidates(x);torch.cuda.synchronize()
            candidate_seconds=time.perf_counter()-start
            torch.cuda.synchronize();start=time.perf_counter()
            final,proposal,guard=guarded_map(x,base,alt,geo);torch.cuda.synchronize()
            runtime_seconds=time.perf_counter()-start
            # Freeze maps before measuring exact errors; labels never enter selection.
            mse=(alt.double()-x.double()).square().reshape(16,-1,64).sum((0,2)) < (base.double()-x.double()).square().reshape(16,-1,64).sum((0,2))
            masks=dict(baseline=torch.zeros_like(final),activation_mse=mse,proposal=proposal,guarded=final)
            case=dict(prompt=label,module=name,activation_sha256=sha(x),guard=guard,
                      candidate_seconds=candidate_seconds,reference_guard_seconds=runtime_seconds,policies={})
            base_loss,base_tokens=exact_loss(x,base,w)
            for policy,mask in masks.items():
                loss,tokens=exact_loss(x,apply_map(base,alt,mask),w)
                case['policies'][policy]=dict(output_error=loss,ratio_to_baseline=loss/base_loss,
                    selected_tiles=int(mask.sum()),mask=mask.int().tolist(),
                    tokens_worse_than_baseline=int((tokens>base_tokens*(1+1e-10)).sum()))
            actual_change=case['policies']['proposal']['output_error']-base_loss
            case['bound_violation']=actual_change>guard['upper_change']+1e-8*max(base_loss,1e-30)
            # Exact 16-map diagnostic on four fixed tiles, keeping all other errors.
            n=final.numel();positions=[0,n//3,2*n//3,n-1]
            oracle=[]
            for bits in range(16):
                mask=torch.zeros_like(final)
                for j,p in enumerate(positions):mask[p]=bool(bits & (1<<j))
                q=apply_map(base,alt,mask);error=q.double()-x.double()
                loss,_=exact_loss(x,q,w)
                bound=upper_change(error,base.double()-x.double(),geo)
                oracle.append(dict(bits=bits,loss=loss,proxy=float(proxy(error,geo)),**bound))
            true_best=min(oracle,key=lambda c:c['loss']);proxy_best=min(oracle,key=lambda c:c['proxy'])
            case['four_tile_oracle']=dict(positions=positions,candidates=oracle,
                 true_best_bits=true_best['bits'],proxy_best_bits=proxy_best['bits'],
                 best_ratio=true_best['loss']/base_loss,
                 true_improving_maps=sum(c['loss']<base_loss*(1-1e-8) for c in oracle),
                 bound_accepted_maps=sum(c['upper_change'] < -1e-10*base_loss for c in oracle),
                 bound_violations=sum(c['loss']-base_loss > c['upper_change']+1e-8*max(base_loss,1e-30) for c in oracle))
            report['packets'].append(case);save(out,report)
            print(f'{label} {name} accepted={guard["accepted"]} proposal_ratio={case["policies"]["proposal"]["ratio_to_baseline"]:.6f} mse_ratio={case["policies"]["activation_mse"]["ratio_to_baseline"]:.6f} oracle_ratio={case["four_tile_oracle"]["best_ratio"]:.6f}',flush=True)
    for handle in handles:handle.remove()
    report['status']='complete'
    packets=report['packets']
    report['summary']=dict(packets=len(packets),accepted_packets=sum(c['guard']['accepted'] for c in packets),
        useful_proposals=sum(c['policies']['proposal']['ratio_to_baseline']<1-1e-8 for c in packets),
        harmful_proposals=sum(c['policies']['proposal']['ratio_to_baseline']>1+1e-8 for c in packets),
        useful_mse_maps=sum(c['policies']['activation_mse']['ratio_to_baseline']<1-1e-8 for c in packets),
        harmful_mse_maps=sum(c['policies']['activation_mse']['ratio_to_baseline']>1+1e-8 for c in packets),
        proposal_beats_mse=sum(c['policies']['proposal']['output_error']<c['policies']['activation_mse']['output_error']*(1-1e-8) for c in packets),
        packets_with_useful_oracle_map=sum(c['four_tile_oracle']['true_improving_maps']>0 for c in packets),
        oracle_useful_maps=sum(c['four_tile_oracle']['true_improving_maps'] for c in packets),
        oracle_bound_accepted_maps=sum(c['four_tile_oracle']['bound_accepted_maps'] for c in packets),
        bound_violations=sum(c['bound_violation']+c['four_tile_oracle']['bound_violations'] for c in packets))
    save(out,report)
    lines=['# Online activation-bound feasibility results','',
        'Three full attention projection matrices, four fixed prompts, one 16-token packet each.',
        'This is a development mechanism test on W4A16 inputs, not an end-to-end W4A4 or domain benchmark.',
        'The analytic bound controls aggregate packet output error; numerical tests are not interval proofs.','',
        '```json',json.dumps(report['summary'],indent=2),'```','',
        '| Prompt / layer | MSE error / baseline | Proposed error / baseline | Accepted | Guarded error / baseline | Four-tile exact optimum / baseline |',
        '|---|---:|---:|---|---:|---:|']
    for c in packets:
        p=c['policies']
        lines.append(f'| {c["prompt"]} / {c["module"]} | {p["activation_mse"]["ratio_to_baseline"]:.6f} | {p["proposal"]["ratio_to_baseline"]:.6f} | {c["guard"]["accepted"]} | {p["guarded"]["ratio_to_baseline"]:.6f} | {c["four_tile_oracle"]["best_ratio"]:.6f} |')
    lines+=['','All maps, bounds, exact oracle values, prompt and metadata hashes, timings and metadata sizes are in report.json.',
            'Timings are unfused float64 reference costs, not production kernel overhead.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    main()
