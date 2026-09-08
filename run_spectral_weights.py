"""Prespecified real-weight feasibility panel; no tokenizer or dataset access."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open

from quantize.spectral_selector import candidates, mse_map, reconstruct, select, spectral_squared
from quantize.spectral_approx import estimate, select_approx

MODELS = {
    'llama-3.2-1b-instruct': '/work/u4320956/hf/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6',
    'llama-3.1-8b': '/work/u4320956/hf/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b',
}


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def sync():
    torch.cuda.synchronize()


def save(out,report):
    temp=out/'report.tmp'
    temp.write_text(json.dumps(report,indent=2)+'\n')
    temp.replace(out/'report.json')


def audit(w,q):
    error=q.float()-w.float()
    sync(); start=time.perf_counter()
    result,_=estimate(error,rank=16,iterations=128)
    sync()
    result['seconds']=time.perf_counter()-start
    # Reduction in float64; no full double copy of a large matrix.
    frob=float(error.square().sum(dtype=torch.float64))
    result.update(frobenius_squared=frob, isotropic_error=frob/w.shape[1],
                  stable_rank_estimate=frob/result['value'] if result['value'] else 0)
    return result


def crop_study(w,b,a):
    records=[]
    for label,frac in [('first',0.),('middle',.5),('last',1.)]:
        row=int((w.shape[0]-32)*frac)//8*8
        col=int((w.shape[1]-128)*frac)//64*64
        sl=(slice(row,row+32),slice(col,col+128))
        # Preserve candidates from whole-matrix quantization, including global scale.
        wc,bc,ac=(x[sl].cpu() for x in (w,b,a))
        start=time.perf_counter(); exact_mask,exact_info=select(wc,bc,ac)
        exact_seconds=time.perf_counter()-start
        start=time.perf_counter(); approx_mask,approx_info=select_approx(wc,bc,ac)
        approx_seconds=time.perf_counter()-start
        policies={'baseline':torch.zeros_like(exact_mask),'mse':mse_map(wc,bc,ac),
                  'reference':exact_mask,'approximate':approx_mask}
        record=dict(position=label,offset=[row,col],reference_seconds=exact_seconds,
                    approximate_seconds=approx_seconds,reference=exact_info,approximate=approx_info,policies={})
        for name,mask in policies.items():
            e=reconstruct(bc,ac,mask).double()-wc.double()
            spec=spectral_squared(e)
            est,_=estimate(e)
            record['policies'][name]=dict(spectral_squared=spec,frobenius_squared=float(e.square().sum()),
                    estimate=est,relative_estimation_error=est['value']/spec-1 if spec else 0.,
                    mask=mask.int().tolist())
        records.append(record)
    return records


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); args=ap.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent
    files=['run_spectral_weights.py','quantize/spectral_selector.py','quantize/spectral_approx.py',
           'quantize/quantizer.py','results/spectral_selection/WEIGHT_PROTOCOL.md']
    report=dict(status='running',slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        torch_version=torch.__version__,device=torch.cuda.get_device_name(),
        source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},matrices=[])
    start_all=time.perf_counter(); save(out,report)
    for model,path in MODELS.items():
        path=Path(path); config=json.loads((path/'config.json').read_text())
        idx=path/'model.safetensors.index.json'
        index=json.loads(idx.read_text())['weight_map'] if idx.exists() else None
        layers=(0,config['num_hidden_layers']//2,config['num_hidden_layers']-1)
        for layer in layers:
            for projection in ('self_attn.q_proj','mlp.down_proj'):
                name=f'model.layers.{layer}.{projection}.weight'
                shard=index[name] if index else 'model.safetensors'
                print(f'LOAD {model} {name}',flush=True)
                with safe_open(str(path/shard),framework='pt',device='cpu') as f:
                    w=f.get_tensor(name).to('cuda')
                torch.cuda.reset_peak_memory_stats()
                sync(); start=time.perf_counter(); b,a=candidates(w); sync()
                case=dict(model=model,revision=path.name,tensor=name,shape=list(w.shape),
                          weight_sha256=sha(w),candidate_seconds=time.perf_counter()-start,
                          candidate_sha256={'baseline':sha(b),'alternative':sha(a)})
                sync(); start=time.perf_counter(); mask,trace=select_approx(w,b,a); sync()
                case.update(selection_seconds=time.perf_counter()-start,selection=trace,
                            tile_count=mask.numel(),selected_tiles=int(mask.sum()))
                filename=f'{model}_layer{layer}_{projection}.pt'
                torch.save(dict(mask=mask.cpu(),shape=list(w.shape),tensor=name,revision=path.name,
                                baseline='nvfp4_4over6',alternative='e0m3_alpha1',tile=[8,64]),out/filename)
                case['map_file']=filename
                # Audit all frozen policies with a larger independent numerical budget.
                case['policies']={}
                for policy,q in [('baseline',b),('mse',reconstruct(b,a,mse_map(w,b,a))),
                                 ('spectral_approximate',reconstruct(b,a,mask))]:
                    case['policies'][policy]=audit(w,q)
                case['crops']=crop_study(w,b,a)
                case['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                report['matrices'].append(case); save(out,report)
                p=case['policies']; sp=p['spectral_approximate']; bp=p['baseline']
                print(f'DONE spectral/base={sp["value"]/bp["value"]:.8f} frobenius/base={sp["frobenius_squared"]/bp["frobenius_squared"]:.8f} tiles={case["selected_tiles"]} seconds={case["selection_seconds"]:.2f}',flush=True)
                del w,b,a,mask,q
                torch.cuda.empty_cache()
    report.update(status='complete',elapsed_seconds=time.perf_counter()-start_all); save(out,report)
    lines=['# Real-weight feasibility results','',
           'No datasets, activations, labels, tokenizer, or model forward passes were used.',
           'Full-matrix spectral values are rank-16/128-iteration estimates, not certificates.',
           'Crops preserve full-matrix candidate normalization. These are development models.', '',
           '| Model / matrix | Spectral / baseline (estimate) | Frobenius / baseline | Tiles | Selection seconds | Stop |',
           '|---|---:|---:|---:|---:|---|']
    for c in report['matrices']:
        p=c['policies']; s=p['spectral_approximate']; b=p['baseline']
        lines.append(f'| {c["model"]} / {c["tensor"]} | {s["value"]/b["value"]:.8f} | {s["frobenius_squared"]/b["frobenius_squared"]:.8f} | {c["selected_tiles"]}/{c["tile_count"]} | {c["selection_seconds"]:.2f} | {c["selection"]["stop_reason"]} |')
    lines+=['','Exact crop objectives, approximation errors, all candidate-policy audits, source hashes, and traces are in report.json.',
            'Maps cover only sampled matrices and must not be treated as a whole-model export.','']
    (out/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
