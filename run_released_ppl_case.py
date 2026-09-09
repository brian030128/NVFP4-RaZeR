"""Invoke the archived released run_ppl.py, with observation-only instrumentation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import runpy
import sys
import time


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--root',required=True)
    ap.add_argument('--case',type=int,required=True); args=ap.parse_args()
    source=Path(args.source).resolve(); root=Path(args.root).resolve()
    case=json.loads((root/'cases.json').read_text())[args.case]
    out=root/case['id']; out.mkdir(parents=True,exist_ok=False)
    # Remove the working repository from import search; quantization/model/eval
    # functions below must resolve to the original release tree.
    working=Path(__file__).resolve().parent
    sys.path[:]=[str(source),*[p for p in sys.path if p and Path(p).resolve()!=working]]
    os.chdir(source)
    import torch
    import torch.nn.functional as F
    import datasets
    import utils
    import quantize
    torch.set_num_threads(4)
    report=dict(status='running',case=case,python=platform.python_version(),
        packages={p:importlib.metadata.version(p) for p in ('torch','transformers','datasets','accelerate','numpy','tokenizers')},
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        slurm_job=os.environ['SLURM_JOB_ID'],slurm_step=os.environ.get('SLURM_STEP_ID'),
        imported_source=dict(utils=utils.__file__,quantize=quantize.__file__),windows=[],datasets=[],
        release_commit=json.loads((root/'manifest.json').read_text())['release_commit'])
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def sha(t): return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
    save()
    utils.model2path.update(json.loads((root/'model_paths.json').read_text()))
    original_dataset=datasets.load_dataset
    def observe_dataset(*a,**kw):
        ds=original_dataset(*a,**kw)
        report['datasets'].append(dict(args=a,kwargs=kw,rows=len(ds),fingerprint=ds._fingerprint,
            python_rng_sha256=hashlib.sha256(repr(random.getstate()).encode()).hexdigest()))
        return ds
    datasets.load_dataset=observe_dataset
    original_loader=utils.load_model_and_tokenizer
    def observe_load(*a,**kw):
        model,tok=original_loader(*a,**kw)
        report['attention_backend']=model.config._attn_implementation
        report['model_class']=type(model).__name__; report['dtype']=str(next(model.parameters()).dtype)
        report['config']=model.config.to_dict()
        def observe_forward(module,inputs,output):
            ids=inputs[0]; logits=output.logits
            with torch.no_grad():
                loss=F.cross_entropy(logits[:,:-1].contiguous().float().view(-1,logits.shape[-1]),ids[:,1:].reshape(-1))
            report['windows'].append(dict(input_sha256=sha(ids),tokens=ids.numel(),nll=float(loss)))
        model.register_forward_hook(observe_forward)
        return model,tok
    utils.load_model_and_tokenizer=observe_load
    original_weight=quantize.quant_weight
    def observe_weight(model,config):
        original_weight(model,config)
        report['quantized_weight_sha256']={n:sha(m.weight) for n,m in model.named_modules()
            if isinstance(m,torch.nn.Linear) and 'head' not in n}
        report['quant_config']=vars(config); save()
    quantize.quant_weight=observe_weight
    policy=case['policy']
    command=['run_ppl.py','--model_name',case['model'],'--datasets','wikitext,c4','--seq_len','2048','--output_dir',str(out/'original_output')]
    if policy=='bf16': command+=['--use_fp16']
    else:
        kind=policy.rsplit('_',1)[0]; w={'nvfp4':'nvfp4','four_over_six':'nvfp4_4over6','razer':'nvfp4_razer_e3m3'}[kind]
        command+=['--w_bits','4','--w_groupsize','16','--w_dtype',w,'--w_outlier','8.0']
        if policy.endswith('w4a4'):
            command+=['--a_bits','4','--a_groupsize','16','--a_dtype','nvfp4_razer_e4m3' if kind=='razer' else w]
    report['argv']=command; save(); sys.argv=command
    started=time.monotonic()
    try:
        ns=runpy.run_path(str(source/'run_ppl.py'),run_name='__main__')
        report['ppl']=ns['res']; report['elapsed_seconds']=time.monotonic()-started
        count=len(report['windows'])-256; assert count>0
        for d,windows in [('wikitext',report['windows'][:count]),('c4',report['windows'][count:])]:
            # Match the released evaluator's FP32 multiply, sum and exponent.
            values=torch.tensor([v['nll'] for v in windows],dtype=torch.float32)*2048
            reconstructed=float(torch.exp(values.sum()/(len(windows)*2048)))
            assert reconstructed==report['ppl'][d],(d,reconstructed,report['ppl'][d])
        report['wiki_windows']=count; report['c4_windows']=256
        report['displayed_match']={d:f'{report["ppl"][d]:.2f}'==f'{case["paper"][d]:.2f}' for d in ('wikitext','c4')}
        report['status']='complete'; save()
        print('REPRODUCTION '+json.dumps(dict(case=case['id'],ppl=report['ppl'],match=report['displayed_match'])),flush=True)
    except BaseException as exc:
        report['status']='failed'; report['error']=repr(exc); save(); raise


if __name__=='__main__': main()
