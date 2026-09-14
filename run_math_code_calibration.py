"""One shared causal math/code scoring pass for adaptive and fixed-count maps."""
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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantize.adaptive_prefix import derive_maps, source_subsets
from quantize.causal_four_over_six import quantize_rows
from quantize.quantizer import quant_mix_4_6, quant_nvfp4_4over6
from run_c4_frozen import digest_file
from run_conditional_format import save, sha

ORIGINS = {
    'qwen4b': 'results/pooled_scale/model_332389_qwen4b',
    'llama8b': 'results/pooled_scale/model_332389_llama8b',
    'qwen27b': 'results/pooled_qwen27b/model_332840',
}


def math_code_data(tok, previous_fit):
    batches, metadata = {}, {}
    for source in ('math', 'code'):
        meta = previous_fit[source]
        assert meta['repo'] == {'math': 'open-web-math/open-web-math',
                                'code': 'codeparrot/codeparrot-clean'}[source]
        wanted = {d['document_sha256']: d['offset'] for d in meta['documents']}
        assert len(wanted) == 64
        stream = load_dataset(meta['repo'], revision=meta['revision'],
                              data_files={'train': meta['path']}, split='train', streaming=True)
        found = {}
        for row in stream:
            text = row['text' if source == 'math' else 'content']
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest not in wanted or digest in found:
                continue
            ids = tok(text, return_tensors='pt').input_ids
            offset = wanted[digest]
            assert ids.shape[1] >= offset + 512
            found[digest] = ids[:, offset:offset+512].clone()
            if len(found) == 64:
                break
        assert set(found) == set(wanted)
        batches[source] = [found[d['document_sha256']] for d in meta['documents']]
        assert [sha(b) for b in batches[source]] == meta['token_sha256']
        metadata[source] = dict(meta)
    assert len({d['document_sha256'] for meta in metadata.values() for d in meta['documents']}) == 128
    return batches, metadata


def load_model(prior, target):
    if target:
        from transformers import Qwen3_5ForConditionalGeneration
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            prior['source'], revision=prior['revision'], dtype=torch.bfloat16,
            attn_implementation='eager', device_map='balanced', max_memory={0:'65GiB', 1:'65GiB'},
            output_loading_info=True)
        assert not loading['missing_keys'] and not loading.get('mismatched_keys') and not loading.get('error_msgs')
        modules = {n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)
                   and 'language_model' in n and 'head' not in n}
    else:
        model = AutoModelForCausalLM.from_pretrained(prior['source'], revision=prior['revision'],
            torch_dtype=torch.bfloat16, attn_implementation='eager', device_map='cuda')
        modules = {n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)
                   and m is not model.get_output_embeddings()}
    model.eval().requires_grad_(False)
    assert list(modules) == list(prior['matrices'])
    return model, modules


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run through Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=ORIGINS, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--adaptive-only', action='store_true')
    ap.add_argument('--allow-source-drift', action='store_true',
        help='Proceed when quantizer sources differ from the origin job, recording exactly which '
             'files differ. Intended for a re-run whose output is itself checked against the '
             'shipped map digest; the file hash is then the weaker of the two guards.')
    args = ap.parse_args()
    target = args.model == 'qwen27b'
    torch.set_num_threads(12 if target else 4); torch.backends.cuda.matmul.allow_tf32 = False
    old = Path(ORIGINS[args.model]); prior = json.loads((old/'report.json').read_text())
    assert prior['status'] == 'complete' and transformers.__version__ == prior['transformers_version']
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    score_dir = out/'scores'; score_dir.mkdir()
    teacher_dir = Path(os.environ['HF_HOME'])/'adaptive_teacher'; teacher_dir.mkdir()
    files = ('run_math_code_calibration.py', 'quantize/adaptive_prefix.py',
             'quantize/causal_four_over_six.py', 'quantize/quantizer.py',
             'results/math_code_adaptive/PROTOCOL.md')
    r = dict(status='running', model=args.model, source=prior['source'], revision=prior['revision'],
        job_id=os.environ['SLURM_JOB_ID'], torch_version=torch.__version__, transformers_version=transformers.__version__,
        origin=str(old), source_sha256={f:digest_file(f) for f in files}, matrices={},
        activation_convention='causal per-token factors for scoring and evaluation',
        calibration_sources=['OpenWebMath','CodeParrot'], uses_c4_calibration=False, uses_wiki_calibration=False,
        subsets=source_subsets(), fixed256_comparison=not args.adaptive_only)
    save(out,r)
    # The origin job pinned the quantizer sources by digest. A later commit can invalidate that
    # digest without changing any function this pass calls -- 384b803 appended
    # quant_nvfp4_4over6_pair and touched nothing else -- so --allow-source-drift downgrades the
    # check to a recorded difference. It is not a way to skip verification: the caller is
    # expected to compare the resulting map digest against the shipped one, which tests the
    # thing the file hash is standing in for.
    drift = {f: dict(origin=prior['source_sha256'][f], now=r['source_sha256'][f])
             for f in ('quantize/causal_four_over_six.py', 'quantize/quantizer.py')
             if r['source_sha256'][f] != prior['source_sha256'][f]}
    r['source_drift'] = drift
    if drift:
        assert args.allow_source_drift, f'quantizer sources differ from origin: {sorted(drift)}'
        print('SOURCE DRIFT ALLOWED: ' + ', '.join(sorted(drift)), flush=True)
    save(out, r)
    model, modules = load_model(prior,target)
    for name,m in modules.items():
        assert sha(m.weight) == prior['matrices'][name]['source_sha256']
        r['matrices'][name] = prior['matrices'][name]
    r['source_weights_verified'] = True
    tok = AutoTokenizer.from_pretrained(r['source'], revision=r['revision'])
    fit,r['fit'] = math_code_data(tok,prior['fit'])
    batches = [b for values in fit.values() for b in values]
    device = model.get_input_embeddings().weight.device
    r['bf16_fit_nll'] = []
    with torch.no_grad():
        for i,batch in enumerate(batches):
            ids=batch.to(device); logits=model(input_ids=ids,use_cache=False).logits
            lp=logits[:,:-1].float().reshape(-1,logits.shape[-1]).log_softmax(-1)
            assert torch.isfinite(lp).all()
            r['bf16_fit_nll'].append(float(F.nll_loss(lp,ids[:,1:].reshape(-1).to(lp.device))))
            torch.save(lp.cpu(),teacher_dir/f'{i:03d}.pt')
            if (i+1)%16 == 0: print(f'TEACHER {i+1}/128',flush=True)
        del ids,logits,lp
    save(out,r)
    base,alt,directions,mse = {},{},{},{}
    cached_bytes = {}
    with torch.no_grad():
        for i,(name,m) in enumerate(modules.items()):
            w=m.weight.detach(); o,k=w.shape
            b=quant_nvfp4_4over6(w,4,16)
            a=quant_mix_4_6(w,4,16,type_block=(8,64),clip='a1',elect='always')
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            diff=(a.float()-w.float()).square()-(b.float()-w.float()).square()
            mse[name]=(diff.reshape(o//8,8,k//64,64).sum((1,3))<0).cpu()
            base[name],alt[name]=(b.cpu().pin_memory(),a.cpu().pin_memory()) if target else (b,a)
            key=str(w.device); size=w.numel()*4; limit=(12 if target else 4)*1024**3
            if cached_bytes.get(key,0)+size <= limit:
                directions[name]=a.float()-b.float(); cached_bytes[key]=cached_bytes.get(key,0)+size
            m.weight.copy_(b)
            if (i+1)%64 == 0: print(f'CANDIDATES {i+1}/{len(modules)}',flush=True)
        del a,b,w,diff
    tables={n:[torch.empty(128,m.weight.numel()//512) for _ in range(3)] for n,m in modules.items()}
    hits={n:[0,0,0] for n in modules}; phase=0; sequence=0

    def act(module,inputs):
        x=inputs[0]; q=quantize_rows(x.detach())
        return (q+(x-x.detach()),*inputs[1:])

    def make_hook(name):
        def forward(module,inputs,output):
            x=inputs[0].detach().reshape(-1,inputs[0].shape[-1])
            def backward(dy):
                grad=dy.detach().reshape(-1,dy.shape[-1]).float().T@x.float()
                d=directions.get(name)
                if d is None:
                    d=alt[name].to(grad.device,non_blocking=True).float()-base[name].to(grad.device,non_blocking=True).float()
                o,k=grad.shape
                value=(grad*d).reshape(o//8,8,k//64,64).sum((1,3)).flatten()
                assert torch.isfinite(value).all(),name
                tables[name][phase][sequence].copy_(value.cpu()); hits[name][phase]+=1
            output.register_hook(backward)
        return forward

    handles=[m.register_forward_pre_hook(act) for m in modules.values()]
    handles += [m.register_forward_hook(make_hook(n)) for n,m in modules.items()]
    generator=None; r['initial_fit_losses']=[]; start=time.perf_counter()
    for sequence,batch in enumerate(batches):
        ids=batch.to(device); embeds=model.get_input_embeddings()(ids).detach().requires_grad_()
        logits=model(inputs_embeds=embeds,use_cache=False).logits
        lp=logits[:,:-1].float().reshape(-1,logits.shape[-1]).log_softmax(-1)
        path=teacher_dir/f'{sequence:03d}.pt'
        teacher=torch.load(path,map_location='cpu',weights_only=True).to(lp.device)
        ce=F.nll_loss(lp,ids[:,1:].reshape(-1).to(lp.device))
        kl=F.kl_div(lp,teacher,reduction='batchmean',log_target=True)
        phase=0; ce.backward(retain_graph=True)
        phase=1; kl.backward(retain_graph=True)
        if generator is None: generator=torch.Generator(device=lp.device).manual_seed(20260930)
        with torch.no_grad(): labels=torch.multinomial(lp.detach().exp(),1,generator=generator).squeeze(-1)
        sampled=F.nll_loss(lp,labels); phase=2; sampled.backward()
        assert all(math.isfinite(float(v)) for v in (ce,kl,sampled))
        r['initial_fit_losses'].append(dict(ce=float(ce),kl=float(kl),sampled_ce=float(sampled)))
        del embeds,logits,lp,teacher,ce,kl,sampled,labels
        path.unlink()
        if (sequence+1)%8 == 0:
            save(out,r); print(f'SCORED {sequence+1}/128 {time.perf_counter()-start:.1f}s',flush=True)
    assert all(v == [128,128,128] for v in hits.values())
    for h in handles: h.remove()
    del directions,model,modules,base,alt
    torch.cuda.empty_cache()
    r['score_seconds']=time.perf_counter()-start
    for i,(name,values) in enumerate(tables.items()):
        torch.save(dict(name=name,ce=values[0],kl=values[1],fisher=values[2]),score_dir/f'{i:03d}.pt')
    maps,stats=derive_maps(((n,r['matrices'][n]['shape'],*values) for n,values in tables.items()),
                           include_fixed=not args.adaptive_only)
    del tables
    total=sum(m['shape'][0]//8*(m['shape'][1]//64) for m in r['matrices'].values())
    r['block_statistics']=stats
    r['block_statistics']['four_over_six']=dict(selected_blocks=0,total_type_blocks=total,selected_fraction=0.)
    mse_count=sum(int(m.sum()) for m in mse.values())
    r['block_statistics']['weight_mse']=dict(selected_blocks=mse_count,total_type_blocks=total,selected_fraction=mse_count/total)
    torch.save(dict(source=r['source'],revision=r['revision'],weight_mse=mse),out/'weight_mse.pt')
    r['weight_mse_sha256']=digest_file(out/'weight_mse.pt')
    (out/'maps.json').write_text(json.dumps(dict(source=r['source'],revision=r['revision'],
        type_block=[8,64],baseline='FourOverSix',alternative='E0M3 alpha1',maps=maps),indent=2)+'\n')
    r['map_sha256']=digest_file(out/'maps.json'); r['maps_frozen']=True
    r['status']='complete'; save(out,r)
    print('FROZEN '+json.dumps({p:s['selected_blocks'] for p,s in r['block_statistics'].items()}),flush=True)


if __name__ == '__main__':
    main()
