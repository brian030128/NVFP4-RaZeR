"""Matched baseline diagnostics; no calibration or E0M3 map selection."""
import argparse
import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from quantize import QuantConfig
from quantize.causal_four_over_six import quantize_rows
from quantize.quantizer import quant_nvfp4, quant_nvfp4_4over6
from run_c4_frozen import REVISION, digest_file, heldout_data
from run_conditional_format import sha
from run_math_code_calibration import load_model
from run_wiki_frozen import WIKI_REVISION


def data(tok, prior, length):
    wiki = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', revision=WIKI_REVISION, split='test')
    ids = tok('\n\n'.join(wiki['text']), return_tensors='pt').input_ids
    # Restrict both lengths to identical full 2048-token spans.
    usable = ids.shape[1] // 2048 * 2048
    batches = {'wiki': [ids[:, i:i+length].clone() for i in range(0, usable, length)]}
    meta = {'wiki': dict(revision=WIKI_REVISION, total_tokens=ids.shape[1], used_tokens=usable,
                         omitted_tail_tokens=ids.shape[1]-usable)}
    ds = load_dataset('allenai/c4', revision=REVISION,
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    rng = random.Random(0); crops = []; docs = []
    for _ in range(256):
        while True:
            index = rng.randint(0, len(ds)-1); text = ds[index]['text']
            tokens = tok(text, return_tensors='pt').input_ids
            if tokens.shape[1] > 2049: break
        offset = rng.randint(0, tokens.shape[1]-2048-1)
        crop = tokens[:, offset:offset+2048]
        crops.extend(crop[:, i:i+length].clone() for i in range(0, 2048, length))
        docs.append(dict(index=index, offset=offset, document_sha256=hashlib.sha256(text.encode()).hexdigest()))
    batches['c4_paper'] = crops
    meta['c4_paper'] = dict(revision=REVISION, path='en/c4-validation.00000-of-00008.json.gz',
                          seed=0, parent_window_tokens=2048, documents=docs)
    if length == 512:
        excluded = {d['document_sha256'] for m in prior['fit'].values() for d in m['documents']}
        batches['c4_study'], meta['c4_study'] = heldout_data(tok, excluded)
    for d, bs in batches.items():
        meta[d].update(windows=len(bs), window_tokens=length, token_sha256=[sha(b) for b in bs])
    return batches, meta


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    ap = argparse.ArgumentParser(); ap.add_argument('--model', choices=('qwen4b','llama8b'), required=True)
    ap.add_argument('--length', type=int, choices=(512,2048), required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--historical-qwen', action='store_true')
    args = ap.parse_args(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    if args.historical_qwen: assert args.model=='qwen4b' and args.length==2048
    prior_path = Path(f'results/math_code_adaptive/calibration_333779_{args.model}/report.json')
    prior = json.loads(prior_path.read_text()); model, modules = load_model(prior, False)
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', model=args.model, length=args.length, source=prior['source'],
        revision=prior['revision'], torch_version=torch.__version__, transformers_version=transformers.__version__,
        dtype='bfloat16', attention='eager', kv_quantization=False, calibration=False,
        source_sha256={p:digest_file(p) for p in ('run_baseline_protocol_audit.py','quantize/quantizer.py',
            'quantize/causal_four_over_six.py','models/qmodule_llama.py','models/qmodule_qwen3.py')},
        evaluation={}, wrapper_equivalence={}, historical_qwen_unquantized_o_proj_input=args.historical_qwen)
    def save(): (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    original = {}
    for n,m in modules.items():
        assert sha(m.weight) == prior['matrices'][n]['source_sha256']
        original[n] = m.weight.cpu().clone()
    batches, report['data'] = data(tok, prior, args.length); save()
    print('DATA '+json.dumps({d:len(bs) for d,bs in batches.items()}), flush=True)
    policies = ['bf16','four_over_six_row','four_over_six_tensor']
    if args.length == 2048: policies += ['nvfp4_tensor']
    if args.historical_qwen: policies=['four_over_six_tensor','nvfp4_tensor']
    for policy in policies:
        fn = quant_nvfp4 if policy == 'nvfp4_tensor' else quant_nvfp4_4over6
        for n,m in modules.items():
            w = original[n].to(m.weight.device)
            m.weight.copy_(w if policy == 'bf16' else fn(w,4,16))
        del w
        handles=[]
        if policy != 'bf16':
            act = quantize_rows if policy.endswith('_row') else lambda x: fn(x,4,16)
            handles = [m.register_forward_pre_hook(lambda module, inputs: (act(inputs[0]), *inputs[1:]))
                       for n,m in modules.items() if not (args.historical_qwen and n.endswith('self_attn.o_proj'))]
        if args.length == 2048 and not args.historical_qwen and policy in ('bf16','four_over_six_tensor'):
            if args.model == 'llama8b':
                from models.qmodule_llama import QuantLlamaForCausalLM as Wrapper
            else:
                from models.qmodule_qwen3 import QuantQwen3ForCausalLM as Wrapper
            config = copy.deepcopy(model.config)
            qc = QuantConfig(a_bits=16 if policy == 'bf16' else 4, a_dtype='nvfp4_4over6', a_groupsize=16)
            with torch.device('meta'): wrapper = Wrapper(config, qc)
            wrapper.load_state_dict(model.state_dict(), assign=True)
            # Rotary frequencies are nonpersistent and absent from state_dict.
            for name, buffer in model.named_buffers():
                parent, _, leaf = name.rpartition('.')
                wrapper.get_submodule(parent)._buffers[leaf] = buffer
            wrapper.eval().requires_grad_(False)
            ids = batches['wiki'][0].cuda()
            a = model(ids, use_cache=False).logits; b = wrapper(ids, use_cache=False).logits
            error = float((a-b).abs().max())
            report['wrapper_equivalence'][policy] = dict(equal=torch.equal(a,b), max_logit_difference=error)
            assert torch.equal(a,b), report['wrapper_equivalence'][policy]
            del wrapper, a, b
        report['evaluation'][policy] = {}; save()
        for domain, bs in batches.items():
            losses=[]
            for i,batch in enumerate(bs):
                ids = batch.cuda(); logits = model(ids, use_cache=False).logits
                loss = float(F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1)))
                assert math.isfinite(loss); losses.append(loss)
                del logits
                if (i+1)%128==0: print(f'EVAL {policy} {domain} {i+1}/{len(bs)}', flush=True)
            report['evaluation'][policy][domain] = dict(ppl=math.exp(sum(losses)/len(losses)), nll=losses,
                scored_tokens=len(losses)*(args.length-1))
            if policy == 'four_over_six_row' and args.length == 512:
                previous = json.loads(Path(f'results/math_code_adaptive/evaluation_333786_{args.model}/report.json').read_text())
                old_domain = 'c4' if domain == 'c4_study' else 'wiki'
                if domain in ('wiki','c4_study'):
                    old = previous['evaluation']['four_over_six'][old_domain]['nll'][:len(losses)]
                    assert previous['data'][old_domain]['token_sha256'][:len(losses)] == report['data'][domain]['token_sha256']
                    error = max(abs(a-b) for a,b in zip(old,losses))
                    report['evaluation'][policy][domain]['previous_max_nll_error'] = error
                    assert error <= 1e-6
            save(); print('PPL '+policy+' '+domain+' '+str(report['evaluation'][policy][domain]['ppl']), flush=True)
        for h in handles: h.remove()
    report['status']='complete'; save(); print('COMPLETE', flush=True)


if __name__ == '__main__': main()
