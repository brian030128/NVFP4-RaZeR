import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
from quantize.quantizer import quant_nvfp4_4over6
from run_wiki_frozen import WIKI_REVISION
torch.set_num_threads(4)
prior=json.loads(Path('results/math_code_adaptive/calibration_333779_llama8b/report.json').read_text())
model=AutoModelForCausalLM.from_pretrained(prior['source'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda:0').eval()
tok=AutoTokenizer.from_pretrained(prior['source'])
ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',revision=WIKI_REVISION,split='test')
ids=tok('\n\n'.join(ds['text']),return_tensors='pt').input_ids[:,:2048].cuda(0)
modules=[m for m in model.modules() if isinstance(m,torch.nn.Linear) and m is not model.get_output_embeddings()]
with torch.no_grad():
    for m in modules:m.weight.copy_(quant_nvfp4_4over6(m.weight,4,16))
    handles=[m.register_forward_pre_hook(lambda m,x:(quant_nvfp4_4over6(x[0],4,16),*x[1:])) for m in modules]
    for tf32 in (False,True):
        torch.backends.cuda.matmul.allow_tf32=tf32
        for cache in (False,True):
            out=model(ids,use_cache=cache)
            nll=float(torch.nn.functional.cross_entropy(out.logits[:,:-1].contiguous().float().view(-1,out.logits.shape[-1]),ids[:,1:].reshape(-1)))
            print('CHECK',dict(tf32=tf32,use_cache=cache,nll=nll),flush=True)
            del out
