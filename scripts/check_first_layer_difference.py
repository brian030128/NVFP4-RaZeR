import os,sys,json
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaModel
from native_model_runtime import Runtime,Linear,encode,decode
assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
data=torch.load('results/task_reorder/full_model_20260920/llama_406730/first_layer_difference_arranged.pt',map_location='cuda',weights_only=True);li=data['layer'];assert torch.equal(data['native_input'],data['reference_input'])
ck=Path(os.environ['HF_HOME'])/'hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b';index=json.loads((ck/'model.safetensors.index.json').read_text())['weight_map']
config=AutoConfig.from_pretrained(ck,local_files_only=True);config.num_hidden_layers=1;config.vocab_size=1;config._attn_implementation='sdpa'
torch.set_default_dtype(torch.bfloat16)
with torch.device('cuda'):model=LlamaModel(config).eval();model.embed_tokens=torch.nn.Identity();model.norm=torch.nn.Identity()
torch.set_default_dtype(torch.float32)
state={};mapping={n:'model.'+n.replace('layers.0.',f'layers.{li}.') for n in model.state_dict()}
for shard in sorted({index[n] for n in mapping.values()}):
 with safe_open(str(ck/shard),framework='pt',device='cuda:0') as f:
  for dest,source in mapping.items():
   if index[source]==shard:state[dest]=f.get_tensor(source)
model.load_state_dict(state,assign=True,strict=True);del state
raw=torch.load('/work/u4320956/task_reorder/transfer_20260920/llama8b/calibration/compact_masks.pt',map_location='cpu',weights_only=True)['raw256'];rt=Runtime('/home/u4320956/.cache/mixfp4-model-runtime/build/libmixfp4_model.so')
for name,module in list(model.named_modules()):
 if isinstance(module,torch.nn.Linear):model.set_submodule(name,Linear(rt,module.weight.data,raw[mapping[name+'.weight'].removesuffix('.weight')]))
rows=[];handles=[]
for name,module in list(model.named_modules()):
 if not isinstance(module,Linear):continue
 def hook(module,inputs,output,name=name):
  x=inputs[0];m=x.numel()//module.k;p,s,g=encode(x.reshape(m,-1));pc,sc,gc=module.activation(m)
  a=decode(p,s,1.);w=decode(module.packed,module.flat_scales,1.,None if module.mask is None else module.mask.bool());alpha=torch.tensor(g*module.global_scale,device='cuda',dtype=torch.float32)
  f32=(a@w.T)*alpha;f64=(a.double()@w.double().T).float()*alpha;y=output.reshape(m,-1)
  row=dict(name=name,code_mismatches=int((pc!=p).sum()),scale_mismatches=int((sc!=s).sum()),fp32_output_mismatches=int((y!=f32.to(torch.bfloat16)).sum()),fp64_output_mismatches=int((y!=f64.to(torch.bfloat16)).sum()),max_fp32_error=float((y.float()-f32).abs().max()))
  rows.append(row);print('PROJECTION',row,flush=True)
 handles.append(module.register_forward_hook(hook))
with torch.inference_mode():native=model(inputs_embeds=data['native_input'],use_cache=False).last_hidden_state
assert torch.equal(native,data['native_output']),'standalone native layer must replay bitwise'
for h in handles:h.remove()
class Reference(torch.nn.Module):
 precision='fp64'
 def __init__(self,lin):super().__init__();self.lin=lin
 def forward(self,x):
  w=self.lin;m=x.numel()//w.k;p,s,g=encode(x.reshape(m,-1));a=decode(p,s,1.);b=decode(w.packed,w.flat_scales,1.,None if w.mask is None else w.mask.bool())
  if self.precision=='fp64':v=(a.double()@b.double().T).float()
  else:v=a@b.T
  return (v*(g*w.global_scale)).to(torch.bfloat16).reshape(*x.shape[:-1],w.n)
for name,module in list(model.named_modules()):
 if isinstance(module,Linear):model.set_submodule(name,Reference(module))
result=dict(layer=li,projections=rows)
with torch.inference_mode():
 for mode in ('fp64','fp32'):
  Reference.precision=mode;ref=model(inputs_embeds=data['native_input'],use_cache=False).last_hidden_state
  result[mode]=dict(mismatches=int((native!=ref).sum()),relative=float(torch.linalg.vector_norm(native.float()-ref.float())/torch.linalg.vector_norm(ref.float())))
  print(mode,result[mode],flush=True)
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
