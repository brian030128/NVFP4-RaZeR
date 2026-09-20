"""Capture first failing real input using only the model's two-layer prefix."""
import os,sys,json,hashlib
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig,AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaModel
from native_model_runtime import Runtime,Linear,encode,decode
assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
ck=Path(os.environ['HF_HOME'])/'hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b'
rt=Runtime('/home/u4320956/.cache/mixfp4-model-runtime/build/libmixfp4_model.so')
config=AutoConfig.from_pretrained(ck,local_files_only=True);config.num_hidden_layers=2;config._attn_implementation='sdpa'
torch.set_default_dtype(torch.bfloat16)
with torch.device('cuda'):model=LlamaModel(config).eval()
torch.set_default_dtype(torch.float32)
index=json.loads((ck/'model.safetensors.index.json').read_text())['weight_map'];wanted={'model.'+n for n in model.state_dict()};state={}
for shard in sorted({index[n] for n in wanted}):
 with safe_open(str(ck/shard),framework='pt',device='cuda:0') as f:
  for n in sorted(wanted):
   if index[n]==shard:state[n.removeprefix('model.')]=f.get_tensor(n)
model.load_state_dict(state,assign=True,strict=True);del state
raw=torch.load('/work/u4320956/task_reorder/transfer_20260920/llama8b/calibration/compact_masks.pt',map_location='cpu',weights_only=True)['raw256']
for name,module in list(model.named_modules()):
 if isinstance(module,torch.nn.Linear):model.set_submodule(name,Linear(rt,module.weight.data,raw['model.'+name]))
tok=AutoTokenizer.from_pretrained(str(ck),local_files_only=True)
text='The researcher measures matrix multiplication and language model inference carefully. Each experiment uses the same input and records the elapsed time. '
ids=tok(text*400,return_tensors='pt').input_ids[:,:32].cuda();saved={}
handles=[]
for name,module in list(model.named_modules()):
 if not isinstance(module,Linear):continue
 def hook(module,inputs,output,name=name):
  x=inputs[0].detach();m=x.numel()//x.shape[-1];pc,sc,gc=module.activation(m);pr,sr,gr=encode(x.reshape(m,-1))
  an=decode(pc,sc,gc);ar=decode(pr,sr,gr);wd=decode(module.packed,module.flat_scales,module.global_scale,None if module.mask is None else module.mask.bool())
  y=output.float().reshape(m,-1);yr=ar@wd.T;yn=an@wd.T
  ep=(decode(pr,sr,1.)@decode(module.packed,module.flat_scales,1.,None if module.mask is None else module.mask.bool()).T)*(gr*module.global_scale)
  row=dict(name=name,code_differences=int((pc!=pr).sum()),scale_differences=int((sc!=sr).sum()),dequant_differences=int((an.to(torch.bfloat16)!=ar.to(torch.bfloat16)).sum()),rel_python=float(torch.linalg.vector_norm(y-yr)/torch.linalg.vector_norm(yr)),rel_native_dump=float(torch.linalg.vector_norm(y-yn)/torch.linalg.vector_norm(yn)),global_native=gc,global_reference=gr,epilogue_relative=float(torch.linalg.vector_norm(y-ep)/torch.linalg.vector_norm(ep)),inside_bf16_mismatches=int((y.to(torch.bfloat16)!=yr.to(torch.bfloat16)).sum()),epilogue_bf16_mismatches=int((y.to(torch.bfloat16)!=ep.to(torch.bfloat16)).sum()))
  saved[name]=row;print('CASE',row,flush=True)
  if name=='layers.1.mlp.down_proj':torch.save(dict(x=x.cpu(),packed_weight=module.packed.cpu(),scales=module.flat_scales.cpu(),global_weight=module.global_scale,mask=None if module.mask is None else module.mask.cpu(),native_codes=pc.cpu(),native_scales=sc.cpu(),reference_codes=pr.cpu(),reference_scales=sr.cpu(),reference_global=gr,native_global=gc),out/(name.replace('.','_')+'.pt'))
 handles.append(model.get_submodule(name).register_forward_hook(hook))
with torch.inference_mode():
 native=model(ids,use_cache=False).last_hidden_state.clone()
for h in handles:h.remove()
class Reference(torch.nn.Module):
 mode='epilogue'
 def __init__(self,lin):super().__init__();self.lin=lin
 def forward(self,x):
  m=x.numel()//x.shape[-1];p,s,g=encode(x.reshape(m,-1));w=self.lin
  if self.mode=='epilogue':y=(decode(p,s,1.)@decode(w.packed,w.flat_scales,1.,None if w.mask is None else w.mask.bool()).T)*(g*w.global_scale)
  else:y=decode(p,s,g)@decode(w.packed,w.flat_scales,w.global_scale,None if w.mask is None else w.mask.bool()).T
  return y.to(torch.bfloat16).reshape(*x.shape[:-1],w.n)
for name,mod in list(model.named_modules()):
 if isinstance(mod,Linear):model.set_submodule(name,Reference(mod))
with torch.inference_mode():
 for mode in ('epilogue','inside'):
  Reference.mode=mode;ref=model(ids,use_cache=False).last_hidden_state
  saved['prefix_'+mode]=dict(relative=float(torch.linalg.vector_norm(native.float()-ref.float())/torch.linalg.vector_norm(ref.float())),mismatches=int((native!=ref).sum()))
  print('PREFIX',mode,saved['prefix_'+mode],flush=True)
(out/'report.json').write_text(json.dumps(saved,indent=2)+'\n')
