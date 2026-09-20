import os,sys,json
from pathlib import Path
import torch
from safetensors import safe_open
from native_model_runtime import Runtime,Linear,encode,decode
assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
rt=Runtime('/home/u4320956/.cache/mixfp4-model-runtime/build/libmixfp4_model.so')
ck=Path(os.environ['HF_HOME'])/'hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b';index=json.loads((ck/'model.safetensors.index.json').read_text())['weight_map']
root=Path('/work/u4320956/task_reorder/transfer_20260920/llama8b');rows=[];out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
with torch.inference_mode():
 for file in sorted((root/'joint192/layouts').glob('*/layout.pt')):
  l=torch.load(file,map_location='cuda',weights_only=True);key=l['name']+'.weight'
  with safe_open(str(ck/index[key]),framework='pt',device='cuda:0') as f:w=f.get_tensor(key)
  which='/work/u4320956/b200/activation_case_406704/layers_1_mlp_down_proj.pt' if w.shape[1]==14336 else '/work/u4320956/b200/activation_case_406608/layers_1_self_attn_q_proj.pt'
  x=torch.load(which,map_location='cuda',weights_only=True)['x'];m=x.numel()//w.shape[1]
  lin=Linear(rt,w,l['mask'],l['row_perm'],l['col_perm']);y=lin(x).float().reshape(m,-1)
  p,s,g=encode(x.reshape(m,-1));a=decode(p,s,1.);b=decode(lin.packed,lin.flat_scales,1.,lin.mask.bool());alpha=g*lin.global_scale
  natural=(a@b[torch.argsort(l['row_perm'])][:,torch.argsort(l['col_perm'])].T)*alpha
  ordered=(a[:,l['col_perm']]@b.T)*alpha;ordered=ordered[:,torch.argsort(l['row_perm'])]
  pc,sc,gc=lin.activation(m);native_a=decode(pc,sc,1.)
  row=dict(name=l['name'],activation_values_different=int((native_a!=a[:,l['col_perm']]).sum()),natural_mismatches=int((y.to(torch.bfloat16)!=natural.to(torch.bfloat16)).sum()),ordered_mismatches=int((y.to(torch.bfloat16)!=ordered.to(torch.bfloat16)).sum()),ordered_relative=float(torch.linalg.vector_norm(y-ordered)/torch.linalg.vector_norm(ordered)))
  print(row,flush=True);rows.append(row);lin.close()
 (out/'report.json').write_text(json.dumps(rows,indent=2)+'\n')
