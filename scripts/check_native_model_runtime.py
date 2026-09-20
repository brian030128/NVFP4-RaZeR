"""Bounded native per-tile/quantization checks; no model or evaluation data."""
import os,sys,json
from pathlib import Path
import torch
from native_model_runtime import Runtime,Linear,encode,decode
assert os.environ.get('SLURM_JOB_ID')
torch.set_num_threads(4);torch.manual_seed(20260920);torch.backends.cuda.matmul.allow_tf32=False
rt=Runtime(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);records=[]
for m,n,k in [(1,256,256),(17,512,768),(129,768,1024),(257,1024,512)]:
 w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16);x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
 mask=torch.randint(0,2,(n//256,k//64),device='cuda',dtype=torch.bool)
 rp=torch.randperm(n,device='cuda');cp=(torch.randperm(k//16,device='cuda')[:,None]*16+torch.arange(16,device='cuda')).flatten()
 for mixed in (False,True):
  lin=Linear(rt,w,mask if mixed else None,rp if mixed else None,cp if mixed else None)
  got=lin(x).float();pa,sa,ga=encode(x);ad=decode(pa,sa,ga);wd=decode(lin.packed,lin.flat_scales,lin.global_scale,None if lin.mask is None else lin.mask.bool())
  ref=ad[:,cp]@wd.T if mixed else ad@wd.T
  if mixed:ref=ref[:,torch.argsort(rp)]
  rel=float(torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref));assert rel<.005,(m,n,k,mixed,rel)
  negative=None
  if mixed:
   wrong=decode(lin.packed,lin.flat_scales,lin.global_scale,None);neg=(ad[:,cp]@wrong.T)[:,torch.argsort(rp)]
   negative=float(torch.linalg.vector_norm(got-neg)/torch.linalg.vector_norm(ref));assert negative>.05,negative
  records.append(dict(m=m,n=n,k=k,mixed=mixed,relative_frobenius=rel,wrong_format_error=negative));lin.close()
  print(records[-1],flush=True)
(out/'report.json').write_text(json.dumps(dict(status='passed',checks=records,threshold=.005),indent=2)+'\n')
# BF16 values with repeated magnitudes exercise nearly tied FourOverSix choices.
activation_checks=[]
for kind in ('normal','discrete','wide'):
 x=torch.randn(257,4096,device='cuda',dtype=torch.bfloat16)
 if kind=='discrete':x=(torch.randint(-64,65,x.shape,device='cuda').float()/32).to(torch.bfloat16)
 if kind=='wide':x=(x.float()*torch.logspace(-3,3,4096,device='cuda')).to(torch.bfloat16)
 lin=Linear(rt,torch.randn(256,4096,device='cuda',dtype=torch.bfloat16));lin(x)
 pc,sc,gc=lin.activation(257);pr,sr,gr=encode(x)
 got=decode(pc,sc,gc).to(torch.bfloat16);ref=decode(pr,sr,gr).to(torch.bfloat16)
 bad=(got!=ref);row=dict(kind=kind,code_bytes_different=int((pc!=pr).sum()),scales_different=int((sc!=sr).sum()),dequant_values_different=int(bad.sum()),global_native=gc,global_reference=gr)
 activation_checks.append(row);print('ACTIVATION_CHECK',row,flush=True)
 if bad.any():
  groups=bad.reshape(-1,16).any(-1).nonzero().flatten()[:32]
  torch.save(dict(input=x.reshape(-1,16)[groups].cpu(),maximum=x.abs().max().cpu(),native=got.reshape(-1,16)[groups].cpu(),reference=ref.reshape(-1,16)[groups].cpu()),out/f'activation_failure_{kind}.pt')
 lin.close()
write=dict(status='passed' if all(x['dequant_values_different']==0 for x in activation_checks) else 'activation_failed',checks=records,activation_checks=activation_checks,threshold=.005)
(out/'report.json').write_text(json.dumps(write,indent=2)+'\n')
assert write['status']=='passed',activation_checks
# Reuse the first failing real-model input instead of reloading the checkpoint.
real_checks=[]
for file in sorted(Path('/work/u4320956/b200/activation_case_406608').glob('*.pt')):
 data=torch.load(file,map_location='cuda',weights_only=True);x=data['x'];n,k=data['scales'].shape;k*=16;m=x.numel()//k
 lin=Linear(rt,torch.zeros(n,k,device='cuda',dtype=torch.bfloat16))
 lin.packed=data['packed_weight'];lin.flat_scales=data['scales'];lin.global_scale=data['global_weight'];lin.swizzled=rt.scales(lin.flat_scales)
 got=lin(x).float().reshape(m,n);pc,sc,gc=lin.activation(m);pr,sr,gr=encode(x.reshape(m,k))
 an=decode(pc,sc,gc);ar=decode(pr,sr,gr);wd=decode(lin.packed,lin.flat_scales,lin.global_scale);ref=ar@wd.T
 row=dict(case=file.name,code_differences=int((pc!=pr).sum()),scale_differences=int((sc!=sr).sum()),dequant_differences=int((an.to(torch.bfloat16)!=ar.to(torch.bfloat16)).sum()),relative_frobenius=float(torch.linalg.vector_norm(got-ref)/torch.linalg.vector_norm(ref)))
 real_checks.append(row);print('REAL_CASE',row,flush=True);lin.close()
write['real_cases']=real_checks;write['status']='passed' if all(x['code_differences']==x['scale_differences']==x['dequant_differences']==0 and x['relative_frobenius']<.005 for x in real_checks) else 'real_case_failed'
(out/'report.json').write_text(json.dumps(write,indent=2)+'\n');assert write['status']=='passed'
