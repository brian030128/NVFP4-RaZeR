"""Full-model native latency, frozen policies, paired actual prefill/cached decode."""
import os,sys,json,time,hashlib,argparse,statistics,ast,gc
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
from native_model_runtime import Runtime,Linear,encode,decode,stream,ptr
from quantize.quantizer import quant_nvfp4_4over6,quant_mix_4_6
from quantize.task_reorder import original_order_weight_reference

ROOT=Path('/work/u4320956/task_reorder/transfer_20260920/llama8b')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,x):path.write_text(json.dumps(x,indent=2)+'\n')

class PolicyLinear(torch.nn.Module):
 policy='base';reference=False;audit=False;audit_rows=[]
 def __init__(self,variants,refs):super().__init__();self.variants=variants;self.refs=refs
 def exact(self,x):
  lin=self.variants[self.policy]
  p,s,g=encode(x.reshape(-1,lin.k));a=decode(p,s,1.)
  w=decode(lin.packed,lin.flat_scales,1.,None if lin.mask is None else lin.mask.bool())
  if lin.cols is not None:
   groups=torch.argsort(lin.cols).long();a=a[:,(groups[:,None]*16+torch.arange(16,device=w.device)).flatten()]
   p=p.reshape(-1,lin.k//16,8)[:,groups].reshape(-1,lin.k//2);s=s[:,groups]
  if self.audit:
   self.expected_encoding=(p,s,g)
   y=(a.double()@w.double().T).float()*(g*lin.global_scale)
  else:y=F.linear(a,w)*(g*lin.global_scale)
  if lin.rows is not None:y=y[:,lin.rows.long()]
  return y.reshape(*x.shape[:-1],lin.n)
 def forward(self,x):
  if self.reference=='exact':return self.exact(x).to(torch.bfloat16)
  if self.reference:return F.linear(quant_nvfp4_4over6(x,4,16),self.refs[self.policy])
  lin=self.variants[self.policy];y=lin(x)
  if self.audit:
   exact=self.exact(x)
   rel=float(torch.linalg.vector_norm(y.float()-exact)/torch.linalg.vector_norm(exact))
   row=dict(name=self.name,policy=self.policy,relative_frobenius=rel)
   pc,sc,gc=lin.activation(x.numel()//lin.k);pr,sr,gr=self.expected_encoding
   row.update(code_mismatches=int((pc!=pr).sum()),scale_mismatches=int((sc!=sr).sum()),global_matches=gc==gr,oracle='FP64 sum rounded toFP32, matched FP32 global product and BF16 output')
   del self.expected_encoding
   self.audit_rows.append(row)
   if rel>=.005:
    print('OPERATOR_FAIL',row,flush=True)
    if not (self.audit_dir/'failed_operator.pt').exists():
     torch.save(dict(name=self.name,policy=self.policy,x=x.cpu(),packed_weight=lin.packed.cpu(),scales=lin.flat_scales.cpu(),global_weight=lin.global_scale,mask=None if lin.mask is None else lin.mask.cpu(),rows=None if lin.rows is None else lin.rows.cpu(),cols=None if lin.cols is None else lin.cols.cpu()),self.audit_dir/'failed_operator.pt')
  return y

def timer(fn):
 torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 t=time.perf_counter();start.record();value=fn();end.record();end.synchronize()
 return value,dict(cuda_ms=start.elapsed_time(end),wall_ms=(time.perf_counter()-t)*1000)

@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--lib',required=True);ap.add_argument('--out',required=True);ap.add_argument('--pilot-only',action='store_true');ap.add_argument('--diagnose',action='store_true');ap.add_argument('--kernel-gate',type=Path,required=True);ap.add_argument('--performance-diagnostic',action='store_true');ap.add_argument('--reuse-audits',type=Path);args=ap.parse_args()
 assert os.environ.get('SLURM_JOB_ID') and torch.cuda.get_device_capability(0)[0]==10
 torch.set_num_threads(4);torch.manual_seed(20260920);torch.backends.cuda.matmul.allow_tf32=False
 out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
 PolicyLinear.audit_dir=out
 rt=Runtime(args.lib);prior=json.loads((ROOT/'calibration/report.json').read_text())
 kernel_gate=args.kernel_gate
 assert json.loads((kernel_gate/'report.json').read_text())['status']=='passed'
 assert digest(args.lib)==json.loads((kernel_gate/'library.json').read_text())['library_sha256']
 masks=torch.load(ROOT/'calibration/compact_masks.pt',map_location='cpu',weights_only=True)
 assert digest(ROOT/'calibration/compact_masks.pt')==prior['compact_mask_sha256']
 assert masks['revision']==prior['revision'] and masks['weight_sha256']=={n:v['source_sha256'] for n,v in prior['matrices'].items()}
 layouts={};hashes={}
 for path in sorted((ROOT/'joint192/layouts').glob('*/layout.pt')):
  l=torch.load(path,map_location='cpu',weights_only=True);layouts[l['name']]=l;hashes[str(path)]=digest(path)
 assert len(layouts)==3
 tiles=sum(int(l['mask'].sum()) for l in layouts.values())+sum(int(m.sum()) for n,m in masks['raw256'].items() if n not in layouts);assert tiles==147
 report=dict(status='preparing',job=os.environ['SLURM_JOB_ID'],device=torch.cuda.get_device_name(0),torch=torch.__version__,source=prior['source'],revision=prior['revision'],layout_sha256=hashes,total_tiles=tiles,plan_sha256=digest('results/task_reorder/full_model_20260920/plan.json'),matrix_audits=[],pilot=[],timings=[])
 report['source_sha256']={p:digest(p) for p in ('scripts/benchmark_native_llama.py','scripts/native_model_runtime.py','native/model_runtime.cu','scripts/build_native_model_runtime.py','quantize/quantizer.py','quantize/task_reorder.py')}
 report['library_sha256']=digest(args.lib)
 report['performance_diagnostic']=args.performance_diagnostic
 write(out/'report.json',report)
 checkpoint=Path(os.environ['HF_HOME'])/'hub'/('models--'+prior['source'].replace('/','--'))/'snapshots'/prior['revision']
 assert checkpoint.is_dir()
 tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(str(checkpoint),dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval()
 policy_modules=[];t0=time.perf_counter()
 for index,(name,meta) in enumerate(prior['matrices'].items()):
  old=model.get_submodule(name);w=old.weight.data
  sha=hashlib.sha256(w.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest();assert sha==meta['source_sha256'],name
  assert old.bias is None,name
  base=Linear(rt,w);base_ref=quant_nvfp4_4over6(w,4,16)
  decoded=decode(base.packed,base.flat_scales,base.global_scale).to(torch.bfloat16);assert torch.equal(decoded,base_ref),(name,'base packing');del decoded
  variants={'base':base};refs={'base':base_ref};rawmask=masks['raw256'][name]
  if rawmask.any():
   raw=Linear(rt,w,rawmask);alt=quant_mix_4_6(w,4,16,type_block=(8,64),clip='a1',elect='always')
   fullmask=rawmask.cuda().repeat_interleave(256,0).repeat_interleave(64,1)[:w.shape[0],:w.shape[1]]
   rawref=torch.where(fullmask,alt,base_ref);del fullmask
   decoded=decode(raw.packed,raw.flat_scales,raw.global_scale,raw.mask.bool()).to(torch.bfloat16);assert torch.equal(decoded,rawref),(name,'raw packing');del decoded
  else:raw=base;rawref=base_ref;alt=None
  variants['raw']=raw;refs['raw']=rawref
  if name in layouts:
   l=layouts[name];arr=Linear(rt,w,l['mask'],l['row_perm'],l['col_perm'])
   if alt is None:alt=quant_mix_4_6(w,4,16,type_block=(8,64),clip='a1',elect='always')
   arrref=original_order_weight_reference(base_ref,alt,l)
   decoded=decode(arr.packed,arr.flat_scales,arr.global_scale,arr.mask.bool() if arr.mask is not None else None).to(torch.bfloat16)
   decoded=decoded[torch.argsort(l['row_perm']).cuda()][:,torch.argsort(l['col_perm']).cuda()]
   assert torch.equal(decoded,arrref),(name,'arranged packing');del decoded
  else:arr=raw;arrref=rawref
  variants['arranged']=arr;refs['arranged']=arrref
  replacement=PolicyLinear(variants,refs);replacement.name=name;model.set_submodule(name,replacement);policy_modules.append(replacement)
  report['matrix_audits'].append(dict(name=name,source_sha256=sha,packed_weights_bitwise=True));del w,old,alt
  if (index+1)%16==0:print('PACKED',index+1,len(prior['matrices']),round(time.perf_counter()-t0,2),flush=True)
 assert len(policy_modules)==len(prior['matrices'])
 report['status']='weights_audited';write(out/'report.json',report)
 text='The researcher measures matrix multiplication and language model inference carefully. Each experiment uses the same input and records the elapsed time. '
 ids=tokenizer(text*400,return_tensors='pt').input_ids[:,:2080].cuda();assert ids.shape==(1,2080)
 report['input_sha256']=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
 if args.reuse_audits:
  old=json.loads(args.reuse_audits.read_text());assert args.performance_diagnostic and old['status']=='complete' and old['performance_diagnostic']
  assert old['library_sha256']==report['library_sha256'] and old['layout_sha256']==report['layout_sha256'] and old['revision']==report['revision'] and old['input_sha256']==report['input_sha256'] and old['torch']==report['torch']
  for name in ('scripts/native_model_runtime.py','native/model_runtime.cu','quantize/quantizer.py','quantize/task_reorder.py'):
   assert old['source_sha256'][name]==report['source_sha256'][name],name
  old_source=args.reuse_audits.parent.parent/f"source_{old['job']}"/'benchmark_native_llama.py'
  def policy_ast(path):return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='PolicyLinear'))
  assert policy_ast(old_source)==policy_ast(Path(__file__))
  assert len(old['operator_audits'])==672 and all(x['relative_frobenius']<.005 and x['code_mismatches']==x['scale_mismatches']==0 and x['global_matches'] for x in old['operator_audits'])
  assert all(x['native_repeat_bitwise'] for x in old['pilot'])
  for key in ('operator_audits','pilot','full_output_equivalence_gate_passed','native_accuracy_established','layer_comparisons'):report[key]=old[key]
  report['audits_reused_from']=dict(path=str(args.reuse_audits),sha256=digest(args.reuse_audits),policy_class_ast_identical=True)
  report['status']='operator_gate_reused';write(out/'report.json',report)
 else:
  traces={};trace_handles=[]
  for li,layer in enumerate(model.model.layers):
   def trace_hook(module,inputs,output,li=li):
    if PolicyLinear.reference=='exact' or PolicyLinear.reference is False:
     mode='exact' if PolicyLinear.reference=='exact' else 'native'
     traces[(PolicyLinear.policy,mode,li)]=(inputs[0].detach().clone(),output.detach().clone())
   trace_handles.append(layer.register_forward_hook(trace_hook))
  if args.diagnose:
   PolicyLinear.reference=False;PolicyLinear.audit=True
   for policy in ('base','arranged'):
    PolicyLinear.policy=policy;native=model(ids[:,:32],use_cache=False).logits
    assert torch.isfinite(native).all()
    report['operator_audits']=PolicyLinear.audit_rows;write(out/'report.json',report)
    if max(x['relative_frobenius'] for x in PolicyLinear.audit_rows)<.005:
     PolicyLinear.reference='exact';reference=model(ids[:,:32],use_cache=False).logits
     rel=float(torch.linalg.vector_norm(native.float()-reference.float())/torch.linalg.vector_norm(reference.float()))
     report.setdefault('exact_full_model',[]).append(dict(policy=policy,relative_frobenius=rel))
     PolicyLinear.reference=False;write(out/'report.json',report)
   report['status']='operator_diagnosis_complete';write(out/'report.json',report);return
  for policy in (('base','raw','arranged') if args.performance_diagnostic else ('base','arranged')):
   PolicyLinear.policy=policy;PolicyLinear.reference=True
   fake=model(ids[:,:32],use_cache=False).logits.float()
   PolicyLinear.reference='exact';reference=model(ids[:,:32],use_cache=False).logits.float()
   PolicyLinear.reference=False;PolicyLinear.audit=True;PolicyLinear.audit_rows=[]
   native=model(ids[:,:32],use_cache=False).logits.float();PolicyLinear.audit=False
   assert torch.isfinite(native).all()
   rel=float(torch.linalg.vector_norm(native-reference)/torch.linalg.vector_norm(reference))
   fake_rel=float(torch.linalg.vector_norm(native-fake)/torch.linalg.vector_norm(fake))
   report['pilot'].append(dict(policy=policy,relative_frobenius=rel,reference='FP32 decoded FP4/FP8 operands; global product applied after GEMM; BF16 projection outputs',bf16_fake_relative_frobenius=fake_rel,bf16_fake_old_gate_passed=fake_rel<.02,max_abs=float((native-reference).abs().max()),max_operator_relative=max(x['relative_frobenius'] for x in PolicyLinear.audit_rows)))
   report.setdefault('operator_audits',[]).extend(PolicyLinear.audit_rows);write(out/'report.json',report)
   print('PILOT',report['pilot'][-1],flush=True)
   repeated=model(ids[:,:32],use_cache=False).logits.float()
   report['pilot'][-1]['native_repeat_bitwise']=torch.equal(native,repeated)
   layer_rows=[];saved_failure=False
   for li in range(32):
    ni,ny=traces[(policy,'native',li)];ri,ry=traces[(policy,'exact',li)]
    different=int((ny!=ry).sum());layer_rows.append(dict(layer=li,mismatches=different,relative=float(torch.linalg.vector_norm(ny.float()-ry.float())/torch.linalg.vector_norm(ry.float()))))
    if different and not saved_failure:
     torch.save(dict(layer=li,policy=policy,native_input=ni.cpu(),reference_input=ri.cpu(),native_output=ny.cpu(),reference_output=ry.cpu(),ids=ids[:,:32].cpu()),out/f'first_layer_difference_{policy}.pt');saved_failure=True
   report.setdefault('layer_comparisons',{})[policy]=layer_rows;write(out/'report.json',report)
   assert report['pilot'][-1]['native_repeat_bitwise'] and report['pilot'][-1]['max_operator_relative']<.005,(policy,report['pilot'][-1])
   assert all(v['code_mismatches']==v['scale_mismatches']==0 and v['global_matches'] for v in PolicyLinear.audit_rows), 'Actual activation encoding must match exactly'
   if not args.performance_diagnostic:assert rel<.02,(policy,report['pilot'][-1])
  for handle in trace_handles:handle.remove()
  traces.clear()
  report['status']='operator_gate_passed' if args.performance_diagnostic else 'pilot_passed'
  report['full_output_equivalence_gate_passed']=all(x['relative_frobenius']<.02 for x in report['pilot'])
  report['native_accuracy_established']=False
  write(out/'report.json',report)
 if args.pilot_only:return
 report['gemm_only']=[]
 for name in layouts:
  layer=model.get_submodule(name)
  for m in (1,128,2048):
   x=torch.randn(m,layer.variants['base'].k,device='cuda',dtype=torch.bfloat16)
   graphs={}
   for policy in ('base','arranged'):
    lin=layer.variants[policy];lin(x);p,_=lin.prepare(m);torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
     for _ in range(100):rt.check(rt.lib.mf_run(p,ptr(x),stream(),1))
    graphs[policy]=g
   for rep in range(5):
    row=dict(name=name,m=m,n=lin.n,k=lin.k,rep=rep)
    for policy in (('base','arranged') if rep%2==0 else ('arranged','base')):
     _,measurement=timer(graphs[policy].replay);row[policy+'_us']=measurement['cuda_ms']*10
    report['gemm_only'].append(row)
   del graphs,x
 write(out/'report.json',report)
 # Release fake-quantized reference weights before timings; no policy setup in the measured region.
 for module in policy_modules:module.refs.clear()
 if not args.reuse_audits:del reference,native
 torch.cuda.empty_cache()
 gc.collect();gc.disable()
 report['timing_repetitions']=6;report['full_request_warmups_per_policy']=2;report['gc_disabled_during_timing']=True
 for length in (128,2048):
  for warm in range(2):
   for policy in ('base','raw','arranged'):
    PolicyLinear.policy=policy;cache=DynamicCache(config=model.config)
    model(ids[:,:length],past_key_values=cache,use_cache=True,logits_to_keep=1)
    for t in range(length,length+32):model(ids[:,t:t+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
  torch.cuda.synchronize()
  orders=(('base','raw','arranged'),('raw','arranged','base'),('arranged','base','raw'),('arranged','raw','base'),('raw','base','arranged'),('base','arranged','raw'))
  for rep,order in enumerate(orders):
   for policy in order:
    PolicyLinear.policy=policy;cache=DynamicCache(config=model.config)
    value,prefill=timer(lambda:model(ids[:,:length],past_key_values=cache,use_cache=True,logits_to_keep=1))
    def decode_steps():
     for t in range(length,length+32):
      value=model(ids[:,t:t+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
     return value
    value,dec=timer(decode_steps)
    assert torch.isfinite(value.logits).all()
    row=dict(policy=policy,prompt_tokens=length,decode_tokens=32,rep=rep,prefill=prefill,decode=dec,request_cuda_ms=prefill['cuda_ms']+dec['cuda_ms'],request_wall_ms=prefill['wall_ms']+dec['wall_ms'])
    report['timings'].append(row);print('TIMING',row,flush=True);write(out/'report.json',report)
 report['status']='complete';report['elapsed_s']=time.perf_counter()-t0;write(out/'report.json',report)
if __name__=='__main__':main()
