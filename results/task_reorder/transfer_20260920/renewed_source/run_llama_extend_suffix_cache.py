"""Capture already-observed rejected sets for pooled development, with exact audits."""
import json,os,tempfile
from pathlib import Path
import torch
from run_llama_suffix_diagnose import write,ce
from run_c4_frozen import digest_file
from run_conditional_format import sha
from run_math_code_calibration import load_model
from run_task_reorder_eval import raw256_masks,mix_coarse
from quantize.quantizer import quant_nvfp4_4over6,quant_mix_4_6
from quantize.task_reorder import original_order_weight_reference
@torch.no_grad()
def main():
 assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 root=Path('/work/u4320956/task_reorder/transfer_20260920/llama8b');out=root/'suffix192';out.mkdir(exist_ok=False)
 original=json.loads((root/'suffix_diagnose/report.json').read_text());cache=Path(original['cache'])
 for n,h in original['cache_sha256'].items():assert digest_file(cache/n)==h
 captures=torch.load(cache/'captures.pt',map_location='cpu',weights_only=True)
 state=torch.load(cache/'state.pt',map_location='cpu',weights_only=True)
 names=state['names'];raw_ce=list(original['metrics']['0']);candidate_ce=[]
 sources=['gate_up_confirm','tile_refine_confirm'];sets=[]
 for source in sources:
  p=root/source;r=json.loads((p/'report.json').read_text());assert r['status']=='complete' and not r['passed'] and digest_file(p/'fresh.pt')==r['fresh_sha256']
  sets.append((source,r,torch.load(p/'fresh.pt',map_location='cpu',weights_only=True)))
 prior=json.loads((root/'calibration/report.json').read_text());raw=raw256_masks(root/'calibration',prior)
 plan=dict(development_only=True,previously_observed_sets=['confirmation']+sources,source_sha256=digest_file(__file__),no_candidate_promotion=True)
 write(out/'plan.json',plan);report=dict(status='running',plan=plan,job_id=os.environ['SLURM_JOB_ID'],audits=0)
 model,modules=load_model(prior,False);model.set_attn_implementation('sdpa')
 for n,m in modules.items():
  assert sha(m.weight)==prior['matrices'][n]['source_sha256']
  w=quant_nvfp4_4over6(m.weight,4,16)
  if raw[n].any():w=mix_coarse(w,quant_mix_4_6(m.weight,4,16,type_block=(8,64),clip='a1',elect='always'),raw[n])
  m.weight.copy_(w)
 last=model.model.layers[-1];captured={};capture=True
 def remember(key):
  def hook(m,inputs):
   if capture:captured[key]=inputs[0].detach().clone()
  return hook
 hooks=[m.register_forward_pre_hook(lambda m,inputs:(quant_nvfp4_4over6(inputs[0],4,16),*inputs[1:])) for m in modules.values()]
 hooks.extend([last.post_attention_layernorm.register_forward_pre_hook(remember('residual')),last.mlp.register_forward_pre_hook(remember('x'))])
 weights={n:{k:v.cuda() for k,v in state['weights'][n].items()} for n in names}
 for source,old,records in sets:
  controls={}
  for p in Path(json.loads((root/source/'plan.json').read_text())['candidate']).glob('*/layout.pt'):
   l=torch.load(p,map_location='cpu',weights_only=True);controls[l['name']]=l
  for i,row in enumerate(records):
   for n in names:modules[n].weight.copy_(weights[n]['raw'])
   capture=True;ids=row['ids'].cuda();got=ce(model(input_ids=ids,use_cache=False).logits,ids)
   assert got==old['metrics']['raw256']['ce'][i],('fullraw',source,i,got)
   capture=False;report['audits']+=1;raw_ce.append(got);captures.append(dict(x=captured['x'].cpu(),residual=captured['residual'].cpu(),**row))
   for n,l in controls.items():modules[n].weight.copy_(original_order_weight_reference(weights[n]['base'],weights[n]['alt'],l))
   got=ce(model.lm_head(model.model.norm(captured['residual']+last.mlp(captured['x']))),ids)
   assert got==old['metrics']['candidate']['ce'][i],('suffixcandidate',source,i,got)
   candidate_ce.append(got);report['audits']+=1
   if (i+1)%16==0:print('CAPTURE',source,i+1,flush=True)
 for h in hooks:h.remove()
 assert len(captures)==192 and len({r['document_sha256'] for r in captures})==192
 dest=Path(tempfile.mkdtemp(prefix='llama_suffix192_',dir=os.environ['TMPDIR']));torch.save(captures,dest/'captures.pt')
 report.update(status='complete',cache=str(dest),state_path=str(cache/'state.pt'),raw_ce=raw_ce,sources=[r['source'] for r in captures],cache_sha256=digest_file(dest/'captures.pt'),state_sha256=digest_file(cache/'state.pt'),original_report_sha256=digest_file(root/'suffix_diagnose/report.json'))
 write(out/'report.json',report);print('COMPLETE',report['audits'],str(dest),flush=True)
if __name__=='__main__':main()
