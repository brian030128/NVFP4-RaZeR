"""Cache-only finite CE diagnosis of down tiles; old confirmation is development."""
import copy,json,os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP,LlamaRMSNorm
from run_c4_frozen import digest_file
from quantize.quantizer import quant_nvfp4_4over6
from quantize.task_reorder import original_order_weight_reference
from run_reorder_replay import paired_bounds

def write(p,r):p.write_text(json.dumps(r,indent=2)+'\n')
@torch.no_grad()
def main():
 assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 root=Path('/work/u4320956/task_reorder/transfer_20260920/llama8b');out=root/'tile_robust';out.mkdir(exist_ok=False)
 parent=json.loads((root/'suffix_diagnose/report.json').read_text());assert parent['status']=='complete' and parent['selected']==3
 cache=Path(parent['cache'])
 for n,h in parent['cache_sha256'].items():assert digest_file(cache/n)==h
 state=torch.load(cache/'state.pt',map_location='cpu',weights_only=True);records=torch.load(cache/'captures.pt',map_location='cpu',weights_only=True)
 cfg=LlamaConfig(**state['config']);mlp=LlamaMLP(cfg).cuda().bfloat16().eval();norm=LlamaRMSNorm(cfg.hidden_size,eps=cfg.rms_norm_eps).cuda().bfloat16().eval();norm.weight.copy_(state['norm'])
 head=state['head'].cuda();names=state['names'];weights={n:{k:v.cuda() for k,v in state['weights'][n].items()} for n in names}
 del state
 handles=[m.register_forward_pre_hook(lambda mod,inputs:(quant_nvfp4_4over6(inputs[0],4,16),)) for m in (mlp.gate_proj,mlp.up_proj,mlp.down_proj)]
 def install(label):
  for j,n in enumerate(names):getattr(mlp,n.split('.')[-1]).weight.copy_(weights[n]['both' if label&(1<<j) else 'raw'])
 def losses(indices):
  values=[]
  for i in indices:
   r=records[i];ids=r['ids'].cuda();x=r['x'].cuda();res=r['residual'].cuda()
   logits=F.linear(norm(res+mlp(x)),head);lp=logits[:,:-1].float().reshape(-1,logits.shape[-1]).log_softmax(-1)
   values.append(float(F.nll_loss(lp,ids[:,1:].reshape(-1))))
  return values
 allidx=list(range(64));fit=list(range(16))+list(range(32,48));hold=list(range(16,32))+list(range(48,64))
 plan=dict(development_only=True,previously_observed_documents=True,source_sha256=digest_file(__file__),parent=3,
  fit=fit,development_check=hold,selection='Greedy minimax of pooled predicted CE+2SE and math/code predicted CE means on fit32; at most20 elected down tiles; cached finite single-tile responses; exact joint remaining32 check; new fresh64 required')
 write(out/'plan.json',plan)
 for label in (0,3,7):
  install(label);got=losses(allidx);assert got==parent['metrics'][str(label)],('cached replay',label)
 print('AUDITS192 PASSED',flush=True)
 install(3);mlp.down_proj.weight.copy_(weights[names[2]]['base']);base=losses(allidx)
 layout=torch.load(root/'compacted/002/layout.pt',map_location='cpu',weights_only=True)
 coordinates=layout['mask'].nonzero().tolist();assert len(coordinates)==39
 report=dict(status='running',job_id=os.environ['SLURM_JOB_ID'],plan=plan,base_down_ce=base,tiles=[],audits=192)
 source=json.loads((root/'tile_diagnose/report.json').read_text());assert source['status']=='complete' and source['audits']==192
 report['tiles']=source['tiles']
 shifts={r['index']:[v-base[i] for v,i in zip(r['fit_ce'],fit)] for r in source['tiles']}
 anchor=[base[i]-parent['metrics']['0'][i] for i in fit]
 def objective(vec):
  p=paired_bounds(vec,[0.]*len(vec))
  return max(p['mean']+2*p['se'],sum(vec[:16])/16,sum(vec[16:])/16)
 chosen=[];vec=anchor.copy();trace=[]
 for step in range(20):
  attempts=[(objective([v+d for v,d in zip(vec,shifts[t])]),t) for t in shifts if t not in chosen]
  value,t=min(attempts)
  if value>=objective(vec):break
  chosen.append(t);vec=[v+d for v,d in zip(vec,shifts[t])];trace.append(dict(tile=t,objective=value))
 # Bounded exact add/remove swaps in the cached additive loss model.
 for step in range(4):
  attempts=[]
  for old in chosen:
   for new in shifts:
    if new in chosen:continue
    v=[a-b+c for a,b,c in zip(vec,shifts[old],shifts[new])]
    attempts.append((objective(v),old,new,v))
  if not attempts:break
  value,old,new,v=min(attempts,key=lambda t:t[0])
  if value>=objective(vec):break
  chosen.remove(old);chosen.append(new);vec=v;trace.append(dict(remove=old,add=new,objective=value))
 selected=[r for r in source['tiles'] if r['index'] in chosen]
 report['fit_search_trace']=trace;report['predicted_fit_objective']=objective(vec)
 trial=copy.deepcopy(layout);trial['mask'].zero_()
 for r in selected:trial['mask'][r['row'],r['col']]=True
 frozen=out/'layouts';frozen.mkdir()
 for j in range(2):
  dest=frozen/f'{j:03d}';dest.mkdir();v=torch.load(root/f'compacted/{j:03d}/layout.pt',map_location='cpu',weights_only=True);torch.save(v,dest/'layout.pt');write(dest/'report.json',dict(status='complete',module=v['name']))
 dest=frozen/'002';dest.mkdir();torch.save(trial,dest/'layout.pt');write(dest/'report.json',dict(status='complete',module=trial['name']))
 install(3);mlp.down_proj.weight.copy_(original_order_weight_reference(weights[names[2]]['base'],weights[names[2]]['alt'],trial))
 joint=losses(allidx);report['joint_ce']=joint
 report['selected']=[r['index'] for r in selected];report['comparisons']={}
 for label,idx in [('all',allidx),('fit',fit),('hold',hold),('hold_math',hold[:16]),('hold_code',hold[16:])]:
  report['comparisons'][label]=paired_bounds([joint[i] for i in idx],[parent['metrics']['0'][i] for i in idx])
 q=report['comparisons'];report['passed_development']=q['hold']['mean']+2*q['hold']['se']<0 and q['hold_math']['mean']<=0 and q['hold_code']['mean']<=0
 report['status']='complete';write(out/'report.json',report);print('RESULT',json.dumps({k:report[k] for k in ('selected','comparisons','passed_development')}),flush=True)
if __name__=='__main__':main()
