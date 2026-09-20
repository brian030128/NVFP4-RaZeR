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
 root=Path('/work/u4320956/task_reorder/transfer_20260920/llama8b');out=root/'tile_refine';out.mkdir(exist_ok=False)
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
 allidx=list(range(64));fit=allidx;hold=allidx
 plan=dict(development_only=True,previously_observed_documents=True,source_sha256=digest_file(__file__),parent=3,
  fit=fit,development_check=hold,selection='NEW candidate: at most2 exact add/remove refinements of previous6tile map on all64 already-observed development documents; minimize max(pooled CE+2SE, each-domain CE+1SE) vs raw; require map change and all64 CE2SE/domain-mean screen before new independent fresh64')
 write(out/'plan.json',plan)
 for label in (0,3,7):
  install(label);got=losses(allidx);assert got==parent['metrics'][str(label)],('cached replay',label)
 print('AUDITS192 PASSED',flush=True)
 install(3);mlp.down_proj.weight.copy_(weights[names[2]]['base']);base=losses(allidx)
 layout=torch.load(root/'compacted/002/layout.pt',map_location='cpu',weights_only=True)
 coordinates=layout['mask'].nonzero().tolist();assert len(coordinates)==39
 report=dict(status='running',job_id=os.environ['SLURM_JOB_ID'],plan=plan,base_down_ce=base,tiles=[],audits=192)
 source=json.loads((root/'tile_diagnose/report.json').read_text());assert source['status']=='complete'
 report['tiles']=source['tiles'];initial=json.loads((root/'tile_joint/report.json').read_text());chosen=initial['selected'].copy();trace=[]
 def objective(values):
  diffs=[v-parent['metrics']['0'][i] for v,i in zip(values,fit)]
  q=paired_bounds(diffs,[0.]*64);a=paired_bounds(diffs[:32],[0.]*32);b=paired_bounds(diffs[32:],[0.]*32)
  return max(q['mean']+2*q['se'],a['mean']+a['se'],b['mean']+b['se'])
 current=objective(initial['joint_ce'])
 for step in range(2):
  options=[]
  for t in range(len(coordinates)):
   proposal=sorted(set(chosen)^{t})
   trial=copy.deepcopy(layout);trial['mask'].zero_()
   for selected_index in proposal:trial['mask'][tuple(coordinates[selected_index])]=True
   w=original_order_weight_reference(weights[names[2]]['base'],weights[names[2]]['alt'],trial)
   mlp.down_proj.weight.copy_(w);del w;values=losses(fit)
   options.append((objective(values),t,values,proposal))
  score,t,values,proposal=min(options,key=lambda x:x[0])
  if score>=current:break
  chosen=proposal;current=score;trace.append(dict(step=step,tile=t,objective=score,development_ce=values))
  report['trace']=trace;write(out/'report.json',report);print('REFINE_STEP',step+1,'tiles',chosen,'objective',score,flush=True)
 report['map_changed']=chosen!=initial['selected']
 selected=[r for r in source['tiles'] if r['index'] in chosen]
 trial=copy.deepcopy(layout);trial['mask'].zero_()
 for r in selected:trial['mask'][r['row'],r['col']]=True
 frozen=out/'layouts';frozen.mkdir()
 for j in range(2):
  dest=frozen/f'{j:03d}';dest.mkdir();v=torch.load(root/f'compacted/{j:03d}/layout.pt',map_location='cpu',weights_only=True);torch.save(v,dest/'layout.pt');write(dest/'report.json',dict(status='complete',module=v['name']))
 dest=frozen/'002';dest.mkdir();torch.save(trial,dest/'layout.pt');write(dest/'report.json',dict(status='complete',module=trial['name']))
 install(3);mlp.down_proj.weight.copy_(original_order_weight_reference(weights[names[2]]['base'],weights[names[2]]['alt'],trial))
 joint=losses(allidx);report['joint_ce']=joint
 report['selected']=[r['index'] for r in selected];report['comparisons']={}
 for label,idx in [('all',allidx),('math',allidx[:32]),('code',allidx[32:])]:
  report['comparisons'][label]=paired_bounds([joint[i] for i in idx],[parent['metrics']['0'][i] for i in idx])
 q=report['comparisons'];report['passed_development']=report['map_changed'] and q['all']['mean']+2*q['all']['se']<0 and q['math']['mean']<=0 and q['code']['mean']<=0
 report['status']='complete';write(out/'report.json',report);print('RESULT',json.dumps({k:report[k] for k in ('selected','comparisons','passed_development')}),flush=True)
if __name__=='__main__':main()
