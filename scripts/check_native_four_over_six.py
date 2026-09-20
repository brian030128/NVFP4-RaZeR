"""Worker-only independent Python quantizer check for the native producer samples."""
import os,json,sys
from pathlib import Path
import torch
from quantize.quantizer import quant_nvfp4_4over6
assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
torch.set_num_threads(4)
p=Path(sys.argv[1]);rows=[]
for k in (5120,17408):
 x=torch.frombuffer(bytearray((p/f'input_k{k}.bin').read_bytes()),dtype=torch.bfloat16).clone().cuda()
 native=torch.frombuffer(bytearray((p/f'dequant_k{k}.bin').read_bytes()),dtype=torch.bfloat16).clone().cuda()
 ref=quant_nvfp4_4over6(x.reshape(1,k),4,16).reshape(-1)
 different=int((ref.view(torch.int16)!=native.view(torch.int16)).sum())
 rows.append(dict(k=k,values=k,differences=different,max_abs=float((ref.float()-native.float()).abs().max())))
report=dict(status='complete',job_id=os.environ['SLURM_JOB_ID'],passed=all(r['differences']==0 for r in rows),checks=rows,scope='Native baseline dequantization on distinct BF16 synthetic values vs independent Python quant_nvfp4_4over6; fused packed codes+SM100scales separately bitwise checked on8shapes.')
(p/'python_reference.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));assert report['passed']
