#!/usr/bin/env python3
"""Resolve omitted historical producers and local Python import dependencies.

Explicit input roots only; writes the review snapshot, never a campaign.
Does not execute imported research code. Rechecks source bytes after copying.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from prepare_public_snapshot import redact_text, discover_private_tokens

p=argparse.ArgumentParser()
p.add_argument('--workspace',type=Path,required=True)
a=p.parse_args()
s=Path(__file__).resolve().parents[1]
root=a.workspace/'research_runs'
roots={
 'primary':root/'mixfp4_n16k64_full_validation_20260911T065444Z',
 'ppl_improvement':root/'mixfp4_n16k64_ppl_improvement_20260914T162041Z',
 'mechanism':root/'mixfp4_mechanism_24h_20260916T205501Z',
 'followup':root/'mixfp4_n16k64_followup_20260917T090438Z',
 'boundary':root/'mixfp4_n16k64_boundary_corruption_20260918T080303Z'}
idx=json.loads((s/'SNAPSHOT_SOURCE_INDEX.json').read_text())
known={r['public_path']:r for r in idx['entries']}
added=[]
def copy(src,dst,key):
 raw=src.read_bytes(); h=hashlib.sha256(raw).hexdigest()
 if src.suffix=='.npz':
  data=raw
 else:
  text=raw.decode('utf-8')
  devices=sorted(set(re.findall(r'GPU-[0-9a-fA-F-]{36}',text)))
  public,counts=redact_text(text,discover_private_tokens([src]),{x:f'GPU-DEVICE-{i:02d}' for i,x in enumerate(devices)})
  data=public.encode()
 rel=dst.relative_to(s).as_posix()
 if rel in known: return
 dst.parent.mkdir(parents=True,exist_ok=True)
 dst.write_bytes(data);dst.chmod(0o755 if data.startswith(b'#!') else 0o644)
 if hashlib.sha256(src.read_bytes()).hexdigest()!=h: raise RuntimeError('source changed')
 row={'public_path':rel,'logical_source':f'campaign-artifact://{key}/{src.relative_to(roots[key]).as_posix()}',
      'source_sha256':h,'public_sha256':hashlib.sha256(data).hexdigest(),'public_bytes':len(data),
      'transform':'byte_identical' if data==raw else 'text_identity_and_path_redaction'}
 known[rel]=row;added.append(row)

for key in ['mechanism','followup','boundary']:
 for folder in ['analysis','job_specs','freeze_history','ops']:
  for src in sorted((roots[key]/folder).glob('*')):
   if src.is_file() and src.suffix in {'.py','.json','.md','.sha256'}:
    copy(src,s/'campaigns'/key/folder/src.name,key)
 for name in ['MAP_MANIFEST.json','BAND_DEFINITIONS.json','CORRUPTION_POOL_DEFINITIONS.json','BIN_DEFINITIONS.json']:
  src=roots[key]/name
  if src.is_file(): copy(src,s/'campaigns'/key/name,key)

for src in sorted((roots['boundary']/'arrays').glob('*paired_cluster_nll.npz')):
 copy(src,s/'campaigns/boundary/arrays'/src.name,'boundary')

audit=s/'validation/CURRENT_EVIDENCE_AUDIT.json'
if audit.exists():
 for row in json.loads(audit.read_text())['five_draw_inputs']:
  src=roots['primary']/row['calibration_manifest']
  copy(src,s/'campaigns/primary/calibration_manifests'/f"{row['model']}_{row['draw']}.json",'primary')

old=roots['ppl_improvement']/'source/NVFP4-RaZeR-main/campaign'
new=roots['ppl_improvement']/'source/NVFP4-RaZeR-extension-v2/campaign'
for src in old.rglob('*.py'):
 if '__pycache__' in src.parts: continue
 rel=src.relative_to(old)
 if not (new/rel).exists() or src.read_bytes()!=(new/rel).read_bytes():
  copy(src,s/'campaigns/ppl_improvement/source_v1_differences'/rel,'ppl_improvement')

for snapshot,key,tree in [('primary','primary','NVFP4-RaZeR-main'),('ppl_extension','ppl_improvement','NVFP4-RaZeR-extension-v2'),('boundary','boundary','NVFP4-RaZeR-main')]:
 source=roots[key]/'source'/tree
 dest=s/'software'/snapshot/'support'
 if key=='boundary':
  for name in ['test_followup_campaign.py','test_boundary_campaign.py']:
   copy(source/'tests'/name,dest/'tests'/name,key)
 # Fixed point over imports, including function-local imports; preserve original modules.
 processed=set()
 while True:
  files=[f for f in (s/'software'/snapshot).rglob('*.py') if f not in processed]
  if not files: break
  for file in files:
   processed.add(file)
   for node in ast.walk(ast.parse(file.read_bytes())):
    modules=[]
    if isinstance(node,ast.Import): modules=[x.name for x in node.names]
    elif isinstance(node,ast.ImportFrom) and not node.level and node.module: modules=[node.module]
    for module in modules:
     relative=Path(*module.split('.')).with_suffix('.py')
     candidate=source/relative
     if candidate.is_file() and not module.startswith('campaign'):
      copy(candidate,dest/relative,key)

idx['entries']=sorted(known.values(),key=lambda r:r['public_path'])
(s/'SNAPSHOT_SOURCE_INDEX.json').write_text(json.dumps(idx,indent=2,sort_keys=True)+'\n')
previous=json.loads((s/'validation/COMPLETENESS_ADDITIONS.json').read_text())['entries'] if (s/'validation/COMPLETENESS_ADDITIONS.json').exists() else []
report={'schema_version':1,'added_files':len(previous)+len(added),'entries':previous+added,
 'reason':'Reverse source audit found standalone mechanism producers, watchdogs, generated plans, definitions and top-level campaign tests omitted. Static local-import closure is now preserved.',
 'execution':'Copied historical scripts are provenance, not authorization to execute them. Paths redacted in supplementary operations scripts are archival placeholders; adapt in a new run root before use.'}
(s/'validation/COMPLETENESS_ADDITIONS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'added':len(added),'total_sources':len(known)}))
