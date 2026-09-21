#!/usr/bin/env python3
"""Inventory original campaign code/config candidates, including symlink roots.

Classifies exclusions rather than equating copy integrity with completeness.
Only the public inventory output is written; no original paths are changed.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from prepare_public_snapshot import origin_for

p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
s=Path(__file__).resolve().parents[1]
inv=json.loads((s/'ARTIFACT_INVENTORY.json').read_text())
roots={r['public_snapshot'].split('/')[1]:a.workspace/'research_runs'/r['id'] for r in inv['campaigns']}
source_index=json.loads((s/'SNAPSHOT_SOURCE_INDEX.json').read_text())['entries']
by_source={};by_hash={}
for row in source_index:
 scheme,tail=row['logical_source'].split('://',1);key,rel=tail.split('/',1)
 src=roots[key]/rel if scheme=='campaign-artifact' else origin_for(Path(row['public_path']),roots)[1]
 by_source[str(src.resolve())]=row['public_path']
 by_hash.setdefault(row['source_sha256'],row['public_path'])
main={}
tracked=subprocess.check_output(['git','-C',str(a.workspace),'ls-files','-z']).decode().split('\0')
for rel in tracked:
 f=a.workspace/rel
 if f.is_file() and f.suffix in {'.py','.sh','.sbatch','.yaml','.yml','.toml','.in','.json'}:
  main.setdefault(hashlib.sha256(f.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),rel)
rows=[];cutoff=datetime.fromisoformat(inv['snapshot_cutoff_utc'].replace('Z','+00:00')).timestamp()
for key,root in roots.items():
 for parent,dirs,files in os.walk(root.resolve(),followlinks=False):
  dirs[:]=[d for d in dirs if d not in {'env','cache','.git','__pycache__','.pytest_cache','node_modules'}]
  for name in files:
   f=Path(parent)/name
   rel=f.relative_to(root.resolve()).as_posix()
   config_json=f.suffix=='.json' and ('job_specs' in f.parts or 'plans' in f.parts or f.parent.name.startswith('NVFP4-RaZeR') or name in {'config.json','configuration.json','EXPERIMENT_SPEC.json'})
   if f.is_symlink() or (f.suffix not in {'.py','.sh','.sbatch','.yaml','.yml','.toml','.in'} and not config_json):continue
   raw=f.read_bytes();h=hashlib.sha256(raw).hexdigest()
   target=by_source.get(str(f.resolve())) or by_hash.get(h)
   base=main.get(hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest())
   if f.stat().st_mtime>cutoff:
    action='defer_post_cutoff';reason='Newer than fixed experimental snapshot cutoff; no automatic promotion.'
   elif target:
    action='included';reason='Source-index path or identical source bytes.'
   elif base:
    action='inherited_main';target='../../'+base;reason='Same content as recorded main (line endings normalized for this comparison).'
   elif '/handoff/' in '/'+rel or rel.startswith('handoff/'):
    action='excluded_duplicate_handoff';reason='Original input archive retained locally with archive hash; historical source lineage indexed separately.'
   elif rel.startswith(('queue/','runs/','plans/')):
    action='excluded_generated_run_record';reason='Per-attempt machine launch/config artifact stays in sealed local campaign; generic launcher and authoritative plans/protocols included.'
   elif rel.startswith('source/') and '/campaign/' not in rel:
    action='excluded_historical_driver';reason='Earlier source driver/test/config outside the frozen campaign entry points and their static local import closure; full original archive remains external.'
   else:
    action='needs_review';reason='Unclassified custom code/config: blocks completeness until reviewed.'
   rows.append({'source':f'campaign-artifact://{key}/{rel}','sha256':h,'bytes':len(raw),'delivery_path':target,'disposition':action,'reason':reason})
report={'schema_version':1,'checked_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'All code/config suffix candidates in five campaign roots including symlink targets, excluding env/cache/git/bytecode; raw outcome classes are in ARTIFACT_INVENTORY.',
 'counts':dict(Counter(r['disposition'] for r in rows)),'entries':rows,
 'unclassified':[r['source'] for r in rows if r['disposition']=='needs_review']}
(s/'validation/REVERSE_SOURCE_INVENTORY.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'counts':report['counts'],'unclassified':report['unclassified']},indent=2))
