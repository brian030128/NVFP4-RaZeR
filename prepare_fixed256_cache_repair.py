"""Carry forward validated C4 and queue only WikiText cases with cache disabled."""
import argparse
import json
import os
from pathlib import Path
import shutil

assert os.environ.get('SLURM_JOB_ID')
ap=argparse.ArgumentParser(); ap.add_argument('--old',type=Path,required=True)
ap.add_argument('--new',type=Path,required=True); args=ap.parse_args()
cases=json.loads((args.new/'cases.json').read_text()); repairs=[]
for i,c in enumerate(cases):
    src=args.old/c['id']; dst=args.new/c['id']
    r=json.loads((src/'report.json').read_text())
    assert 'c4' in r['evaluation'] and r['source_weights_verified'],(c['id'],r.get('error'))
    shutil.copytree(src,dst)
    if not r.get('wiki_use_cache',False): repairs.append(i)
    else: assert r['status']=='complete'
(args.new/'repair_cases.txt').write_text(''.join(str(i)+'\n' for i in repairs))
(args.new/'repair_provenance.json').write_text(json.dumps(dict(original=str(args.old),
    repaired_wiki_cases=repairs,reused_complete_cases=[i for i in range(len(cases)) if i not in repairs]),indent=2)+'\n')
print('WIKI REPAIRS',repairs,flush=True)
