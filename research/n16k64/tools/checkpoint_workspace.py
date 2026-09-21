#!/usr/bin/env python3
"""Local recovery checkpoint; never edits the input workspace or follows symlinks.

Large evidence and environments remain in place. Their omission is explicit.
The checkpoint is private and must not be published.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime, timezone

p = argparse.ArgumentParser()
p.add_argument('--workspace', type=Path, required=True)
p.add_argument('--delivery', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
def git(*args):
    return subprocess.check_output(['git', '-C', str(a.workspace), *args])
state = {k: git(*v).decode() for k,v in {
    'head':['rev-parse','HEAD'], 'branch':['branch','--show-current'],
    'status':['status','--porcelain=v1','--untracked-files=all'],
    'worktrees':['worktree','list','--porcelain'],
}.items()}
state['created_utc'] = datetime.now(timezone.utc).isoformat()
(a.output/'tracked.patch').write_bytes(git('diff','--binary','HEAD'))
subprocess.run(['git','-C',str(a.workspace),'bundle','create',str(a.output/'history.bundle'),'--all'],check=True)
extensions={'.py','.json','.jsonl','.md','.csv','.yaml','.yml','.toml','.sh','.sbatch','.txt','.sha256','.lock'}
skip={'env','cache','.git','__pycache__','.pytest_cache','models','datasets','node_modules'}
entries=[]
with tarfile.open(a.output/'recovery.tar.gz','w:gz') as tar:
    for label,root in [('original',a.workspace),('delivery',a.delivery)]:
        for parent,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if d not in skip]
            for name in files:
                f=Path(parent)/name
                rel=f.relative_to(root)
                if f.is_symlink():
                    tar.add(f,arcname=str(Path(label)/rel),recursive=False)
                    continue
                if f.suffix not in extensions or f.stat().st_size > 5_000_000: continue
                before=hashlib.sha256(f.read_bytes()).hexdigest()
                tar.add(f,arcname=str(Path(label)/rel),recursive=False)
                after=hashlib.sha256(f.read_bytes()).hexdigest()
                if before != after: raise RuntimeError('unstable checkpoint source: '+str(rel))
                entries.append({'source':label+'/'+str(rel),'sha256':before,'bytes':f.stat().st_size})
state['files']=entries
state['exclusions']='Symlink campaign targets, models, datasets, environments, caches, binaries and files >5 MB stay untouched in place; this is not a full raw-evidence backup.'
(a.output/'CHECKPOINT.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps({'checkpoint':str(a.output),'files':len(entries),'head':state['head'].strip()}))
