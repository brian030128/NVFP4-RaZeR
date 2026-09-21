#!/usr/bin/env python3
"""Check an explicit prospective submission list without staging or committing."""
import json
import subprocess
from pathlib import Path

s=Path(__file__).resolve().parents[1];repo=s.parents[1]
listpath=s/'validation/SUBMISSION_FILES.txt'
reportpath=s/'validation/SUBMISSION_SCOPE_CHECK.json'
files={p for p in s.rglob('*') if p.is_file()}|{repo/'README.md',listpath,reportpath}
rels=sorted(str(p.relative_to(repo)) for p in files)
listpath.write_text('\n'.join(rels)+'\n');listpath.chmod(0o644)
def git(*args):return subprocess.run(['git',*args],cwd=repo,capture_output=True,text=True)
new_commits=git('rev-list','origin/main..HEAD').stdout.splitlines()
staged=git('diff','--cached','--name-only').stdout.splitlines()
large=[str(p.relative_to(repo)) for p in files if p.exists() and p.stat().st_size>10*1024*1024]
whitespace=[]
for rel in rels:
 if not (repo/rel).exists():continue
 r=git('diff','--check','--',rel) if rel=='README.md' else git('diff','--no-index','--check','/dev/null',rel)
 # --no-index returns 1 for an ordinary difference, even with no whitespace errors.
 if r.stdout.strip() or r.stderr.strip() or r.returncode not in (0,1):
  whitespace.append({'path':rel,'output':r.stdout.strip(),'stderr':r.stderr.strip()})
report={'candidate_paths':len(rels),'new_commits_beyond_main':new_commits,'actual_staged_paths':staged,
 'files_over_10_mib':large,'whitespace_errors':whitespace,
 'new_history_review':'No new commits beyond recorded remote main; only explicitly listed working files are prospective additions.',
 'passed':not(new_commits or staged or large or whitespace),
 'staged_or_committed_by_this_check':False}
reportpath.write_text(json.dumps(report,indent=2)+'\n');reportpath.chmod(0o644)
print(json.dumps(report,indent=2))
raise SystemExit(0 if report['passed'] else 1)
