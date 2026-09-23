"""Bounded filename inventory, including ignored files and linked campaigns.

Discovery is not hash/protocol acceptance. No tensor loading or GPU execution.
"""
from common import *
import subprocess

def main():
    start=now()
    command=['rg','--files','--hidden','--no-ignore','--follow','research_runs',
        '-g','*score*.pt','-g','*moment*.pt','-g','*score*.npz','-g','*covariance*',
        '-g','!**/cache/**','-g','!**/site-packages/**','-g','!**/.git/**']
    found=subprocess.run(command,cwd=REPO,check=True,capture_output=True,text=True)
    rows=[]
    for name in sorted(found.stdout.splitlines()):
        path=REPO/name
        rows.append(dict(path=name,resolved_path=str(path.resolve()),bytes=path.stat().st_size,
            discovery_only=True,accepted_by_this_search=False))
    result=dict(checked_utc=start,command=command,entries=rows,
        full_score_candidates=[r['path'] for r in rows if 'raw_scores_full' in r['path']],
        scope='All research_runs campaign directories, following symlinks and ignored paths; filename patterns shown above. Cache/site-packages/.git excluded. Not a claim to identify arbitrary unnamed tensors.',
        qualification='Sampled scores, combined additional-calibration moments, and rejected regeneration parents cannot replace verified seed0 full per-sequence child scores. Read metadata and validate source hashes/protocol before any reuse.',
        nested_archive_index='results/RAW_ARCHIVE_SEARCH.json')
    dest=OUT/'results/RAW_SCORE_SEARCH_EXPANDED.json';jsonout(dest,result)
    log('T0-raw-discovery-expanded','python scripts/raw_artifact_search.py',start,[dest])
    print('Candidates inventoried',len(rows),'full-score filename candidates',result['full_score_candidates'])

if __name__=='__main__':main()
