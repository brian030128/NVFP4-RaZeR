"""Additional fail-closed cadence guard around unchanged pinned launchers.

No scientific payload changes. Original adapter hashes remain recorded;
cadence_guard.json additionally identifies this runtime policy wrapper.
"""
import argparse
import runpy
import time
import sys
from common import *

PINNED={
    'accuracy_run.py':'431a256b849b34a517f4e08b009ab58d3f0d94bd7f4db4f3a6204319617f4ed8',
    'gpu_run_wait.py':'7c3d65eecf72b83d0a8cf4416d846593ea8d7f8f1e4dfca4f8db17abe27ac8e4',
}

class Cadence:
    def __init__(self):self.previous=None
    def enforce(self,record,begin,end):
        if record['phase']!='during':return record
        reasons=list(record['reasons'])
        if end-begin>60:reasons.append('GPU ownership query duration exceeds 60 seconds')
        if self.previous is not None and begin-self.previous>60:
            reasons.append('GPU ownership monitoring interval exceeds 60 seconds')
        self.previous=begin
        result=dict(record,passed=record['passed'] and not reasons,reasons=reasons)
        result['cadence_guard']=dict(query_seconds=end-begin,maximum_seconds=60)
        return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True,choices=PINNED)
    a,rest=ap.parse_known_args();entry=OUT/'scripts'/a.entry
    assert sha(entry)==PINNED[a.entry]
    assert '--name' in rest and '--inspect-only' not in rest
    name=rest[rest.index('--name')+1];assert name.replace('_','').replace('-','').isalnum()
    root=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z/runs'/name
    assert not root.exists(),'Never overwrite a prior attempt or waiting directory'
    sys.path.insert(0,str(PRIMARY/'source/NVFP4-RaZeR-main'))
    from campaign import gpu_preflight as gp
    original=gp.check;guard=Cadence();recorded=False
    def checked(*args,**kwargs):
        nonlocal recorded
        if not recorded:
            assert root.is_dir()
            jsonout(root/'cadence_guard.json',dict(wrapper=str(Path(__file__).relative_to(REPO)),
                wrapper_sha256=sha(Path(__file__)),entry=a.entry,entry_sha256=PINNED[a.entry],
                maximum_seconds=60,scientific_payload_changed=False,created_utc=now()))
            recorded=True
        begin=time.monotonic();result=original(*args,**kwargs);end=time.monotonic()
        return guard.enforce(result,begin,end)
    gp.check=checked
    sys.argv=[str(entry)]+rest
    runpy.run_path(str(entry),run_name='__main__')

if __name__=='__main__':main()
