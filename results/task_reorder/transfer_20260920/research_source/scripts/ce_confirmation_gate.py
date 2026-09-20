"""Verify the prospective CE-primary fresh confirmation, including frozen artifacts."""
import argparse
import json
import math
from pathlib import Path
from scripts.frozen_confirmation_gate import digest

def passes(paired):
    ce=[paired[ref]['all']['ce'] for ref in ('raw256','identity')]
    means=[paired['raw256'][domain]['ce']['mean'] for domain in ('math','code')]
    return all(math.isfinite(q['mean']) and math.isfinite(q['se']) and q['se']>=0 and q['mean']+2*q['se']<0 for q in ce) and all(math.isfinite(v) and v<=0 for v in means)

def eligible(experiment):
    report=json.loads((experiment/'report.json').read_text());plan=json.loads((experiment/'plan.json').read_text())
    assert plan['protocol_family']=='ce_primary_after_positive_control'
    assert report['status']=='complete' and report['completed_sequences']==64
    assert report['candidates']==1 and report['fixed_candidate_before_fresh_data'] and report['new_windows']
    assert digest(experiment/'plan.json')==report['plan_sha256']
    assert report['layout_sha256']==plan['layout_sha256']
    for p,h in report['layout_sha256'].items():assert digest(p)==h
    for file,key in [('fresh.pt','fresh_sha256'),('fresh_manifest.json','fresh_manifest_sha256')]:assert digest(experiment/file)==report[key]
    passed=passes(report['paired']);assert passed==report['passed']
    return passed

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path)
    if eligible(p.parse_args().experiment):print('CE_PRIMARY_PASSED')
