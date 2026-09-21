"""Verify a completed frozen confirmation before any PPL allocation (metadata only)."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eligible(experiment):
    report = json.loads((experiment / 'report.json').read_text())
    if report['status'] != 'complete' or report['completed_sequences'] != 64:
        raise ValueError('Confirmation incomplete')
    if report['candidates'] != 1 or not report['fixed_candidate_before_fresh_data'] or not report['new_windows']:
        raise ValueError('Not a frozen independent single-candidate confirmation')
    if digest(experiment / 'plan.json') != report['plan_sha256']:
        raise ValueError('Plan changed')
    plan = json.loads((experiment / 'plan.json').read_text())
    if report['layout_sha256'] != plan['layout_sha256']:
        raise ValueError('Confirmation layouts differ from frozen plan')
    for path, expected in report['layout_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Confirmed layout changed')
    for file, key in [('fresh.pt', 'fresh_sha256'), ('fresh_manifest.json', 'fresh_manifest_sha256')]:
        if digest(experiment / file) != report[key]:
            raise ValueError('Confirmation data changed')
    paired = report['paired']
    upper = [paired[ref]['all'][q]['upper'] for ref in ('raw256', 'identity') for q in ('ce', 'kl')]
    means = [paired['raw256'][domain][q]['mean'] for domain in ('math', 'code') for q in ('ce', 'kl')]
    passed = all(math.isfinite(v) and v < 0 for v in upper) and all(math.isfinite(v) and v <= 0 for v in means)
    if passed != report['passed']:
        raise ValueError('Reported decision contradicts frozen criterion')
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    if eligible(parser.parse_args().experiment):
        print('identity domain_pruned')
