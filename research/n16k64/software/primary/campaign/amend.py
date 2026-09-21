"""Append a hash-chained entry to provenance/PROTOCOL_AMENDMENTS.jsonl. The frozen protocol file itself is never edited.

Each entry: {seq, utc, kind, affects_confirmatory, results_viewed_before, title, detail, source_files{path: sha256}, prev_sha256, entry_sha256}
where entry_sha256 = sha256(canonical JSON of the entry without entry_sha256)."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

CR = Path(os.environ['CAMPAIGN_ROOT'])
LOG = CR / 'provenance' / 'PROTOCOL_AMENDMENTS.jsonl'
KINDS = ('implementation_fix', 'evaluation_addition', 'operational', 'scope_note')


def canon(d):
    return json.dumps(d, sort_keys=True, separators=(',', ':')).encode()


def entries():
    return [json.loads(l) for l in LOG.read_text().splitlines()] if LOG.exists() else []


def verify():
    prev = None
    for i, e in enumerate(entries()):
        body = {k: v for k, v in e.items() if k != 'entry_sha256'}
        if e['seq'] != i + 1 or e['prev_sha256'] != prev or hashlib.sha256(canon(body)).hexdigest() != e['entry_sha256']:
            raise SystemExit(f'amendment chain broken at seq {i + 1}')
        prev = e['entry_sha256']
    return prev


def append(kind, title, detail, affects_confirmatory, results_viewed_before, files, utc=None):
    if kind not in KINDS:
        raise ValueError(kind)
    prev = verify()
    src = CR / 'source' / 'NVFP4-RaZeR-main'
    body = dict(seq=len(entries()) + 1, utc=utc or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), kind=kind, title=title, detail=detail,
                affects_confirmatory=affects_confirmatory, results_viewed_before=results_viewed_before,
                protocol_freeze_sha256=(CR / 'freeze' / 'PROTOCOL_FREEZE.sha256').read_text().split()[0],
                source_files={f: hashlib.sha256((src / f).read_bytes()).hexdigest() for f in files}, prev_sha256=prev)
    body['entry_sha256'] = hashlib.sha256(canon(body)).hexdigest()
    with open(LOG, 'a') as fh:
        fh.write(json.dumps(body, sort_keys=True) + '\n')
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', required=True, choices=KINDS)
    ap.add_argument('--title', required=True)
    ap.add_argument('--detail', required=True)
    ap.add_argument('--affects-confirmatory', required=True, choices=('yes', 'no'))
    ap.add_argument('--results-viewed-before', required=True)
    ap.add_argument('--utc')
    ap.add_argument('--files', nargs='*', default=[])
    a = ap.parse_args()
    e = append(a.kind, a.title, a.detail, a.affects_confirmatory == 'yes', a.results_viewed_before, a.files, a.utc)
    print(e['seq'], e['entry_sha256'])


if __name__ == '__main__':
    main()
