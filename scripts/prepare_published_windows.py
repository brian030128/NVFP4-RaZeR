"""Freeze the published evaluation windows to a file the native runner can load.

The native runtime lives in a minimal venv that has torch and transformers but
not `datasets`, and that venv is shared with the gated latency jobs, so it should
not grow dependencies for this. Building the windows here instead is also the
better experiment: the native run then consumes byte-identical token tensors to
the simulated run rather than re-deriving them from the datasets, and each window
carries the same `token_sha256` the published protocol records.

Windows come from `run_baseline_protocol_audit.data`, unchanged: WikiText-2-raw
test joined on blank lines and cut into full 2048-token spans, and 256 crops of
2048 tokens drawn from C4 validation shard 0 under seed 0.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from run_baseline_protocol_audit import data
from run_c4_frozen import digest_file
from run_conditional_format import sha


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--calibration', type=Path,
                    default=Path('/work/u4320956/task_reorder/transfer_20260920/llama8b/calibration'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--length', type=int, default=2048)
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run through Slurm')

    prior = json.loads((args.calibration / 'report.json').read_text())
    checkpoint = Path(os.environ['HF_HOME']) / 'hub' / (
        'models--' + prior['source'].replace('/', '--')) / 'snapshots' / prior['revision']
    if not checkpoint.is_dir():
        raise RuntimeError(f'Missing checkpoint snapshot {checkpoint}')
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)

    batches, meta = data(tokenizer, prior, args.length)
    # Keep only the two published evaluation domains; c4_study is a 512-token
    # construct and is not part of this protocol.
    batches = {d: batches[d] for d in ('wiki', 'c4_paper')}
    meta = {d: meta[d] for d in ('wiki', 'c4_paper')}
    for domain, windows in batches.items():
        recorded = meta[domain]['token_sha256']
        if [sha(w) for w in windows] != recorded:
            raise RuntimeError(f'{domain} window digests do not match their metadata')
        if any(w.shape != (1, args.length) for w in windows):
            raise RuntimeError(f'{domain} contains a window of the wrong length')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(batches=batches, meta=meta, length=args.length,
                    source=prior['source'], revision=prior['revision']), args.out)
    report = dict(status='complete', job_id=os.environ['SLURM_JOB_ID'],
                  out=str(args.out), windows_sha256=digest_file(args.out),
                  source=prior['source'], revision=prior['revision'], length=args.length,
                  windows={d: len(b) for d, b in batches.items()})
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print('RESULT ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
