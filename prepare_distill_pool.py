"""Larger distillation budgets (cost study arm C3) from the fit set's own OpenWebMath / CodeParrot files.

Streams prior['fit'][source] (repo, revision, file path) in file order, the files
math_code_data reads, and keeps documents that are not fit or development documents
(document sha256), have at least 512 tokens, and whose window is not a known window
(token sha256). One 512-token window per document is cut at an offset drawn from a
seeded torch.Generator, the convention of run_fine_row_research.fresh_data that produced
the development sets. The first --per-source windows of each source are kept, in file
order; smaller budgets use prefixes of each source. Records have the fresh.pt layout.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import try_to_load_from_cache
from transformers import AutoTokenizer

from run_c4_frozen import digest_file
from run_conditional_format import sha
from run_multiround import data_paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--per-source', type=int, required=True)
    ap.add_argument('--seed', type=int, default=20260923)
    ap.add_argument('--out', type=Path, required=True, help='pool.pt; the manifest is written next to it')
    args = ap.parse_args()
    calibration, development = data_paths('llama8b', args.data_root)
    prior = json.loads((calibration / 'report.json').read_text())
    excluded = {d['document_sha256'] for m in prior['fit'].values() for d in m['documents']}
    known = {t for m in prior['fit'].values() for t in m['token_sha256']}
    dev_documents = 0
    for directory in development:
        report = json.loads((directory / 'report.json').read_text())
        assert digest_file(directory / 'fresh.pt') == report['fresh_sha256'], directory
        for r in torch.load(directory / 'fresh.pt', map_location='cpu', weights_only=True):
            excluded.add(r['document_sha256']); known.add(r['token_sha256']); dev_documents += 1
    assert len(excluded) == 128 + dev_documents == 320
    tok = AutoTokenizer.from_pretrained(prior['source'], revision=prior['revision'])
    generator = torch.Generator().manual_seed(args.seed)
    records, stats, files = [], {}, {}
    for source in ('math', 'code'):
        meta = prior['fit'][source]
        cached = try_to_load_from_cache(meta['repo'], meta['path'], revision=meta['revision'], repo_type='dataset')
        files[source] = dict(repo=meta['repo'], revision=meta['revision'], path=meta['path'],
                             sha256=digest_file(cached) if isinstance(cached, str) else None)
        stream = load_dataset(meta['repo'], revision=meta['revision'], data_files={'train': meta['path']},
                              split='train', streaming=True)
        s = dict(scanned=0, excluded=0, short=0, known_window=0, kept=0)
        for row in stream:
            s['scanned'] += 1
            content = row['text' if source == 'math' else 'content']
            digest = hashlib.sha256(content.encode()).hexdigest()
            if digest in excluded:
                s['excluded'] += 1
                continue
            ids = tok(content, return_tensors='pt').input_ids
            if ids.shape[1] < 512:
                s['short'] += 1
                continue
            offset = int(torch.randint(ids.shape[1] - 512 + 1, (1,), generator=generator))
            ids = ids[:, offset:offset + 512].clone()
            token_hash = sha(ids)
            if token_hash in known:
                s['known_window'] += 1
                continue
            records.append(dict(source=source, document_sha256=digest, offset=offset, token_sha256=token_hash, ids=ids))
            excluded.add(digest); known.add(token_hash); s['kept'] += 1
            if s['kept'] == args.per_source:
                break
        s['exhausted'] = s['kept'] < args.per_source
        stats[source] = s
        print(source, json.dumps(s), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, args.out)
    manifest = dict(pool=str(args.out), pool_sha256=digest_file(args.out), seed=args.seed, per_source=args.per_source,
                    window_tokens=512, files=files, stats=stats, excluded_fit_documents=128,
                    excluded_development_documents=dev_documents,
                    records=[{k: v for k, v in r.items() if k != 'ids'} for r in records])
    args.out.with_name(args.out.stem + '_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('POOL', args.out, manifest['pool_sha256'], {k: v['kept'] for k, v in stats.items()}, flush=True)


if __name__ == '__main__':
    main()
