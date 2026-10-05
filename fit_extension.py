"""run_train_map.py --fit-windows N: the paper's 128 fit windows, then N - 128 more, for cost runs
(results/topk_cal/fit_windows/NOTE.md).

The extension is flipquant's rule (github brian030128/flipquant, flipquant/data.py ``calibration_sets_extended`` at
b1c4123), applied on top of the paper's fit set instead of flipquant's own:
- The streams are the same shards as the paper's fit record: OpenWebMath train-00000 @ fde8ef8 and CodeParrot-clean
  file-000000000001 @ 35a59fb. Each source gives (N - 128) / 2 windows, math first.
- ``stream_windows`` is flipquant's ``_stream_windows`` verbatim, with seed ``SEED + 2`` = 20260930:
  - it reads the stream in file order and skips a document whose sha256 is in the skip set, or that it has already
    seen;
  - it skips documents shorter than 512 tokens;
  - it takes one window at ``random.Random(seed).randrange(len - 512 + 1)`` from each of the first ``count``
    documents.
- The skip set is the paper's fit documents, the paper's 192 development documents (the model's three development
  records) and its published C4 evaluation documents. flipquant skips its own fit and development documents instead.
  WikiText-2 is another corpus, so it needs no skip set.
Every window is recorded by document sha256, offset and token sha256.
"""
import hashlib
import json
import random
from pathlib import Path

SEED = 20260928          # flipquant/data.py SEED; the extension draws with SEED + 2
FIELD = {'math': 'text', 'code': 'content'}
TOKENS = 512


def token_sha(t):
    """The fit record's token hash (run_conditional_format.sha)."""
    from run_conditional_format import sha
    return sha(t)


def stream_windows(tok, meta, source, count, skip, seed):
    """flipquant's _stream_windows on the paper record's stream of ``source``."""
    from datasets import load_dataset
    stream = load_dataset(meta['repo'], revision=meta['revision'], data_files={'train': meta['path']}, split='train',
                          streaming=True)
    rng, seen, out = random.Random(seed), set(skip), []
    for row in stream:
        text = row[FIELD[source]]
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        ids = tok(text, return_tensors='pt').input_ids
        if ids.shape[1] < TOKENS:
            continue
        offset = rng.randrange(ids.shape[1] - TOKENS + 1)
        out.append(dict(source=source, document_sha256=digest, offset=offset, ids=ids[:, offset:offset + TOKENS].clone()))
        if len(out) == count:
            return out
    raise RuntimeError(f'{source}: only {len(out)} usable documents')


def skip_documents(prior, development_dirs, published_report):
    """The paper's fit, development and C4 evaluation document sha256 values, by kind."""
    fit = {d['document_sha256'] for s in ('math', 'code') for d in prior['fit'][s]['documents']}
    dev = set()
    for d in development_dirs:
        dev |= {r['document_sha256'] for r in json.loads((Path(d) / 'fresh_manifest.json').read_text())['records']}
    c4 = set()
    if published_report is not None:
        c4 = {d['document_sha256'] for d in json.loads(Path(published_report).read_text())['data']['c4_paper']['documents']}
    return dict(fit=fit, development=dev, c4_evaluation=c4)


def extend(tok, prior, fit_windows, skip):
    """The N - 128 extra windows (math then code) and their records."""
    assert fit_windows % 2 == 0 and fit_windows > 128, fit_windows
    used = set().union(*skip.values())
    extra = []
    for source in ('math', 'code'):
        extra += stream_windows(tok, prior['fit'][source], source, fit_windows // 2 - 64, used, SEED + 2)
    records = [dict(source=r['source'], document_sha256=r['document_sha256'], offset=r['offset'],
                    token_sha256=token_sha(r['ids'])) for r in extra]
    docs = [r['document_sha256'] for r in records]
    assert len(set(docs)) == len(docs) and not set(docs) & used
    return [r['ids'] for r in extra], dict(
        rule="flipquant calibration_sets_extended's _stream_windows (seed SEED + 2 = 20260930) on the paper record's "
             'streams, on top of the paper fit set; skipping the paper fit, development and published C4 documents',
        flipquant_reference='brian030128/flipquant flipquant/data.py calibration_sets_extended @ b1c4123',
        seed=SEED + 2, windows=len(records), per_source=fit_windows // 2 - 64,
        skipped_documents={k: len(v) for k, v in skip.items()}, records=records)
