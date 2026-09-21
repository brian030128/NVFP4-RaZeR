"""Offline, hash-verified calibration and evaluation inputs.

All readers use the pinned files in $HF_HUB_CACHE (see provenance/INPUT_FETCH_MANIFEST.json) and
reproduce the archived `datasets.load_dataset` row order. Token tensors are hashed exactly like the
archived code (`sha256(tensor.view(uint8))` on the int64 [1, L] tensor).
"""
import bisect
import gzip
import hashlib
import json
import os
import random
import re
from pathlib import Path

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]
WIKI = ('Salesforce/wikitext', 'b08601e04326c79dfdd32d625aee71d232d685c3', 'wikitext-2-raw-v1/test-00000-of-00001.parquet')
C4 = ('allenai/c4', '1588ec454efa1a09f29cd18ddd04fe05fc8653a2', 'en/c4-validation.00000-of-00008.json.gz')
MATH = ('open-web-math/open-web-math', 'fde8ef8de2300f5e778f56261843dab89f230815', 'data/train-00000-of-00114-5a023365406cb9c4.parquet')
CODE = ('codeparrot/codeparrot-clean', '35a59fb025bc0a102f7d96eac09d145b896d487b', 'file-000000000001.json.gz')
PG19 = ('emozilla/pg19-test', 'c5e39bf32e33f9111323aa68d7d9000d22722035')
ARXIV = ('ccdv/arxiv-summarization', '240aaf1a969b3f8cd0ade6986bfad0cd730ee288')
GOVREPORT = ('ccdv/govreport-summarization', '4e21184e01ae8017e2c036e180fe5e541fef60a0')


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def hub_file(repo, revision, path):
    hub = Path(os.environ.get('HF_HUB_CACHE', Path(os.environ['HF_HOME']) / 'hub'))
    p = hub / ('datasets--' + repo.replace('/', '--')) / 'snapshots' / revision / path
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def hub_snapshot(repo, revision):
    hub = Path(os.environ.get('HF_HUB_CACHE', Path(os.environ['HF_HOME']) / 'hub'))
    p = hub / ('datasets--' + repo.replace('/', '--')) / 'snapshots' / revision
    if not p.is_dir():
        raise FileNotFoundError(p)
    return p


def parquet_column(path, column):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=[column]).column(column).to_pylist()


def jsonl_gz_column(path, column):
    out = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line)[column])
    return out


def source_texts(domain):
    if domain == 'math':
        return parquet_column(hub_file(*MATH), 'text')
    if domain == 'code':
        return jsonl_gz_column(hub_file(*CODE), 'content')
    raise ValueError(domain)


# ---------------------------------------------------------------- evaluation windows (archived protocol)

_ARTICLE = re.compile(r'^ = [^=].* = \n$')


def wiki_windows(tok, length=2048):
    lines = parquet_column(hub_file(*WIKI), 'text')
    text = '\n\n'.join(lines)
    enc = tok(text, return_tensors='pt', return_offsets_mapping=True)
    ids = enc.input_ids
    offsets = enc['offset_mapping'][0]
    usable = ids.shape[1] // 2048 * 2048
    windows = [ids[:, i:i + length].clone() for i in range(0, usable, length)]
    # Article boundaries: character offsets of " = Title = " lines in the joined string.
    starts, pos = [], 0
    for ln in lines:
        if _ARTICLE.match(ln):
            starts.append(pos)
        pos += len(ln) + 2
    def article_of(char):
        return max(0, bisect.bisect_right(starts, char) - 1)
    meta_windows = []
    for w, i in enumerate(range(0, usable, length)):
        first = int(offsets[i][0]); last = int(offsets[i + length - 1][0])
        meta_windows.append(dict(window=w, token_start=i, first_article=article_of(first), last_article=article_of(last)))
    meta = dict(repo=WIKI[0], revision=WIKI[1], path=WIKI[2], total_tokens=int(ids.shape[1]), used_tokens=usable,
                omitted_tail_tokens=int(ids.shape[1] - usable), windows=len(windows), window_tokens=length,
                articles=len(starts), window_articles=meta_windows, token_sha256=[sha(b) for b in windows],
                text_sha256=text_sha(text))
    return windows, meta


def c4_windows(tok, length=2048, count=256, seed=0):
    texts = jsonl_gz_column(hub_file(*C4), 'text')
    rng = random.Random(seed)
    crops, docs = [], []
    for _ in range(count):
        while True:
            index = rng.randint(0, len(texts) - 1)
            text = texts[index]
            tokens = tok(text, return_tensors='pt').input_ids
            if tokens.shape[1] > 2049:
                break
        offset = rng.randint(0, tokens.shape[1] - 2048 - 1)
        crop = tokens[:, offset:offset + 2048]
        crops.extend(crop[:, i:i + length].clone() for i in range(0, 2048, length))
        docs.append(dict(index=index, offset=offset, document_sha256=text_sha(text)))
    meta = dict(repo=C4[0], revision=C4[1], path=C4[2], seed=seed, parent_window_tokens=2048, documents=docs,
                rows_in_file=len(texts), windows=len(crops), window_tokens=length, token_sha256=[sha(b) for b in crops],
                unique_documents=len({d['document_sha256'] for d in docs}))
    return crops, meta


def long_context_windows(tok, length, count, max_books=None):
    """Non-overlapping windows from PG19 test books in file order; clusters are books."""
    snap = hub_snapshot(*PG19)
    files = sorted(p for p in snap.rglob('*.parquet'))
    windows, meta_w = [], []
    book = 0
    for f in files:
        for text in parquet_column(f, 'text'):
            ids = tok(text, return_tensors='pt').input_ids
            for start in range(0, ids.shape[1] - length + 1, length):
                windows.append(ids[:, start:start + length].clone())
                meta_w.append(dict(book=book, book_sha256=text_sha(text), token_start=start))
                if len(windows) == count:
                    return windows, dict(repo=PG19[0], revision=PG19[1], window_tokens=length, windows=count,
                                         window_meta=meta_w, token_sha256=[sha(b) for b in windows])
            book += 1
            if max_books and book >= max_books:
                break
    raise RuntimeError(f'insufficient PG19 tokens for {count} x {length}')


def pg19_frozen_books(tok, length, minimum_books=20, eligibility_tokens=8193):
    """One nested prefix window from each frozen eligible PG19 book.

    Eligibility is evaluated at 8193 tokens for both the 4K and 8K endpoints,
    so the same first ``minimum_books`` in file/row order are used at both
    lengths.  This prevents a few long books from supplying many windows while
    satisfying the document-level inference unit locked by the extension.
    """
    if length not in (4096, 8192) or eligibility_tokens < 8193:
        raise ValueError("PG19 frozen-book geometry differs from protocol")
    snap = hub_snapshot(*PG19)
    windows, meta_w, eligible_index = [], [], 0
    for file_index, f in enumerate(sorted(p for p in snap.rglob('*.parquet'))):
        for row_index, text in enumerate(parquet_column(f, 'text')):
            ids = tok(text, return_tensors='pt').input_ids
            if ids.shape[1] < eligibility_tokens:
                continue
            windows.append(ids[:, :length].clone())
            meta_w.append(dict(book=eligible_index, source_file_index=file_index,
                               source_row_index=row_index, book_sha256=text_sha(text),
                               token_start=0, eligible_tokens_minimum=eligibility_tokens))
            eligible_index += 1
            if len(windows) == minimum_books:
                return windows, dict(repo=PG19[0], revision=PG19[1],
                                     rule='first 20 file-order books with at least 8193 tokenizer tokens; one nested prefix per book',
                                     window_tokens=length, windows=len(windows),
                                     source_books=len(windows), minimum_documents=minimum_books,
                                     window_meta=meta_w, token_sha256=[sha(b) for b in windows])
    raise RuntimeError(f'insufficient PG19 books: {len(windows)} < {minimum_books}')


# ---------------------------------------------------------------- calibration sequences

def archived_manifest(key):
    from campaign.models import REGISTRY
    rep = json.loads((SOURCE_ROOT / REGISTRY[key]['archived_calibration'] / 'report.json').read_text())
    return rep['fit'], rep


def crops_from_manifest(tok, fit, length=512):
    """Rebuild archived calibration crops (math then code) from document hashes and token offsets."""
    batches, meta = {}, {}
    for domain in ('math', 'code'):
        m = fit[domain]
        wanted = {d['document_sha256']: d['offset'] for d in m['documents']}
        found = {}
        for text in source_texts(domain):
            h = text_sha(text)
            if h not in wanted or h in found:
                continue
            ids = tok(text, return_tensors='pt').input_ids
            off = wanted[h]
            if ids.shape[1] < off + length:
                raise ValueError(f'{domain} document {h} too short for offset {off}')
            found[h] = ids[:, off:off + length].clone()
            if len(found) == len(wanted):
                break
        if set(found) != set(wanted):
            raise ValueError(f'{domain}: {len(set(wanted) - set(found))} manifest documents not found')
        batches[domain] = [found[d['document_sha256']] for d in m['documents']]
        got = [sha(b) for b in batches[domain]]
        if 'token_sha256' in m and got != m['token_sha256']:
            raise ValueError(f'{domain}: token hashes differ from the manifest')
        meta[domain] = dict(repo=m['repo'], revision=m['revision'], path=m['path'], documents=m['documents'], token_sha256=got)
    return batches, meta


def builder_seed0(tok, length=512, count=64):
    """The archived run_pooled_scale.shared_data math/code rule, applied to any tokenizer."""
    batches, meta = {}, {}
    seen = set()
    for domain, spec in (('math', MATH), ('code', CODE)):
        rng = random.Random(20260926)
        docs, bs = [], []
        for text in source_texts(domain):
            h = text_sha(text)
            if h in seen:
                continue
            seen.add(h)
            ids = tok(text, return_tensors='pt').input_ids
            if ids.shape[1] < length:
                continue
            off = rng.randrange(ids.shape[1] - length + 1)
            bs.append(ids[:, off:off + length].clone())
            docs.append(dict(document_sha256=h, offset=off))
            if len(bs) == count:
                break
        if len(bs) != count:
            raise RuntimeError(f'{domain}: only {len(bs)} eligible documents')
        batches[domain] = bs
        meta[domain] = dict(repo=spec[0], revision=spec[1], path=spec[2], documents=docs, token_sha256=[sha(b) for b in bs],
                            rule='run_pooled_scale.shared_data: file order, skip <512 tokens, offset Random(20260926).randrange')
    return batches, meta


def keyed_draw(tok, draw, exclude, length=512, count=64, domains=('math', 'code')):
    """Independent draw `draw` (e.g. 'draw1'): documents ordered by sha256(f'{draw}:{doc_sha256}'), skipping any
    excluded hash or document shorter than `length` tokens; offset = int(sha256(f'{draw}:offset:{doc_sha256}'), 16)
    mod (n_tokens - length + 1)."""
    batches, meta = {}, {}
    for domain in domains:
        texts = source_texts(domain) if domain in ('math', 'code') else heldout_texts(domain)
        keyed = sorted(((hashlib.sha256(f'{draw}:{text_sha(t)}'.encode()).hexdigest(), i) for i, t in enumerate(texts)))
        docs, bs, used = [], [], set()
        for _, i in keyed:
            text = texts[i]
            h = text_sha(text)
            if h in exclude or h in used:
                continue
            ids = tok(text, return_tensors='pt').input_ids
            if ids.shape[1] < length:
                continue
            off = int(hashlib.sha256(f'{draw}:offset:{h}'.encode()).hexdigest(), 16) % (ids.shape[1] - length + 1)
            bs.append(ids[:, off:off + length].clone())
            docs.append(dict(document_sha256=h, offset=off))
            used.add(h)
            if len(bs) == count:
                break
        if len(bs) != count:
            raise RuntimeError(f'{domain}: only {len(bs)} eligible documents for {draw}')
        batches[domain] = bs
        meta[domain] = dict(draw=draw, documents=docs, token_sha256=[sha(b) for b in bs],
                            rule='keyed sha256 order, excluded hashes skipped, >=512 tokens, keyed offset')
    return batches, meta


def domain_eval_windows(tok, domain, exclude, length=2048, count=64):
    """V52 in-domain evaluation (amendment A03): keyed draw 'domain_eval' of `count` documents with >= `length` tokens from
    the calibration source shard of `domain` ('math' or 'code'), skipping every document used by this tokenizer's seed0 and
    draw1-4 calibration sets; one window per document (cluster = document)."""
    b, meta = keyed_draw(tok, 'domain_eval', exclude, length=length, count=count, domains=(domain,))
    m = meta[domain]
    spec = MATH if domain == 'math' else CODE
    return b[domain], dict(repo=spec[0], revision=spec[1], path=spec[2], window_tokens=length, windows=count, documents=m['documents'],
                           window_meta=[dict(document_sha256=d['document_sha256'], offset=d['offset']) for d in m['documents']],
                           token_sha256=m['token_sha256'], excluded_calibration_documents=len(exclude), rule=m['rule'] + '; draw key domain_eval')


def heldout_texts(domain):
    if domain == 'arxiv':
        snap = hub_snapshot(*ARXIV)
        files = sorted(snap.rglob('*test*.parquet')) or sorted(snap.rglob('*.parquet'))
        return [t for f in files for t in parquet_column(f, 'article')]
    if domain == 'govreport':
        snap = hub_snapshot(*GOVREPORT)
        files = sorted(snap.rglob('*test*.parquet')) or sorted(snap.rglob('*.parquet'))
        return [t for f in files for t in parquet_column(f, 'report')]
    raise ValueError(domain)


def manifest_sha256(meta):
    return hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
