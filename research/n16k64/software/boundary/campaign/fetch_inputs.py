"""Download every pinned model/dataset input and write a hash-verified manifest.

Host-side network I/O only (no GPU). Files land in $HF_HOME; the manifest records
the repository, pinned revision, every file's size and SHA-256, and whether the
local bytes match the Hub's LFS digest.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

MODELS = {
    'llama8b': ('meta-llama/Llama-3.1-8B', 'd04e592bb4f6aa9cfee91e2e20afa771667e1d4b'),
    'qwen4b': ('Qwen/Qwen3-4B', '1cfa9a7208912126459214e8b04321603b3df60c'),
    'qwen27b': ('Qwen/Qwen3.8-27B', '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'),
    'mistral7b': ('mistralai/Mistral-7B-v0.3', 'caa1feb0e54d415e2df31207e5f4e273e33509b1'),
    'phi4': ('microsoft/phi-4', '2db69c1c3e91a05d2c64a3185acfbaf36f744e25'),
    'olmo2_13b': ('allenai/OLMo-2-1124-13B', '3fefddc1bf18a30e1d9b91000271630718f2aa8b'),
}
MODEL_ALLOW = ['*.json', '*.safetensors', 'tokenizer*', '*.model', 'merges.txt', 'vocab.json',
               '*.tiktoken', '*.txt', '*.py', '*.jinja']
MODEL_IGNORE = ['consolidated*', 'original/*', '*.pth', '*.bin', '*.gguf', '*.onnx', '*.msgpack', '*.h5']

# Exact files used by the archived protocol (revision, path) plus auxiliary corpora.
DATA_FILES = [
    ('Salesforce/wikitext', 'b08601e04326c79dfdd32d625aee71d232d685c3', 'wikitext-2-raw-v1/test-00000-of-00001.parquet'),
    ('allenai/c4', '1588ec454efa1a09f29cd18ddd04fe05fc8653a2', 'en/c4-validation.00000-of-00008.json.gz'),
    ('open-web-math/open-web-math', 'fde8ef8de2300f5e778f56261843dab89f230815',
     'data/train-00000-of-00114-5a023365406cb9c4.parquet'),
    ('codeparrot/codeparrot-clean', '35a59fb025bc0a102f7d96eac09d145b896d487b', 'file-000000000001.json.gz'),
]
DATA_SNAPSHOTS = [
    ('emozilla/pg19-test', 'c5e39bf32e33f9111323aa68d7d9000d22722035', None),
    ('ccdv/arxiv-summarization', '240aaf1a969b3f8cd0ade6986bfad0cd730ee288', ['document/test*']),
    ('ccdv/govreport-summarization', '4e21184e01ae8017e2c036e180fe5e541fef60a0', ['document/test*']),
]


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_snapshot(api, repo, revision, local_dir, repo_type):
    info = api.repo_info(repo, revision=revision, repo_type=repo_type, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    files = []
    for p in sorted(Path(local_dir).rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir).as_posix()
        s = remote.get(rel)
        digest = sha256_file(p.resolve())
        lfs = getattr(s, 'lfs', None) if s is not None else None
        expected = (lfs.sha256 if lfs is not None else None)
        files.append(dict(path=rel, size=p.resolve().stat().st_size, sha256=digest,
                          hub_lfs_sha256=expected,
                          hub_lfs_match=(None if expected is None else expected == digest),
                          hub_blob_id=(getattr(s, 'blob_id', None) if s is not None else None)))
    bad = [f['path'] for f in files if f['hub_lfs_match'] is False]
    return files, bad, info.sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', default='')
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), hf_home=os.environ.get('HF_HOME'),
                    models={}, datasets={}, failures=[])
    only = set(filter(None, args.only.split(',')))

    def dump():
        tmp = out.with_suffix('.tmp')
        tmp.write_text(json.dumps(manifest, indent=1) + '\n')
        tmp.replace(out)

    for key, (repo, rev) in MODELS.items():
        if only and key not in only:
            continue
        t0 = time.time()
        try:
            path = snapshot_download(repo, revision=rev, allow_patterns=MODEL_ALLOW, ignore_patterns=MODEL_IGNORE)
            files, bad, resolved = verify_snapshot(api, repo, rev, path, 'model')
            manifest['models'][key] = dict(repo=repo, revision=rev, resolved_revision=resolved, local_path=path,
                                           files=files, lfs_mismatches=bad, seconds=time.time() - t0)
            if bad or resolved != rev:
                manifest['failures'].append(dict(item=key, bad=bad, resolved=resolved))
            print(f'MODEL {key} ok={not bad} files={len(files)} {time.time()-t0:.0f}s', flush=True)
        except Exception as exc:  # recorded, never silently skipped
            manifest['failures'].append(dict(item=key, error=repr(exc)))
            print(f'MODEL {key} FAILED {exc!r}', flush=True)
        dump()

    if not only or 'data' in only:
        for repo, rev, path in DATA_FILES:
            t0 = time.time()
            try:
                local = hf_hub_download(repo, path, revision=rev, repo_type='dataset')
                info = api.repo_info(repo, revision=rev, repo_type='dataset', files_metadata=True)
                s = {x.rfilename: x for x in info.siblings}.get(path)
                digest = sha256_file(local)
                exp = s.lfs.sha256 if (s is not None and s.lfs is not None) else None
                manifest['datasets'][f'{repo}:{path}'] = dict(repo=repo, revision=rev, path=path, local_path=local,
                    size=Path(local).stat().st_size, sha256=digest, hub_lfs_sha256=exp,
                    hub_lfs_match=(None if exp is None else exp == digest), seconds=time.time() - t0)
                print(f'DATA {repo}:{path} ok', flush=True)
            except Exception as exc:
                manifest['failures'].append(dict(item=f'{repo}:{path}', error=repr(exc)))
                print(f'DATA {repo}:{path} FAILED {exc!r}', flush=True)
            dump()
        for repo, rev, allow in DATA_SNAPSHOTS:
            t0 = time.time()
            try:
                local = snapshot_download(repo, revision=rev, repo_type='dataset', allow_patterns=allow)
                files, bad, resolved = verify_snapshot(api, repo, rev, local, 'dataset')
                manifest['datasets'][repo] = dict(repo=repo, revision=rev, resolved_revision=resolved,
                                                  local_path=local, files=files, lfs_mismatches=bad,
                                                  seconds=time.time() - t0)
                print(f'DATA {repo} ok files={len(files)}', flush=True)
            except Exception as exc:
                manifest['failures'].append(dict(item=repo, error=repr(exc)))
                print(f'DATA {repo} FAILED {exc!r}', flush=True)
            dump()
    manifest['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    dump()
    print('FAILURES', json.dumps(manifest['failures']), flush=True)
    sys.exit(1 if manifest['failures'] else 0)


if __name__ == '__main__':
    main()
