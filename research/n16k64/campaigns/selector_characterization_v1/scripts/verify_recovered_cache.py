"""Read-only cache verification; never loads models or downloads files."""
from common import *

def main():
    start = now()
    lineage = FOLLOWUP / 'INPUT_PROVENANCE.json'
    caches = load(lineage)['caches']
    rows = []
    for model in ['mistral7b', 'qwen4b']:
        spec = caches[model]
        snapshot = Path(spec['path'])
        assert snapshot.name == spec['revision'] and snapshot.is_dir()
        index = load(snapshot / 'model.safetensors.index.json')
        required = set(index['weight_map'].values())
        required.update(p.name for p in snapshot.iterdir()
                        if not p.name.endswith('.safetensors'))
        for name in sorted(required):
            path = snapshot / name
            target = path.resolve(strict=True)
            assert target.is_file() and target.parent == (snapshot.parent.parent / 'blobs').resolve(strict=True)
            size = target.stat().st_size
            sha256 = hashlib.sha256()
            gitblob = hashlib.sha1(b'blob ' + str(size).encode() + b'\0')
            with target.open('rb') as f:
                for block in iter(lambda: f.read(8 << 20), b''):
                    sha256.update(block)
                    gitblob.update(block)
            expected = target.name
            actual = sha256.hexdigest() if len(expected) == 64 else gitblob.hexdigest()
            assert actual == expected, (model, name, 'cache blob identity mismatch')
            rows.append(dict(model=model, revision=spec['revision'], path=str(path),
                             size=size, sha256=sha256.hexdigest(), blob_identity_verified=True))
        print(model, 'verified', len(required), 'files', flush=True)
    output = OUT / 'results/RECOVERED_CACHE_VERIFICATION.json'
    jsonout(output, dict(status='verified', checked_utc=now(), files=rows,
                        lineage_source=str(lineage.relative_to(REPO)), lineage_sha256=sha(lineage),
                        qualification='Content-addressed cache identity verified; numerical historical equivalence remains a separate gate'))
    log('cache_recovery', 'python scripts/verify_recovered_cache.py', start, [output])

if __name__ == '__main__':
    main()
