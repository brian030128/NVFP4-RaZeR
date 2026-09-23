"""Pin future jobs to one explicitly chosen device; never weaken ownership checks.

The unchanged launcher is hash-pinned and transformed only at GPU selection
and provenance recording. This avoids the two devices repeatedly interrupted
by foreign jobs. It neither reserves a GPU against other users nor makes
co-tenancy safe: all original fail-closed checks remain active.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

BASE_SHA256='35e40c567436c78bcab5b51116751c58f66e6c4b970349409c148565d3e2ffa4'

def transformed(uuid):
    if not uuid.startswith('GPU-') or any(c not in 'GPU-0123456789abcdef' for c in uuid):
        raise ValueError('Expected a literal GPU UUID')
    base=Path(__file__).with_name('gpu_run.py');raw=base.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==BASE_SHA256,'Re-audit changed base launcher'
    source=raw.decode()
    old="if g['uuid'] in active or g['name'] not in gp.ALLOWED_MODELS:continue"
    assert source.count(old)==1
    source=source.replace(old,"if g['uuid'] != "+repr(uuid)+" or g['uuid'] in active or g['name'] not in gp.ALLOWED_MODELS:continue",1)
    marker="    code={str(f.relative_to(SOURCE)):sha(f)"
    assert source.count(marker)==1
    source=source.replace(marker,"    rec.update(launcher_base_sha256="+repr(BASE_SHA256)+",allowed_uuid="+repr(uuid)+")\n"+marker,1)
    ast.parse(source)
    return source

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--only-uuid',required=True)
    parser.add_argument('--inspect-only',action='store_true')
    args,rest=parser.parse_known_args()
    source=transformed(args.only_uuid)
    if args.inspect_only:
        print(json.dumps(dict(base_sha256=BASE_SHA256,allowed_uuid=args.only_uuid,
            transformed_sha256=hashlib.sha256(source.encode()).hexdigest(),
            scope='CPU source inspection only; no GPU query, allocation or evaluation')))
        return 0
    sys.argv=[__file__]+rest
    exec(compile(source,__file__,'exec'),dict(__name__='__main__',__file__=__file__))

if __name__=='__main__':raise SystemExit(main())
