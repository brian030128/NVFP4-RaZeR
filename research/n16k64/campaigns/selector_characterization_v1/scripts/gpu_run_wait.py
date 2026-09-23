"""Wait on inventory-query failure before launch; never weaken in-run checks.

New adapter, not an edit to a running launcher's code. The parent and original
launcher stay hash-pinned. A query error before a lease/model exists is recorded
as resource unavailability, not a fabricated GPU attempt or a successful check.
"""
import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PARENT_SHA='6601e10310d5e0d5c3a94b40056c1bbf10f5aa9b9bc1f0778728721033300c8d'

def inventory_or_wait(gp, root):
    try:
        return gp.smi_gpus()
    except gp.QueryError as exc:
        row=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),status='unavailable_before_launch',
                 error=str(exc),gpu_launched=False)
        with (root/'inventory_query_errors.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('WAIT GPU inventory unavailable; no lease or model launched',flush=True)
        return []

def transformed(uuid):
    parent=Path(__file__).with_name('gpu_run_on_uuid.py')
    assert hashlib.sha256(parent.read_bytes()).hexdigest()==PARENT_SHA,'Re-audit changed parent adapter'
    from gpu_run_on_uuid import transformed as parent_transform
    source=parent_transform(uuid)
    assert source.count('gp.smi_gpus()')==1
    source=source.replace('gp.smi_gpus()','inventory_or_wait(gp,root)',1)
    marker="    code={str(f.relative_to(SOURCE)):sha(f)"
    assert source.count(marker)==1
    source=source.replace(marker,"    rec.update(wait_adapter_parent_sha256="+repr(PARENT_SHA)+")\n"+marker,1)
    ast.parse(source)
    return source

def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--only-uuid',required=True)
    parser.add_argument('--inspect-only',action='store_true');args,rest=parser.parse_known_args()
    source=transformed(args.only_uuid)
    if args.inspect_only:
        print(json.dumps(dict(parent_sha256=PARENT_SHA,allowed_uuid=args.only_uuid,
            transformed_sha256=hashlib.sha256(source.encode()).hexdigest(),no_gpu_query=True)))
        return 0
    sys.argv=[__file__]+rest
    exec(compile(source,__file__,'exec'),dict(__name__='__main__',__file__=__file__,inventory_or_wait=inventory_or_wait))

if __name__=='__main__':raise SystemExit(main())
