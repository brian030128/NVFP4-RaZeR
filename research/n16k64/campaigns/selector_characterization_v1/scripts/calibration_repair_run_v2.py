"""Prepared Mistral path-only repair; no further GPU attempt is authorized.

--inspect-only is CPU-only. Actual execution requires a separately recorded
explicit user grant beyond the exhausted two-repair limit. A record is
provenance, not permission for an agent to invent approval.
"""
import argparse
import ast
import hashlib
import time
import sys
from common import *

PARENT_SHA='8700b280c27de24bb8dfd02928a55a22eea26ba7b7163a9892bcf89459968fa4'
STREAM_SHA='e49d53cb9adab02f217cc3cb78f5f6012a6bf0dada7093ccbe2721dae2cb92a5'
GUARD_SHA='d8b33ad150bfebdd2749a1fc661123c3745d9bd39ee1fb4ca9b94d626ec71ee5'
RUN='calibration_mistral7b_attempt4_authorized'

def require_authorization(record):
    assert set(record)=={'authorization_type','scope','run_name','user_evidence'},'Unexpected authorization schema'
    assert record['authorization_type']=='explicit_user_additional_retry'
    assert record['scope']=='one_additional_mistral_seed0_calibration_attempt'
    assert record['run_name']==RUN
    assert isinstance(record['user_evidence'],str) and record['user_evidence'].strip()

def transformed(uuid):
    assert sha(OUT/'scripts/calibration_repair_run.py')==PARENT_SHA
    assert sha(OUT/'scripts/stream_calibrate_historical_v2.py')==STREAM_SHA
    assert sha(OUT/'scripts/cadence_launch.py')==GUARD_SHA
    from calibration_repair_run import transformed as parent
    source=parent(uuid);assert source.count('stream_calibrate_historical.py')==2
    source=source.replace('stream_calibrate_historical.py','stream_calibrate_historical_v2.py')
    marker="    code={str(f.relative_to(SOURCE)):sha(f)"
    assert source.count(marker)==1
    source=source.replace(marker,"    rec.update(additional_retry_authorization_sha256=AUTH_SHA, calibration_repair_launcher_parent_sha256=PARENT_SHA)\n"+marker,1)
    ast.parse(source);return source

def main():
    ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--only-uuid',required=True)
    ap.add_argument('--inspect-only',action='store_true');ap.add_argument('--authorization-record',type=Path)
    ap.add_argument('--model',choices=['mistral7b']);ap.add_argument('--name',choices=[RUN])
    ap.add_argument('--calibrate-stream',action='store_true');ap.add_argument('--device-type',choices=['a6000'],default='a6000')
    ap.add_argument('--hub-cache',type=Path);a=ap.parse_args()
    source=transformed(a.only_uuid)
    if a.inspect_only:
        print('CPU source validated; no GPU query or approval',hashlib.sha256(source.encode()).hexdigest());return
    if a.authorization_record is None:ap.error('Explicit additional-retry authorization record required; no GPU launch')
    auth=load(a.authorization_record);require_authorization(auth);auth_sha=sha(a.authorization_record)
    auth_bytes=a.authorization_record.read_bytes();assert hashlib.sha256(auth_bytes).hexdigest()==auth_sha
    assert a.model=='mistral7b' and a.name==RUN,'Explicit scoped model/run required'
    assert a.calibrate_stream and a.hub_cache and a.hub_cache.is_dir()
    assert sha(OUT/'FROZEN_PROTOCOL.yaml')=='dbf2a3cb353e168a9bbe104309a345781f51f7d04ca281c4415e8097463e2306'
    rest=['--model',a.model,'--name',a.name,'--calibrate-stream','--device-type',a.device_type,'--hub-cache',str(a.hub_cache)]
    root=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z/runs'/RUN
    assert not root.exists(),'Never overwrite or duplicate an attempt'
    sys.path.insert(0,str(PRIMARY/'source/NVFP4-RaZeR-main'))
    from campaign import gpu_preflight as gp
    from cadence_launch import Cadence
    from gpu_run_wait import inventory_or_wait
    guard=Cadence();original=gp.check;recorded=False
    def checked(*args,**kwargs):
        nonlocal recorded
        if not recorded:
            assert root.is_dir()
            (root/'additional_retry_authorization.json').write_bytes(auth_bytes)
            assert sha(root/'additional_retry_authorization.json')==auth_sha
            assert load(root/'additional_retry_authorization.json')==auth
            jsonout(root/'cadence_guard.json',dict(wrapper=str(Path(__file__).relative_to(REPO)),wrapper_sha256=sha(Path(__file__)),
                entry='calibration_repair_run_v2.py',entry_sha256=sha(Path(__file__)),maximum_seconds=60,
                scientific_payload_changed=False,created_utc=now(),cadence_implementation_sha256=GUARD_SHA))
            recorded=True
        begin=time.monotonic();result=original(*args,**kwargs);end=time.monotonic()
        return guard.enforce(result,begin,end)
    gp.check=checked
    # The run-local authorization copy is byte-identical to the supplied record.
    sys.argv=[__file__]+rest
    exec(compile(source,__file__,'exec'),dict(__name__='__main__',__file__=__file__,inventory_or_wait=inventory_or_wait,
        AUTH_SHA=auth_sha,PARENT_SHA=PARENT_SHA))

if __name__=='__main__':main()
