"""One explicitly authorized Llama repair; unchanged historical score gates."""
import argparse
import ast
import sys
import time
from common import *

PARENT_SHA='7c3d65eecf72b83d0a8cf4416d846593ea8d7f8f1e4dfca4f8db17abe27ac8e4'
STREAM_SHA='4768375813b3ce54d0776ad5d6c732585c12f7cf6c61b90106eaf3bd947f1bcd'
GUARD_SHA='d8b33ad150bfebdd2749a1fc661123c3745d9bd39ee1fb4ca9b94d626ec71ee5'
RUN='calibration_llama8b_attempt4_authorized'

def require_authorization(record):
    assert record['authorization_type']=='explicit_user_additional_retry'
    assert isinstance(record['user_evidence'],str) and record['user_evidence'].strip()
    matches=[r for r in record['scopes'] if r.get('run_name')==RUN]
    assert matches==[dict(model='llama8b',task='seed0 calibration',additional_attempts=1,run_name=RUN)]

def transformed(uuid):
    assert sha(OUT/'scripts/gpu_run_wait.py')==PARENT_SHA
    assert sha(OUT/'scripts/stream_calibrate_llama_historical.py')==STREAM_SHA
    assert sha(OUT/'scripts/cadence_launch.py')==GUARD_SHA
    from gpu_run_wait import transformed as parent
    source=parent(uuid)
    old="str(OUT/'scripts/stream_calibrate.py'),'--model',a.model,'--draw','seed0','--raw','none',"
    new="str(OUT/'scripts/stream_calibrate_llama_historical.py'),'--model',a.model,'--draw','seed0','--raw','sample',"
    assert source.count(old)==1;source=source.replace(old,new,1)
    marker="    code={str(f.relative_to(SOURCE)):sha(f)"
    assert source.count(marker)==1
    source=source.replace(marker,"    rec.update(additional_retry_authorization_sha256=AUTH_SHA, calibration_repair_launcher_parent_sha256=PARENT_SHA, calibration_repair_adapter_sha256=STREAM_SHA, recovery_function_sha256=sha(OUT/'scripts/calibration_identity_diagnostic.py'))\n"+marker,1)
    ast.parse(source);return source

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only-uuid',required=True)
    ap.add_argument('--inspect-only',action='store_true');ap.add_argument('--authorization-record',type=Path)
    ap.add_argument('--model',choices=['llama8b']);ap.add_argument('--name',choices=[RUN])
    ap.add_argument('--calibrate-stream',action='store_true');ap.add_argument('--device-type',choices=['a6000'],default='a6000')
    a=ap.parse_args();source=transformed(a.only_uuid)
    if a.inspect_only:
        print('CPU source validated; no GPU query',hashlib.sha256(source.encode()).hexdigest());return
    if a.authorization_record is None:ap.error('Explicit additional-retry authorization record required; no GPU launch')
    auth=load(a.authorization_record);require_authorization(auth);auth_bytes=a.authorization_record.read_bytes();auth_sha=sha(a.authorization_record)
    assert hashlib.sha256(auth_bytes).hexdigest()==auth_sha
    assert a.model=='llama8b' and a.name==RUN and a.calibrate_stream
    assert sha(OUT/'FROZEN_PROTOCOL.yaml')=='dbf2a3cb353e168a9bbe104309a345781f51f7d04ca281c4415e8097463e2306'
    historical=PRIMARY/'runs/V30_calib_llama8b_seed0_attempt1'
    old=load(historical/'launch_record.json')['command']
    assert old[old.index('--raw')+1]=='sample' and '--subset-moments' not in old
    assert '210d182478d9bab9ce6c9b8e6d8ff831420ca83f31a2ecf170998c13ca641aca  campaign/quant.py' in (historical/'source_manifest.txt').read_text()
    root=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z/runs'/RUN
    assert not root.exists(),'Never overwrite or duplicate an attempt'
    sys.path.insert(0,str(PRIMARY/'source/NVFP4-RaZeR-main'))
    from campaign import gpu_preflight as gp
    from cadence_launch import Cadence
    from gpu_run_wait import inventory_or_wait
    original=gp.check;guard=Cadence();recorded=False
    def checked(*args,**kwargs):
        nonlocal recorded
        if not recorded:
            assert root.is_dir()
            (root/'additional_retry_authorization.json').write_bytes(auth_bytes)
            assert sha(root/'additional_retry_authorization.json')==auth_sha
            jsonout(root/'cadence_guard.json',dict(wrapper=str(Path(__file__).relative_to(REPO)),wrapper_sha256=sha(Path(__file__)),
                entry='calibration_llama_retry_run.py',entry_sha256=sha(Path(__file__)),maximum_seconds=60,
                scientific_payload_changed=False,created_utc=now(),cadence_implementation_sha256=GUARD_SHA))
            recorded=True
        begin=time.monotonic();result=original(*args,**kwargs);end=time.monotonic()
        return guard.enforce(result,begin,end)
    gp.check=checked
    sys.argv=[__file__,'--model',a.model,'--name',a.name,'--calibrate-stream','--device-type','a6000']
    exec(compile(source,__file__,'exec'),dict(__name__='__main__',__file__=__file__,inventory_or_wait=inventory_or_wait,
        AUTH_SHA=auth_sha,PARENT_SHA=PARENT_SHA,STREAM_SHA=STREAM_SHA))

if __name__=='__main__':main()
