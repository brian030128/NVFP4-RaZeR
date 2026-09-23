"""Final Mistral calibration repair, isolated from all live launchers."""
import argparse
import ast
import hashlib
import sys
from pathlib import Path
from gpu_run_wait import transformed as wait_transform, inventory_or_wait

def transformed(uuid):
    source=wait_transform(uuid)
    old="str(OUT/'scripts/stream_calibrate.py'),'--model',a.model,'--draw','seed0','--raw','none',"
    new="str(OUT/'scripts/stream_calibrate_historical.py'),'--model',a.model,'--draw','seed0','--raw','sample','--subset-moments',"
    assert source.count(old)==1
    source=source.replace(old,new)
    marker="    code={str(f.relative_to(SOURCE)):sha(f)"
    assert source.count(marker)==1
    source=source.replace(marker,"    rec.update(calibration_repair_adapter_sha256=sha(OUT/'scripts/stream_calibrate_historical.py'), recovery_function_sha256=sha(OUT/'scripts/calibration_identity_diagnostic.py'))\n"+marker,1)
    ast.parse(source)
    return source

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--only-uuid',required=True);parser.add_argument('--inspect-only',action='store_true')
    a,rest=parser.parse_known_args()
    assert '--calibrate-stream' in rest and '--model' in rest
    assert rest[rest.index('--model')+1]=='mistral7b'
    assert '--name' in rest and rest[rest.index('--name')+1]=='calibration_mistral7b_attempt3'
    assert '--limit-sequences' not in rest
    source=transformed(a.only_uuid)
    if a.inspect_only:
        print('CPU transformed source validated',hashlib.sha256(source.encode()).hexdigest());return
    sys.argv=[__file__]+rest
    exec(compile(source,__file__,'exec'),dict(__name__='__main__',__file__=__file__,inventory_or_wait=inventory_or_wait))

if __name__=='__main__':main()
