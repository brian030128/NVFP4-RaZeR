"""Observe Llama scores using hash-proven historical quant text/storage flags.

Separate from the live Mistral adapter: Llama did not use subset moments.
Exact sequence-score and N8/N16 admission is still mandatory.
"""
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_identity_diagnostic import historical_quant_text, OLD_QUANT_SHA

def main():
    from campaign import quant, runtime
    from common import sha
    import stream_calibrate
    assert sys.argv[sys.argv.index('--model')+1]=='llama8b'
    assert sys.argv[sys.argv.index('--raw')+1]=='sample'
    assert '--subset-moments' not in sys.argv
    original=Path(quant.__file__).read_text();recovered=historical_quant_text(original)
    record=dict(status='hash_verified_source_restore_not_numerical_acceptance',
        current_quant_sha256=hashlib.sha256(original.encode()).hexdigest(),
        recovered_quant_sha256=OLD_QUANT_SHA,repair_adapter_sha256=sha(__file__),
        recovery_function_sha256=sha(Path(__file__).with_name('calibration_identity_diagnostic.py')),
        observer_adapter_sha256=sha(stream_calibrate.__file__),raw='sample',subset_moments=False,
        qualification='Llama V30 historical raw allocation and quant text restored; CPU parent observer is additional. No guarantee of exact backward reproduction.')
    runtime.atomic_json(runtime.out_dir('parent_moments')/'historical_repair.json',record)
    exec(compile(recovered,str(quant.__file__)+'::sha_verified_historical','exec'),quant.__dict__)
    stream_calibrate.main()

if __name__=='__main__':main()
