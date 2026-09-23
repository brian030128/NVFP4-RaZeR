"""Final repair: recovered historical quant source and original storage options.

Only this process's imported module is replaced, after full source-hash proof.
No repository/historical file is rewritten. The original observer and scoring
expressions are unchanged; outputs still require exact historical admission.
"""
import hashlib
import sys
from pathlib import Path
from calibration_identity_diagnostic import historical_quant_text, OLD_QUANT_SHA

def main():
    from campaign import quant, runtime
    original = Path(quant.__file__).read_text()
    recovered = historical_quant_text(original)
    assert '--raw' in sys.argv and sys.argv[sys.argv.index('--raw')+1]=='sample'
    assert '--subset-moments' in sys.argv
    from common import sha
    import stream_calibrate
    record = dict(status='hash_verified_source_restore_not_numerical_acceptance',
        current_quant_sha256=hashlib.sha256(original.encode()).hexdigest(),
        recovered_quant_sha256=OLD_QUANT_SHA,repair_adapter_sha256=sha(__file__),
        recovery_function_sha256=sha(Path(__file__).with_name('calibration_identity_diagnostic.py')),
        observer_adapter_sha256=sha(stream_calibrate.__file__),
        raw='sample',subset_moments=True,
        qualification='Original raw/subset allocations restored; CPU parent observer remains additional. No guarantee of bitwise backward reproducibility.')
    runtime.atomic_json(runtime.out_dir('parent_moments')/'historical_repair.json',record)
    exec(compile(recovered,str(quant.__file__)+'::sha_verified_historical','exec'),quant.__dict__)
    stream_calibrate.main()

if __name__=='__main__':main()
