"""CPU-only checks for scoped Llama calibration recovery; no GPU calls."""
import os
import runpy
import subprocess
import sys
import unittest
from common import OUT,REPO,PRIMARY,load,sha
from calibration_llama_retry_run import transformed,require_authorization,RUN
from gpu_run_wait import transformed as parent

UUID='GPU-9cec7336-5b30-3f86-35e7-06919156e7da'

class LlamaRepair(unittest.TestCase):
    def test_exact_command_changes(self):
        before=parent(UUID);after=transformed(UUID)
        old="str(OUT/'scripts/stream_calibrate.py'),'--model',a.model,'--draw','seed0','--raw','none',"
        new="str(OUT/'scripts/stream_calibrate_llama_historical.py'),'--model',a.model,'--draw','seed0','--raw','sample',"
        marker="    code={str(f.relative_to(SOURCE)):sha(f)"
        record="    rec.update(additional_retry_authorization_sha256=AUTH_SHA, calibration_repair_launcher_parent_sha256=PARENT_SHA, calibration_repair_adapter_sha256=STREAM_SHA, recovery_function_sha256=sha(OUT/'scripts/calibration_identity_diagnostic.py'))\n"
        self.assertEqual(after,before.replace(old,new,1).replace(marker,record+marker,1))
        self.assertNotIn('--subset-moments',after)
    def test_real_grant_scope(self):
        record=load(OUT/'plans/additional_retry_authorization_20260923.json');require_authorization(record)
        for scopes in [[],[dict(model='llama8b',task='seed0 calibration',additional_attempts=2,run_name=RUN)]]:
            with self.assertRaises(AssertionError):require_authorization(dict(record,scopes=scopes))
    def test_no_grant_no_gpu_directory(self):
        root=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z/runs'/RUN
        before=root.stat().st_mtime_ns if root.exists() else None
        r=subprocess.run([sys.executable,str(OUT/'scripts/calibration_llama_retry_run.py'),'--only-uuid',UUID],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=20)
        self.assertEqual(r.returncode,2);self.assertIn('authorization record required',r.stderr)
        self.assertEqual(root.stat().st_mtime_ns if root.exists() else None,before)
    def test_runpy_import_and_historical_options(self):
        ns=runpy.run_path(str(OUT/'scripts/stream_calibrate_llama_historical.py'),run_name='cpu_import_only')
        self.assertTrue(callable(ns['main']))
        old=load(PRIMARY/'runs/V30_calib_llama8b_seed0_attempt1/launch_record.json')['command']
        self.assertEqual(old[old.index('--raw')+1],'sample');self.assertNotIn('--subset-moments',old)

if __name__=='__main__':unittest.main()
