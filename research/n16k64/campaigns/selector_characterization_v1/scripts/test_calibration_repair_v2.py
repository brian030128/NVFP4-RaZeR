"""CPU checks for a prepared, not-authorized additional calibration entry."""
import os
import subprocess
import sys
import unittest
from common import OUT,REPO
from calibration_repair_run_v2 import transformed,require_authorization,RUN
from calibration_repair_run import transformed as parent

UUID='GPU-9cec7336-5b30-3f86-35e7-06919156e7da'

class RepairV2(unittest.TestCase):
    def test_narrow_source_change(self):
        before=parent(UUID);after=transformed(UUID)
        expected=before.replace('stream_calibrate_historical.py','stream_calibrate_historical_v2.py')
        marker="    code={str(f.relative_to(SOURCE)):sha(f)"
        expected=expected.replace(marker,"    rec.update(additional_retry_authorization_sha256=AUTH_SHA, calibration_repair_launcher_parent_sha256=PARENT_SHA)\n"+marker,1)
        self.assertEqual(after,expected)
        self.assertIn("'--raw','sample','--subset-moments'",after)
    def test_no_authorization_refuses_before_run_directory(self):
        root=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z/runs'/RUN
        before=root.stat().st_mtime_ns if root.exists() else None
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
        r=subprocess.run([sys.executable,str(OUT/'scripts/calibration_repair_run_v2.py'),'--only-uuid',UUID,'--calibrate-stream'],env=env,capture_output=True,text=True,timeout=20)
        self.assertEqual(r.returncode,2,r.stderr);self.assertIn('authorization record required',r.stderr)
        self.assertEqual(root.stat().st_mtime_ns if root.exists() else None,before)
    def test_scope_must_be_exact(self):
        # In-memory fixture only: no approval file or GPU call is created.
        good=dict(authorization_type='explicit_user_additional_retry',scope='one_additional_mistral_seed0_calibration_attempt',run_name=RUN,user_evidence='CPU fixture; not a real user grant')
        require_authorization(good)
        for key,value in [('authorization_type','template'),('scope','any_GPU_job'),('run_name','other_run'),('user_evidence','')]:
            with self.subTest(key=key),self.assertRaises(AssertionError):require_authorization(dict(good,**{key:value}))

if __name__=='__main__':unittest.main()
