"""CPU-only regression test for the actual runpy sibling-import failure."""
import os
import subprocess
import sys
import unittest
from common import OUT, PRIMARY, sha

class HistoricalRunpy(unittest.TestCase):
    def test_old_failure_and_path_only_fix_without_model_execution(self):
        old=OUT/'scripts/stream_calibrate_historical.py'
        new=OUT/'scripts/stream_calibrate_historical_v2.py'
        self.assertEqual(sha(old),'3a3ba8401fa512dfc353ace0fe3fb0b536bba806da36844c77cb06c6c7a2f571')
        needle='from calibration_identity_diagnostic import historical_quant_text, OLD_QUANT_SHA'
        bootstrap=("# runpy.run_path does not add a script's directory to sys.path.\n"
                   "# Path bootstrap only; historical quant/scoring/observer code is unchanged.\n"
                   "sys.path.insert(0, str(Path(__file__).resolve().parent))\n")
        self.assertEqual(new.read_text(),old.read_text().replace(needle,bootstrap+needle))
        probe="import runpy,sys; runpy.run_path(sys.argv[1],run_name='cpu_import_probe'); import torch; assert not torch.cuda.is_initialized(); print('CPU runpy import PASS; main not called')"
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
        cwd=PRIMARY/'source/NVFP4-RaZeR-main'
        before=subprocess.run([sys.executable,'-I','-B','-c',probe,str(old)],cwd=cwd,env=env,capture_output=True,text=True,timeout=30)
        self.assertNotEqual(before.returncode,0)
        self.assertIn("No module named 'calibration_identity_diagnostic'",before.stderr)
        after=subprocess.run([sys.executable,'-I','-B','-c',probe,str(new)],cwd=cwd,env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(after.returncode,0,after.stderr)
        self.assertIn('main not called',after.stdout)

if __name__=='__main__':unittest.main()
