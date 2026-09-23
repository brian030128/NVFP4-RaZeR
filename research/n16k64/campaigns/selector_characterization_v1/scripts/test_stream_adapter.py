"""Exercise the actual observer body without importing/loading any model."""
import ast
import unittest
import numpy as np
import torch
from common import OUT,PRIMARY,aggregate_scores
from validate_parent_moments import require_monitoring

class StreamingAdapter(unittest.TestCase):
    def test_final_repair_is_hash_verified_and_storage_only(self):
        from calibration_identity_diagnostic import historical_quant_text, OLD_QUANT_SHA
        from calibration_repair_run import transformed
        import hashlib
        source=(PRIMARY/'source/NVFP4-RaZeR-main/campaign/quant.py').read_text()
        old=historical_quant_text(source)
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),OLD_QUANT_SHA)
        self.assertIn('q = q + (x - x.detach())',old)
        with self.assertRaises(AssertionError):historical_quant_text(source+'\n')
        text=transformed('GPU-9cec7336-5b30-3f86-35e7-06919156e7da')
        self.assertIn("'--raw','sample','--subset-moments'",text)
        self.assertIn('stream_calibrate_historical.py',text)
        self.assertIn("child.terminate()",text)
        self.assertIn('len(active)<3',text)
    def test_calibration_admission_rejects_gap_and_invalid_monitor(self):
        lr=dict(status='complete',started_utc='2026-09-22T00:00:00Z',finished_utc='2026-09-22T00:02:00Z')
        during=[dict(passed=True,timestamp_utc='2026-09-22T00:00:30Z'),dict(passed=True,timestamp_utc='2026-09-22T00:01:30Z')]
        require_monitoring(lr,[dict(passed=True)],during,dict(passed=True))
        with self.assertRaisesRegex(AssertionError,'monitoring gap'):
            require_monitoring(lr,[dict(passed=True)],during[:1],dict(passed=True))
        during[1]['passed']=False
        with self.assertRaises(AssertionError):
            require_monitoring(lr,[dict(passed=True)],during,dict(passed=True))
    def test_actual_observer_with_correlated_children(self):
        torch.set_num_threads(1)
        tree=ast.parse((OUT/'scripts/stream_calibrate.py').read_text())
        observer=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='observer')
        store={N:{} for N in [32,64,128,256]};ns={'torch':torch,'store':store}
        module=ast.Module(body=[observer],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),'<actual stream observer>','exec'),ns)
        rng=np.random.default_rng(231);x=rng.normal(size=(128,32)).astype(np.float32)
        x[:,1]=x[:,0];x[:,2]=-x[:,0];y=(x*.25+rng.normal(size=x.shape)).astype(np.float32)
        for c,k in zip(x,y):ns['observer']('test',torch.from_numpy(c),torch.from_numpy(k),(256,64))
        for N,modules in store.items():
            s=modules['test'];ce=aggregate_scores(x.astype(np.float64),256,64,N);kl=aggregate_scores(y.astype(np.float64),256,64,N)
            self.assertEqual(s['n'],128)
            for key,expected in [('ce_sum',ce.sum(0)),('ce_sq',(ce*ce).sum(0)),('kl_sum',kl.sum(0)),('kl_sq',(kl*kl).sum(0)),('cross',(ce*kl).sum(0))]:
                np.testing.assert_allclose(s[key].numpy(),expected,rtol=1e-13,atol=1e-12)
    def test_single_insertion_compiles_without_execution(self):
        source=(PRIMARY/'source/NVFP4-RaZeR-main/campaign/calibrate.py').read_text()
        needle='            c8c, k8c = c8.cpu(), k8.cpu()\n'
        self.assertEqual(source.count(needle),1)
        altered=source.replace(needle,needle+'            stream_observer(n, c8c, k8c, shapes[n])\n')
        compile(altered,'<instrumented calibrator>','exec')

if __name__=='__main__':unittest.main()
