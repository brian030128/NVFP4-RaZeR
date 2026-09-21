"""Statistical gate boundary tests; no torch or model work."""
import copy
import unittest
from scripts.ce_confirmation_gate import passes

class TestCEGate(unittest.TestCase):
    def setUp(self):
        self.p={r:{d:{'ce':dict(mean=-.003,se=.001),'kl':dict(mean=.002,se=.001)} for d in ('all','math','code')} for r in ('raw256','identity')}
    def test_kl_remains_diagnostic(self):self.assertTrue(passes(self.p))
    def test_two_se_boundary_fails(self):
        self.p['identity']['all']['ce']['mean']=-.002;self.assertFalse(passes(self.p))
    def test_domain_regression_fails(self):
        self.p['raw256']['code']['ce']['mean']=.000001;self.assertFalse(passes(self.p))
    def test_nonfinite_fails(self):
        self.p['raw256']['all']['ce']['se']=float('nan');self.assertFalse(passes(self.p))
    def test_negative_se_fails(self):
        self.p['raw256']['all']['ce']['se']=-1;self.assertFalse(passes(self.p))

if __name__=='__main__':unittest.main()
