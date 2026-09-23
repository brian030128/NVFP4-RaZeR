"""CPU fixtures for fail-closed intervals; no GPU query/import/launch."""
import unittest
from cadence_launch import Cadence

class TestCadence(unittest.TestCase):
    def row(self,passed=True):return dict(phase='during',passed=passed,reasons=[] if passed else ['foreign PID'])
    def test_long_query(self):
        r=Cadence().enforce(self.row(),0,61)
        self.assertFalse(r['passed']);self.assertIn('query duration',r['reasons'][0])
    def test_gap(self):
        g=Cadence();self.assertTrue(g.enforce(self.row(),0,1)['passed'])
        self.assertFalse(g.enforce(self.row(),61,62)['passed'])
    def test_boundaries_and_prior_failure(self):
        g=Cadence();self.assertTrue(g.enforce(self.row(),0,1)['passed'])
        self.assertTrue(g.enforce(self.row(),60,61)['passed'])
        r=g.enforce(self.row(False),90,91);self.assertFalse(r['passed']);self.assertEqual(r['reasons'],['foreign PID'])
    def test_prelaunch_not_counted(self):
        g=Cadence();r=dict(phase='prelaunch',passed=True,reasons=[])
        self.assertEqual(g.enforce(r,0,1),r);self.assertTrue(g.enforce(self.row(),1000,1001)['passed'])

if __name__=='__main__':unittest.main()
