"""CPU safety fixtures; never signal a process or query a GPU."""
import unittest
from reassign_guarded_waiting import validate,OUT,REPO

class TestGuardedReassignment(unittest.TestCase):
    def fixture(self):
        return dict(uid=123,start_ticks='456',script=str(OUT/'scripts/cadence_launch.py'),cwd=str(REPO),run='test',
            launch_record=False,leases=[],children=[],argv=['python','cadence_launch.py','--entry','accuracy_run.py','--name','test','--only-uuid','GPU-abc'])
    def test_accept_only_both_pinned_entry_types(self):
        for entry in ['accuracy_run.py','gpu_run_wait.py']:
            s=self.fixture();s['argv'][3]=entry;validate(s,123,'test','456')
    def test_reject_identity_activity_and_scope(self):
        for key,value in [('uid',124),('start_ticks','457'),('script','/tmp/other.py'),('cwd','/tmp'),('run','other'),('launch_record',True),('leases',['lease']),('children',['child'])]:
            s=self.fixture();s[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):validate(s,123,'test','456')
    def test_reject_unsupported_payload(self):
        s=self.fixture();s['argv'][3]='unrelated.py'
        with self.assertRaises(AssertionError):validate(s,123,'test','456')

if __name__=='__main__':unittest.main()
