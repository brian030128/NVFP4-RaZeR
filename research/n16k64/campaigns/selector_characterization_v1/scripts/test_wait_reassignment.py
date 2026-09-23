import unittest
from common import OUT,REPO
from reassign_waiting import validate_snapshot

class WaitingReassignment(unittest.TestCase):
    def test_only_exact_owned_unlaunched_process_is_eligible(self):
        s=dict(uid=1000,start_ticks='123',script=str(OUT/'scripts/gpu_run_wait.py'),cwd=str(REPO),run='test',
               launch_record=False,leases=[],children=[])
        for entry in ['gpu_run_wait.py','accuracy_run.py']:
            s['script']=str(OUT/'scripts'/entry)
            validate_snapshot(s,1000,'test','123')
            for key,value in [('uid',1001),('start_ticks','124'),('script','/unrelated.py'),
                              ('script',str(OUT/'scripts/calibration_repair_run.py')),('cwd','/elsewhere'),('run','other'),
                              ('launch_record',True),('leases',['lease']),('children',['42'])]:
                with self.subTest(entry=entry,key=key),self.assertRaises(AssertionError):
                    validate_snapshot(dict(s,**{key:value}),1000,'test','123')

if __name__=='__main__':unittest.main()
