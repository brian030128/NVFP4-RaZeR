"""Independent rejection fixtures; no inference or changes to historical evidence."""
import copy
import unittest
import numpy as np
from quality import baseline_agrees, validated_records, result_status, RUNTIME

class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.old=dict(model='llama8b',corpus='wiki',policy='four_over_six',
            source='research_runs/primary/runs/V31_ppl_primary_llama8b_attempt1/ppl/ppl_report.json',
            compatibility_key='same',cluster_ids=['a','b'],weight_sha256='same',
            window_tokens=np.array([10,20]),window_nll=np.array([2.,3.]))
        self.new=copy.deepcopy(self.old)
        self.new['source']=f'research_runs/{RUNTIME.name}/runs/full/ppl/ppl_report.json'

    def test_new_status_and_exact_anchor(self):
        self.assertEqual(result_status(self.old),'validated_reuse')
        self.assertEqual(result_status(self.new),'validated_new')
        self.assertTrue(baseline_agrees(self.new,self.old))
        kept,excluded=validated_records([self.old,self.new],[])
        self.assertEqual(len(kept),2);self.assertFalse(excluded)

    def test_window_drift_cannot_cancel_in_aggregate(self):
        self.new['window_nll']+=np.array([.02,-.01])
        self.assertAlmostEqual(float(self.new['window_nll']@self.new['window_tokens']),80.)
        self.assertFalse(baseline_agrees(self.new,self.old))
        kept,excluded=validated_records([self.old,self.new],[])
        self.assertEqual(len(kept),1);self.assertEqual(len(excluded),1)

    def test_missing_baseline_or_changed_identity_rejected(self):
        candidate=copy.deepcopy(self.new);candidate['policy']='new_map'
        kept,excluded=validated_records([self.old,candidate],[])
        self.assertEqual(len(kept),1);self.assertEqual(len(excluded),1)
        for key,value in [('compatibility_key','different'),('weight_sha256','different'),('cluster_ids',['b','a'])]:
            other=copy.deepcopy(self.new);other[key]=value
            self.assertFalse(baseline_agrees(other,self.old))

if __name__=='__main__':unittest.main()
