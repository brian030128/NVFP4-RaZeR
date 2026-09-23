import unittest
from common import OUT
from accuracy_run import transform

class AccuracyAdapterTests(unittest.TestCase):
    def setUp(self):
        self.source=(OUT/'scripts/gpu_run.py').read_text();self.adapted=transform(self.source)

    def test_lease_and_preflight_unchanged(self):
        a=self.source.index("    leases=RUNTIME/'leases'");b=self.source.index("    env['CAMPAIGN_LEASE_ID']")
        self.assertEqual(self.adapted.count('inventory_or_wait(gp,root)'),1)
        restored=self.adapted.replace('inventory_or_wait(gp,root)','gp.smi_gpus()',1)
        self.assertIn(self.source[a:b],restored)

    def test_monitor_and_cleanup_unchanged(self):
        a=self.source.index('    t0=time.monotonic()');b=self.source.index('        with (OUT/\'RUN_LOG.jsonl\')')
        self.assertIn(self.source[a:b],self.adapted)

    def test_only_frozen_accuracy_payload(self):
        self.assertIn("'-m','campaign.job_wrapper','--gpu-run','--'",self.adapted)
        self.assertIn("'-m','campaign.evaluate_lmeval'",self.adapted)
        self.assertIn("'--suite','full8'",self.adapted)
        self.assertNotIn("'-m','campaign.evaluate_ppl'",self.adapted)
        with self.assertRaises(AssertionError):transform(self.source+'\n')

    def test_restricted_uuid_only_changes_selection_and_provenance(self):
        uuid='GPU-9cec7336-5b30-3f86-35e7-06919156e7da'
        text=transform(self.source,uuid)
        text=text.replace("if g['uuid'] != "+repr(uuid)+" or g['uuid'] in active", "if g['uuid'] in active",1)
        text=text.replace("    rec.update(allowed_uuid="+repr(uuid)+")\n",'',1)
        self.assertEqual(text,self.adapted)
        with self.assertRaises(ValueError):transform(self.source,'not-a-uuid')

if __name__=='__main__':unittest.main()
