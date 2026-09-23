"""Reject misleading aggregate matches and GPU-monitor gaps, CPU fixtures only."""
import copy
import tempfile
import unittest
from unittest.mock import patch
from accuracy_new_results import *

class AccuracyAdmission(unittest.TestCase):
    def fixture(self):
        data={t:dict(ids=np.array(['a','b']),correct=np.array([0.,1.]),response_sha256=np.array(['r0','r1'])) for t in TASKS}
        identity={k:k for k in IDENTITY_KEYS}
        return data,identity

    def test_exact_baseline_passes(self):
        data,i=self.fixture();require_baseline_identity(data,copy.deepcopy(data),i,dict(i))

    def test_same_score_different_ids_rejected(self):
        data,i=self.fixture();new=copy.deepcopy(data);new['piqa']['ids']=np.array(['c','d'])
        with self.assertRaises(AssertionError):require_baseline_identity(data,new,i,i)

    def test_same_accuracy_different_responses_rejected(self):
        data,i=self.fixture();new=copy.deepcopy(data);new['mmlu']['response_sha256'][0]='zz'
        with self.assertRaises(AssertionError):require_baseline_identity(data,new,i,i)

    def test_swapped_correctness_with_same_mean_rejected(self):
        data,i=self.fixture();new=copy.deepcopy(data);new['boolq']['correct']=np.array([1.,0.])
        with self.assertRaises(AssertionError):require_baseline_identity(data,new,i,i)

    def test_runtime_change_rejected(self):
        data,i=self.fixture();j=dict(i,environment='different')
        with self.assertRaises(AssertionError):require_baseline_identity(data,data,i,j)

    def test_monitor_gap_rejected_even_when_checks_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            jsonout(run/'launch_record.json',dict(status='complete',invalid_reasons=None,
                started_utc='2026-09-22T00:00:00+00:00',finished_utc='2026-09-22T00:02:01+00:00'))
            (run/'preflight_candidates.jsonl').write_text(json.dumps(dict(passed=True))+'\n')
            (run/'gpu_monitor.jsonl').write_text(json.dumps(dict(passed=True,timestamp_utc='2026-09-22T00:00:01Z'))+'\n')
            jsonout(run/'postflight.json',dict(passed=True))
            with self.assertRaisesRegex(AssertionError,'monitoring gap'):require_gpu_record(run)

    def test_coverage_only_fills_actual_new_results(self):
        row=dict(model='fixture',draw='seed0',policy='ce_matched',task='piqa',status='coverage_gap')
        self.assertEqual(merge_coverage([row],[dict(row)]),[row])
        fresh=dict(row,status='validated_new',value=.5)
        self.assertEqual(merge_coverage([row],[fresh]),[fresh])
        self.assertEqual(row['status'],'coverage_gap')

    def test_duplicate_or_historical_overwrite_rejected(self):
        row=dict(model='fixture',draw='seed0',policy='ce_matched',task='piqa',status='coverage_gap')
        fresh=dict(row,status='validated_new',value=.5)
        with self.assertRaisesRegex(AssertionError,'duplicate'):merge_coverage([row],[fresh,fresh])
        with self.assertRaisesRegex(AssertionError,'overwrite'):merge_coverage([dict(row,status='sealed_report_reuse')],[fresh])

if __name__=='__main__':unittest.main()
