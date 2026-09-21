import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_taskfit_diagnosis import collect_hashes, shifted_weight


class TaskfitTests(unittest.TestCase):
    def test_exclusions_cover_scalar_and_list_forms(self):
        value = {'fit': {'token_sha256': ['one', 'two'],
                         'documents': [{'document_sha256': 'document'}]},
                 'records': [{'token_sha256': 'three', 'document_sha256': 'other'}]}
        self.assertEqual(collect_hashes(value, 'token_sha256'), {'one', 'two', 'three'})
        self.assertEqual(collect_hashes(value, 'document_sha256'), {'document', 'other'})

    def test_counterfactual_uses_candidate_minus_raw_not_source_error(self):
        original = torch.ones(2, 4, dtype=torch.bfloat16)
        raw = torch.full_like(original, .75)
        candidate = torch.full_like(original, .875)
        self.assertTrue(torch.equal(shifted_weight(original, raw, candidate), torch.full_like(original, 1.125)))
        self.assertTrue(torch.equal(shifted_weight(original, raw, candidate, -1), candidate))
        self.assertTrue(torch.equal(shifted_weight(original, raw, raw), original))
        self.assertTrue(torch.equal(original, torch.ones_like(original)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
