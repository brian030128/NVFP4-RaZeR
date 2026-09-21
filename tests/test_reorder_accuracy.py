import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize_reorder_accuracy import paired_accuracy, statistics


class AccuracyTests(unittest.TestCase):
    def test_paired_wins_losses_and_exact_null(self):
        a = [dict(doc_id=i, content_sha256=str(i), acc=v) for i, v in enumerate([1, 0, 1, 0])]
        b = [dict(doc_id=i, content_sha256=str(i), acc=v) for i, v in enumerate([0, 1, 1, 0])]
        r = statistics(paired_accuracy(a, b, 'acc'))
        self.assertEqual((r['wins'], r['losses'], r['delta'], r['p_exact']), (1, 1, 0., 1.))
        self.assertEqual(statistics(paired_accuracy(a, a, 'acc'))['two_se'], 0.)

    def test_questions_must_match_exactly(self):
        a = [dict(doc_id=0, content_sha256='a', acc=1), dict(doc_id=1, content_sha256='b', acc=0)]
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            paired_accuracy(a, [dict(r, content_sha256='changed') for r in a], 'acc')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            paired_accuracy(a, [a[0], a[0]], 'acc')


if __name__ == '__main__':
    unittest.main(verbosity=2)
