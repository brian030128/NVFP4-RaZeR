"""Run under Slurm: python tests/test_reorder_objective_ablation.py."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantize.task_reorder import SearchConfig, search_layout
from run_reorder_objective_ablation import (VARIANTS, balanced_signs, run, shrink_,
                                            transform)


def write_scores(directory, ce, kl, name='m.weight', atom=(1, 16)):
    """Emit a score shard set in the layout run_task_reorder.load_scores expects."""
    directory.mkdir(parents=True, exist_ok=True)
    sequences = ce.shape[0]
    manifest = dict(schema='mixfp4_reorder_scores_v1', status='complete', name=name,
                    atom_shape=list(atom),
                    weight_shape=[ce.shape[1] * atom[0], ce.shape[2] * atom[1]],
                    sequence_ids=[f'{i:064x}' for i in range(sequences)],
                    sequence_sources=['math' if i % 2 else 'code' for i in range(sequences)])
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    for i in range(sequences):
        torch.save(dict(sequence_id=manifest['sequence_ids'][i], ce=ce[i], kl=kl[i]),
                   directory / f'{i:03d}.pt')
    return manifest


class ObjectiveAblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_ce_kl_variant_is_exactly_the_deployed_transform(self):
        # The reproduction arm must not perturb the scores at all, otherwise the
        # ablation cannot be compared with the recorded deployed results.
        ce, kl = torch.randn(6, 4, 3), torch.randn(6, 4, 3)
        out_ce, out_kl, detail = transform('ce_kl', ce, kl, 1234, 'fit')
        self.assertTrue(torch.equal(out_ce, ce))
        self.assertTrue(torch.equal(out_kl, kl))
        self.assertEqual(detail, {})

    def test_single_objective_variants_collapse_the_conjunction(self):
        ce, kl = torch.randn(6, 4, 3), torch.randn(6, 4, 3)
        only_kl = transform('kl', ce, kl, 1234, 'fit')
        self.assertTrue(torch.equal(only_kl[0], kl))
        self.assertTrue(torch.equal(only_kl[1], kl))
        only_ce = transform('ce', ce, kl, 1234, 'fit')
        self.assertTrue(torch.equal(only_ce[0], ce))
        self.assertTrue(torch.equal(only_ce[1], ce))
        # Duplication must not alias, or an in-place shrink would hit both arms.
        self.assertIsNot(only_kl[0], only_kl[1])
        self.assertIsNot(only_ce[0], only_ce[1])

    def test_balanced_signs_cancel_a_constant_signal_exactly(self):
        signs = balanced_signs(64, 7)
        self.assertEqual(int((signs < 0).sum()), 32)
        self.assertEqual(float(signs.sum()), 0.)
        self.assertTrue(torch.equal(signs, balanced_signs(64, 7)))
        self.assertFalse(torch.equal(signs, balanced_signs(64, 8)))
        with self.assertRaises(ValueError):
            balanced_signs(7, 0)

    def test_placebo_destroys_the_mean_and_keeps_the_spread(self):
        ce = -3 + torch.randn(64, 5, 4)
        out_ce, out_kl, detail = transform('placebo', ce, ce.clone(), 1234, 'fit')
        self.assertEqual(detail['flipped_sequences'], 32)
        # The planted -3 cancels exactly; only the noise term survives.
        self.assertLess(float(out_ce.mean(0).abs().max()), 1.0)
        self.assertGreater(float(ce.mean(0).abs().min()), 2.0)
        self.assertTrue(torch.equal(out_ce.abs(), ce.abs()))
        self.assertTrue(torch.equal(out_ce, out_kl))

    def test_shrinkage_erases_noise_atoms_and_keeps_strong_ones(self):
        # Column 0 is pure noise, column 1 is a large real effect.
        scores = torch.randn(64, 1, 2)
        scores[:, 0, 1] -= 50.
        before_std = scores.std(0)
        before_deviation = scores - scores.mean(0, keepdim=True)
        shrink_(scores)
        self.assertLess(abs(float(scores.mean(0)[0, 0])), .2)
        self.assertLess(abs(float(scores.mean(0)[0, 1]) + 50.), 1.)
        # Only the mean moves: per-sequence deviations and spread are preserved,
        # so the noise the search must beat is unchanged.
        self.assertTrue(torch.allclose(scores - scores.mean(0, keepdim=True),
                                       before_deviation, atol=1e-5))
        self.assertTrue(torch.allclose(scores.std(0), before_std, atol=1e-5))
        with self.assertRaises(ValueError):
            shrink_(torch.randn(2, 1, 2))

    def test_shrinkage_survives_zero_and_tiny_atoms(self):
        # Padding rows and dead channels give all-zero atoms; squaring a tiny
        # float32 mean underflows. Neither may produce a nonfinite score, which
        # quantize/task_reorder._features would reject outright.
        scores = torch.zeros(8, 1, 3)
        scores[:, 0, 1] = 1e-24
        scores[:, 0, 2] = torch.linspace(-1e-24, 1e-24, 8)
        shrink_(scores)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertEqual(float(scores[:, 0, 0].abs().max()), 0.)

    def test_placebo_fit_objective_is_far_below_a_planted_layout(self):
        # Planted structure both axes must recover, plus per-sequence noise.
        rows = torch.tensor([1., -1.]).repeat(256)
        cols = torch.tensor([1., -1.]).repeat(4)
        signal = -(rows[:, None] * cols[None, :])
        ce = signal[None] + .05 * torch.randn(8, 512, 8)
        config = SearchConfig(rounds=2, starts=2, swap_samples=128)
        real = search_layout(ce, ce.clone(), config)
        null_ce, null_kl, _ = transform('placebo', ce, ce.clone(), 1234, 'fit')
        null = search_layout(null_ce, null_kl, config)
        self.assertGreater(real['fit_objective'], 10 * max(null['fit_objective'], 1e-9))

    def test_runner_reports_generalization_and_refuses_bad_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Real structure in the fit half only: election must not confirm it.
            ce = .4 * torch.randn(8, 8, 4)
            write_scores(root / 'scores', ce, ce.clone())
            config = SearchConfig(tile_rows=4, tile_cols=32, rounds=1, starts=2,
                                  swap_samples=16)
            report = run(root / 'scores', root / 'out', 'ce_kl', config)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['model_forward_passes'], 0)
            self.assertEqual(report['fit_sequences'], 4)
            self.assertEqual(report['election_sequences'], 4)
            self.assertIn('election_fit_ratio', report)
            self.assertEqual(report['election_transform_variant'], 'ce_kl')
            self.assertTrue((root / 'out' / 'layout.pt').exists())
            # A second run into the same directory must not silently overwrite.
            with self.assertRaises(FileExistsError):
                run(root / 'scores', root / 'out', 'ce_kl', config)
            with self.assertRaises(ValueError):
                run(root / 'scores', root / 'out2', 'nope', config)

    def test_shrinkage_leaves_held_out_election_scores_untouched(self):
        # Shrinkage regularizes the search; scoring the frozen layout on held-out
        # sequences must stay on raw scores or the generalization test is circular.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ce = .4 * torch.randn(8, 8, 4)
            write_scores(root / 'scores', ce, ce.clone())
            config = SearchConfig(tile_rows=4, tile_cols=32, rounds=1, starts=2,
                                  swap_samples=16)
            report = run(root / 'scores', root / 'out', 'shrunk', config)
            self.assertEqual(report['election_transform_variant'], 'ce_kl')
            self.assertEqual(report['election_transform'], {})
            self.assertGreater(report['fit_transform']['mean_shrinkage_ce'], 0.)
            self.assertLessEqual(report['fit_transform']['mean_shrinkage_ce'], 1.)

    def test_every_declared_variant_is_accepted(self):
        ce, kl = torch.randn(4, 2, 2), torch.randn(4, 2, 2)
        for variant in VARIANTS:
            transform(variant, ce.clone(), kl.clone(), 1234, 'fit')


if __name__ == '__main__':
    unittest.main(verbosity=2)
