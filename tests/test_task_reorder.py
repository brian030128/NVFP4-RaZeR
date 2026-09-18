"""Run under Slurm: python tests/test_task_reorder.py."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantize.task_reorder import (SearchConfig, elect_layout, reordered_weight_reference,
                                   scale_block_scores, search_layout)
from run_task_reorder import run, split_sequences


class TaskReorderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_joint_search_recovers_scrambled_256x64_rectangles(self):
        # Both axes must change: every original 256-row group and every original
        # four-chunk K group has equal positive and negative mass.
        rows = torch.tensor([1., -1.]).repeat(256)
        cols = torch.tensor([1., -1.]).repeat(4)
        score = -(rows[:, None] * cols[None, :])
        ce = score[None] * torch.linspace(.98, 1.02, 8)[:, None, None]
        result = search_layout(ce, ce * .3, SearchConfig(rounds=3, swap_samples=128))
        self.assertEqual(result['identity_objective'], 0.)
        self.assertGreater(result['fit_objective'], 1900.)
        self.assertEqual(result['fit_mask'].shape, (2, 2))
        self.assertEqual(int(result['fit_mask'].sum()), 2)
        self.assertFalse(torch.equal(result['row_perm'], torch.arange(512)))
        self.assertFalse(torch.equal(result['col_perm'], torch.arange(128)))

    def test_aggregate_sequences_before_standard_errors(self):
        noise = torch.tensor([-10., 10., -10., 10.])
        ce = torch.stack((-1 + noise, -1 - noise), dim=-1)[:, :, None]
        layout = search_layout(ce, ce, SearchConfig(tile_rows=2, tile_cols=16, rounds=0))
        elected = elect_layout(ce, ce, layout)
        self.assertTrue(bool(elected['mask'][0, 0]))
        self.assertAlmostEqual(float(elected['upper_ce'][0, 0]), -2.)
        # Individual cells do not pass; summing their SEs would wrongly reject.
        upper = ce.mean(0) + 3 * ce.std(0) / 2
        self.assertTrue(bool((upper > 0).all()))

    def test_independent_election_rejects_fit_winners(self):
        ce = -torch.ones(4, 8, 8)
        config = SearchConfig(tile_rows=4, rounds=1, starts=2, swap_samples=16)
        layout = search_layout(ce, ce, config)
        self.assertTrue(bool(layout['fit_mask'].all()))
        before = layout['row_perm'].clone()
        result = elect_layout(ce, -ce, layout)
        self.assertFalse(bool(result['mask'].any()))
        self.assertTrue(torch.equal(before, layout['row_perm']))

    def test_permutation_contract_and_task_contractions(self):
        base, alternative = torch.randn(8, 128), torch.randn(8, 128)
        gradient = torch.randn(6, 8, 128)
        atoms = torch.stack([scale_block_scores(g, alternative-base) for g in gradient])
        config = SearchConfig(tile_rows=4, rounds=2, starts=3, swap_samples=64)
        layout = search_layout(atoms, atoms * .5, config)
        election = elect_layout(atoms, atoms * .5, layout)
        mask = election['mask']
        qp = reordered_weight_reference(base, alternative, layout, mask)
        rp, cp = layout['row_perm'], layout['col_perm']
        q = qp[layout['inverse_row_perm']][:, layout['inverse_col_perm']]
        x = torch.randn(3, 128)
        yp = x[:, cp] @ qp.T
        self.assertTrue(torch.allclose(yp[:, layout['inverse_row_perm']], x @ q.T,
                                       atol=1e-5, rtol=1e-5))
        # Independent direct elementwise contraction in the deployed basis.
        d = (alternative-base)[rp][:, cp]
        per_tile = torch.stack([(g[rp][:, cp] * d).reshape(2, 4, 2, 64).sum((1, 3))
                                for g in gradient]).double()
        upper = per_tile.mean(0) + 3 * per_tile.std(0) / (6 ** .5)
        self.assertTrue(torch.allclose(upper, election['upper_ce'], atol=2e-5, rtol=1e-5))

    def test_partial_tiles_survive_export_and_never_lose_to_identity(self):
        scores = torch.randn(6, 11, 7) - .1
        config = SearchConfig(tile_rows=4, rounds=3, starts=3, swap_samples=64)
        layout = search_layout(scores, scores * .6, config)
        self.assertGreaterEqual(layout['fit_objective'], layout['identity_objective'])
        self.assertEqual(layout['padded_shape'], [12, 128])
        result = elect_layout(scores, scores * .6, layout)
        self.assertAlmostEqual(layout['fit_objective'], result['objective'], places=8)
        rp, cp = layout['row_perm'], layout['col_atom_perm']
        perm = scores.double()[:, rp][:, :, cp]
        padded = torch.nn.functional.pad(perm, (0, 1, 0, 1))
        tiled = padded.reshape(6, 3, 4, 2, 4).sum((2, 4))
        expected = tiled.mean(0) + 3 * tiled.std(0) / (6 ** .5)
        self.assertTrue(torch.allclose(expected, result['upper_ce'], atol=1e-12, rtol=1e-12))
        previous = {}
        for entry in layout['trace']:
            key = entry['start'], entry['temperature']
            value = entry['continuation_objective']
            if key in previous:
                self.assertGreaterEqual(value + 1e-9, previous[key])
            previous[key] = value

    def test_candidate_quantization_commutes_with_scale_group_permutation(self):
        from quantize.quantizer import quant_nvfp4_4over6, quant_mix_4_6
        weight = torch.randn(16, 128).bfloat16()
        rows = torch.randperm(16)
        groups = torch.randperm(8)
        columns = (groups[:, None] * 16 + torch.arange(16)).flatten()
        permuted = weight[rows][:, columns]
        for quantizer in (lambda w: quant_nvfp4_4over6(w, 4, 16),
                          lambda w: quant_mix_4_6(w, 4, 16, type_block=(8, 64),
                                                  clip='a1', elect='always')):
            self.assertTrue(torch.equal(quantizer(permuted), quantizer(weight)[rows][:, columns]))

    def test_determinism_and_axis_constraints(self):
        scores = torch.randn(4, 12, 8)
        for axes in ('rows', 'cols'):
            config = SearchConfig(tile_rows=4, rounds=2, starts=2, swap_samples=32, axes=axes)
            state = torch.random.get_rng_state().clone()
            one = search_layout(scores, scores, config)
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            two = search_layout(scores, scores, config)
            self.assertTrue(torch.equal(one['row_perm'], two['row_perm']))
            self.assertTrue(torch.equal(one['col_perm'], two['col_perm']))
            key, n = ('col_perm', 128) if axes == 'rows' else ('row_perm', 12)
            self.assertTrue(torch.equal(one[key], torch.arange(n)))

    def test_invalid_inputs_and_degenerate_scores(self):
        scores = torch.zeros(4, 1, 1)
        result = search_layout(scores, scores, SearchConfig(rounds=1, starts=2))
        self.assertFalse(bool(result['fit_mask'].any()))
        with self.assertRaises(ValueError):
            search_layout(scores[:1], scores[:1])
        with self.assertRaises(ValueError):
            search_layout(scores, scores + float('nan'))
        with self.assertRaises(ValueError):
            search_layout(scores, scores, SearchConfig(atom_cols=8))
        with self.assertRaises(ValueError):
            search_layout(scores, scores, objective_scales=[0., 1.])

    def test_worker_runner_and_split_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'scores'
            directory.mkdir()
            manifest = dict(schema='mixfp4_reorder_scores_v1', status='complete', name='test.weight',
                            atom_shape=[1, 16], weight_shape=[8, 128],
                            sequence_ids=[f'seq{i}' for i in range(16)],
                            sequence_sources=['math']*8 + ['code']*8)
            (directory / 'manifest.json').write_text(json.dumps(manifest))
            for i, identifier in enumerate(manifest['sequence_ids']):
                ce = torch.randn(8, 8) - .2
                torch.save(dict(ce=ce, kl=ce*.2, sequence_id=identifier), directory / f'{i:03d}.pt')
            fit, election = split_sequences(manifest, .5, 11)
            self.assertFalse(set(fit) & set(election))
            self.assertEqual(len(fit), 8)
            out = Path(temporary) / 'out'
            report = run(directory, out, SearchConfig(tile_rows=4, rounds=1, starts=2, swap_samples=16))
            artifact = torch.load(out / 'layout.pt', weights_only=True)
            self.assertEqual(report['status'], 'complete')
            self.assertFalse(set(artifact['fit_sequence_ids']) & set(artifact['election_sequence_ids']))
            self.assertIn('mask', artifact)
            self.assertFalse(report['quality_evaluated'])


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Tests must run on a Slurm worker')
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
