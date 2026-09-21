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
                                   original_order_weight_reference, scale_block_scores, search_layout)
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
        layout['mask'] = mask
        self.assertTrue(torch.equal(q, original_order_weight_reference(base, alternative, layout)))
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

    def test_quality_controls_and_illegal_scale_group(self):
        base, alternative = torch.randn(512, 128), torch.randn(512, 128)
        scores = torch.randn(4, 512, 8)
        layout = search_layout(scores, scores, SearchConfig(rounds=0, starts=1))
        layout['mask'] = torch.tensor([[True, False], [False, True]])
        layout['identity_mask'] = layout['mask']
        layout['identity_8x64_mask'] = torch.rand(64, 2) > .5
        for policy, row_size in (('identity', 256), ('identity_8x64', 8)):
            element_mask = layout[policy + '_mask'].repeat_interleave(row_size, 0).repeat_interleave(64, 1)
            expected = torch.where(element_mask, alternative, base)
            self.assertTrue(torch.equal(expected, original_order_weight_reference(base, alternative, layout, policy)))
        layout['col_perm'][0], layout['col_perm'][16] = layout['col_perm'][16].clone(), layout['col_perm'][0].clone()
        with self.assertRaisesRegex(ValueError, 'scale group'):
            original_order_weight_reference(base, alternative, layout)

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
                            sequence_sources=['math']*8 + ['code']*8,
                            weight_sha256='test_digest', revision='test_revision')
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
            self.assertIn('identity_mask', artifact)
            self.assertEqual(artifact['identity_8x64_mask'].shape, (1, 2))
            self.assertFalse(report['quality_evaluated'])

    def test_evaluator_requires_matched_frozen_layouts(self):
        from run_task_reorder_eval import load_layouts, paired
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior = dict(reorder_modules={'test.weight': 'scores'}, revision='test_revision',
                         matrices={'test.weight': {'source_sha256': 'test_digest'}})
            layout = search_layout(-torch.ones(4, 256, 4), -torch.ones(4, 256, 4),
                                   SearchConfig(rounds=0, starts=1))
            layout.update(name='test.weight', fit_sequence_ids=['fit'], election_sequence_ids=['elect'],
                          provenance=dict(status='complete', weight_sha256='test_digest',
                                          revision='test_revision', sequence_ids=['fit', 'elect']),
                          mask=torch.ones(1, 1, dtype=torch.bool), identity_mask=torch.ones(1, 1, dtype=torch.bool),
                          identity_8x64_mask=torch.ones(32, 1, dtype=torch.bool))
            for label in ('both', 'rows'):
                path = root / label / '000'
                path.mkdir(parents=True)
                torch.save(layout, path / 'layout.pt')
                (path / 'report.json').write_text('{"status": "complete"}')
            specs = [f'{label}={root / label}' for label in ('both', 'rows')]
            panels, hashes = load_layouts(specs, prior)
            self.assertEqual(len(hashes), 2)
            layout['election_sequence_ids'] = ['fit']
            torch.save(layout, root / 'rows/000/layout.pt')
            with self.assertRaisesRegex(ValueError, 'leakage'):
                load_layouts(specs, prior)
            layout['election_sequence_ids'] = ['elect']
            layout['identity_mask'].zero_()
            torch.save(layout, root / 'rows/000/layout.pt')
            with self.assertRaisesRegex(ValueError, 'Identity controls'):
                load_layouts(specs, prior)
        result = paired([1., 2., 3.], [1.5, 2.5, 3.5])
        self.assertEqual(result['delta_nll'], -.5)
        self.assertEqual(result['two_se'], 0.)

    def test_raw256_control_preserves_covariance_and_tail(self):
        from run_task_reorder_eval import coarse_mask, mix_coarse
        scores = torch.ones(128, 34, 2)
        # Opposing sequence noise cancels within a coarse tile. Testing the
        # constituent 8x64 bounds before summing would reject these tiles.
        noise = torch.tensor([-100., 100.]).repeat(64)
        scores[:, :32, 0] = -1
        scores[:, 0, 0] += noise
        scores[:, 1, 0] -= noise
        scores[:, 32:, 1] = -1
        mask = coarse_mask(scores.flatten(1), scores.flatten(1) * .3, [272, 128])
        self.assertTrue(torch.equal(mask, torch.tensor([[True, False], [False, True]])))
        base, alternative = torch.zeros(272, 128), torch.ones(272, 128)
        mixed = mix_coarse(base, alternative, mask)
        self.assertEqual(float(mixed.sum()), 256 * 64 + 16 * 64)

    def test_published_windows_are_tokenizer_specific(self):
        import copy
        from run_task_reorder_eval import validate_evaluation_data
        for model, job, windows in [('qwen27b', '336969', 145), ('llama8b', '336566', 141)]:
            reference = json.loads(Path(f'results/kse_paper/job_{job}/{model}/report.json').read_text())['data']
            self.assertEqual(reference['wiki']['windows'], windows)
            actual = copy.deepcopy(reference)
            validate_evaluation_data(actual, reference)
            actual['wiki']['token_sha256'][0] = 'wrong-window'
            with self.assertRaisesRegex(ValueError, 'token_sha256'):
                validate_evaluation_data(actual, reference)

    def test_shared_aggregation_matches_direct_permuted_tiles(self):
        from quantize.shared_mlp_reorder import aggregate_atoms, phis
        p = torch.randperm(32)
        labels = torch.empty(32, dtype=torch.long)
        labels[p] = torch.arange(32) // 4
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            n, c = (256, 32) if name == 'down_proj' else (512, 16)
            ce, kl = torch.randn(4, n, c), torch.randn(4, n, c)
            features = aggregate_atoms(ce, kl, name)
            actual = phis({name: features}, labels)[name].reshape(-1, features.shape[1], 2, 4)
            row_perm = torch.arange(n) if name == 'down_proj' else (p[:, None] * 16 + torch.arange(16)).flatten()
            col_perm = p if name == 'down_proj' else torch.arange(c)
            for objective, scores in enumerate((ce, kl)):
                arranged = scores.double()[:, row_perm][:, :, col_perm]
                direct = arranged.reshape(4, n // 256, 256, c // 4, 4).sum((2, 4))
                if name == 'down_proj':
                    direct = direct.transpose(1, 2)
                self.assertTrue(torch.allclose(actual[:, :, objective].permute(2, 0, 1), direct, atol=1e-12, rtol=1e-12))

    def test_shared_search_rejects_nontransferring_train_gain(self):
        from quantize.shared_mlp_reorder import SharedConfig, search_shared
        score = -torch.tensor([1., -1.]).repeat(16)[:, None] * torch.tensor([1., -1.])[None]
        x = score[:, :, None].repeat(1, 1, 16).double()
        train = {name: x.clone() for name in ('gate_proj', 'up_proj', 'down_proj')}
        config = SharedConfig(starts=3, rounds=3, swap_samples=64, swap_passes=1)
        rejected = search_shared(train, {name: -v for name, v in train.items()}, config)
        self.assertTrue(torch.equal(rejected['group_perm'], torch.arange(32)))
        self.assertGreater(max(c['fit_objective'] for c in rejected['candidates']), 0.)
        accepted = search_shared(train, train, config)
        self.assertGreater(accepted['selected_validation']['score'], 0.)
        self.assertFalse(torch.equal(accepted['group_perm'], torch.arange(32)))
        self.assertTrue(torch.equal(accepted['group_perm'].sort().values, torch.arange(32)))

    def test_shared_mlp_fold_with_quantized_activation_groups(self):
        from quantize.quantizer import quant_nvfp4_4over6
        x = torch.randn(3, 256).double() * .1
        gate, up = torch.randn(512, 256).double() * .1, torch.randn(512, 256).double() * .1
        down = torch.randn(256, 512).double() * .1
        bias_gate, bias_up = torch.randn(512).double(), torch.randn(512).double()
        p = (torch.randperm(32)[:, None] * 16 + torch.arange(16)).flatten()
        def forward(g, u, d, bg, bu):
            hidden = torch.nn.functional.silu(x @ g.T + bg) * (x @ u.T + bu)
            return quant_nvfp4_4over6(hidden, 4, 16).double() @ d.T
        original = forward(gate, up, down, bias_gate, bias_up)
        folded = forward(gate[p], up[p], down[:, p], bias_gate[p], bias_up[p])
        self.assertTrue(torch.allclose(original, folded, atol=1e-10, rtol=1e-10))

    def test_finite_replay_selection_requires_both_losses(self):
        from run_reorder_replay import paired_bounds, select_policy
        reference = [1., 2., 3., 4.]
        good = paired_bounds([v - .1 for v in reference], reference)
        bad = paired_bounds([v + .1 for v in reference], reference)
        scores = {'four_over_six': {'ce': bad, 'kl': bad},
                  'ce_only': {'ce': paired_bounds([v - 1 for v in reference], reference), 'kl': bad},
                  'joint': {'ce': good, 'kl': good}}
        self.assertEqual(select_policy(scores, list(scores)), 'joint')
        self.assertEqual(select_policy(scores, ['four_over_six', 'ce_only']), 'four_over_six')

    def test_confirmation_cannot_switch_to_another_winner(self):
        from run_reorder_replay import confirm_choices
        scores = {'four_over_six': {'ce': {'upper': 0}, 'kl': {'upper': 0}},
                  'both': {'ce': {'upper': -.5}, 'kl': {'upper': .1}},
                  'shared': {'ce': {'upper': -1}, 'kl': {'upper': -1}}}
        choices = {'selected_any': 'both', 'selected_foldable': 'shared'}
        self.assertEqual(confirm_choices(scores, choices),
                         {'selected_any': 'four_over_six', 'selected_foldable': 'shared'})

    def test_rowband_evaluation_scope_and_atom_guards(self):
        from run_task_reorder_eval import load_layouts
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / '000'
            path.mkdir()
            layout = search_layout(-torch.ones(4, 32, 2), -torch.ones(4, 32, 2),
                SearchConfig(atom_rows=8, atom_cols=64, axes='rows', rounds=0, starts=1))
            layout.update(name='expanded', fit_sequence_ids=['fit'], election_sequence_ids=['elect'],
                provenance=dict(status='complete', weight_sha256='digest', revision='revision',
                                sequence_ids=['fit', 'elect']),
                mask=torch.ones(1, 2, dtype=torch.bool), identity_mask=torch.ones(1, 2, dtype=torch.bool),
                identity_8x64_mask=torch.ones(32, 2, dtype=torch.bool))
            prior = dict(reorder_modules={'original': ''}, revision='revision',
                         matrices={'expanded': {'source_sha256': 'digest'}, 'original': {}})
            (path / 'report.json').write_text('{"status": "complete"}')
            torch.save(layout, path / 'layout.pt')
            spec = [f'bands={root}']
            with self.assertRaisesRegex(ValueError, 'Unexpected'):
                load_layouts(spec, prior)
            with self.assertRaisesRegex(ValueError, 'atoms'):
                load_layouts(spec, prior, ['expanded'])
            load_layouts(spec, prior, ['expanded'], True)
            layout['col_perm'] = layout['col_perm'].roll(16)
            torch.save(layout, path / 'layout.pt')
            with self.assertRaisesRegex(ValueError, 'cannot reorder columns'):
                load_layouts(spec, prior, ['expanded'], True)
            layout['col_perm'] = torch.arange(128)
            layout['row_perm'][[0, 8]] = layout['row_perm'][[8, 0]]
            torch.save(layout, path / 'layout.pt')
            with self.assertRaisesRegex(ValueError, 'splits'):
                load_layouts(spec, prior, ['expanded'], True)

    def test_parallel_evaluation_merge_rejects_mismatch_and_recomputes_pairs(self):
        import copy
        from merge_reorder_evaluations import merge_reports
        common = dict(status='complete', model='test', source='test', revision='pinned',
            background='raw256', activation='fixed', attention='sdpa', length=2048,
            data={'tokens': ['a', 'b']}, layout_sha256={'file': 'hash'}, source_sha256={'code': 'hash'},
            pilot_modules=['mlp'], elected_tiles={'raw256': 1, 'rows': 2},
            pilot_elected_tiles={'rows': 1}, mask_selection={'rows': 'k3'})
        reference = dict(common, policies=['raw256'], evaluation={'raw256': {'wiki': {'nll': [2., 3.], 'ppl': 12.}}})
        changed = dict(common, policies=['rows'], evaluation={'rows': {'wiki': {'nll': [1.9, 2.9], 'ppl': 11.}}})
        merged = merge_reports([reference, changed], ['raw256', 'rows'])
        self.assertAlmostEqual(merged['paired']['rows - raw256']['wiki']['delta_nll'], -.1)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            merge_reports([reference, reference], ['raw256'])
        wrong = copy.deepcopy(changed)
        wrong['data']['tokens'][0] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'mismatch: data'):
            merge_reports([reference, wrong], ['raw256', 'rows'])
        with self.assertRaisesRegex(ValueError, 'cover'):
            merge_reports([reference], ['raw256', 'rows'])

    def test_one_by_64_scores_preserve_individual_row_elections(self):
        from run_fine_row_research import row_direction_scores
        x, dy, direction = torch.randn(2, 5, 128), torch.randn(2, 5, 256), torch.randn(256, 128)
        actual = row_direction_scores(x, dy, direction)
        gradient = dy.flatten(0, 1).T @ x.flatten(0, 1)
        atoms = scale_block_scores(gradient, direction)
        self.assertTrue(torch.allclose(actual, atoms.reshape(256, 2, 4).sum(-1), atol=1e-5, rtol=1e-5))
        fine = -torch.ones(4, 256, 8)
        order = torch.randperm(256)
        layout = search_layout(fine, fine, SearchConfig(rounds=0, starts=1, axes='rows'))
        layout['row_atom_perm'] = order
        expected = elect_layout(fine, fine, layout)['mask']
        grouped = fine.reshape(4, 256, 2, 4).sum(-1)
        layout['config']['atom_cols'] = 64
        layout['col_atom_perm'] = torch.arange(2)
        self.assertTrue(torch.equal(elect_layout(grouped, grouped, layout)['mask'], expected))

    def test_truncated_frozen_prefix_preserves_selected_gradients(self):
        from run_fine_row_research import arm_gradient
        layers = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.SiLU(),
                                     torch.nn.Linear(16, 16), torch.nn.SiLU(), torch.nn.Linear(16, 8))
        layers.requires_grad_(False)
        x = torch.randn(3, 16)
        def collect(full):
            gradients = {}
            handles = []
            for index in (2, 4):
                def hook(module, inputs, output, index=index):
                    return arm_gradient(output, lambda dy: gradients.__setitem__(index, dy.detach().clone()))
                handles.append(layers[index].register_forward_hook(hook))
            y = layers(x.clone().requires_grad_(full))
            y.square().mean().backward()
            for handle in handles:
                handle.remove()
            return y.detach(), gradients
        full, fg = collect(True)
        short, sg = collect(False)
        self.assertTrue(torch.equal(full, short))
        for index in fg:
            self.assertTrue(torch.equal(fg[index], sg[index]))

    def test_fresh_selection_requires_both_matched_and_raw_improvement(self):
        from run_fine_row_validate import choose_fresh
        def candidate(raw_ce, raw_kl, identity_ce, identity_kl):
            return {'raw256': {'ce': {'upper': raw_ce, 'mean': raw_ce}, 'kl': {'upper': raw_kl}},
                    'identity': {'ce': {'upper': identity_ce}, 'kl': {'upper': identity_kl}}}
        scores = {'good': candidate(-.2, -.1, -.1, -.1),
                  'raw_only': candidate(-1., -.2, .1, -.1),
                  'ce_only': candidate(-2., .1, -1., .1)}
        self.assertEqual(choose_fresh(scores, list(scores)), 'good')
        self.assertEqual(choose_fresh(scores, ['raw_only', 'ce_only']), 'raw256')

    def test_compact_masks_match_raw_election_and_verify_provenance(self):
        from run_task_reorder_eval import coarse_mask, raw256_masks, validate_compact_masks
        from run_c4_frozen import digest_file
        prior = dict(revision='pinned', matrices={'module': {'shape': [272, 128], 'source_sha256': 'weight'}})
        scores = -torch.ones(128, 34 * 2)
        expected = coarse_mask(scores, scores, [272, 128])
        bundle = dict(schema='mixfp4_compact_masks_v1', sequences=128, k=3, revision='pinned',
                      weight_sha256={'module': 'weight'}, raw256={'module': expected},
                      fine8x64={'module': torch.ones(34, 2, dtype=torch.bool)})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.save(bundle, root / 'compact_masks.pt')
            prior['compact_mask_sha256'] = digest_file(root / 'compact_masks.pt')
            actual = raw256_masks(root, prior)
            self.assertTrue(torch.equal(actual['module'], expected))
            bundle['weight_sha256']['module'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'source weights'):
                validate_compact_masks(bundle, prior)
            prior['compact_mask_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'digest'):
                raw256_masks(root, prior)

    def test_threshold_proposals_keep_layout_and_record_new_election(self):
        from run_reorder_map_refine import proposal_layout
        ce = -torch.ones(4, 256, 8)
        ce[0, :, :4] = 2
        source = search_layout(ce, ce.clone(), SearchConfig(rounds=0, starts=1))
        source['row_perm'] = source['row_perm'].flip(0)
        source['row_atom_perm'] = source['row_perm']
        strict = proposal_layout(source, ce, ce.clone(), 3)
        loose = proposal_layout(source, ce, ce.clone(), 0)
        self.assertTrue(torch.equal(strict['row_perm'], source['row_perm']))
        self.assertTrue(torch.equal(strict['mask'], torch.tensor([[False, True]])))
        self.assertTrue(bool(loose['mask'].all()))
        self.assertEqual(loose['config']['k'], 3.)
        self.assertEqual(loose['mask_selection']['proposal_k'], 0)
        self.assertFalse(loose['mask_selection']['outer_election_used'])
        plain = proposal_layout(source, ce, ce.clone(), 0, identity=True)
        self.assertTrue(torch.equal(plain['row_perm'], torch.arange(256)))
        self.assertFalse(torch.equal(source['row_perm'], plain['row_perm']))


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Tests must run on a Slurm worker')
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
