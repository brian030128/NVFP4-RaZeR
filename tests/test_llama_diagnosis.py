import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_llama_taskfit_diagnosis import shifted_weight, summarize
from quantize.quantizer import quant_nvfp4_4over6


class DiagnosisTests(unittest.TestCase):
    def test_frozen_residual_baseline_and_derivative(self):
        x0 = torch.randn(4, 64, device='cuda', dtype=torch.bfloat16)
        q0 = quant_nvfp4_4over6(x0, 4, 16)
        x = x0.clone().requires_grad_(True)
        result = q0 + (x - x0)
        self.assertTrue(torch.equal(result, q0))
        result.float().sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_shift_uses_effective_quantized_difference(self):
        w = torch.ones(2, 16, dtype=torch.bfloat16)
        raw, candidate = w * .75, w * .875
        self.assertTrue(torch.equal(shifted_weight(w, raw, candidate), w * 1.125))
        self.assertTrue(torch.equal(shifted_weight(w, raw, candidate, -1), candidate))

    def test_interaction_subtracts_raw_once_per_isolated_delta(self):
        rows = [dict(split='fresh', source='general') for _ in range(2)]
        values = {'quant_raw': 10., 'quant_gate': 9., 'quant_up': 8., 'quant_down': 7.,
                  'quant_both': 4., 'bf16': 3., 'bf16_plus_both': 3., 'bf16_minus_both': 3.}
        metrics = {p: {k: [v, v] for k in ('ce', 'kl')} for p, v in values.items()}
        result = summarize(metrics, rows)
        self.assertEqual(result['joint_minus_sum_isolated']['fresh_general']['ce']['mean'], 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
