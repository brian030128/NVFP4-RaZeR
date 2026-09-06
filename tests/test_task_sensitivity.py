"""Small structural checks; run through slurm/task_sensitivity.sbatch."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Tests must run in a Slurm allocation.')
import torch
from analyze_task_sensitivity import IdentityBackward, tile_sum, expand_mask, paired, score_tiles, loss, trust_masks, matched_random, apply_type_map

torch.manual_seed(71)
x = torch.randn(16, 128, requires_grad=True)
q = torch.round(x.detach())
y = IdentityBackward.apply(x, q)
assert torch.equal(y, q)
y.square().sum().backward()
assert torch.equal(x.grad, 2*q)

# The tile gradient predicts the directional derivative of a smooth linear loss.
w = torch.randn(16, 128, dtype=torch.float64, requires_grad=True)
d = torch.randn_like(w)
a = torch.randn(3, 128, dtype=torch.float64)
target = torch.randn(3, 16, dtype=torch.float64)
objective = lambda v: (a @ v.T - target).square().mean()
g, = torch.autograd.grad(objective(w), w)
mask = torch.tensor([[True, False], [False, True]])
step = d * expand_mask(mask)
pred = tile_sum(g*d)[mask].sum()
eps = 1e-5
actual = (objective(w+eps*step)-objective(w-eps*step))/(2*eps)
assert torch.allclose(pred, actual, atol=1e-7, rtol=1e-7)
# Wrong tile layout must be distinguishable, rather than testing only a total sum.
wrong = (g*d).reshape(2, 2, 8, 64).sum((2, 3))[mask].sum()
assert not torch.allclose(wrong, actual, atol=1e-4)
assert paired([1., 2.], [1., 2.])['mean'] == 0

# Exercise the actual streaming hooks against an ordinary autograd weight gradient.
from types import SimpleNamespace
class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 64)
        self.proj = torch.nn.Linear(64, 64, bias=False)
        self.head = torch.nn.Linear(64, 16, bias=False)
        self.device = torch.device('cpu')
    def get_input_embeddings(self):
        return self.embed
    def forward(self, ids, use_cache=False):
        return SimpleNamespace(logits=self.head(self.proj(self.embed(ids)).tanh()))

toy = Toy().eval()
bs = [torch.randint(0, 16, (1, 8)) for _ in range(3)]
base = {'proj': toy.proj.weight.detach().clone()}
alt = {'proj': base['proj'] + torch.randn_like(base['proj'])*0.02}
expected = []
for ids in bs:
    grad, = torch.autograd.grad(loss(toy, ids), toy.proj.weight)
    expected.append(tile_sum(grad*(alt['proj']-base['proj'])))
scores, ses, _ = score_tiles(toy, {'proj': toy.proj}, base, alt, bs)
expected = torch.stack(expected)
assert torch.allclose(scores['proj'], expected.mean(0), atol=1e-7, rtol=1e-5)
assert torch.allclose(ses['proj'], expected.std(0)/(3**0.5), atol=1e-7, rtol=1e-5)

means = {'a': torch.tensor([[-.06, -.03, -.02, -.05, .03]])}
ses = {'a': torch.tensor([[.001, .001, .001, .04, .001]])}
selected = trust_masks(means, ses)
assert torch.equal(selected['a'], torch.tensor([[True, True, False, False, False]]))
assert -float(means['a'][selected['a']].sum()) <= .1
random_mask = matched_random(selected, 1)
assert int(random_mask['a'].sum()) == 2

# An exported sparse type map reconstructs the intended legal grid per tile.
from quantize.quantizer import quant_nvfp4, quant_mix_4_6
# E4M3 subnormal scales must not let NVFP4 emit the nonexistent E2M1 code 8.
subnormal_fixture = torch.zeros(16, 64, dtype=torch.bfloat16)
subnormal_fixture[0, 0] = 0.361328125
subnormal_fixture[8:, :] = 3.933906555175781e-06
assert torch.equal(quant_nvfp4(subnormal_fixture, 4, 16),
                   quant_mix_4_6(subnormal_fixture, 4, 16, type_block=(8,64), clip='a1', elect='never'))
toy = Toy().eval()
toy.embed = torch.nn.Embedding(16, 128)
toy.proj = torch.nn.Linear(128, 64, bias=False)
pristine = toy.proj.weight.detach().clone()
head_before = toy.head.weight.detach().clone()
q2 = quant_nvfp4(pristine, 4, 16)
q3 = quant_mix_4_6(pristine, 4, 16, type_block=(8,64), clip='a1', elect='always')
spec = {'weight_type_block': [8,64], 'scale_block': 16, 'alpha': 1., 'default': 'E2M1',
        'modules': {'proj': {'tile_grid_shape': [8,2], 'e0m3_flat_indices': [5]}}}
apply_type_map(toy, spec)
expected = q2.clone()
expected[16:24, 64:128] = q3[16:24, 64:128]
assert torch.equal(toy.proj.weight, expected.float())
assert torch.equal(toy.head.weight, head_before)

# FourOverSix maps must preserve the stronger default, not silently fall
# back to the alpha=1 NVFP4 baseline when replayed.
from quantize.quantizer import quant_nvfp4_4over6
with torch.no_grad():
    toy.proj.weight.copy_(pristine)
four_spec = {k: v for k, v in spec.items() if k != 'alpha'}
four_spec.update(baseline_weight_dtype='nvfp4_4over6', e2m1_alphas=[1., 1.5], e0m3_alpha=1.)
apply_type_map(toy, four_spec)
expected = quant_nvfp4_4over6(pristine, 4, 16)
expected[16:24, 64:128] = q3[16:24, 64:128]
assert torch.equal(toy.proj.weight, expected.float())
assert torch.equal(toy.head.weight, head_before)

# Exercise the calibration-only CLI, including its independent validation
# forecast, without downloading a model or writing research results.
import json
import sys
import tempfile
from unittest.mock import patch
import analyze_task_sensitivity as task
toy = Toy().eval()
toy.config = SimpleNamespace(use_cache=False)
with tempfile.TemporaryDirectory(dir=os.environ['HF_HOME']) as out:
    argv = ['analyze_task_sensitivity.py', '--model', 'toy', '--out', out,
            '--fit', '3', '--val', '3', '--seq-len', '8', '--calibrate-only', '--validate-selected']
    with patch.object(sys, 'argv', argv), \
         patch.object(task, 'load_model_and_tokenizer', return_value=(toy, None)), \
         patch.object(task, 'data_splits', return_value=(bs, bs, bs[:2], {'fit':'test', 'val':'test', 'probe':'test'})):
        task.main()
    report = json.load(open(os.path.join(out, 'report.json')))
    assert report['calibration_only'] and report['complete']
    assert report['validation_forecast']['n'] == 3
    assert 'final' not in report and report['results'] == {}
    assert os.path.isfile(os.path.join(out, 'type_map.json'))
    if not report['validation_forecast']['improvement_supported']:
        assert report['export_decision'] == 'fallback_nvfp4'
        assert report['selection']['tiles'] == 0
        assert json.load(open(os.path.join(out, 'type_map.json')))['modules'] == {}

# Increasing calibration preserves the original fit prefix and independent
# validation/probe text, rather than silently changing all three sets.
import random
ids = torch.arange(3*400*8).reshape(1, -1)
tokenizer = lambda _text, return_tensors: SimpleNamespace(input_ids=ids)
with patch('datasets.load_dataset', return_value={'text': ['synthetic unit fixture']}):
    f16, v16, p16, h16 = task.data_splits(tokenizer, 8, 16, 16, 71)
    f64, v64, p64, h64 = task.data_splits(tokenizer, 8, 64, 16, 71)
legacy_rng = random.Random(71)
windows = list(ids.split(8, 1))
legacy_fit = legacy_rng.sample(windows[:400], 16)
legacy_val = legacy_rng.sample(windows[400:800], 16)
legacy_probe = legacy_rng.sample(windows[800:], 8)
for got, expected in [(f16, legacy_fit), (v16, legacy_val), (p16, legacy_probe), (f64[:16], f16)]:
    assert torch.equal(torch.cat(got, 1), torch.cat(expected, 1))
assert h16['val'] == h64['val'] and h16['probe'] == h64['probe']
assert len({int(x[0,0]) for x in f64}) == 64
# Discrete trust calibration rejects a harmful step, accepts a supported
# smaller step, and falls back when none earn their predicted improvement.
fake_means = {'a': torch.tensor([[-0.02, -0.02, -0.02, -0.02]])}
fake_ses = {'a': torch.zeros(1, 4)}
with patch.object(task, 'install'), patch.object(task, 'evaluate', side_effect=[[1.1]*3, [.98]*3]):
    mask, history = task.calibrated_trust_masks(None, {}, {}, {}, fake_means, fake_ses, [], [1.]*3)
assert len(history) == 2 and not history[0]['accepted'] and history[1]['accepted']
assert int(mask['a'].sum()) == 2
with patch.object(task, 'install'), patch.object(task, 'evaluate', return_value=[1.1]*3):
    mask, history = task.calibrated_trust_masks(None, {}, {}, {}, fake_means, fake_ses, [], [1.]*3)
assert not mask['a'].any() and not any(h['accepted'] for h in history)
print('task sensitivity structural checks passed', flush=True)
