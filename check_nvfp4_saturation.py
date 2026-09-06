"""Regression and saved-target-tensor checks; compute node only."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Submit through Slurm.')
import json
from pathlib import Path
import torch
from quantize.quantizer import quant_nvfp4, quant_mix_4_6

torch.set_num_threads(12)
fixture = torch.zeros(16, 64, dtype=torch.bfloat16, device='cuda')
fixture[0, 0] = 0.361328125
fixture[8:, :] = 3.933906555175781e-06
report = {}
for name, w in [('synthetic', fixture), ('qwen38_layer10_qkv', torch.load(
        'results/task_sensitivity_qwen38_probe/mismatch_weight.pt', weights_only=True).cuda())]:
    x = w.float().reshape(-1, 16)
    global_scale = x.abs().amax() / (6*448)
    scaled = x / global_scale
    scale = (scaled.abs().amax(-1, keepdim=True)/6).clamp(min=2**-9, max=448).to(torch.float8_e4m3fn).float()
    normalized = scaled/scale
    exponent = torch.floor(torch.log2(normalized.abs()+(normalized == 0).float())).clamp(min=0)
    mantissa = normalized/(2**exponent)*2
    old_code = torch.sign(mantissa)*torch.floor(mantissa.abs()+.5)*(2**exponent)/2
    illegal = int((old_code.abs() > 6).sum())
    assert illegal > 0
    expected = (old_code.clamp(-6,6)*scale*global_scale).reshape_as(w).bfloat16()
    fixed = quant_nvfp4(w, 4, 16)
    mixed = quant_mix_4_6(w, 4, 16, type_block=(8,64), clip='a1', elect='never')
    assert torch.equal(fixed, expected)
    assert torch.equal(fixed, mixed)
    report[name] = {'previous_illegal_codes': illegal, 'fixed_equals_saturated_reference': True,
                    'fixed_equals_mix_e2m1': True}
Path('results/task_sensitivity_qwen38_probe/saturation_check.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
