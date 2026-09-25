import sys
from pathlib import Path

import pytest
import torch

SM120 = Path(__file__).resolve().parents[1]
REPO = SM120.parent
for p in (str(SM120), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    config.addinivalue_line('markers', 'gpu: needs an SM120 GPU and a built kernel library')


@pytest.fixture(scope='session')
def device():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip('needs an SM120 GPU')
    return torch.device('cuda')


def weight_cases(device, seed=0):
    """Named weight tensors covering the numeric corner cases the quantizers must agree on."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    cases = {}
    cases['normal'] = torch.randn(256, 1024, generator=g) * 0.02
    t = torch.distributions.StudentT(torch.tensor(3.0)).sample((128, 512)) * 0.01
    cases['student_t'] = t
    w = torch.randn(192, 768, generator=g) * 0.02
    w[:, torch.randperm(768, generator=g)[:6]] *= 40.0            # outlier input channels
    cases['outlier_cols'] = w
    w = torch.randn(64, 256, generator=g) * 0.02
    w[3] = 0.0                                                     # an all-zero row
    w[5, 16:48] = 0.0                                              # all-zero scale blocks
    w[7, :16] = 1e-6                                               # tiny block -> subnormal E4M3 scale
    cases['zeros_and_tiny'] = w
    w = torch.randn(48, 2560, generator=g) * 0.05                  # N=48 rows (Qwen3.8 linear-attn style)
    cases['n48'] = w
    w = torch.randn(32, 128, generator=g)
    w[0, 0] = 3.0e4                                                # one huge value: most scales subnormal
    cases['dynamic_range'] = w
    return {k: v.to(device=device, dtype=torch.bfloat16) for k, v in cases.items()}
