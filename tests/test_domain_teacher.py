"""Check soft-target gradient, token normalization and causally shifted support."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Tests run in Slurm.')
import torch
from domain_teacher_loss import teacher_kl

torch.manual_seed(17)
x = torch.randn(2, 5, 7, requires_grad=True)
t = torch.randn_like(x, requires_grad=True)
loss = teacher_kl(x, t)
loss.backward()
expected = (x.detach()[:, :-1].softmax(-1)-t.detach()[:, :-1].softmax(-1))/8
torch.testing.assert_close(x.grad[:, :-1], expected, atol=1e-7, rtol=1e-5)
assert torch.count_nonzero(x.grad[:, -1]) == 0 and t.grad is None
torch.testing.assert_close(teacher_kl(x.repeat(2, 1, 1), t.repeat(2, 1, 1)), loss)
assert abs(teacher_kl(x, x.detach()).item()) < 1e-7
changed = t.detach().clone()
changed[:, -1] += torch.randn_like(changed[:, -1])*100
torch.testing.assert_close(teacher_kl(x, changed), loss)
print('PASS: teacher KL gradient, teacher stop-gradient, token normalization and final-position exclusion', flush=True)
