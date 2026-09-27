"""Why the non-deterministic runs' setup phases are slower (REPORT.md): GPU -> host copies into fresh pageable memory.

Under torch.use_deterministic_algorithms(True), PyTorch fills newly allocated memory
(torch.utils.deterministic.fill_uninitialized_memory), which touches the host pages before the copy runs. This times
.cpu() of 8.4 GB of bf16 teacher-like log-probabilities (8 chunks) with determinism off, on, and on without the fill:

    python results/nodev_cost/d2h_determinism.py off|on|on-nofill
"""
import sys
import time

import torch

mode = sys.argv[1]
if mode.startswith('on'):
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = mode == 'on'
x = torch.randn(64, 511, 128256, device='cuda').bfloat16()
torch.cuda.synchronize()
t = time.time()
out = [x[i:i + 8].cpu() for i in range(0, 64, 8)]
torch.cuda.synchronize()
dt = time.time() - t
print(f'{mode}: fill_uninitialized_memory={torch.utils.deterministic.fill_uninitialized_memory and mode.startswith("on")} '
      f'{dt:.2f} s for {x.numel() * 2 / 1e9:.1f} GB ({x.numel() * 2 / 1e9 / dt:.2f} GB/s)')
