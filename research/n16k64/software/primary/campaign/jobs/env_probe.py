"""Minimal launched job: record the container's view of devices, run one tiny CUDA op per GPU."""
import json, os, time
from campaign import runtime
import torch
out = {'visible': [], 'env': runtime.environment()}
runtime.phase('cuda_probe')
for i in range(torch.cuda.device_count()):
    x = torch.randn(1024, 1024, device=f'cuda:{i}')
    out['visible'].append(dict(index=i, name=torch.cuda.get_device_name(i), uuid=str(torch.cuda.get_device_properties(i).uuid),
                               checksum=float((x @ x).sum())))
time.sleep(float(os.environ.get('PROBE_SLEEP', '5')))
runtime.atomic_json(runtime.out_dir() / 'probe.json', out)
print(json.dumps(out['visible']))
