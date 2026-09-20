"""Export small frozen layout metadata; no model loading or search."""
import hashlib, json
from pathlib import Path
import torch
torch.set_num_threads(1)
root=Path('/work/u4320956/task_reorder/pilot_20260919/qwen27b/fine_rows_v2/both_exact_compaction/frozen')
out=Path('results/task_reorder/transfer_20260920/permutations');out.mkdir(exist_ok=True)
manifest={}
for p in sorted(root.rglob('layout.pt')):
    v=torch.load(p,map_location='cpu',weights_only=True)
    name=v['name'].split('.')[-1]
    n,k=v['weight_shape']
    assert torch.equal(v['row_perm'].sort().values,torch.arange(n))
    assert torch.equal(v['col_perm'].sort().values,torch.arange(k))
    for axis in ('row','col'):
        values=list(map(str,v[f'{axis}_perm'].tolist()))
        (out/f'{name}_{axis}.txt').write_text('\n'.join(' '.join(values[i:i+16]) for i in range(0,len(values),16))+'\n')
    manifest[name]=dict(source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),n=n,k=k,tiles=int(v['mask'].sum()),rows_moved=int((v['row_perm']!=torch.arange(n)).sum()),columns_moved=int((v['col_perm']!=torch.arange(k)).sum()))
assert set(manifest)=={'gate_proj','up_proj','down_proj'},manifest
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
