import json
from pathlib import Path
a=json.loads(Path('results/released_reproduction/job_335297/llama-3.1-8b_four_over_six_w4a4/report.json').read_text())
b=json.loads(Path('results/fixed256_paper_eval/job_335407/llama8b_four_over_six/report.json').read_text())
print('weight hash differences', [k for k,v in b['quantized_weight_sha256'].items() if v!=a['quantized_weight_sha256'][k]])
print('first 5 archived losses', [w['nll'] for w in a['windows'][:5]])
print('first 5 current losses', b['evaluation']['wiki']['nll'][:5])
print('archived packages',a['packages'])
print('current versions',b['torch_version'],b['transformers_version'])
