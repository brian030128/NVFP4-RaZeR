"""Calibration record for Llama-3.1-8B-Instruct, derived from the base Llama-3.1-8B record.

The Instruct model shares the base tokenizer, so the same 128 OpenWebMath/CodeParrot
calibration windows (and 192 development documents) apply unchanged; only the weight
hashes and model identity differ. Downloads the weights into HF_HOME as a side effect.
"""
import hashlib
import json
import os
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

BASE = Path('/work/u4320956/task_reorder/transfer_20260920/llama8b/calibration/report.json')
OUT = Path('/work/u4320956/mixfp4_potential/llama8b_ins_calibration/report.json')
SOURCE, REVISION = 'meta-llama/Llama-3.1-8B-Instruct', '0e9e39f249a16976918f6564b8830bc894c89659'


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    assert os.environ.get('SLURM_JOB_ID')
    base = json.loads(BASE.read_text())
    model = AutoModelForCausalLM.from_pretrained(SOURCE, revision=REVISION, torch_dtype=torch.bfloat16)
    modules = {n: m for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    assert list(modules) == list(base['matrices'])
    prior = dict(status='complete', maps_frozen=True, model='llama8b_ins', source=SOURCE, revision=REVISION,
                 derived_from=str(BASE), job_id=os.environ['SLURM_JOB_ID'],
                 transformers_version=transformers.__version__, torch_version=torch.__version__,
                 uses_c4_calibration=False, uses_wiki_calibration=False, fit=base['fit'],
                 matrices={n: dict(shape=list(m.weight.shape), source_sha256=sha(m.weight)) for n, m in modules.items()})
    assert transformers.__version__ == base['transformers_version']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(prior, indent=2) + '\n')
    print(f'PRIOR {OUT} matrices={len(prior["matrices"])}', flush=True)


if __name__ == '__main__':
    main()
