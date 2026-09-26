"""Calibration record for a Llama Instruct model, derived from the base Llama-3.1-8B record.

Llama-3.1-8B-Instruct and Llama-3.2-1B/3B-Instruct share the base Llama-3 tokenizer, so the
same 128 OpenWebMath/CodeParrot calibration windows (and 192 development documents)
apply unchanged; math_code_data() re-checks every window's token hash on use. Only the
weight hashes and model identity differ. Downloads the weights into HF_HOME as a side effect.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

BASE = Path('/work/u4320956/task_reorder/transfer_20260920/llama8b/calibration/report.json')
MODELS = {
    'llama8b_ins': dict(source='meta-llama/Llama-3.1-8B-Instruct', revision='0e9e39f249a16976918f6564b8830bc894c89659',
                        out=Path('/work/u4320956/mixfp4_potential/llama8b_ins_calibration/report.json')),
    'llama1b_ins': dict(source='meta-llama/Llama-3.2-1B-Instruct', revision='9213176726f574b556790deb65791e0c5aa438b6',
                        out=Path('/work/u4320956/mixfp4_potential/llama1b_ins_calibration/report.json')),
    'llama3b_ins': dict(source='meta-llama/Llama-3.2-3B-Instruct', revision='0cb88a4f764b7a12671c53f0838cd831a0843b95',
                        out=Path('/work/u4320956/mixfp4_potential/llama3b_ins_calibration/report.json')),
}


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    assert os.environ.get('SLURM_JOB_ID')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=tuple(MODELS), default='llama8b_ins')
    args = ap.parse_args()
    spec = MODELS[args.model]
    base = json.loads(BASE.read_text())
    model = AutoModelForCausalLM.from_pretrained(spec['source'], revision=spec['revision'], torch_dtype=torch.bfloat16)
    modules = {n: m for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and m is not model.get_output_embeddings()}
    if args.model == 'llama8b_ins':
        # Same architecture as the base record; the 1B model has its own layer list.
        assert list(modules) == list(base['matrices'])
    prior = dict(status='complete', maps_frozen=True, model=args.model, source=spec['source'], revision=spec['revision'],
                 derived_from=str(BASE), job_id=os.environ['SLURM_JOB_ID'],
                 transformers_version=transformers.__version__, torch_version=torch.__version__,
                 uses_c4_calibration=False, uses_wiki_calibration=False, fit=base['fit'],
                 matrices={n: dict(shape=list(m.weight.shape), source_sha256=sha(m.weight)) for n, m in modules.items()})
    assert transformers.__version__ == base['transformers_version']
    spec['out'].parent.mkdir(parents=True, exist_ok=True)
    spec['out'].write_text(json.dumps(prior, indent=2) + '\n')
    print(f'PRIOR {spec["out"]} matrices={len(prior["matrices"])}', flush=True)


if __name__ == '__main__':
    main()
