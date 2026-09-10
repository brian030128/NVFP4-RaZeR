"""Declare BF16 reference cases under the paper-aligned 2048-token protocol.

Qwen3.8-27B has no BF16 row in the released reproduction, because the released
evaluator predates its architecture. Llama-3.1-8B and Qwen3-4B do, and they are
run here as validation: an unquantized pass touches no quantizer at all, so if
this path is faithful it must reproduce those released values exactly.
"""
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

# 27B first so its longer run overlaps the shorter ones if steps are parallelised.
TARGETS = (('qwen27b', '333787'), ('llama8b', '333779'), ('qwen4b', '333779'))


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = []
    for model, job in TARGETS:
        old = Path(f'results/math_code_adaptive/calibration_{job}_{model}')
        prior = json.loads((old / 'report.json').read_text())
        assert prior['status'] == 'complete' and prior['maps_frozen']
        path = prior['source']
        if not Path(path).is_dir():
            path = snapshot_download(path, revision=prior['revision'],
                                     allow_patterns=['*.json', '*.safetensors', '*.model',
                                                     'tokenizer.*', 'merges.txt', 'vocab.json'])
        cases.append(dict(id=f'{model}_bf16', model=model, policy='bf16',
                          calibration=str(old), model_path=path))
    (args.out / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')

    from run_c4_frozen import REVISION
    from run_wiki_frozen import WIKI_REVISION
    load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', revision=WIKI_REVISION, split='test')
    load_dataset('allenai/c4', revision=REVISION,
                 data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                 split='validation')
    print(f'PREPARED {len(cases)} cases', flush=True)


if __name__ == '__main__':
    main()
