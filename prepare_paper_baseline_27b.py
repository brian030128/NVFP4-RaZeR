"""Declare the missing Qwen3.8-27B NVFP4 baseline under the paper-aligned protocol.

The four_over_six case is re-run alongside it as a regression check against the
published fixed-256 evaluation (job 335428).
"""
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    old = Path('results/math_code_adaptive/calibration_333787_qwen27b')
    prior = json.loads((old / 'report.json').read_text())
    assert prior['status'] == 'complete' and prior['maps_frozen']
    assert not prior['uses_c4_calibration'] and not prior['uses_wiki_calibration']
    path = prior['source']
    if not Path(path).is_dir():
        path = snapshot_download(path, revision=prior['revision'],
                                 allow_patterns=['*.json', '*.safetensors', '*.model',
                                                 'tokenizer.*', 'merges.txt', 'vocab.json'])
    cases = [dict(id=f'qwen27b_{p}', model='qwen27b', policy=p, calibration=str(old),
                  model_path=path) for p in ('nvfp4', 'four_over_six')]
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
