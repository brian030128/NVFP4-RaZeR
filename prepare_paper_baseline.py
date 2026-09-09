"""Declare the paper-aligned NVFP4 / FourOverSix / MixFP4 panel for Llama and Qwen.

Adds the missing plain-NVFP4 baseline under the corrected full-W4A4 protocol and
re-runs two already-published cases per model as regression checks that the
evaluator change is numerically inert.
"""
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

# fixed256_math_code128 is the largest balanced math+code calibration budget in
# the frozen set; it is named here on budget, not on its measured perplexity.
PRIMARY_MIXFP4 = 'fixed256_math_code128'
REGRESSION = ('four_over_six', PRIMARY_MIXFP4)


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = []
    for model, job in [('llama8b', '333779'), ('qwen4b', '333779')]:
        old = Path(f'results/math_code_adaptive/calibration_{job}_{model}')
        prior = json.loads((old / 'report.json').read_text())
        bundle = json.loads((old / 'maps.json').read_text())
        assert not prior['uses_c4_calibration'] and not prior['uses_wiki_calibration']
        assert PRIMARY_MIXFP4 in bundle['maps']
        path = prior['source']
        if not Path(path).is_dir():
            path = snapshot_download(path, revision=prior['revision'],
                                     allow_patterns=['*.json', '*.safetensors', '*.model',
                                                     'tokenizer.*', 'merges.txt', 'vocab.json'])
        for policy in ('nvfp4', *REGRESSION):
            cases.append(dict(id=f'{model}_{policy}', model=model, policy=policy,
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
