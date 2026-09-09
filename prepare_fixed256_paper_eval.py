"""Pin existing maps/checkpoints and declare fixed-256-only evaluation cases."""
import argparse
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset


def main():
    assert os.environ.get('SLURM_JOB_ID')
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True); cases=[]
    # Start larger models first to overlap their work with the smaller cases.
    for model,job in [('qwen27b','333787'),('llama8b','333779'),('qwen4b','333779')]:
        old=Path(f'results/math_code_adaptive/calibration_{job}_{model}')
        prior=json.loads((old/'report.json').read_text()); bundle=json.loads((old/'maps.json').read_text())
        path=prior['source']
        if not Path(path).is_dir():
            path=snapshot_download(path,revision=prior['revision'],
                allow_patterns=['*.json','*.safetensors','*.model','tokenizer.*','merges.txt','vocab.json'])
        policies=['four_over_six',*[p for p in bundle['maps'] if p.startswith('fixed256_')]]
        assert len(policies)==11
        cases.extend(dict(id=model+'_'+p,model=model,policy=p,calibration=str(old),model_path=path) for p in policies)
    (args.out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    from run_c4_frozen import REVISION
    from run_wiki_frozen import WIKI_REVISION
    load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',revision=WIKI_REVISION,split='test')
    load_dataset('allenai/c4',revision=REVISION,data_files={'validation':'en/c4-validation.00000-of-00008.json.gz'},split='validation')
    print('PREPARED 33 cases',flush=True)


if __name__=='__main__': main()
