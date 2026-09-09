"""Prepare pinned checkpoints and fixed reproduction cases on a Slurm worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download

RELEASE='e230099bf8006f571614b6c9a71cfd6556c49ef0'
PAPER={
    'llama-3.1-8b': {'bf16':(6.24,8.96), 'nvfp4_w4a16':(6.63,9.48), 'four_over_six_w4a16':(6.60,9.42),
        'razer_w4a16':(6.50,9.29), 'nvfp4_w4a4':(6.95,9.94), 'four_over_six_w4a4':(6.88,9.83), 'razer_w4a4':(6.74,9.63)},
    'qwen3-4b': {'bf16':(13.66,16.65), 'nvfp4_w4a16':(13.83,16.85), 'four_over_six_w4a16':(13.83,16.85),
        'razer_w4a16':(13.83,16.85), 'nvfp4_w4a4':(13.88,17.21), 'four_over_six_w4a4':(13.88,17.21), 'razer_w4a4':(13.82,17.11)}}


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--out',required=True)
    args=ap.parse_args(); source=Path(args.source); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    paths={}; revisions={}
    for nickname,key in [('qwen3-4b','qwen4b'),('llama-3.1-8b','llama8b')]:
        old=json.loads(Path(f'results/math_code_adaptive/calibration_333779_{key}/report.json').read_text())
        path=old['source']
        if not Path(path).is_dir():
            path=snapshot_download(old['source'],revision=old['revision'],
                allow_patterns=['*.json','*.safetensors','*.model','tokenizer.*','merges.txt','vocab.json'])
        paths[nickname]=path; revisions[nickname]=dict(source=old['source'],revision=old['revision'])
    (out/'model_paths.json').write_text(json.dumps(paths,indent=2)+'\n')
    # Populate the shared job-local cache before concurrent readers start.
    load_dataset('wikitext','wikitext-2-raw-v1',split='test')
    load_dataset('allenai/c4',data_files={'validation':'en/c4-validation.00000-of-00008.json.gz'},split='validation')
    cases=[]
    for model in PAPER:
        for policy in ('bf16','nvfp4_w4a4','four_over_six_w4a4','razer_w4a4',
                       'nvfp4_w4a16','four_over_six_w4a16','razer_w4a16'):
            cases.append(dict(id=f'{model}_{policy}',model=model,policy=policy,paper=dict(zip(('wikitext','c4'),PAPER[model][policy]))))
    (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yml','.sh')}
    (out/'manifest.json').write_text(json.dumps(dict(release_commit=RELEASE,model_revisions=revisions,
        source_sha256=hashes,job_id=os.environ['SLURM_JOB_ID'],account=os.environ.get('SLURM_JOB_ACCOUNT'),
        calibration=False,seq_len=2048,seed=0,cases=len(cases)),indent=2)+'\n')
    print(f'PREPARED {len(cases)} cases',flush=True)


if __name__=='__main__': main()
