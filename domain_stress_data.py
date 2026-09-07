"""Fixed math/code language-modeling examples, shared by experiment runners."""
import hashlib
import random
from datasets import load_dataset
from huggingface_hub import HfApi


def load_stress(tok, reference=None):
    data, metadata = {}, {}
    for name, repo, config in [('math', 'openai/gsm8k', 'main'), ('code', 'google-research-datasets/mbpp', 'full')]:
        revision = reference[name]['revision'] if reference else HfApi().dataset_info(repo).sha
        ds = load_dataset(repo, config, revision=revision, split='test')
        indices = reference[name]['row_indices'] if reference else random.Random(20260917).sample(range(len(ds)), 128)
        batches = []
        for i in indices:
            row = ds[i]
            text = ('Question: '+row['question']+'\nAnswer: '+row['answer'] if name == 'math'
                    else 'Problem: '+row['text']+'\nCode:\n'+row['code'])
            ids = tok(text, return_tensors='pt').input_ids[:, :2048]
            assert ids.numel() >= 2
            batches.append(ids)
        data[name] = batches
        metadata[name] = {'repo': repo, 'config': config, 'revision': revision,
                          'row_indices': indices, 'lengths': [b.numel() for b in batches],
                          'token_sha256': [hashlib.sha256(b.numpy().tobytes()).hexdigest() for b in batches]}
    return data, metadata
