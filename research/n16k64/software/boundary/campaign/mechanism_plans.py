"""Materialize and hash-freeze evaluation plans from the verified map manifest."""
import argparse
import hashlib
import json
from pathlib import Path


DEV = [
    'four_over_six', 'full', 'group_only_strongest', 'group_only_weakest', 'group_only_random',
    'full_minus_strongest', 'full_minus_weakest', 'full_minus_random',
    'full_plus_kl_vetoed_ce_approved', 'full_plus_kl_vetoed_matched_random',
    'full_plus_ce_vetoed_kl_approved', 'full_plus_ce_vetoed_matched_random',
    'attention_only', 'mlp_only', 'matched_random_attention', 'matched_random_mlp', 'matched_random_union']
MISTRAL = [
    'four_over_six', 'full', 'group_only_strongest', 'group_only_weakest',
    'full_minus_strongest', 'full_minus_weakest',
    'full_plus_kl_vetoed_ce_approved', 'full_plus_kl_vetoed_matched_random',
    'attention_only', 'mlp_only', 'matched_random_attention', 'matched_random_mlp', 'matched_random_union']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); entries = json.loads(Path(args.manifest).read_text()); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    by = {(e['model'], e['policy']): e for e in entries}; records = []
    for model in ('llama8b', 'qwen4b', 'mistral7b'):
        names = MISTRAL if model == 'mistral7b' else DEV
        plan = []
        for name in names:
            if name == 'four_over_six': plan.append({'name': name, 'kind': name}); continue
            e = by[(model, name)]
            plan.append({'name': name, 'kind': 'map', 'map_path': e['path'], 'map_sha256': e['sha256'],
                         'map_policy': name, 'type_block': e['type_block'], 'expected_total_tiles': e['total_tiles'],
                         'protocol_id': 'aligned-analysis'})
        path = out / f'{model}_mechanism_full.json'; path.write_text(json.dumps(plan, indent=1, sort_keys=True)+'\n')
        records.append({'model': model, 'role': 'mistral_one_shot' if model == 'mistral7b' else 'development_full',
                        'path': str(path), 'sha256': sha(path), 'policies': len(plan), 'names': names})
    manifest = out / 'PLAN_MANIFEST.json'; manifest.write_text(json.dumps({'schema':'mechanism-plan-manifest/v1','plans':records},indent=1,sort_keys=True)+'\n')
    print(sha(manifest), manifest)


if __name__ == '__main__': main()
