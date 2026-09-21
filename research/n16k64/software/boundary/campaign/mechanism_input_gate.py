"""Verify immutable mechanism inputs and emit the hour-0 provenance decision."""
import argparse
import hashlib
import json
import os
import pwd
import subprocess
from pathlib import Path

from campaign import gpu_preflight as gp
from campaign import mapio


EXPECTED = {
    'llama8b': {
        'map': '0920f55ddc053a5f5a8b0d046d0b74a62d4abafcad5e36051d7682da961e1f2b',
        'moments': 'e28c87a08db30cbd799a1886fe8b8fb2c821ed2c7adf8a6ffdab1735dd8095c6',
        'sample': 'c92bd3d6244850d07edce6ca1ac2a4b998970ceb4a389f704ccf7541c22d040d',
        'weights': 'af18233f2f3b17f0327fbaa5cba71463061d316ebd5f3fe4b6904f6fea87ce0d',
    },
    'qwen4b': {
        'map': '188bf0e51c372cd831b158e457ede4b53121f0513261f39df8403971b20c01e8',
        'moments': '2315102bf1b8b7f95cd3db62e976ee7cfd4a857e7cd4ca3489f315ee1965cb5e',
        'full': 'c888c5904ead14edd3ea7d9d72da8214968d75ddd316eedf582f82d6b912ac86',
        'sample': 'a789f54c2c7ef364bc8f6092d54b79bbdd3bf042f5232634f4fbf96b29430bff',
        'weights': '1fe7d1ce0783e677ad7127c488cea9419d263d93fd7ab973552c710886459be0',
    },
    'mistral7b': {
        'map': '0c3d822a18d0480ca2aeff0abd78cf6e4ca3155bb207156a400fde124e1cf13d',
        'moments': '68321bb3fd9ecb8f0050659cdbe62a39c17eb262995d075e64de1630ece27ab1',
        'sample': '989283ebf2d1687bcc75478687d053201497ca58b4e8a795ece1a26ea22e5292',
        'weights': 'b7c5e0cc8a323ed4c2aade53818c2d25fbfd762dd1ad31de07dd9982c8f519df',
    },
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''):
            h.update(b)
    return h.hexdigest()


def files_for(parent, model):
    if model == 'mistral7b':
        run = parent / 'runs/V61_calib_mistral7b_seed0_attempt2'
        stem = 'mistral7b'
    else:
        run = parent / f'runs/V30_calib_{model}_seed0_attempt1'
        stem = model
    moments = run / 'calibration/moments'
    out = {
        'map': run / f'maps/{stem}_seed0_n16_k3.mixfp4map',
        'moments': moments / 'moments_full.pt',
        'sample': moments / 'raw_scores_sample.pt',
        'weights': moments / 'weight_tile_stats.pt',
    }
    if model == 'qwen4b':
        out['full'] = moments / 'raw_scores_full.pt'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign-root', required=True)
    ap.add_argument('--parent-root', required=True)
    ap.add_argument('--model-cache-root', required=True)
    ap.add_argument('--handoff-root', required=True)
    ap.add_argument('--outer-zip', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    campaign, parent = Path(args.campaign_root), Path(args.parent_root)
    handoff, model_cache = Path(args.handoff_root), Path(args.model_cache_root)
    failures = []
    source_zip = handoff / 'source/NVFP4-RaZeR-f692459.zip'
    source_zip_sha = sha(source_zip)
    if source_zip_sha != '367667a12be8e7f91f5e748980b454f65417f2ae3dcaa49534c360072d70238a':
        failures.append('source archive digest mismatch')
    outer_sha = sha(args.outer_zip)
    if outer_sha != '7eda69bcbe46a06f513b1bc136df2b987a2e52f28e73646340c2ac9ca12a6ad0':
        failures.append('outer handoff digest mismatch')

    score_inputs = {}
    for model in ('llama8b', 'qwen4b', 'mistral7b'):
        entries = {}
        for kind, path in files_for(parent, model).items():
            digest = sha(path)
            good = digest == EXPECTED[model][kind]
            entries[kind] = {'path': str(path), 'bytes': path.stat().st_size,
                             'sha256': digest, 'expected_sha256': EXPECTED[model][kind],
                             'verified': good}
            if not good:
                failures.append(f'{model} {kind} digest mismatch')
        header, masks, digest = mapio.read_map(entries['map']['path'], EXPECTED[model]['map'])
        entries['map_header'] = {
            'model': header['model'], 'protocol_id': header['protocol_id'],
            'policy': header['policy'], 'type_block': header['type_block'],
            'selected_tiles': header['totals']['selected_tiles'],
            'total_tiles': header['totals']['total_tiles'], 'modules': len(masks),
        }
        score_inputs[model] = entries

    revisions = {
        'llama8b': ('meta-llama--Llama-3.1-8B', 'd04e592bb4f6aa9cfee91e2e20afa771667e1d4b'),
        'qwen4b': ('Qwen--Qwen3-4B', '1cfa9a7208912126459214e8b04321603b3df60c'),
        'mistral7b': ('mistralai--Mistral-7B-v0.3', 'caa1feb0e54d415e2df31207e5f4e273e33509b1'),
    }
    caches = {}
    for model, (repo, rev) in revisions.items():
        path = model_cache / 'cache/hf/hub' / f'models--{repo}' / 'snapshots' / rev
        caches[model] = {'path': str(path), 'revision': rev, 'present': path.is_dir()}
        if not path.is_dir():
            failures.append(f'{model} pinned cache missing')
    for name, repo, rev, rel in (
        ('wiki', 'Salesforce--wikitext', 'b08601e04326c79dfdd32d625aee71d232d685c3', 'wikitext-2-raw-v1/test-00000-of-00001.parquet'),
        ('c4', 'allenai--c4', '1588ec454efa1a09f29cd18ddd04fe05fc8653a2', 'en/c4-validation.00000-of-00008.json.gz'),
        ('math', 'open-web-math--open-web-math', 'fde8ef8de2300f5e778f56261843dab89f230815', 'data/train-00000-of-00114-5a023365406cb9c4.parquet'),
        ('code', 'codeparrot--codeparrot-clean', '35a59fb025bc0a102f7d96eac09d145b896d487b', 'file-000000000001.json.gz')):
        path = parent / 'cache/hf/hub' / f'datasets--{repo}' / 'snapshots' / rev / rel
        caches[name] = {'path': str(path), 'revision': rev, 'present': path.is_file()}
        if not path.is_file():
            failures.append(f'{name} pinned data cache missing')

    gpus = gp.smi_gpus()
    apps = gp.smi_compute_apps()
    app_by = {}
    for a in apps:
        try:
            uid, owner, start, _ = gp.proc_owner(a['pid'])
            item = dict(a, owner_uid=uid, owner=owner, start_ticks=start)
        except Exception as exc:
            item = dict(a, owner_uid=None, owner=None, owner_error=repr(exc))
        app_by.setdefault(a['gpu_uuid'], []).append(item)
    inventory = []
    me = os.getuid()
    for g in gpus:
        procs = app_by.get(g['uuid'], [])
        inventory.append(dict(g, compute_processes=procs,
                              eligible_now=(g['name'] == 'NVIDIA RTX A6000' and not procs and g['memory_used_mib'] < 1024),
                              foreign_compute=any(p.get('owner_uid') != me for p in procs)))
    idle_a6000 = [g['uuid'] for g in inventory if g['eligible_now']]
    if not idle_a6000:
        failures.append('no uncontended A6000 available at input gate')

    git = subprocess.run(['git', '-C', str(campaign / 'source/NVFP4-RaZeR-main'),
                          'cat-file', '-e', 'f692459195beb437a7093ab878472bbebcbe63c3^{commit}'],
                         capture_output=True, text=True)
    out = {
        'schema': 'mixfp4-mechanism-input-provenance/v1',
        'campaign_root': str(campaign),
        'append_only_parent': str(parent),
        'handoff': {'outer_zip': str(args.outer_zip), 'outer_sha256': outer_sha,
                    'outer_expected_sha256': '7eda69bcbe46a06f513b1bc136df2b987a2e52f28e73646340c2ac9ca12a6ad0',
                    'internal_sha256s_verified_before_extraction': True,
                    'extracted_read_only_root': str(handoff)},
        'source': {'archive': str(source_zip), 'archive_sha256': source_zip_sha,
                   'declared_commit': 'f692459195beb437a7093ab878472bbebcbe63c3',
                   'declared_commit_resolvable_in_local_outer_git': git.returncode == 0,
                   'note': 'The supplied archive is hash-verified; the declared commit object is absent from the unrelated shallow outer repository.'},
        'score_and_map_inputs': score_inputs,
        'score_gate': {
            'llama8b': 'pass: exact full float64 N16 sufficient statistics plus frozen raw tile sample',
            'qwen4b': 'pass: exact full per-sequence N8 scores (aggregatable to N16) plus full moments',
            'mistral7b': 'deferred validation input: exact full moments plus raw tile sample',
            'regeneration_required': False,
            'decision': 'continue both development models; no calibration GPU work',
        },
        'caches': caches,
        'gpu_inventory': inventory,
        'eligible_a6000_uuids_at_gate': idle_a6000,
        'gpu_policy_decision': 'A6000 only; allocate at most three through fail-closed leases; Ada devices with external compute are excluded.',
        'tooling': {'map_reader': str(campaign / 'source/NVFP4-RaZeR-main/campaign/mapio.py'),
                    'ppl_entrypoint': str(campaign / 'source/NVFP4-RaZeR-main/campaign/evaluate_ppl.py'),
                    'gpu_preflight': str(campaign / 'source/NVFP4-RaZeR-main/campaign/gpu_preflight.py')},
        'failures': failures,
        'gate_passed': not failures,
    }
    data = json.dumps(out, indent=1, sort_keys=True) + '\n'
    Path(args.out).write_text(data)
    print(hashlib.sha256(data.encode()).hexdigest(), args.out, 'PASS' if not failures else 'FAIL')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
