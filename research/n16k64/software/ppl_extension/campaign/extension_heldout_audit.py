"""Hash and compatibility audit for the prospectively reserved held-out panel (CPU only)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from campaign import models
from campaign import runtime


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def audit_one(reserved: dict) -> dict:
    key, repo, revision = reserved['key'], reserved['repository'], reserved['revision']
    spec = models.REGISTRY[key]
    if spec['model_id'] != repo or spec['revision'] != revision:
        raise RuntimeError(f'{key}: source registry differs from frozen reservation')
    snapshot = models.snapshot_path(key)
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f'{key}: hub resolved {info.sha} != frozen {revision}')
    remote = {item.rfilename: item for item in info.siblings}
    files = []
    for path in sorted(p for p in snapshot.iterdir() if p.is_file()):
        item = remote.get(path.name)
        digest = sha256_file(path)
        lfs_sha = getattr(getattr(item, 'lfs', None), 'sha256', None)
        row = {'path': path.name, 'size': path.stat().st_size, 'sha256': digest,
               'hub_size': getattr(item, 'size', None), 'hub_lfs_sha256': lfs_sha,
               'hub_lfs_match': (digest == lfs_sha if lfs_sha else None)}
        if row['hub_size'] is not None and row['size'] != row['hub_size']:
            raise RuntimeError(f'{key}/{path.name}: size mismatch')
        if lfs_sha and digest != lfs_sha:
            raise RuntimeError(f'{key}/{path.name}: LFS SHA-256 mismatch')
        files.append(row)
        print(f'HELDOUT HASH {key}/{path.name} {path.stat().st_size / 2**30:.2f} GiB', flush=True)
    config_path = snapshot / 'config.json'
    config_sha = sha256_file(config_path)
    if config_sha != reserved['config_sha256_from_hub_revision']:
        raise RuntimeError(f'{key}: config hash mismatch')
    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    tokenizer_sha, tokenizer_files = models.tokenizer_manifest(key)
    with init_empty_weights():
        empty = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    empty_model_class = type(empty).__name__
    tokenizer_class = type(tokenizer).__name__
    modules = models.scope(empty, key)
    shapes = [{'name': name, 'shape': list(module.weight.shape), 'bias': module.bias is not None}
              for name, module in modules.items()]
    total8, bad8 = models.tile_totals(modules, (8, 64))
    total16, bad16 = models.tile_totals(modules, (16, 64))
    del tokenizer, empty
    if bad8 or bad16:
        raise RuntimeError(f'{key}: nondivisible scoped modules: n8={bad8[:3]} n16={bad16[:3]}')
    index_path = snapshot / 'model.safetensors.index.json'
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        required_shards = sorted(set(index['weight_map'].values()))
    else:
        required_shards = sorted(p.name for p in snapshot.glob('*.safetensors'))
    absent_shards = [name for name in required_shards if not (snapshot / name).is_file()]
    if absent_shards:
        raise RuntimeError(f'{key}: absent weight shards {absent_shards}')
    architecture = (config.architectures or [empty_model_class])[0]
    return {
        'key': key, 'family': reserved['family'], 'repository': repo, 'revision': revision,
        'tokenizer_revision': reserved['tokenizer_revision'], 'resolved_hub_revision': info.sha,
        'hub_card_license': (info.card_data.license if info.card_data else None),
        'reservation_license': reserved['license'], 'snapshot': str(snapshot),
        'config_sha256': config_sha, 'config_class': type(config).__name__, 'architecture': architecture,
        'transformers_model_class': empty_model_class, 'transformers_version': __import__('transformers').__version__,
        'tokenizer_class': tokenizer_class,
        'tokenizer_manifest_sha256': tokenizer_sha, 'tokenizer_files': tokenizer_files,
        'files': files, 'file_manifest_sha256': canonical_sha(files),
        'required_weight_shards': required_shards, 'required_weight_shard_count': len(required_shards),
        'quantized_scope': {'rule': 'all text torch.nn.Linear except output head', 'modules': len(modules),
                            'module_shape_manifest_sha256': canonical_sha(shapes), 'n8_total_tiles': total8,
                            'n16_total_tiles': total16, 'n8_nondivisible': bad8, 'n16_nondivisible': bad16},
        'compatibility_passed': True,
    }


def main() -> None:
    campaign_root = Path(os.environ['CAMPAIGN_ROOT'])
    reservation_path = campaign_root / 'provenance' / 'HELDOUT_PANEL_RESERVATION.json'
    reservation_sha = sha256_file(reservation_path)
    sidecar_sha = (campaign_root / 'provenance' / 'HELDOUT_PANEL_RESERVATION.sha256').read_text().split()[0]
    if reservation_sha != sidecar_sha:
        raise RuntimeError('held-out reservation sidecar mismatch')
    reservation = json.loads(reservation_path.read_text())
    out = runtime.out_dir('heldout_input_audit')
    started = time.time()
    audited = [audit_one(item) for item in reservation['models']]
    report = {
        'schema_version': '1.0', 'status': 'complete', 'quality_values_accessed': False,
        'reservation_path': str(reservation_path), 'reservation_sha256': reservation_sha,
        'models': audited, 'model_count': len(audited), 'all_compatibility_passed': all(x['compatibility_passed'] for x in audited),
        'wall_seconds': time.time() - started,
    }
    path = out / 'HELDOUT_INPUT_AUDIT.json'
    runtime.atomic_json(path, report)
    source_manifest = json.loads((runtime.run_dir / 'launch_record.json').read_text())['source_manifest_sha256']
    runtime.atomic_json(runtime.run_dir / 'job_result.json', {
        'protocol_id': 'ppl-extension-heldout-reservation-audit', 'protocol_freeze_sha256': '',
        'source': {'model_id': 'prospective-heldout-panel', 'model_revision': reservation_sha,
                   'tokenizer_revision': reservation_sha, 'model_class': 'multiple',
                   'module_manifest_sha256': canonical_sha([x['quantized_scope'] for x in audited]),
                   'source_manifest_sha256': source_manifest},
        'environment': runtime.environment(),
        'data': {'calibration_manifest_sha256': None, 'evaluation_manifest_sha256': None,
                 'token_hashes': {}, 'overlap_audit': None},
        'policies': [],
        'results': {'raw_outputs': [str(path)], 'summary': {'all_compatibility_passed': report['all_compatibility_passed']},
                    'uncertainty': {}, 'attempted_endpoints': ['P70_HELDOUT_INPUT_COMPATIBILITY'],
                    'missing_endpoints': []},
        'logs': [], 'failures': [],
    })
    print(json.dumps({'reservation_sha256': reservation_sha, 'all_compatibility_passed': report['all_compatibility_passed']}), flush=True)


if __name__ == '__main__':
    main()
