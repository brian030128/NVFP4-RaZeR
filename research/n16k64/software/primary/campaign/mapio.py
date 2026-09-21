"""Canonical, hashable, self-describing E0M3 tile-map files.

Format (`MIXFP4MAP/1`):
    bytes 0..11   b'MIXFP4MAP/1\\n'
    bytes 12..19  little-endian uint64 header length H
    bytes 20..    UTF-8 JSON header (sort_keys=True, separators=(',', ':'), ASCII only)
    then          payload: for each module in header['modules'] order, numpy.packbits of the
                  row-major flattened bool grid with bitorder='little', padded to whole bytes.

The file is a pure function of the header and masks: no timestamps, run ids, hostnames or
floating-point score values, so two runs that elect identical masks under the same frozen
model/policy/source produce byte-identical files. Run provenance lives in a sidecar JSON.
"""
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch

MAGIC = b'MIXFP4MAP/1\n'
REQUIRED_HEADER = ('format', 'protocol_id', 'policy', 'model', 'type_block', 'scale_block', 'baseline',
                   'alternative', 'modules', 'totals', 'source_manifest_sha256', 'calibration_manifest_sha256')


class MapVerificationError(RuntimeError):
    pass


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


def build_header(*, protocol_id, policy, model, type_block, masks, weight_shapes, source_manifest_sha256,
                 calibration_manifest_sha256, baseline='FourOverSix E2M1 (quant_nvfp4_4over6, alpha {1,1.5})',
                 alternative='E0M3 signed uniform alpha=1 (quant_mix_4_6 clip=a1 elect=always)', scale_block=16):
    modules, sel, tot = [], 0, 0
    bm, bk = type_block
    for name, mask in masks.items():
        o, k = weight_shapes[name]
        grid = [o // bm, k // bk]
        if o % bm or k % bk or list(mask.shape) != grid:
            raise ValueError(f'{name}: mask {tuple(mask.shape)} inconsistent with weight {(o, k)} and block {type_block}')
        s = int(mask.sum())
        modules.append(dict(name=name, weight_shape=[int(o), int(k)], grid_shape=grid, selected=s, total=grid[0] * grid[1]))
        sel += s
        tot += grid[0] * grid[1]
    return dict(format='MIXFP4MAP/1', protocol_id=protocol_id, policy=policy, model=model,
                type_block=[int(bm), int(bk)], scale_block=int(scale_block), baseline=baseline, alternative=alternative,
                modules=modules, totals=dict(selected_tiles=sel, total_tiles=tot, selected_weights=sel * bm * bk,
                                             total_weights=tot * bm * bk, selected_fraction=(sel / tot if tot else 0.0)),
                source_manifest_sha256=source_manifest_sha256, calibration_manifest_sha256=calibration_manifest_sha256)


def serialize(header, masks):
    missing = [k for k in REQUIRED_HEADER if k not in header]
    if missing:
        raise ValueError(f'header missing {missing}')
    names = [m['name'] for m in header['modules']]
    if names != list(masks):
        raise ValueError('mask order must equal header module order')
    payload = []
    for m in header['modules']:
        mask = masks[m['name']]
        if mask.dtype != torch.bool or list(mask.shape) != m['grid_shape']:
            raise ValueError(f'{m["name"]}: bad mask dtype/shape')
        payload.append(np.packbits(mask.detach().cpu().contiguous().numpy().reshape(-1), bitorder='little').tobytes())
    head = _canon(header)
    return MAGIC + struct.pack('<Q', len(head)) + head + b''.join(payload)


def payload_sha256(masks):
    h = hashlib.sha256()
    for name, mask in masks.items():
        h.update(name.encode())
        h.update(np.packbits(mask.detach().cpu().contiguous().numpy().reshape(-1), bitorder='little').tobytes())
    return h.hexdigest()


def write_map(path, header, masks, provenance=None):
    """Write atomically, refuse to overwrite, return (sha256, path)."""
    data = serialize(header, masks)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)
    digest = hashlib.sha256(data).hexdigest()
    if provenance is not None:
        side = dict(provenance, map_file=path.name, map_sha256=digest, mask_payload_sha256=payload_sha256(masks))
        path.with_suffix(path.suffix + '.provenance.json').write_text(json.dumps(side, indent=1, sort_keys=True) + '\n')
    return digest, str(path)


def read_map(path, expected_sha256=None):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise MapVerificationError(f'map digest {digest} != expected {expected_sha256} for {path}')
    if not data.startswith(MAGIC):
        raise MapVerificationError('bad magic')
    (hlen,) = struct.unpack('<Q', data[len(MAGIC):len(MAGIC) + 8])
    start = len(MAGIC) + 8
    header = json.loads(data[start:start + hlen].decode('ascii'))
    if _canon(header) != data[start:start + hlen]:
        raise MapVerificationError('header is not canonical JSON')
    off = start + hlen
    masks = {}
    for m in header['modules']:
        n = m['grid_shape'][0] * m['grid_shape'][1]
        nbytes = (n + 7) // 8
        chunk = np.frombuffer(data[off:off + nbytes], dtype=np.uint8)
        if chunk.size != nbytes:
            raise MapVerificationError('truncated payload')
        bits = np.unpackbits(chunk, bitorder='little')[:n].astype(bool)
        if nbytes * 8 > n and np.unpackbits(chunk, bitorder='little')[n:].any():
            raise MapVerificationError(f'{m["name"]}: nonzero padding bits')
        t = torch.from_numpy(bits.copy()).reshape(m['grid_shape'])
        if int(t.sum()) != m['selected']:
            raise MapVerificationError(f'{m["name"]}: selected count mismatch')
        masks[m['name']] = t
        off += nbytes
    if off != len(data):
        raise MapVerificationError('trailing bytes after payload')
    if sum(m['selected'] for m in header['modules']) != header['totals']['selected_tiles']:
        raise MapVerificationError('totals mismatch')
    return header, masks, digest


def verify_for_model(header, *, model_id, model_revision, tokenizer_revision, weight_shapes, type_block,
                     protocol_id, policy_name, known_source_manifests=None, expected_total_tiles=None):
    """Refuse to install a map that does not describe exactly this model/policy/protocol."""
    errs = []
    if header['model'].get('model_id') != model_id:
        errs.append(f'model id {header["model"].get("model_id")} != {model_id}')
    if header['model'].get('revision') != model_revision:
        errs.append(f'model revision {header["model"].get("revision")} != {model_revision}')
    if header['model'].get('tokenizer_revision') != tokenizer_revision:
        errs.append(f'tokenizer revision {header["model"].get("tokenizer_revision")} != {tokenizer_revision}')
    if list(header['type_block']) != list(type_block):
        errs.append(f'type block {header["type_block"]} != {list(type_block)}')
    if header['protocol_id'] != protocol_id:
        errs.append(f'protocol {header["protocol_id"]} != {protocol_id}')
    if header['policy'].get('name') != policy_name:
        errs.append(f'policy {header["policy"].get("name")} != {policy_name}')
    names = [m['name'] for m in header['modules']]
    if names != list(weight_shapes):
        errs.append('module names/order differ from the loaded model scope')
    for m in header['modules']:
        shp = weight_shapes.get(m['name'])
        if shp is None or list(shp) != m['weight_shape']:
            errs.append(f'{m["name"]}: weight shape {shp} != {m["weight_shape"]}')
    if expected_total_tiles is not None and header['totals']['total_tiles'] != expected_total_tiles:
        errs.append(f'total tiles {header["totals"]["total_tiles"]} != expected {expected_total_tiles}')
    if known_source_manifests is not None and header['source_manifest_sha256'] not in known_source_manifests:
        errs.append(f'source manifest {header["source_manifest_sha256"]} is not an archived campaign source state')
    if errs:
        raise MapVerificationError('; '.join(errs))
    return True
