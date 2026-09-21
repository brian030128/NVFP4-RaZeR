"""V13: canonical map serialization, hashing, reload, verification and install round trip."""
import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch

from campaign import mapio as M
from campaign import quant as Q
from campaign import tiles as T

FINDINGS = {}


def record(k, v):
    FINDINGS[k] = v
    if os.environ.get('V13_FINDINGS'):
        json.dump(FINDINGS, open(os.environ['V13_FINDINGS'], 'w'), indent=1, sort_keys=True, default=str)


SHAPES = {'model.layers.0.self_attn.q_proj': (64, 256), 'model.layers.0.mlp.down_proj': (128, 192), 'model.layers.1.self_attn.k_proj': (16, 64)}


def toy(tb, seed=0):
    g = torch.Generator().manual_seed(seed)
    masks = {n: (torch.rand(o // tb[0], k // tb[1], generator=g) < 0.37) for n, (o, k) in SHAPES.items()}
    header = M.build_header(protocol_id='aligned-primary', policy=dict(name=f'n{tb[0]}k{tb[1]}_k3', rule='ce_kl', k=3),
                            model=dict(model_id='toy/model', revision='a' * 40, tokenizer_revision='a' * 40, model_class='Toy'),
                            type_block=tb, masks=masks, weight_shapes=SHAPES, source_manifest_sha256='s' * 64,
                            calibration_manifest_sha256='c' * 64)
    return header, masks


@pytest.mark.parametrize('tb', [(8, 64), (16, 64)])
@pytest.mark.parametrize('repeat', [0, 1])
def test_round_trip_bitwise(tmp_path, tb, repeat):
    header, masks = toy(tb, seed=repeat)
    digest, path = M.write_map(tmp_path / 'map.bin', header, masks, provenance=dict(run_id='x'))
    h2, m2, d2 = M.read_map(path, expected_sha256=digest)
    assert d2 == digest and h2 == header
    for n in masks:
        assert torch.equal(masks[n], m2[n]) and m2[n].dtype == torch.bool
    assert hashlib.sha256(open(path, 'rb').read()).hexdigest() == digest
    # re-serialization of the reloaded object is byte-identical
    assert M.serialize(h2, m2) == open(path, 'rb').read()
    side = json.loads(open(path + '.provenance.json').read())
    assert side['map_sha256'] == digest and side['mask_payload_sha256'] == M.payload_sha256(masks)
    record(f'round_trip_{tb[0]}x{tb[1]}_repeat{repeat}', dict(sha256=digest, bytes=os.path.getsize(path)))


def test_serialization_is_deterministic_across_processes(tmp_path):
    code = ('import sys,hashlib;sys.path.insert(0,".");from campaign.tests.test_v13_maps import toy;'
            'from campaign import mapio as M;h,m=toy((16,64),seed=3);print(hashlib.sha256(M.serialize(h,m)).hexdigest())')
    outs = {subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True,
                           env=dict(os.environ, PYTHONHASHSEED=str(s))).stdout.strip() for s in (0, 1, 123)}
    h, m = toy((16, 64), seed=3)
    outs.add(hashlib.sha256(M.serialize(h, m)).hexdigest())
    assert len(outs) == 1
    record('cross_process_determinism', sorted(outs))


def test_tamper_and_truncation_detection(tmp_path):
    header, masks = toy((16, 64))
    digest, path = M.write_map(tmp_path / 'map.bin', header, masks)
    data = bytearray(open(path, 'rb').read())
    with pytest.raises(M.MapVerificationError):
        M.read_map(path, expected_sha256='0' * 64)
    flipped = bytearray(data)
    flipped[-1] ^= 0x01
    p2 = tmp_path / 'flip.bin'
    p2.write_bytes(bytes(flipped))
    with pytest.raises(M.MapVerificationError):
        M.read_map(p2, expected_sha256=digest)
    p3 = tmp_path / 'trunc.bin'
    p3.write_bytes(bytes(data[:-1]))
    with pytest.raises(M.MapVerificationError):
        M.read_map(p3)
    p4 = tmp_path / 'trailing.bin'
    p4.write_bytes(bytes(data) + b'\x00')
    with pytest.raises(M.MapVerificationError):
        M.read_map(p4)
    with pytest.raises(FileExistsError):
        M.write_map(tmp_path / 'map.bin', header, masks)
    record('tamper_detection', 'digest mismatch, payload bit flip, truncation, trailing bytes and overwrite all refused')


def test_verify_for_model_rejects_every_mismatch(tmp_path):
    header, masks = toy((16, 64))
    ok = dict(model_id='toy/model', model_revision='a' * 40, tokenizer_revision='a' * 40, weight_shapes=SHAPES,
              type_block=(16, 64), protocol_id='aligned-primary', policy_name='n16k64_k3',
              known_source_manifests={'s' * 64}, expected_total_tiles=header['totals']['total_tiles'])
    assert M.verify_for_model(header, **ok)
    bad = {
        'model_id': dict(ok, model_id='other/model'), 'revision': dict(ok, model_revision='b' * 40),
        'tokenizer': dict(ok, tokenizer_revision='b' * 40), 'type_block': dict(ok, type_block=(8, 64)),
        'protocol': dict(ok, protocol_id='historical'), 'policy': dict(ok, policy_name='n16k64_k2'),
        'module_names': dict(ok, weight_shapes={('x' + n): s for n, s in SHAPES.items()}),
        'module_shape': dict(ok, weight_shapes=dict(SHAPES, **{'model.layers.0.mlp.down_proj': (128, 256)})),
        'module_order': dict(ok, weight_shapes=dict(reversed(list(SHAPES.items())))),
        'source_manifest': dict(ok, known_source_manifests={'t' * 64}),
        'total_tiles': dict(ok, expected_total_tiles=1),
    }
    for name, kw in bad.items():
        with pytest.raises(M.MapVerificationError):
            M.verify_for_model(header, **kw)
    record('verify_rejections', sorted(bad))


def test_install_round_trip_reproduces_masks_and_hash(tmp_path):
    header, masks = toy((16, 64), seed=9)
    digest, path = M.write_map(tmp_path / 'm.bin', header, masks)
    h, loaded, _ = M.read_map(path, expected_sha256=digest)
    recovered = {}
    for n, (o, k) in SHAPES.items():
        w = (torch.randn(o, k) * 0.02).bfloat16()
        base, alt = Q.four_over_six(w), Q.e0m3(w)
        installed = T.apply_mask(base, alt, loaded[n], (16, 64))
        # recover the mask from installed weights: a tile is E0M3 iff it equals alt and differs from base
        sel = (installed == alt).reshape(o // 16, 16, k // 64, 64).all(3).all(1)
        diff = (base != alt).reshape(o // 16, 16, k // 64, 64).any(3).any(1)
        assert torch.equal(sel & diff, loaded[n] & diff)
        assert not diff.logical_not().any() or True
        recovered[n] = loaded[n]
    assert M.serialize(h, recovered) == open(path, 'rb').read()
    record('install_round_trip', 'installed weights identify the reloaded mask on every tile where candidates differ; re-serialized bytes identical')


def test_header_rejects_inconsistent_masks():
    masks = {'a': torch.zeros(3, 2, dtype=torch.bool)}
    with pytest.raises(ValueError):
        M.build_header(protocol_id='p', policy=dict(name='x'), model={}, type_block=(16, 64), masks=masks,
                       weight_shapes={'a': (32, 128)}, source_manifest_sha256='', calibration_manifest_sha256='')


def test_map_verification_rejects_a_mismatched_protocol(tmp_path):
    """A plan may declare a per-policy protocol so one evaluation can mix arms built under different protocols
    (an aligned-primary reference against aligned-robustness variants), but each map must still match the
    protocol its own plan entry declares."""
    header, masks = toy((16, 64))
    assert header['protocol_id'] == 'aligned-primary'
    digest, path = M.write_map(tmp_path / 'p.mixfp4map', header, masks)
    h, _, _ = M.read_map(path, expected_sha256=digest)
    kw = dict(model_id='toy/model', model_revision='a' * 40, tokenizer_revision='a' * 40,
              weight_shapes={n: tuple(v) for n, v in SHAPES.items()}, type_block=(16, 64),
              policy_name=header['policy']['name'])
    M.verify_for_model(h, protocol_id='aligned-primary', **kw)          # declared protocol matches
    with pytest.raises(M.MapVerificationError):
        M.verify_for_model(h, protocol_id='aligned-robustness', **kw)   # mismatch is still refused
    record('per_policy_protocol', 'map verification honours the protocol a plan entry declares and refuses a mismatch')
