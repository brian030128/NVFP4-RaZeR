"""Exact fake-quant emulation of a Model Optimizer (modelopt) mixed NVFP4/FP8 checkpoint.

Used to start MixFP4 map training from nvidia/Qwen3.8-27B-NVFP4 instead of from our
own FourOverSix quantization. The checkpoint stores, per quantized linear layer,

  NVFP4 (MLP, lm_head)  weight        uint8, two E2M1 codes per byte (low nibble first)
                        weight_scale  float8_e4m3fn, one per 16 elements along K
                        weight_scale_2 float32 scalar (global scale)
                        input_scale   float32 scalar: static activation global scale
  FP8 (attention)       weight        float8_e4m3fn, weight_scale float32 scalar
                        input_scale   float32 scalar: static per-tensor activation scale

Weights are dequantized to BF16 exactly as packed_candidates.decode_base does for our
own candidates (the E2M1 code layout is identical), so nvidia's NVFP4 weights become
the E2M1 candidate B of the tile map bitwise. The E0M3 candidate A is block_max/7 on
the ORIGINAL BF16 weight, under nvidia's global scale, so both candidates of a tile
share one tensor scale. Activations follow the checkpoint's recipe: NVFP4 with the
static input_scale as global scale and dynamic E4M3 scales per 16 elements; FP8 with
one static scale per tensor. Both are per-token independent, so the same quantizer
is causal and serves training, development evaluation and PPL evaluation alike.
"""
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from quantize.packed_candidates import LEVELS, _fp8, _pack_nibbles, decode_alt, decode_base

REPO = 'nvidia/Qwen3.8-27B-NVFP4'
REVISION = '482ca0f3832238542f8f5295dde86b5f22711d80'
# Per-model modelopt checkpoints (repo, pinned revision). Nemotron's release is uniform
# NVFP4 with an exclude list and was further trained with quantization-aware distillation.
CHECKPOINTS = {'qwen27b': (REPO, REVISION),
               'nemotron9b': ('nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4', '8556c9164ddb43fe1f4f4ad730593b3c5e3f7328')}
# Checkpoint-name prefixes that transformers renames on load (conversion_mapping.py).
RENAMES = (('backbone.', 'model.'),)
_MIDS = {}


def to_model_name(name):
    for old, new in RENAMES:
        if name.startswith(old):
            return new + name[len(old):]
    return name


def download(repo=REPO, revision=REVISION):
    """Snapshot into the job-local /tmp (the HF cache in $HOME has no room for 22 GB)."""
    from huggingface_hub import snapshot_download
    root = Path(os.environ.get('TMPDIR', '/tmp')) / 'modelopt_ckpt'
    return Path(snapshot_download(repo, revision=revision, cache_dir=str(root),
                                  allow_patterns=['*.json', '*.safetensors']))


def _levels(device):
    if device not in _MIDS:
        levels = torch.tensor(LEVELS, device=device)
        _MIDS[device] = (levels, (levels[:-1] + levels[1:]) / 2)
    return _MIDS[device]


@torch.no_grad()
def act_nvfp4(x, gs):
    """NVFP4 activation: static FP32 global scale gs, dynamic E4M3 scale per 16 elements."""
    shape = x.shape
    scaled = x.float().reshape(-1, 16) / gs
    peak = scaled.abs().amax(-1, keepdim=True)
    s = _fp8(peak / 6).float()
    levels, mids = _levels(x.device)
    q = levels[torch.bucketize((scaled / s).abs().contiguous(), mids, right=False)] * scaled.sign()
    return (q * s * gs).reshape(shape).to(torch.bfloat16)


@torch.no_grad()
def act_fp8(x, scale):
    """FP8 E4M3 activation with one static scale per tensor (saturating)."""
    return ((x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).to(torch.bfloat16)


@torch.no_grad()
def encode_alt(w, gs):
    """E0M3 alpha=1 codes of the BF16 weight w under a given global scale (mirrors packed_candidates.encode)."""
    scaled = w.reshape(-1, 16).float() / gs
    s0 = _fp8(scaled.abs().amax(-1, keepdim=True) / 7)
    e0m3 = ((scaled / s0.float()).round().clamp(-7, 7) + 8).to(torch.int64)
    return _pack_nibbles(e0m3), s0.reshape(-1)


def reference_dequant(codes_u8, scale, gs, shape):
    """Independent NVFP4 dequantization (table lookup per nibble) used to audit decode_base."""
    lo, hi = (codes_u8 & 15).long(), (codes_u8 >> 4).long()
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                         device=codes_u8.device)
    vals = torch.stack([table[lo], table[hi]], -1).reshape(shape[0], -1)
    return (vals * scale.float().repeat_interleave(16, 1) * gs).to(torch.bfloat16)


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.config = json.loads((self.path / 'hf_quant_config.json').read_text())['quantization']
        index = json.loads((self.path / 'model.safetensors.index.json').read_text())['weight_map']
        # Everything is keyed by the MODEL's tensor names; self.original maps back to the file.
        self.original = {to_model_name(n): n for n in index}
        assert len(self.original) == len(index)
        self.index = {to_model_name(n): f for n, f in index.items()}
        if 'quantized_layers' in self.config:
            self.layers = {to_model_name(n): v for n, v in self.config['quantized_layers'].items()}
        else:
            # Uniform config (quant_algo + exclude_modules): the quantized modules are the
            # ones that carry scales; check that agrees with the exclude list.
            algo = self.config['quant_algo']
            assert algo == 'NVFP4', algo
            self.layers = {n[:-len('.weight_scale_2')]: dict(quant_algo='NVFP4')
                           for n in self.index if n.endswith('.weight_scale_2')}
            assert not any(n.endswith('.weight_scale') and n[:-len('.weight_scale')] not in self.layers
                           for n in self.index)
            excluded = {to_model_name(n) for n in self.config.get('exclude_modules', [])}
            assert not excluded & set(self.layers), excluded & set(self.layers)
        self._files = {}

    def get(self, name):
        f = self.index[name]
        if f not in self._files:
            self._files[f] = safe_open(str(self.path / f), framework='pt', device='cpu')
        return self._files[f].get_tensor(self.original[name])

    def kind(self, module_name):
        v = self.layers.get(module_name)
        return None if v is None else v['quant_algo']

    @torch.no_grad()
    def apply(self, model, device_of):
        """Load every checkpoint tensor into the model: plain tensors copied, FP8 weights
        dequantized. NVFP4 weights are NOT written here (see nvfp4_candidate).
        Returns {module: ('nvfp4'|'fp8', input_scale)} and a load audit."""
        params = dict(model.named_parameters())
        acts, audit = {}, dict(plain_copied=0, plain_changed=0, plain_max_abs_diff=0.0, skipped=[])
        names = set(self.index)
        for n in sorted(names):
            if n.startswith('mtp.') or n.startswith('model.mtp.'):
                audit['skipped'].append(n)
                continue
            if n.endswith(('.weight_scale', '.weight_scale_2', '.input_scale')):
                continue
            module = n[:-len('.weight')] if n.endswith('.weight') else None
            kind = self.kind(module) if module else None
            p = params[n]
            if kind == 'NVFP4':
                acts[module] = ('nvfp4', float(self.get(module + '.input_scale')))
                continue
            t = self.get(n)
            if kind == 'FP8':
                assert t.dtype == torch.float8_e4m3fn and t.shape == p.shape, n
                w = (t.float() * self.get(module + '.weight_scale').float()).to(p.dtype)
                nmse = float((w.to(p.device).float() - p.float()).square().sum() / p.float().square().sum())
                assert nmse < 0.01, (n, nmse)
                audit['fp8_max_nmse'] = max(audit.get('fp8_max_nmse', 0.0), nmse)
                acts[module] = ('fp8', float(self.get(module + '.input_scale')))
            else:
                assert t.shape == p.shape, (n, t.shape, p.shape)
                if t.dtype != p.dtype:
                    audit.setdefault('dtype_cast', []).append([n, str(t.dtype), str(p.dtype)])
                w = t.to(p.dtype)
                d = (t.to(p.device).float() - p.float()).abs().max().item()
                audit['plain_copied'] += 1
                audit['plain_changed'] += int(d > 0)
                audit['plain_max_abs_diff'] = max(audit['plain_max_abs_diff'], d)
            p.copy_(w.to(p.device))
        assert set(acts) == set(self.layers), set(self.layers) ^ set(acts)
        return acts, audit

    @torch.no_grad()
    def nvfp4_candidate(self, module, w_bf16, alt_source='bf16'):
        """Packed candidates for one NVFP4 layer: base = nvidia's codes/scales (bitwise),
        alt = E0M3 under nvidia's global scale, of the original BF16 weight (alt_source
        'bf16') or of nvidia's dequantized weight ('base'; for QAD checkpoints, whose
        NVFP4 weights were trained and no longer approximate the BF16 release)."""
        dev = w_bf16.device
        codes = self.get(module + '.weight').to(dev)
        scale = self.get(module + '.weight_scale').to(dev)
        gs = self.get(module + '.weight_scale_2').float().to(dev).reshape(())
        shape = tuple(w_bf16.shape)
        assert codes.dtype == torch.uint8 and codes.shape == (shape[0], shape[1] // 2), module
        assert scale.dtype == torch.float8_e4m3fn and scale.shape == (shape[0], shape[1] // 16), module
        p = dict(shape=shape, gs=gs, e2m1=codes.reshape(-1), s2=scale.reshape(-1))
        base = decode_base(p)
        assert alt_source in ('bf16', 'base'), alt_source
        p['e0m3'], p['s0'] = encode_alt(w_bf16 if alt_source == 'bf16' else base, gs)
        assert torch.equal(base.view(torch.int16),
                           reference_dequant(codes, scale, gs, shape).view(torch.int16)), module
        stats = dict(gs=float(gs), gs_over_amax_rule=float(gs / (w_bf16.float().abs().amax() / (6 * 448))),
                     base_nmse=float((base.float() - w_bf16.float()).square().sum() / w_bf16.float().square().sum()),
                     alt_nmse=float((decode_alt(p).float() - w_bf16.float()).square().sum()
                                    / w_bf16.float().square().sum()))
        # Guards the nibble order and code table, which the bitwise audit above shares.
        # (A wrong nibble order gives NMSE ~2; QAD drift from the BF16 release is allowed.)
        assert stats['base_nmse'] < 0.1, (module, stats)
        return p, stats
