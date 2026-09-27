"""Native deployment artifacts: frozen format map + weights -> packed codes, tagged scales, metadata.

An artifact directory holds
    weights.safetensors   per module: <name>.packed  uint8 [out, in/2]   (nibbles, 2j in the low nibble)
                                      <name>.scales  uint8 [out, in/16]  (UE4M3, bit 7 = E0M3 tag)
                                      <name>.global_scale float32 [1]
                                      <name>.bias    bf16 [out]          (only if the Linear has one)
    artifact.json         format, model identity, map identity (sha256 + header), quantization and
                          numerics spec, kernel compatibility, per-module shapes / tile counts / tensor
                          hashes, byte sizes, and the sha256 of weights.safetensors.

The stored form is canonical (row-major scales), not kernel-specific: placing scale bytes into a
kernel's layout is a gather done once at load (NativeLinear), so one artifact serves every build
whose weight granule it satisfies. Nothing is re-quantized at load.

Invariants checked on export AND on load (a violation raises):
  * decode(artifact) equals the fake-quant weight the map defines, bit for bit (export only; needs
    the source weights);
  * every E0M3 tag is uniform over whole type-block tiles (all 16 x 4 scale bytes of a N16K64
    tile), and the tagged tiles are exactly the map's (count per module = map header count, and the
    tile set hashes to the map's mask);
  * tensor hashes and the safetensors file hash match the manifest.
"""
import datetime
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from . import mapio
from . import numerics as N

FORMAT = 'MIXFP4-SM120-ARTIFACT/1'
WEIGHT_KIND_ACT = {'map': 'four_over_six_rows', 'four_over_six': 'four_over_six_rows', 'nvfp4': 'nvfp4_rows'}


class ArtifactError(RuntimeError):
    pass


def tensor_sha256(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def tile_flags(scales, type_block):
    """Per-tile E0M3 flag from tagged scale bytes; raises if any tile is not uniformly tagged."""
    bm, bk = type_block
    n, kb = scales.shape
    per = bk // N.SCALE_BLOCK
    if n % bm or kb % per:
        raise ArtifactError(f'scales {tuple(scales.shape)} not divisible into {type_block} tiles')
    f = (scales >> 7).reshape(n // bm, bm, kb // per, per)
    lo, hi = f.amin((1, 3)), f.amax((1, 3))
    if not torch.equal(lo, hi):
        bad = int((lo != hi).sum())
        raise ArtifactError(f'{bad} tiles carry a mixed E0M3 tag inside one {type_block} granule; the kernel '
                            f'requires uniform tags (non-uniform tags below one MMA atom can hang the GPU)')
    return hi.bool()


@dataclass
class PackedWeight:
    name: str
    packed: torch.Tensor       # uint8 [n, k/2]
    scales: torch.Tensor       # uint8 [n, k/16], bit 7 = E0M3
    global_scale: float
    bias: torch.Tensor | None  # bf16 [n]
    type_block: tuple | None
    e0m3_tiles: int

    @property
    def shape(self):
        return self.packed.shape[0], self.packed.shape[1] * 2

    def decode(self):
        return N.decode_fake_order(N.unpack_nibbles(self.packed), self.scales, self.global_scale)


def _git_head():
    try:
        root = Path(__file__).resolve().parents[2]
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout.strip())
        return head, dirty
    except OSError:
        return None, None


@torch.no_grad()
def pack_module(name, weight, bias, kind, mask=None, type_block=None, expected=None):
    """Quantize + pack one Linear. `expected` (bf16) is the fake-quant weight to match bit for bit."""
    nib, sbytes, gs = N.quantize_weight(weight, kind, mask, type_block)
    pw = PackedWeight(name, N.pack_nibbles(nib), sbytes, float(gs),
                      None if bias is None else bias.detach().to(torch.bfloat16).contiguous(),
                      tuple(type_block) if type_block else None, int(mask.sum()) if mask is not None else 0)
    if expected is not None:
        got = N.decode_fake_order(nib, sbytes, gs)
        if not torch.equal(got, expected):
            raise ArtifactError(f'{name}: packed weight differs from the fake-quant weight in '
                                f'{int((got != expected).sum())} elements')
    return pw


def fake_quant_weight(weight, kind, mask=None, type_block=None):
    """The bf16 weight the fake-quant path installs (quantize/quantizer.py + campaign apply_mask)."""
    from quantize.quantizer import quant_mix_4_6, quant_nvfp4, quant_nvfp4_4over6
    if kind == 'nvfp4':
        return quant_nvfp4(weight, 4, 16)
    base = quant_nvfp4_4over6(weight, 4, 16)
    if kind == 'four_over_six' or mask is None or not bool(mask.any()):
        return base
    alt = quant_mix_4_6(weight, 4, 16, type_block=(8, 64), clip='a1', elect='always')
    bm, bk = type_block
    return torch.where(mask.to(weight.device).repeat_interleave(bm, 0).repeat_interleave(bk, 1), alt, base)


@torch.no_grad()
def export(out_dir, linears, *, kind='map', map_path=None, map_sha256=None, model_info=None,
           verify_fake=True, device='cuda', note=None):
    """Export `linears` ({name: nn.Linear}, in model order) to an artifact directory.

    kind='map' requires `map_path` (MIXFP4MAP/1); its module list and shapes must equal `linears`.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ArtifactError(f'{out_dir} is not empty')
    out_dir.mkdir(parents=True, exist_ok=True)
    header, masks, digest, type_block = None, None, None, None
    if kind == 'map':
        header, masks, digest = mapio.read_map(map_path, expected_sha256=map_sha256)
        type_block = tuple(header['type_block'])
        names = [m['name'] for m in header['modules']]
        if names != list(linears):
            missing = sorted(set(names) - set(linears))
            extra = sorted(set(linears) - set(names))
            raise ArtifactError(f'map modules differ from the model scope: missing {missing[:5]}, extra {extra[:5]}')
        for m in header['modules']:
            if list(linears[m['name']].weight.shape) != m['weight_shape']:
                raise ArtifactError(f'{m["name"]}: weight {tuple(linears[m["name"]].weight.shape)} != map {m["weight_shape"]}')
        if model_info is not None:
            for key in ('model_id', 'revision'):
                if key in model_info and header['model'].get(key) != model_info[key]:
                    raise ArtifactError(f'map is for {header["model"]}, model is {model_info}')
    elif kind not in ('four_over_six', 'nvfp4'):
        raise ArtifactError(f'unknown kind {kind!r}')

    tensors, modules = {}, []
    tot = dict(packed=0, scales=0, global_scales=0, bias=0, bf16_weights=0)
    for name, lin in linears.items():
        w = lin.weight.detach().to(device)
        mask = masks[name] if masks is not None else None
        exp = fake_quant_weight(w, kind, mask, type_block) if verify_fake else None
        pw = pack_module(name, w, lin.bias, kind, mask, type_block, exp)
        del exp
        if type_block is not None:
            flags = tile_flags(pw.scales, type_block)
            if not torch.equal(flags.cpu(), mask.cpu()):
                raise ArtifactError(f'{name}: tagged tiles differ from the map mask')
        tensors[f'{name}.packed'] = pw.packed.cpu()
        tensors[f'{name}.scales'] = pw.scales.cpu()
        tensors[f'{name}.global_scale'] = torch.tensor([pw.global_scale], dtype=torch.float32)
        if pw.bias is not None:
            tensors[f'{name}.bias'] = pw.bias.cpu()
        n, k = pw.shape
        modules.append(dict(name=name, shape=[n, k], bias=pw.bias is not None, e0m3_tiles=pw.e0m3_tiles,
                            global_scale=pw.global_scale, packed_sha256=tensor_sha256(pw.packed),
                            scales_sha256=tensor_sha256(pw.scales),
                            bias_sha256=None if pw.bias is None else tensor_sha256(pw.bias)))
        tot['packed'] += pw.packed.numel()
        tot['scales'] += pw.scales.numel()
        tot['global_scales'] += 4
        tot['bias'] += 0 if pw.bias is None else pw.bias.numel() * 2
        tot['bf16_weights'] += n * k * 2
    wpath = out_dir / 'weights.safetensors'
    save_file(tensors, str(wpath), metadata={'format': FORMAT})
    head, dirty = _git_head()
    meta = dict(
        format=FORMAT, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        source_revision=head, source_dirty=dirty, torch=torch.__version__, note=note,
        model=model_info or (header['model'] if header else None),
        weight_kind=kind, activation_quantizer=WEIGHT_KIND_ACT[kind],
        numerics=dict(spec='sm120/NUMERICS.md', weight_baseline='quant_nvfp4_4over6(w, 4, 16)' if kind != 'nvfp4' else 'quant_nvfp4(w, 4, 16)',
                      weight_alternative='quant_mix_4_6(w, 4, 16, clip=a1, elect=always)' if kind == 'map' else None,
                      scale_block=16, scale_encoding='UE4M3, bit 7 = E0M3 tag', global_scale='FP32 amax / (6 * 448) per tensor',
                      nibble_order='element 2j in the low nibble', verified_against_fake_quant=bool(verify_fake)),
        type_block=list(type_block) if type_block else None,
        map=None if header is None else dict(file=Path(map_path).name, sha256=digest, format=header['format'],
                                             protocol_id=header['protocol_id'], policy=header['policy'],
                                             totals=header['totals'], baseline=header['baseline'],
                                             alternative=header['alternative'],
                                             mask_payload_sha256=mapio.payload_sha256(masks)),
        kernel_requirements=dict(weight_granule=list(type_block) if type_block else None,
                                 compatible_configs=[c for c in _compatible_configs(type_block)]),
        modules=modules,
        sizes_bytes=dict(tot, total=tot['packed'] + tot['scales'] + tot['global_scales'] + tot['bias'],
                         safetensors_file=wpath.stat().st_size),
        weights_sha256=file_sha256(wpath),
    )
    (out_dir / 'artifact.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
    return meta


def _compatible_configs(type_block):
    from . import configs as CFG
    out = []
    for c in CFG.CONFIGS.values():
        if type_block is None:
            out.append(c.name)                       # E2M1-only weights run on every build
        elif c.type_block is not None and c.type_block[1] <= type_block[1] and type_block[0] % c.type_block[0] == 0 \
                and type_block[1] % c.type_block[1] == 0:
            out.append(c.name)                       # the map's tiles are unions of the kernel's granules
    return out


def load(art_dir, device='cuda', verify=True):
    """-> (meta, {name: PackedWeight}); verifies hashes, tag uniformity and tile counts."""
    art_dir = Path(art_dir)
    meta = json.loads((art_dir / 'artifact.json').read_text())
    if meta.get('format') != FORMAT:
        raise ArtifactError(f'unknown artifact format {meta.get("format")}')
    wpath = art_dir / 'weights.safetensors'
    if verify and file_sha256(wpath) != meta['weights_sha256']:
        raise ArtifactError('weights.safetensors does not match artifact.json')
    tensors = load_file(str(wpath), device='cpu')
    tb = tuple(meta['type_block']) if meta['type_block'] else None
    out = {}
    for m in meta['modules']:
        name = m['name']
        packed, scales = tensors[f'{name}.packed'], tensors[f'{name}.scales']
        gs = float(tensors[f'{name}.global_scale'][0])
        bias = tensors.get(f'{name}.bias')
        if verify:
            if tensor_sha256(packed) != m['packed_sha256'] or tensor_sha256(scales) != m['scales_sha256']:
                raise ArtifactError(f'{name}: tensor hash mismatch')
            if list(packed.shape) != [m['shape'][0], m['shape'][1] // 2] or gs != m['global_scale']:
                raise ArtifactError(f'{name}: shape / global scale mismatch')
            if tb is not None:
                tiles = int(tile_flags(scales, tb).sum())
                if tiles != m['e0m3_tiles']:
                    raise ArtifactError(f'{name}: {tiles} tagged tiles, manifest says {m["e0m3_tiles"]}')
            elif bool((scales >> 7).any()):
                raise ArtifactError(f'{name}: E0M3 tags in an E2M1-only artifact')
        out[name] = PackedWeight(name, packed.to(device), scales.to(device), gs,
                                 None if bias is None else bias.to(device), tb, m['e0m3_tiles'])
    if verify and meta.get('map') is not None:
        if sum(m['e0m3_tiles'] for m in meta['modules']) != meta['map']['totals']['selected_tiles']:
            raise ArtifactError('tagged tile total differs from the map header')
    return meta, out


def masks_from_artifact(weights):
    """{name: tile mask} recovered from the tags (to compare with the map it was exported from)."""
    return {n: tile_flags(pw.scales, pw.type_block).cpu() for n, pw in weights.items() if pw.type_block}
