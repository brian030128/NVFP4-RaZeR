"""Deployment-format storage of the two MixFP4 weight candidates.

The FourOverSix (E2M1) and E0M3 alpha=1 candidates are kept as packed 4-bit codes
plus one FP8 E4M3 block scale per 16 elements and one FP32 global scale, i.e.
about 1.1 bytes per weight for both candidates together instead of 4 bytes of
dequantized BF16. Decoding reproduces the reference quantizers' BF16 output
bitwise; `pack` checks this against the supplied reference tensors and refuses
to pack otherwise.
"""
import torch

LEVELS = (0., .5, 1., 1.5, 2., 3., 4., 6.)


def _fp8(x):
    return x.clamp(min=2 ** -9, max=448).to(torch.float8_e4m3fn)


def _pack_nibbles(codes):
    flat = codes.reshape(-1).to(torch.uint8)
    return flat[0::2] | (flat[1::2] << 4)


def _unpack_nibbles(packed):
    out = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = packed & 15
    out[1::2] = packed >> 4
    return out


@torch.no_grad()
def encode(w):
    """Codes and scales for both candidates, mirroring quant_nvfp4_4over6 in float32."""
    x = w.reshape(-1, 16).float()
    gs = x.abs().amax() / (6 * 448)
    scaled = x / gs
    peak = scaled.abs().amax(-1, keepdim=True)
    levels = torch.tensor(LEVELS, device=w.device)
    mids = (levels[:-1] + levels[1:]) / 2
    sign = scaled.sign()
    choices = []
    for qmax in (6., 4.):
        s = _fp8(peak / qmax)
        index = torch.bucketize((scaled / s.float()).abs().contiguous(), mids, right=False)
        error = (levels[index] * sign * s.float() - scaled).square().sum(-1, keepdim=True)
        choices.append((s, index, error))
    use4 = choices[1][2] < choices[0][2]
    s2 = torch.where(use4, choices[1][0].float(), choices[0][0].float()).to(torch.float8_e4m3fn)
    index = torch.where(use4, choices[1][1], choices[0][1])
    e2m1 = index | ((sign < 0).to(index.dtype) << 3)
    s0 = _fp8(peak / 7)
    e0m3 = ((scaled / s0.float()).round().clamp(-7, 7) + 8).to(torch.int64)
    return dict(shape=tuple(w.shape), gs=gs, e2m1=_pack_nibbles(e2m1), s2=s2.reshape(-1),
                e0m3=_pack_nibbles(e0m3), s0=s0.reshape(-1))


@torch.no_grad()
def decode_base(p):
    codes = _unpack_nibbles(p['e2m1']).reshape(-1, 16).long()
    levels = torch.tensor(LEVELS, device=codes.device)
    q = levels[codes & 7] * torch.where(codes >= 8, -1., 1.).to(levels)
    return (q * p['s2'].float()[:, None] * p['gs']).reshape(p['shape']).to(torch.bfloat16)


@torch.no_grad()
def decode_alt(p):
    codes = _unpack_nibbles(p['e0m3']).reshape(-1, 16).float() - 8
    return (codes * p['s0'].float()[:, None] * p['gs']).reshape(p['shape']).to(torch.bfloat16)


@torch.no_grad()
def pack(w, base_ref, alt_ref):
    """Pack both candidates; return None if decoding is not bitwise identical."""
    p = encode(w)
    if torch.equal(decode_base(p), base_ref) and torch.equal(decode_alt(p), alt_ref):
        return p
    return None


def nbytes(p):
    return sum(v.numel() * v.element_size() for v in p.values() if torch.is_tensor(v))
