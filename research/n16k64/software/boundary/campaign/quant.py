"""Weight candidates and activation quantizers used by every campaign protocol.

Weights (dequantized BF16, fake quantization only):
  four_over_six  quantize.quantizer.quant_nvfp4_4over6(w, 4, 16)           baseline of every MixFP4 map
  e0m3           quant_mix_4_6(w, 4, 16, type_block=(8,64), clip='a1', elect='always')
  nvfp4          quant_nvfp4(w, 4, 16)
  nover6         quant_nvfp4_nover6(w, 4, 16)                              (9 alphas in [1,1.5])
  razer_e3m3     quant_nvfp4_razer_e3m3(w, 4, 16, outlier=8.0)             (released RaZeR weight format)

Activations:
  four_over_six_rows  quantize.causal_four_over_six.quantize_rows          aligned protocol (causal, per token)
  nvfp4_rows / nover6_rows / razer_e4m3_rows   per-token-global-scale versions of the tensor-wide quantizers
  four_over_six_tensor / nvfp4_tensor           historical tensor-wide (non-causal) evaluation convention
"""
import functools

import torch

from quantize.causal_four_over_six import quantize_rows
from quantize.quantizer import (NOVER6_ALPHAS, _quant_e2m1, quant_mix_4_6, quant_nvfp4, quant_nvfp4_4over6,
                                quant_nvfp4_nover6, quant_nvfp4_razer_e3m3, quant_nvfp4_razer_e4m3)


class CandidateError(ValueError):
    pass


def _guard(fn):
    """Reject misaligned/non-finite weights before and non-finite candidates after quantization.

    The released quantizers flatten with reshape(-1, 16); a (O, K) matrix with K % 16 != 0 but
    O*K % 16 == 0 would silently form scale blocks that straddle rows, so it is refused here."""
    @functools.wraps(fn)
    def wrapped(w):
        if w.ndim != 2:
            raise CandidateError(f'{fn.__name__}: expected a 2-D weight, got shape {tuple(w.shape)}')
        if w.shape[1] % 16:
            raise CandidateError(f'{fn.__name__}: in_features {w.shape[1]} not divisible by the 16-element scale block')
        if not torch.isfinite(w).all():
            raise CandidateError(f'{fn.__name__}: non-finite input weight')
        if not bool(w.detach().abs().amax() > 0):
            raise CandidateError(f'{fn.__name__}: all-zero weight has an undefined global scale')
        out = fn(w)
        if out.shape != w.shape or out.dtype != torch.bfloat16 or not torch.isfinite(out).all():
            raise CandidateError(f'{fn.__name__}: invalid candidate (shape/dtype/non-finite)')
        return out
    return wrapped


@_guard
def four_over_six(w):
    return quant_nvfp4_4over6(w, 4, 16)


@_guard
def e0m3(w):
    return quant_mix_4_6(w, 4, 16, type_block=(8, 64), clip='a1', elect='always')


@_guard
def nvfp4(w):
    return quant_nvfp4(w, 4, 16)


@_guard
def nover6(w):
    return quant_nvfp4_nover6(w, 4, 16)


@_guard
def razer_e3m3(w):
    return quant_nvfp4_razer_e3m3(w, 4, 16, outlier=8.0)


WEIGHT = dict(four_over_six=four_over_six, e0m3=e0m3, nvfp4=nvfp4, nover6=nover6, razer_e3m3=razer_e3m3)


@torch.no_grad()
def nvfp4_rows(x):
    """quant_nvfp4 with the FP32 global scale computed per token row (causal)."""
    shape = x.shape
    if shape[-1] % 16:
        raise ValueError('Input width must be divisible by 16')
    b = x.float().reshape(-1, shape[-1] // 16, 16)
    gs = (b.abs().amax((1, 2), keepdim=True) / (6.0 * 448)).clamp_min(torch.finfo(torch.float32).tiny)
    s = b / gs
    bmax = s.abs().amax(-1, keepdim=True)
    scale = (bmax / 6.0).clamp(max=448, min=2 ** (-9)).to(torch.float8_e4m3fn).to(s.dtype)
    return (_quant_e2m1(s, scale) * gs).reshape(shape).bfloat16()


@torch.no_grad()
def nover6_rows(x, alphas=NOVER6_ALPHAS):
    shape = x.shape
    if shape[-1] % 16:
        raise ValueError('Input width must be divisible by 16')
    b = x.float().reshape(-1, shape[-1] // 16, 16)
    gs = (b.abs().amax((1, 2), keepdim=True) / (6.0 * 448)).clamp_min(torch.finfo(torch.float32).tiny)
    s = b / gs
    bmax = s.abs().amax(-1, keepdim=True)
    best, err = None, None
    for a in alphas:
        scale = (bmax * (a / 6.0)).clamp(max=448.0, min=2 ** (-9)).to(torch.float8_e4m3fn).to(s.dtype)
        dq = _quant_e2m1(s, scale)
        e = (dq - s).pow(2).sum(-1, keepdim=True)
        if best is None:
            best, err = dq, e
        else:
            take = e < err
            best, err = torch.where(take, dq, best), torch.where(take, e, err)
    return (best * gs).reshape(shape).bfloat16()


@torch.no_grad()
def razer_e4m3_rows(x):
    """Vectorized per-token version of the released quant_nvfp4_razer_e4m3.

    Identical arithmetic to the released function applied to each token row on its own
    (tested row-by-row), including the released selection expression
    `(w_q_razer_tmp - block_scale_q).pow(2).mean(-1)`, which compares codes with the block
    scale rather than with the scaled input. That expression is preserved, not corrected,
    because the matched baseline must not change the released algorithm; it is documented
    in the baseline audit (V80)."""
    shape = x.shape
    if shape[-1] % 16:
        raise ValueError('Input width must be divisible by 16')
    b = x.float().reshape(-1, shape[-1] // 16, 16)
    gs = b.abs().amax((1, 2), keepdim=True) / (6.0 * 448)
    s = b / gs
    bmax = s.abs().amax(-1, keepdim=True)
    bsq = (bmax / 6.0).clamp(max=448, min=2 ** (-9)).to(torch.float8_e4m3fn).to(s.dtype)
    ws = s / bsq
    exp = torch.floor(torch.log2(torch.abs(ws) + (ws == 0).type(ws.dtype))).clamp(min=0)
    wm = ws / (2 ** exp) * 2
    wm = torch.sign(wm) * torch.floor(torch.abs(wm) + 0.5)
    wq_fp4 = wm * (2 ** exp) / 2
    error = torch.full(ws.shape[:-1], float('inf'), dtype=ws.dtype, device=ws.device)
    wq = torch.zeros_like(ws)
    for special in (-5.0, 5.0):
        tmp = torch.where((ws - wq_fp4).abs() < (ws - special).abs(), wq_fp4, torch.full_like(ws, special))
        qe = (tmp - bsq).pow(2).mean(-1)
        upd = qe < error
        error = torch.where(upd, qe, error)
        wq = torch.where(upd[..., None], tmp, wq)
    return (wq * bsq * gs).reshape(shape).bfloat16()


def four_over_six_tensor(x):
    return quant_nvfp4_4over6(x, 4, 16)


def nvfp4_tensor(x):
    return quant_nvfp4(x, 4, 16)


ACTIVATION = dict(four_over_six_rows=quantize_rows, nvfp4_rows=nvfp4_rows, nover6_rows=nover6_rows,
                  razer_e4m3_rows=razer_e4m3_rows, four_over_six_tensor=four_over_six_tensor, nvfp4_tensor=nvfp4_tensor)


ROW_WISE = ('four_over_six_rows', 'nvfp4_rows', 'nover6_rows', 'razer_e4m3_rows')


def chunked_rows(fn, x, max_rows=4096):
    """Apply a row-independent quantizer over token-row chunks. Bitwise identical to fn(x) (rows never interact);
    bounds the temporary memory of the quantizer for long batched prompts."""
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    if flat.shape[0] <= max_rows:
        return fn(x)
    return torch.cat([fn(flat[i:i + max_rows]) for i in range(0, flat.shape[0], max_rows)]).reshape(shape)


class ActivationQuant:
    """Forward pre-hooks that quantize every scoped Linear input. `ste=True` passes identity gradients."""

    def __init__(self, modules, kind='four_over_six_rows', ste=False, max_rows=4096):
        if kind not in ACTIVATION:
            raise ValueError(kind)
        self.kind, self.fn, self.ste = kind, ACTIVATION[kind], ste
        self.max_rows = max_rows if kind in ROW_WISE else None
        self.handles = [m.register_forward_pre_hook(self._pre) for m in modules.values()]

    def _pre(self, module, inputs):
        x = inputs[0]
        q = chunked_rows(self.fn, x.detach(), self.max_rows) if self.max_rows else self.fn(x.detach())
        if q.dtype != x.dtype:  # BF16 models: no-op; float32 diagnostics: keep the model dtype
            q = q.to(x.dtype)
        if self.ste:
            q = q + (x - x.detach())
        return (q, *inputs[1:])

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def tensor_sha256(t):
    import hashlib
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
