"""Install an artifact's NativeLinears into a loaded Hugging Face model, with a coverage report.

Scope rule (the selector campaign's, campaign/models.py `scope`): every text nn.Linear except the
output head. The head, embeddings, norms, attention (SDPA) and any non-Linear op stay BF16; that
is the documented, intended precision boundary, not a fallback.

There is no silent fallback: a scoped Linear that the artifact does not cover, or cannot run on
the chosen kernel, raises (strict=True, the default). With strict=False such modules are left in
BF16 and listed under `fallback` in the report, with the reason.
"""
from dataclasses import dataclass, field

import torch

from . import artifact as A
from .lib import Kernel
from .linear import NativeLinear
from .select import KernelSet


def resolve_kernel(kernel):
    """'auto' -> the mixed KernelSet with this GPU's tile table; 'auto_stock' -> the stock NVFP4 set;
    a configuration name -> that single build; Kernel / KernelSet instances pass through."""
    if isinstance(kernel, (Kernel, KernelSet)):
        return kernel
    if kernel in ('auto', 'auto_mixed'):
        return KernelSet('mixed')
    if kernel == 'auto_stock':
        return KernelSet('stock')
    return Kernel.load(kernel)


def scope(model, loader='causal_lm'):
    if loader == 'qwen3_5_conditional':
        return {n: m for n, m in model.named_modules()
                if isinstance(m, torch.nn.Linear) and 'language_model' in n and 'head' not in n}
    head = model.get_output_embeddings()
    return {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not head}


def _set_module(model, name, new):
    parent_name, _, child = name.rpartition('.')
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, new)


@dataclass
class InstallReport:
    kernel: str
    kernel_sha256: str
    artifact_weights_sha256: str
    map_sha256: str | None
    native: list = field(default_factory=list)
    fallback: list = field(default_factory=list)      # (name, reason)
    bf16_by_design: list = field(default_factory=list)
    e0m3_tiles: int = 0
    device_bytes: dict = field(default_factory=lambda: dict(packed=0, scales_placed=0, scales_padding=0, bias=0))
    kernel_set: dict | None = None

    def as_dict(self):
        return dict(kernel=self.kernel, kernel_sha256=self.kernel_sha256,
                    artifact_weights_sha256=self.artifact_weights_sha256, map_sha256=self.map_sha256,
                    native_modules=len(self.native), fallback=self.fallback, bf16_by_design=self.bf16_by_design,
                    e0m3_tiles=self.e0m3_tiles, device_bytes=self.device_bytes, kernel_set=self.kernel_set)


@torch.no_grad()
def install(model, art_dir, kernel='auto', loader='causal_lm', strict=True, device='cuda'):
    """Replace every scoped Linear by a NativeLinear from the artifact. Returns an InstallReport.

    Modules that already are NativeLinears (a previous install) are replaced as well, so artifacts
    can be swapped on one loaded model."""
    kern = resolve_kernel(kernel)
    meta, weights = A.load(art_dir, device=device)
    act_kind = meta['activation_quantizer']
    name = kern.cfg.name if isinstance(kern, Kernel) else f'KernelSet({kern.family})'
    rep = InstallReport(name, kern.sha256, meta['weights_sha256'], (meta.get('map') or {}).get('sha256'))
    rep.kernel_set = kern.describe() if isinstance(kern, KernelSet) else None
    mods = dict(scope(model, loader))
    mods.update(native_modules(model))
    mods = {n: m for n, m in model.named_modules() if n in mods}      # model order
    head = model.get_output_embeddings()
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and n not in mods:
            rep.bf16_by_design.append(n if m is not head else f'{n} (output head)')
    missing = [n for n in mods if n not in weights]
    extra = [n for n in weights if n not in mods]
    if extra:
        raise A.ArtifactError(f'artifact modules not in the model scope: {extra[:5]}')
    for n in missing:
        if strict:
            raise A.ArtifactError(f'scoped Linear {n} is not in the artifact (strict install)')
        rep.fallback.append((n, 'not in artifact'))
    for n, pw in weights.items():
        lin = mods[n]
        shape = (lin.out_features, lin.in_features)
        if shape != tuple(pw.shape):
            raise A.ArtifactError(f'{n}: model weight {shape} != artifact {pw.shape}')
        if (lin.bias is None) != (pw.bias is None):
            raise A.ArtifactError(f'{n}: bias presence differs between model and artifact')
        try:
            nl = NativeLinear(pw, kern, act_kind, name=n)
        except ValueError as e:
            if strict:
                raise
            rep.fallback.append((n, str(e)))
            continue
        _set_module(model, n, nl)
        rep.native.append(n)
        rep.e0m3_tiles += pw.e0m3_tiles
        rep.device_bytes['packed'] += nl.packed.numel()
        rep.device_bytes['scales_placed'] += nl.sf.numel()
        rep.device_bytes['scales_padding'] += nl.sf.numel() - pw.scales.numel()
        rep.device_bytes['bias'] += 0 if nl.bias_bf16 is None else nl.bias_bf16.numel() * 2
        del lin
    torch.cuda.empty_cache()
    return rep


def native_modules(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, NativeLinear)}


def reset_counters(model):
    for m in native_modules(model).values():
        m.calls = 0
        m.tokens = 0


def coverage(model):
    """Per-forward evidence: which NativeLinears ran, and that no scoped nn.Linear remains."""
    nat = native_modules(model)
    head = model.get_output_embeddings()
    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m is not head]
    return dict(native=len(nat), native_called=sum(1 for m in nat.values() if m.calls > 0),
                native_not_called=[n for n, m in nat.items() if m.calls == 0][:20],
                calls=sum(m.calls for m in nat.values()), tokens=sum(m.tokens for m in nat.values()),
                remaining_bf16_linears=linears)
