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
from .lib import Kernel, LibraryError
from .linear import NativeLinear
from .select import KernelSet


def _deployed(family, paper):
    """The adopted kernel-opt set if its builds are in the build directory, else the paper set; (set, note)."""
    try:
        return KernelSet(family), f'{family} (kernel-opt adoption, amendment 7)'
    except LibraryError as e:
        return KernelSet(paper), f'{family} not built ({e}) -> {paper} (paper set)'


def _deployed_wB():
    """kernel-opt 8x64 plan P6 (decision b): the adopted 8x64 set 'mixed_wB_ko' (amendments 11-12b: t0, the adopted
    table's 8x64 widths) if its builds are in the build directory, else the paper kernel n8k64_wB; (kernel, note). The
    dispatch variant is the build directory's: the adopted one is #2's (MIXFP4_DISPATCH_FREQ=1), and the note says
    which the loaded builds carry."""
    try:
        ks = KernelSet('mixed_wB_ko')
    except LibraryError as e:
        return Kernel.load('n8k64_wB'), f'mixed_wB_ko not built ({e}) -> n8k64_wB (paper kernel)'
    freq = all((k.manifest.get('extra_defines') or {}).get('MIXFP4_DISPATCH_FREQ') == 1 for k in ks.kernels.values())
    return ks, ('mixed_wB_ko (kernel-opt 8x64 adoption, amendments 11-12b; '
                + ("#2's dispatch" if freq else "default dispatch: this build directory lacks MIXFP4_DISPATCH_FREQ=1") + ')')


def resolve_kernel(kernel, meta=None):
    """Returns (kernel, routing note or None).

    'auto' -> the deployed mixed set for the artifact's map unit (artifact.map_unit):
      - 8x64 (rows not a multiple of 16; the weights-on-B family): 'mixed_wB_ko', the kernel-opt 8x64 adoption
        (amendments 11-12b: t0, the adopted table's 8x64 widths, #2's dispatch from its build directory), else the paper
        kernel n8k64_wB (the same outputs bit for bit).
      - 16x64 (and finer): 'mixed_ko', the kernel-opt adoption of 2026-10-01 (amendment 7): no site-0 prmt tags (t0),
        #4's 64 x 64 epilogue tile at width 128, the 4b widths and per-call scheduler rows (the '<gpu>.ko.json' table).
        If its builds are not in the build directory, the paper 'mixed' set (the same outputs bit for bit).
      - whole 128-row panels (256x64): the 4-arm 'mixed256' set (kernel-opt A', adopted 2026-09-30) on the paper
        table if built, else the paper 'mixed' set -- unchanged by amendment 7 (no 256x64 work for now).
    'auto_stock' -> the stock NVFP4 set tuned the same way: 'stock_ko' (#4 + 4b), else the paper 'stock'.
    'auto_mixed' -> the 16x64 set of 'auto' for any map; 'auto_wB' -> the 8x64 kernel of 'auto' for any map;
    'auto_256' -> 'mixed256' (NativeLinear verifies the tags are uniform over its 128-row panels).
    'paper_mixed' / 'paper_stock' / 'paper_256' -> the paper sets 'mixed' / 'stock' / 'mixed256' with the paper table;
    'paper_wB' -> the paper's 8x64 kernel n8k64_wB.
    A configuration name -> that single build; Kernel / KernelSet instances pass through."""
    if isinstance(kernel, (Kernel, KernelSet)):
        return kernel, None
    if kernel == 'auto':
        unit = A.map_unit(meta) if meta is not None else None
        if unit is not None and unit[0] % 16:
            k, note = _deployed_wB()
            return k, f'auto: {unit[0]}x{unit[1]} map -> {note}'
        if unit is not None and unit[0] % 128 == 0:
            try:
                return KernelSet('mixed256'), f'auto: {unit[0]}x{unit[1]} map -> mixed256 (kernel-opt A\')'
            except LibraryError as e:
                return KernelSet('mixed'), f'auto: {unit[0]}x{unit[1]} map, mixed256 not built ({e}) -> mixed'
        ks, note = _deployed('mixed_ko', 'mixed')
        return ks, f'auto -> {note}'
    if kernel == 'auto_mixed':
        ks, note = _deployed('mixed_ko', 'mixed')
        return ks, f'auto_mixed -> {note}'
    if kernel == 'auto_stock':
        ks, note = _deployed('stock_ko', 'stock')
        return ks, f'auto_stock -> {note}'
    if kernel == 'auto_wB':
        k, note = _deployed_wB()
        return k, f'auto_wB -> {note}'
    if kernel == 'paper_wB':
        return Kernel.load('n8k64_wB'), None
    if kernel == 'auto_256':
        return KernelSet('mixed256'), None     # 256x64 maps, 32-row granules, 4 arms (kernel-opt A')
    if kernel in ('paper_mixed', 'paper_stock', 'paper_256'):
        return KernelSet({'paper_mixed': 'mixed', 'paper_stock': 'stock', 'paper_256': 'mixed256'}[kernel]), None
    return Kernel.load(kernel), None


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
    activation_quantizer: str | None = None
    activation_quantizer_override: bool = False
    routing: str | None = None

    def as_dict(self):
        return dict(kernel=self.kernel, kernel_sha256=self.kernel_sha256,
                    artifact_weights_sha256=self.artifact_weights_sha256, map_sha256=self.map_sha256,
                    native_modules=len(self.native), fallback=self.fallback, bf16_by_design=self.bf16_by_design,
                    e0m3_tiles=self.e0m3_tiles, device_bytes=self.device_bytes, kernel_set=self.kernel_set,
                    activation_quantizer=self.activation_quantizer,
                    activation_quantizer_override=self.activation_quantizer_override, routing=self.routing)


@torch.no_grad()
def install(model, art_dir, kernel='auto', loader='causal_lm', strict=True, device='cuda', activation_quantizer=None):
    """Replace every scoped Linear by a NativeLinear from the artifact. Returns an InstallReport.

    Modules that already are NativeLinears (a previous install) are replaced as well, so artifacts
    can be swapped on one loaded model.

    activation_quantizer: None (the artifact's own, the calibrated configuration) or an explicit override
    ('nvfp4_rows' / 'four_over_six_rows'). An override is for latency measurements only: the weights were calibrated
    with the artifact's activation quantizer, so an overridden install is not an evaluated configuration. The report
    records it (activation_quantizer_override=True)."""
    meta, weights = A.load(art_dir, device=device)
    kern, routing = resolve_kernel(kernel, meta)
    act_kind = meta['activation_quantizer'] if activation_quantizer is None else activation_quantizer
    if act_kind not in ('nvfp4_rows', 'four_over_six_rows'):
        raise ValueError(f'unknown activation quantizer {act_kind!r}')
    name = kern.cfg.name if isinstance(kern, Kernel) else f'KernelSet({kern.family})'
    rep = InstallReport(name, kern.sha256, meta['weights_sha256'], (meta.get('map') or {}).get('sha256'))
    rep.kernel_set = kern.describe() if isinstance(kern, KernelSet) else None
    rep.routing = routing
    rep.activation_quantizer = act_kind
    rep.activation_quantizer_override = act_kind != meta['activation_quantizer']
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
