"""NativeLinear: an nn.Linear replacement that runs the patched SM120 mixed FP4 GEMM.

forward(x) = two launches issued by ONE ctypes call (sm120_linear), no other device work:
  1. the per-token quantizer (csrc/quant_act.cuh; the Triton quant_act.quantize is the fallback for
     libraries built without it -- both are bit-identical to the reference): FP32 global scale per
     token, E2M1 codes packed two per byte, UE4M3 scale bytes written directly into the kernel's
     scale-factor layout (padding zeroed in-kernel);
  2. the GEMM with the fused epilogue
        weights on A (n16k64_wA):  D[out, t] = bf16((gs_w * gs_x[t]) * acc + bias[out]),
        D column-major, i.e. the row-major [t, out] tensor returned as is (no transpose, no copy);
        weights on B (n8k64_wB):   D[t, out] = bf16((gs_x[t] * gs_w) * acc + bias[out]), row-major.
The activation scale, the weight global scale and the bias are applied in FP32 before the single
rounding to bf16 (see NUMERICS.md for how this differs from the fake-quant order).
"""
import torch

from . import quant_act
from .artifact import PackedWeight
from .lib import Kernel, sf_buffer_size, sf_offset_formula
from .select import KernelSet


def place_scales(scales, k):
    """Row-major [rows, k/16] scale bytes -> the kernel's block-scaled layout (zero padding)."""
    rows = scales.shape[0]
    buf = torch.zeros(sf_buffer_size(rows, k), dtype=torch.uint8, device=scales.device)
    r = torch.arange(rows, device=scales.device)[:, None]
    kb = torch.arange(k // 16, device=scales.device)[None, :]
    buf[sf_offset_formula(r, kb, k).reshape(-1)] = scales.reshape(-1)
    return buf


class NativeLinear(torch.nn.Module):
    """Holds only packed weights; `calls` / `tokens` count native GEMMs for coverage evidence."""

    def __init__(self, pw: PackedWeight, kernel, act_kind: str, name: str = ''):
        """kernel: a lib.Kernel, or a select.KernelSet choosing the CTA-tile width per call."""
        super().__init__()
        n, k = pw.shape
        if k % 32:
            raise ValueError(f'{name}: in_features {k} is not a multiple of 32')
        if pw.type_block is not None:
            if kernel.type_block is None:
                raise ValueError(f'{name}: {kernel.cfg.name} cannot execute E0M3 tiles')
            if pw.type_block[0] % kernel.type_block[0] or pw.type_block[1] % kernel.type_block[1]:
                raise ValueError(f'{name}: map tile {pw.type_block} is not a union of kernel granules {kernel.type_block}')
        elif bool((pw.scales >> 7).any()):
            raise ValueError(f'{name}: E0M3 tags without a type block')
        self.name, self.kernel, self.act_kind = name, kernel, act_kind
        self.in_features, self.out_features = k, n
        self.global_scale = float(pw.global_scale)
        self.e0m3_tiles = pw.e0m3_tiles
        self.register_buffer('packed', pw.packed.contiguous(), persistent=False)
        self.register_buffer('sf', place_scales(pw.scales, k), persistent=False)
        self.register_buffer('bias_bf16', None if pw.bias is None else pw.bias.to(torch.bfloat16).contiguous(),
                             persistent=False)
        self.weights_on_a = kernel.weight_operand == 0
        self.kernel_set = kernel if isinstance(kernel, KernelSet) else None
        self.quant_mode = Kernel.QUANT_MODES[act_kind]
        self.fused = True          # use sm120_linear when the selected library provides it
        self.calls = 0
        self.tokens = 0

    @property
    def weight(self):
        raise AttributeError(f'{self.name}: NativeLinear holds packed FP4 weights only; a caller tried to read a '
                             f'BF16 weight (this would be an undocumented fallback)')

    @property
    def bias(self):
        return self.bias_bf16

    def extra_repr(self):
        return (f'in={self.in_features}, out={self.out_features}, kernel={self.kernel.cfg.name}, '
                f'act={self.act_kind}, e0m3_tiles={self.e0m3_tiles}')

    def forward(self, x):
        k, n = self.in_features, self.out_features
        if x.shape[-1] != k:
            raise ValueError(f'{self.name}: input width {x.shape[-1]} != {k}')
        lead = x.shape[:-1]
        x2 = x.reshape(-1, k)
        t = x2.shape[0]
        self.calls += 1
        self.tokens += t
        if t == 0:
            return x.new_empty((*lead, n))
        kern = self.kernel if self.kernel_set is None else self.kernel_set.pick(n, k, t)
        if self.fused and kern.has_quant:
            if x2.dtype != torch.bfloat16 or x2.stride(1) != 1:
                x2 = x2.to(torch.bfloat16).contiguous()
            dev = x2.device
            # one scratch allocation for the quantized activation: packed codes | scale bytes | gs,
            # each 256-byte aligned (TMA needs 16-byte aligned global addresses). Stream-ordered reuse
            # by the caching allocator is safe: both kernels run before any later user on this stream.
            off_sf = (t * k // 2 + 255) // 256 * 256
            off_gs = (off_sf + sf_buffer_size(t, k) + 255) // 256 * 256
            scratch = torch.empty(off_gs + 4 * t, dtype=torch.uint8, device=dev)
            base = scratch.data_ptr()
            y = torch.empty((t, n), dtype=torch.bfloat16, device=dev)
            ws = kern.workspace(n, t, k, dev) if self.weights_on_a else kern.workspace(t, n, k, dev)
            bias = self.bias_bf16
            rc = kern.lib.sm120_linear(x2.data_ptr(), x2.stride(0), t, k, self.quant_mode, base,
                                       base + off_sf, base + off_gs, self.packed.data_ptr(), self.sf.data_ptr(),
                                       self.global_scale, None if bias is None else bias.data_ptr(), n, y.data_ptr(),
                                       None if ws is None else ws.data_ptr(), 0 if ws is None else ws.numel(),
                                       torch.cuda.current_stream(dev).cuda_stream)
            if rc != 0:
                raise RuntimeError(f'{self.name}: sm120_linear failed with code {rc}')
            y = y.view(*lead, n)
            return y if y.dtype == x.dtype else y.to(x.dtype)
        packed_x, sf_x, gs_x = quant_act.quantize(x2, self.act_kind)
        if self.weights_on_a:
            y = kern.gemm(self.packed, self.sf, packed_x, sf_x, n, t, k,
                          scale_m_default=self.global_scale, scale_n=gs_x, bias=self.bias_bf16, check=False)
        else:
            y = kern.gemm(packed_x, sf_x, self.packed, self.sf, t, n, k,
                          scale_m=gs_x, scale_n_default=self.global_scale, bias=self.bias_bf16, check=False)
        y = y.view(*lead, n)
        return y if y.dtype == x.dtype else y.to(x.dtype)
