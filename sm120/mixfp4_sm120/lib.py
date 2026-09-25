"""ctypes binding to one built kernel library (sm120/build.py), with load-time verification.

A library is only loaded through its build manifest: the file's SHA-256 must equal the manifest's,
the compiled description must match configs.py, and a mixed library must have been patched. This
is what prevents silently running an unpatched binary (which computes E2M1 wherever E0M3 was
requested and returns plausible numbers).
"""
import ctypes
import hashlib
import json
import os
import threading
from pathlib import Path

import torch

from . import configs as CFG

BUILD_ROOT = Path(os.environ.get('SM120_BUILD_DIR', Path(__file__).resolve().parents[1] / 'build'))

_c_int, _c_i64, _c_ptr, _c_size, _c_float = ctypes.c_int, ctypes.c_int64, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_float


class LibraryError(RuntimeError):
    pass


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sf_offset_formula(rows_index, kb_index, k):
    """Byte offset of scale factor (row r, 16-element block kb) in the SM1xx block-scaled layout.

    CUTLASS tiles scale factors in atoms of 128 rows x 4 blocks (64 K), 512 bytes each, row tiles
    outermost: within an atom, row r lands at (r % 32) * 16 + ((r // 32) % 4) * 4 and block kb at
    kb % 4. Verified against the library's own layout function (Kernel.check_sf_formula).
    Works on ints or integer tensors.
    """
    k_atoms = (k // 16 + 3) // 4
    r, kb = rows_index, kb_index
    return ((r // 128) * k_atoms + kb // 4) * 512 + (r % 32) * 16 + ((r // 32) % 4) * 4 + kb % 4


def sf_buffer_size(rows, k):
    return ((rows + 127) // 128) * ((k // 16 + 3) // 4) * 512


class Kernel:
    """One configuration's GEMM: D = bf16((s_m * s_n) * decode(A) decode(B)^T + bias)."""

    _cache = {}
    _lock = threading.Lock()

    @classmethod
    def load(cls, name=CFG.DEFAULT, build_root=None, allow_unpatched=False):
        key = (name, str(build_root), allow_unpatched)
        with cls._lock:
            k = cls._cache.get(key)
            if k is None:
                k = cls(name, build_root, allow_unpatched)
                cls._cache[key] = k
            return k

    def __init__(self, name, build_root=None, allow_unpatched=False):
        self.cfg = CFG.get(name)
        root = Path(build_root or BUILD_ROOT) / name
        mpath = root / 'manifest.json'
        if not mpath.exists():
            raise LibraryError(f'no build manifest at {mpath}; run: python sm120/build.py --config {name}')
        self.manifest = json.loads(mpath.read_text())
        path = root / self.manifest['library']
        want = self.manifest['library_sha256']
        if allow_unpatched:
            # Negative controls only: the unpatched binary of the same build.
            path = Path(f'{path}.unpatched')
            want = self.manifest['unpatched_sha256']
        got = _sha256(path)
        if got != want:
            raise LibraryError(f'{path}: sha256 {got} != manifest {want} (stale or modified library)')
        if self.cfg.mixed and self.cfg.type_block is not None and not allow_unpatched:
            if not self.manifest['patch'].get('patched'):
                raise LibraryError(f'{name}: manifest says the library was not patched')
        self.path = path
        self.patched = not allow_unpatched and bool(self.manifest['patch'].get('patched'))
        self.sha256 = got
        lib = ctypes.CDLL(str(path))
        lib.sm120_describe.argtypes, lib.sm120_describe.restype = [ctypes.c_char_p, _c_int], _c_int
        lib.sm120_sf_size.argtypes, lib.sm120_sf_size.restype = [_c_int, _c_int, _c_int, _c_int], _c_i64
        lib.sm120_sf_offsets.argtypes, lib.sm120_sf_offsets.restype = [_c_int, _c_int, _c_int, _c_int, _c_ptr], _c_int
        lib.sm120_granule_map.argtypes, lib.sm120_granule_map.restype = [_c_int, _c_ptr, _c_int], _c_int
        lib.sm120_workspace_size.argtypes, lib.sm120_workspace_size.restype = [_c_int, _c_int, _c_int], _c_size
        lib.sm120_gemm.argtypes = [_c_ptr, _c_ptr, _c_ptr, _c_ptr, _c_ptr, _c_int, _c_int, _c_int,
                                   _c_ptr, _c_float, _c_ptr, _c_float, _c_ptr, _c_ptr, _c_size, _c_ptr]
        lib.sm120_gemm.restype = _c_int
        self.lib = lib
        buf = ctypes.create_string_buffer(4096)
        lib.sm120_describe(buf, 4096)
        self.desc = json.loads(buf.value.decode())
        if self.desc != self.manifest['compiled_description']:
            raise LibraryError(f'{name}: loaded library describes itself as {self.desc}, manifest says '
                               f'{self.manifest["compiled_description"]}')
        self.weight_operand = self.cfg.weight_operand
        self.act_operand = 1 - self.weight_operand
        self.d_colmajor = bool(self.desc['d_colmajor'])
        self.type_block = self.cfg.type_block
        self._ws = {}
        self._offsets = {}

    # -------------------------------------------------------------------------------- layouts

    def sf_size(self, operand, m, n, k):
        return int(self.lib.sm120_sf_size(operand, m, n, k))

    def sf_offsets(self, operand, m, n, k):
        """int64 [rows, k/16] byte offsets of every scale factor in the kernel's layout (host)."""
        key = (operand, m, n, k)
        hit = self._offsets.get(key)
        if hit is None:
            rows = m if operand == 0 else n
            hit = torch.empty((rows, k // 16), dtype=torch.int64)
            if self.lib.sm120_sf_offsets(operand, m, n, k, ctypes.c_void_p(hit.data_ptr())) != 0:
                raise LibraryError('sm120_sf_offsets failed')
            self._offsets[key] = hit
        return hit

    def check_sf_formula(self, shapes):
        """Assert sf_offset_formula / sf_buffer_size reproduce the library's layout on `shapes`."""
        for m, n, k in shapes:
            for operand in (0, 1):
                rows = m if operand == 0 else n
                lib_off = self.sf_offsets(operand, m, n, k)
                r = torch.arange(rows)[:, None]
                kb = torch.arange(k // 16)[None, :]
                if not torch.equal(sf_offset_formula(r, kb, k), lib_off):
                    raise LibraryError(f'scale-factor layout formula differs from the kernel for {(operand, m, n, k)}')
                if sf_buffer_size(rows, k) != self.sf_size(operand, m, n, k):
                    raise LibraryError(f'scale-factor buffer size differs for {(operand, m, n, k)}')

    def granule_map(self, operand):
        buf = (ctypes.c_int * 4096)()
        n = self.lib.sm120_granule_map(operand, buf, 4096)
        if n < 0:
            raise LibraryError('granule map buffer too small')
        return list(buf[:n])

    def workspace(self, m, n, k, device):
        key = (m, n, k, device)
        ws = self._ws.get(key)
        if ws is None:
            size = int(self.lib.sm120_workspace_size(m, n, k))
            ws = torch.empty(max(size, 16), dtype=torch.uint8, device=device) if size else None
            self._ws[key] = ws
        return ws

    # -------------------------------------------------------------------------------- GEMM

    def gemm(self, a, sfa, b, sfb, m, n, k, *, scale_m=None, scale_m_default=1.0, scale_n=None,
             scale_n_default=1.0, bias=None, out=None, check=True):
        """D = bf16((s_m[m] * s_n[n]) * decode(A) @ decode(B)^T + bias).

        a: uint8 [m, k/2], b: uint8 [n, k/2] (element 2j in the low nibble); sfa / sfb: placed scale
        bytes (bit 7 = E0M3 tag). Returns D as a [m, n] tensor, or for column-major-D builds as its
        row-major transpose [n, m] (the same memory)."""
        dev = a.device
        if check:
            for name, t in (('a', a), ('sfa', sfa), ('b', b), ('sfb', sfb)):
                if not (t.is_cuda and t.dtype == torch.uint8 and t.is_contiguous()):
                    raise ValueError(f'{name} must be a contiguous CUDA uint8 tensor')
            if a.shape != (m, k // 2) or b.shape != (n, k // 2):
                raise ValueError(f'operand shapes {tuple(a.shape)}, {tuple(b.shape)} do not match m={m} n={n} k={k}')
            if sfa.numel() < sf_buffer_size(m, k) or sfb.numel() < sf_buffer_size(n, k):
                raise ValueError('scale-factor buffer too small')
            if k % 32 or m <= 0 or n <= 0:
                raise ValueError(f'unsupported problem m={m} n={n} k={k} (K must be a multiple of 32)')
            for name, t, size in (('scale_m', scale_m, m), ('scale_n', scale_n, n)):
                if t is not None and not (t.is_cuda and t.dtype == torch.float32 and t.is_contiguous() and t.numel() >= size):
                    raise ValueError(f'{name} must be a contiguous CUDA float32 vector of length >= {size}')
            if bias is not None:
                blen = n if self.weight_operand == 1 else m
                if not (bias.is_cuda and bias.dtype == torch.bfloat16 and bias.is_contiguous() and bias.numel() == blen):
                    raise ValueError(f'bias must be a contiguous CUDA bf16 vector of length {blen}')
        shape = (n, m) if self.d_colmajor else (m, n)
        if out is None:
            out = torch.empty(shape, dtype=torch.bfloat16, device=dev)
        elif check and (out.shape != shape or out.dtype != torch.bfloat16 or not out.is_contiguous()):
            raise ValueError(f'out must be a contiguous bf16 {shape} tensor')
        ws = self.workspace(m, n, k, dev)
        stream = torch.cuda.current_stream(dev).cuda_stream
        ptr = lambda t: None if t is None else t.data_ptr()  # noqa: E731
        rc = self.lib.sm120_gemm(ptr(a), ptr(sfa), ptr(b), ptr(sfb), ptr(out), m, n, k,
                                 ptr(scale_m), float(scale_m_default), ptr(scale_n), float(scale_n_default),
                                 ptr(bias), ptr(ws), 0 if ws is None else ws.numel(), stream)
        if rc != 0:
            raise LibraryError(f'sm120_gemm({m}, {n}, {k}) failed with code {rc}')
        return out
