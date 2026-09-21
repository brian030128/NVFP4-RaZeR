"""Independent scalar reference quantizers (numpy float32 scalars + explicit rounding rules).

Nothing here imports the vectorized quantizers. Floating-point steps that are exact IEEE
operations in float32 (division, multiplication) are replicated with numpy float32 scalars;
format rounding (UE4M3 scale, E2M1/E0M3 codes, BF16 output) is implemented from first principles.
The only non-replicable step is the order of a 16-term float32 error sum used for the
FourOverSix alpha choice; near-ties within a relative 1e-5 band are reported as ambiguous.
"""
import math

import numpy as np

F32 = np.float32
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MIDS = tuple((E2M1_LEVELS[i] + E2M1_LEVELS[i + 1]) / 2 for i in range(7))


def e4m3_values():
    vals = set()
    for e in range(16):
        for m in range(8):
            if e == 15 and m == 7:
                continue  # NaN encoding in float8_e4m3fn
            v = (m / 8.0) * 2.0 ** (-6) if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
            vals.add((v, m))
    return sorted(vals)


_E4M3 = e4m3_values()
_E4M3_V = [v for v, _ in _E4M3]


def round_e4m3(x):
    """Nearest float8_e4m3fn value, ties to even mantissa. Input is non-negative and <= 448."""
    x = float(x)
    if x <= 0:
        return 0.0
    import bisect
    i = bisect.bisect_left(_E4M3_V, x)
    if i < len(_E4M3_V) and _E4M3_V[i] == x:
        return x
    lo, hi = _E4M3[i - 1], _E4M3[min(i, len(_E4M3) - 1)]
    dlo, dhi = x - lo[0], hi[0] - x
    if dlo < dhi:
        return lo[0]
    if dhi < dlo:
        return hi[0]
    return lo[0] if lo[1] % 2 == 0 else hi[0]


def round_bf16(v):
    """Nearest bfloat16 value (1 sign, 8 exponent, 7 fraction bits), ties to even."""
    v = float(v)
    if v == 0.0 or not math.isfinite(v):
        return v
    e = math.floor(math.log2(abs(v)))
    # guard against log2 rounding at exact powers of two
    if 2.0 ** e > abs(v):
        e -= 1
    if 2.0 ** (e + 1) <= abs(v):
        e += 1
    if e < -126:
        raise ValueError('bf16 subnormal range is outside the tested domain')
    unit = 2.0 ** (e - 7)
    q = abs(v) / unit
    fl = math.floor(q)
    frac = q - fl
    if frac > 0.5 or (frac == 0.5 and fl % 2 == 1):
        fl += 1
    return math.copysign(fl * unit, v)


def clamp_scale(x, lo=2.0 ** -9, hi=448.0):
    return min(max(float(x), lo), hi)


def e2m1_nearest_ties_down(a):
    """FourOverSix bucket rule on a magnitude: level i iff mids[i-1] < a <= mids[i]; above 5 -> 6."""
    for i, mid in enumerate(E2M1_MIDS):
        if a <= mid:
            return E2M1_LEVELS[i]
    return 6.0


def e2m1_nearest_ties_up(a):
    """Ideal rule: nearest E2M1 level, ties away from zero, saturating at 6."""
    best = None
    for lv in E2M1_LEVELS:
        d = abs(a - lv)
        if best is None or d < best[0] or (d == best[0] and lv > best[1]):
            best = (d, lv)
    return min(best[1], 6.0)


def e2m1_released(a):
    """Released `_quant_e2m1`/`quant_nvfp4` arithmetic on a float32 magnitude: exponent
    floor(log2 a) clamped at 0, mantissa x*2/2^exp, float32 floor(|m| + 0.5), saturate at 6.
    Differs from `e2m1_nearest_ties_up` only within one float32 ulp below a tie."""
    a = float(a)
    if a == 0.0:
        return 0.0
    _, e = math.frexp(a)
    exp = max(e - 1, 0)
    xm = F32(F32(a) / F32(2.0 ** exp) * F32(2.0))
    k = math.floor(float(F32(xm + F32(0.5))))
    return min(k * 2.0 ** exp / 2.0, 6.0)


def round_half_even(x):
    fl = math.floor(x)
    frac = x - fl
    if frac > 0.5 or (frac == 0.5 and fl % 2 == 1):
        return fl + 1
    return fl


def _blocks(w):
    w = np.asarray(w, dtype=np.float32)
    if w.shape[-1] % 16:
        raise ValueError('width not divisible by 16')
    return w.reshape(-1, 16)


def ref_four_over_six(w):
    """Returns (dq float64 array of the BF16 output, ambiguous_block_count)."""
    b = _blocks(w)
    amax = F32(np.abs(b).max())
    gs = F32(amax / F32(2688.0))
    out = np.zeros(b.shape, dtype=np.float64)
    ambiguous = 0
    for r in range(b.shape[0]):
        ws = [F32(x / gs) for x in b[r]]
        bmax = max(abs(float(x)) for x in ws)
        cands = []
        for qmax in (6.0, 4.0):
            s = F32(round_e4m3(clamp_scale(F32(F32(bmax) / F32(qmax)), hi=448.0)))
            codes = [math.copysign(e2m1_nearest_ties_down(abs(float(F32(x / s)))), float(x)) if x != 0 else 0.0 for x in ws]
            deq = [F32(F32(c) * s) for c in codes]
            err = sum((float(d) - float(x)) ** 2 for d, x in zip(deq, ws))
            cands.append((codes, s, err))
        (c6, s6, e6), (c4, s4, e4) = cands
        if abs(e4 - e6) <= 1e-5 * max(e4, e6, 1e-30):
            ambiguous += 1
        codes, s = (c4, s4) if e4 < e6 else (c6, s6)
        out[r] = [round_bf16(F32(F32(F32(c) * s) * gs)) for c in codes]
    return out.reshape(np.asarray(w).shape), ambiguous


def ref_e0m3(w):
    b = _blocks(w)
    amax = F32(np.abs(b).max())
    gs = max(F32(amax / F32(2688.0)), F32(np.finfo(np.float32).tiny))
    out = np.zeros(b.shape, dtype=np.float64)
    for r in range(b.shape[0]):
        ws = [F32(x / gs) for x in b[r]]
        bmax = max(abs(float(x)) for x in ws)
        s = F32(round_e4m3(clamp_scale(F32(F32(bmax) * F32(1.0 / 7.0)))))
        codes = [min(max(round_half_even(float(F32(x / s))), -7), 7) for x in ws]
        out[r] = [round_bf16(F32(F32(F32(F32(c) * s)) * gs)) for c in codes]
    return out.reshape(np.asarray(w).shape)


def ref_nvfp4(w):
    b = _blocks(w)
    amax = F32(np.abs(b).max())
    gs = F32(amax / F32(2688.0))
    out = np.zeros(b.shape, dtype=np.float64)
    for r in range(b.shape[0]):
        ws = [F32(x / gs) for x in b[r]]
        bmax = max(abs(float(x)) for x in ws)
        s = F32(round_e4m3(clamp_scale(F32(F32(bmax) / F32(6.0)))))
        codes = [math.copysign(e2m1_released(abs(float(F32(x / s)))), float(x)) if x != 0 else 0.0 for x in ws]
        out[r] = [round_bf16(F32(F32(F32(c) * s) * gs)) for c in codes]
    return out.reshape(np.asarray(w).shape)


def ref_four_over_six_rows(x):
    """Causal per-token FourOverSix activation quantizer (quantize_rows) reference."""
    x = np.asarray(x, dtype=np.float32)
    rows = x.reshape(-1, x.shape[-1])
    out = np.zeros(rows.shape, dtype=np.float64)
    amb = 0
    for i in range(rows.shape[0]):
        row = rows[i]
        blocks = row.reshape(-1, 16)
        gs = max(F32(F32(np.abs(blocks).max()) / F32(2688.0)), F32(np.finfo(np.float32).tiny))
        for j in range(blocks.shape[0]):
            ws = [F32(v / gs) for v in blocks[j]]
            peak = max(abs(float(v)) for v in ws)
            cands = []
            for qmax in (6.0, 4.0):
                s = F32(round_e4m3(clamp_scale(F32(F32(peak) / F32(qmax)))))
                codes = [math.copysign(e2m1_nearest_ties_down(abs(float(F32(v / s)))), float(v)) if v != 0 else 0.0 for v in ws]
                deq = [F32(F32(c) * s) for c in codes]
                cands.append((deq, sum((float(d) - float(v)) ** 2 for d, v in zip(deq, ws))))
            (d6, e6), (d4, e4) = cands
            if abs(e4 - e6) <= 1e-5 * max(e4, e6, 1e-30):
                amb += 1
            deq = d4 if e4 < e6 else d6
            out[i, j * 16:(j + 1) * 16] = [round_bf16(F32(d * gs)) for d in deq]
    return out.reshape(x.shape), amb
