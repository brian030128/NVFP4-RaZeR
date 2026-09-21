"""Type-block (tile) geometry, N8->N16 score aggregation, sufficient statistics and elections.

Conventions (identical to the archived calibration code):
  * a weight matrix has shape (O, K) = (out_features, in_features);
  * a type block is (block_m rows) x (block_k columns);
  * tile scores are flattened row-major over (O // block_m, K // block_k).
"""
import math

import torch

N8 = (8, 64)
N16 = (16, 64)


def check_type_block(type_block):
    if not isinstance(type_block, (tuple, list)) or len(type_block) != 2:
        raise ValueError(f'type_block must be a (rows, cols) pair, got {type_block!r}')
    bm, bk = int(type_block[0]), int(type_block[1])
    if bm <= 0 or bk <= 0 or bm != type_block[0] or bk != type_block[1]:
        raise ValueError(f'type_block dimensions must be positive integers, got {type_block!r}')
    return bm, bk


def grid_shape(weight_shape, type_block):
    bm, bk = check_type_block(type_block)
    if len(weight_shape) != 2:
        raise ValueError(f'expected a 2-D weight shape, got {tuple(weight_shape)}')
    o, k = int(weight_shape[0]), int(weight_shape[1])
    if o % bm or k % bk:
        raise ValueError(f'weight shape {(o, k)} is not divisible by type block {(bm, bk)}')
    return o // bm, k // bk


def apply_mask(base, alt, mask, type_block=N8):
    """Tile-wise select `alt` where mask is True. Bitwise identical to the archived 8x64 helper."""
    if base.ndim != 2 or alt.ndim != 2:
        raise ValueError(f'base/alt must be 2-D, got ranks {base.ndim}/{alt.ndim}')
    if base.shape != alt.shape:
        raise ValueError(f'base shape {tuple(base.shape)} != alt shape {tuple(alt.shape)}')
    if base.dtype != alt.dtype:
        raise ValueError(f'base dtype {base.dtype} != alt dtype {alt.dtype}')
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError('mask must be a torch.bool tensor')
    if mask.ndim != 2:
        raise ValueError(f'mask must be 2-D, got rank {mask.ndim}')
    gs = grid_shape(base.shape, type_block)
    if tuple(mask.shape) != gs:
        raise ValueError(f'mask shape {tuple(mask.shape)} != tile grid {gs} for weight {tuple(base.shape)} and block {tuple(type_block)}')
    bm, bk = type_block
    expanded = mask.to(base.device).repeat_interleave(bm, 0).repeat_interleave(bk, 1)
    return torch.where(expanded, alt, base)


def tile_sum(values, type_block):
    """Sum a (O, K) tensor over tiles -> flattened (O//bm)*(K//bk) vector (row-major)."""
    bm, bk = check_type_block(type_block)
    go, gk = grid_shape(values.shape, type_block)
    return values.reshape(go, bm, gk, bk).sum((1, 3)).flatten()


def directional_scores(grad, direction, type_block):
    """Per-tile directional derivative <grad_W, D> over one tile, as in the archived hook."""
    if grad.shape != direction.shape:
        raise ValueError('gradient and direction shapes differ')
    return tile_sum(grad * direction, type_block)


def aggregate_n8_to_n16(scores, out_features, in_features):
    """Per-sequence N8 tile scores [S, (O/8)(K/64)] -> N16 [S, (O/16)(K/64)] by summing vertical pairs.

    This is the only permitted N16 construction: sum child scores per sequence, then compute
    moments. Child SEs are never combined and N8 masks are never OR/AND-ed.
    """
    if scores.ndim != 2:
        raise ValueError('scores must be [sequences, tiles]')
    if out_features % 16 or in_features % 64:
        raise ValueError(f'matrix {(out_features, in_features)} is not N16K64-divisible')
    s = scores.shape[0]
    if scores.shape[1] != (out_features // 8) * (in_features // 64):
        raise ValueError('score width does not match the N8K64 tile grid')
    x = scores.reshape(s, out_features // 8, in_features // 64)
    x = x.reshape(s, out_features // 16, 2, in_features // 64)
    return x.sum(dim=2).reshape(s, -1)


def aggregate_rows_1d(values, out_features, in_features, factor=2, child=N8):
    """Vector version (one sequence or a moment that is additive) for [ (O/8)(K/64) ] inputs."""
    return aggregate_n8_to_n16(values.reshape(1, -1), out_features, in_features).reshape(-1)


class Moments:
    """Exact float64 running sufficient statistics over sequences for two objectives."""

    def __init__(self, tiles, device='cpu'):
        z = lambda: torch.zeros(tiles, dtype=torch.float64, device=device)
        self.n = 0
        self.ce_sum, self.ce_sq, self.kl_sum, self.kl_sq, self.cross = z(), z(), z(), z(), z()

    def update(self, ce, kl):
        c, k = ce.double(), kl.double()
        self.n += 1
        self.ce_sum += c
        self.ce_sq += c * c
        self.kl_sum += k
        self.kl_sq += k * k
        self.cross += c * k

    def add(self, other):
        self.n += other.n
        for a in ('ce_sum', 'ce_sq', 'kl_sum', 'kl_sq', 'cross'):
            getattr(self, a).add_(getattr(other, a))
        return self

    def state(self):
        return dict(n=self.n, ce_sum=self.ce_sum, ce_sq=self.ce_sq, kl_sum=self.kl_sum, kl_sq=self.kl_sq, cross=self.cross)

    @classmethod
    def from_state(cls, st):
        m = cls(0)
        m.n = int(st['n'])
        for a in ('ce_sum', 'ce_sq', 'kl_sum', 'kl_sq', 'cross'):
            setattr(m, a, st[a].double())
        return m

    @classmethod
    def from_raw(cls, ce, kl):
        """Accumulate in sequence order so raw and streamed paths are bitwise identical."""
        m = cls(ce.shape[1], device=ce.device)
        for i in range(ce.shape[0]):
            m.update(ce[i], kl[i])
        return m

    def mean_se(self, which):
        s, q = (self.ce_sum, self.ce_sq) if which == 'ce' else (self.kl_sum, self.kl_sq)
        n = self.n
        if n < 2:
            raise ValueError('need at least two sequences')
        mean = s / n
        var = ((q - s * s / n) / (n - 1)).clamp_min(0.0)
        return mean, (var / n).sqrt()

    def correlation(self):
        n = self.n
        mc, mk = self.ce_sum / n, self.kl_sum / n
        cov = (self.cross - n * mc * mk) / (n - 1)
        vc = ((self.ce_sq - self.ce_sum * self.ce_sum / n) / (n - 1)).clamp_min(0.0)
        vk = ((self.kl_sq - self.kl_sum * self.kl_sum / n) / (n - 1)).clamp_min(0.0)
        denom = (vc * vk).sqrt()
        return torch.where(denom > 0, cov / denom, torch.zeros_like(cov))


RULES = ('ce_kl', 'ce_only', 'kl_only', 'mean_only')


def upper_bound(moments, k, rule='ce_kl'):
    """Selection statistic; a tile is elected iff the returned value is < 0."""
    if rule not in RULES:
        raise ValueError(f'unknown rule {rule!r}')
    mc, sc = moments.mean_se('ce')
    mk, sk = moments.mean_se('kl')
    if rule == 'ce_only':
        return mc + k * sc
    if rule == 'kl_only':
        return mk + k * sk
    if rule == 'mean_only':
        return torch.maximum(mc, mk)
    return torch.maximum(mc + k * sc, mk + k * sk)


def elect(moments, k, rule='ce_kl'):
    u = upper_bound(moments, k, rule)
    if not torch.isfinite(u).all():
        raise ValueError('non-finite selection statistic')
    return u < 0


def archived_upper_scores(ce, kl, k):
    """The archived float32 formula from run_kse_paper.upper_scores (historical protocol only)."""
    out = []
    for x in (ce, kl):
        out.append(x.mean(0) + k * x.std(0, unbiased=True) / math.sqrt(x.shape[0]))
    return torch.maximum(*out)
