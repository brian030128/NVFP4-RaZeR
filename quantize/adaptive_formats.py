"""IF4 (Cook et al.) and MixFP4 (Zou et al.) weight quantizers: per 16-element block, or elected per hardware tile.

Experiment A (results/paper_extra/A/PROTOCOL.md, check in A1_CHECK.md). Both formats pick, per 16-element block, the
candidate with the lower squared error; this module implements the originals exactly, and a coarse rule that elects
one candidate per tile of rows x cols elements.

IF4 (arXiv 2603.28765; official code github.com/mit-han-lab/fouroversix @ dadfad69,
src/fouroversix/quantize/pytorch/reference.py: compute_nv_scale_factors, fake_quantize_to_e2m1, fake_quantize_to_int4,
select_intfloat), reproduced operation by operation:
  alpha = amax / (6 * 448); one block scale s = e4m3(bmax / 6 * (6 * 448 / amax)) for both candidates;
  x_b = x / ((amax / (6 * 448)) * s)  (0 where s == 0);
  FP4 candidate: E2M1, round to nearest even (the official piecewise torch.round);
  INT4 candidate: round(clamp(x_b * 1.16666666, -7, 7)) (half to even), dequantized x 0.8571428571 (the official
  truncated 7/6 and 6/7 constants);
  error: sum of squared errors per block in the original units; INT4 only when strictly lower (ties keep FP4).
MixFP4 (arXiv 2605.31035, Algorithm 1; no official code):
  s32 = amax / 2688; x8 = x / s32; E2M1 scale e4m3(bmax / 6), E1M2 scale e4m3(bmax / 7) with the x2 remap to the
  integers -7..7; both rounded to nearest even; error: squared error per block (in x8 units); E2M1 only when strictly
  lower (ties go to E1M2, Algorithm 1's else branch). E4M3 underflow is unspecified in the paper: a zero scale gives a
  zero block (as the official IF4 code does), and such blocks are counted.
Zou rule + FourOverSix E2M1 (rule zoufo6; a user-requested arm that separates the selection rule from the FourOverSix
base): the E2M1 candidate is the repo's FourOverSix (quant_nvfp4_4over6: per block the max/6 or max/4 scale, whichever
has the lower squared error; the E2M1 base of the TM-OPT+TC maps); the uniform candidate is Zou's E1M2 path above; per
block (or tile) the lower squared error, ties to the uniform format as in Zou. quant_nvfp4_4over6 returns BF16, so
both candidates' errors are taken on the weights as installed (BF16), in the original units.
E2M1 bases (never the uniform format), each rule's own base for experiment A's gain comparison: rule e2m1 is IF4's FP
candidate everywhere (E2M1 at scale e4m3(bmax / 6), i.e. NVFP4 weights, round to nearest even); rule e2m1zou is Zou's
E2M1 candidate everywhere. They differ only in the operation order of the scale and dequantization (0.33 % of BF16
elements on Llama-3.1-8B, a1_check.json).
Tiles (the coarse rule): for each candidate, the squared errors of all elements of a tile are summed, and the tile
takes the candidate with the smaller sum, with the original rule's tie-breaking; every 16-block inside keeps that
candidate's own scale rule. Rows are padded with zeros to a whole tile (zero error for both candidates).
"""
import torch

IF4_INT_EXPANSION = 0.8571428571       # 6 / 7, as fouroversix kernels/constants.py
IF4_INT_EXPANSION_RCP = 1.16666666     # 7 / 6
RULES = ('if4', 'zou', 'zoufo6', 'e2m1', 'e2m1zou')


def e2m1_rne(x):
    """fouroversix fake_quantize_to_e2m1 (round_style nearest): E2M1 values, round to nearest even."""
    a = x.abs()
    step1 = torch.round(2 * a) / 2
    step2 = torch.round(a)
    step3 = 2 * torch.round(a / 2)
    m1 = a < 2
    m2 = a < 4
    return x.sign() * (step1 * m1 + step2 * (~m1) * m2 + step3 * (~m1) * (~m2))


def _e4m3(t):
    return t.to(torch.float8_e4m3fn).to(torch.float32)


def candidates(w, rule):
    """The two candidates of every 16-element block of a 2-D weight [rows, cols] (cols % 16 == 0).

    Returns (dequantized FP candidate, dequantized uniform candidate, squared error per block of each, zero-scale
    blocks), all float32; the per-block errors are [rows, cols / 16]."""
    rows, cols = w.shape
    x = w.float().reshape(-1, 16)
    amax = w.float().abs().max()
    if rule == 'if4':
        if amax == 0:
            s = torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)
        else:
            encode = torch.tensor(6 * 448 * 1, dtype=torch.float32, device=x.device) / amax
            s = _e4m3(x.abs().max(dim=-1).values / torch.tensor(6 * 1, dtype=torch.float32, device=x.device) * encode)
        decode = 1 / (torch.tensor(6 * 448 * 1, dtype=torch.float32, device=x.device) / amax)
        xb = torch.where(s.unsqueeze(1) != 0, x * (1 / (decode * s.unsqueeze(1))), 0)
        q_fp = e2m1_rne(xb)
        q_int = (xb * IF4_INT_EXPANSION_RCP).clamp(min=-7, max=7).round()
        d_fp = (q_fp * s.unsqueeze(1) * amax) / (6 * 448 * 1)
        d_int = ((q_int * s.unsqueeze(1) * amax) * IF4_INT_EXPANSION) / (6 * 448 * 1)
        e_fp = ((d_fp - x) ** 2).sum(dim=-1)
        e_int = ((d_int - x) ** 2).sum(dim=-1)
        zero = int(((s == 0) & (x.abs().max(dim=-1).values != 0)).sum())      # underflow of a nonzero block
    elif rule == 'zou':
        s32 = amax / 2688
        x8 = x / s32 if amax != 0 else torch.zeros_like(x)
        bmax = x8.abs().max(dim=-1).values
        s_fp, s_int = _e4m3(bmax / 6), _e4m3(bmax / 7)
        q_fp = e2m1_rne(torch.where(s_fp.unsqueeze(1) != 0, x8 / s_fp.unsqueeze(1), 0))
        q_int = torch.where(s_int.unsqueeze(1) != 0, x8 / s_int.unsqueeze(1), 0).round().clamp(min=-7, max=7)
        d8_fp, d8_int = q_fp * s_fp.unsqueeze(1), q_int * s_int.unsqueeze(1)
        e_fp = ((d8_fp - x8) ** 2).sum(dim=-1)
        e_int = ((d8_int - x8) ** 2).sum(dim=-1)
        d_fp, d_int = d8_fp * s32, d8_int * s32
        zero = int(((s_fp == 0) & (bmax != 0)).sum() + ((s_int == 0) & (bmax != 0)).sum())
    elif rule in ('e2m1', 'e2m1zou'):
        d_fp, _, e_fp, _, zero = candidates(w, 'if4' if rule == 'e2m1' else 'zou')
        d_fp, e_fp = d_fp.reshape(-1, 16), e_fp.reshape(-1)
        d_int, e_int = d_fp, torch.full_like(e_fp, float('inf'))       # never chosen
    elif rule == 'zoufo6':
        from quantize.quantizer import quant_nvfp4_4over6
        d_fp = quant_nvfp4_4over6(w, 4, 16).float().reshape(-1, 16)
        _, d_zou, _, _, zero_zou = candidates(w, 'zou')
        d_int = d_zou.to(w.dtype).float().reshape(-1, 16)          # both candidates as installed
        e_fp = ((d_fp - x) ** 2).sum(dim=-1)
        e_int = ((d_int - x) ** 2).sum(dim=-1)
        zero = zero_zou
    else:
        raise ValueError(rule)
    return (d_fp.reshape(rows, cols), d_int.reshape(rows, cols), e_fp.reshape(rows, cols // 16),
            e_int.reshape(rows, cols // 16), zero)


def select(e_fp, e_int, rule, tile=None):
    """Blocks [rows, cols / 16] that take the uniform (INT4 / E1M2) candidate: per block (tile None or (1, 16)), or
    elected per tile (rows_t, cols_t) from the summed squared errors; ties follow the original rule."""
    uniform_wins = (lambda a, b: a < b) if rule == 'if4' else (lambda a, b: a <= b)   # a = uniform error, b = FP
    if tile is None or tuple(tile) == (1, 16):
        return uniform_wins(e_int, e_fp)
    tr, tc = tile
    assert tc % 16 == 0, tile
    rows, kb = e_fp.shape
    bc = tc // 16
    assert kb % bc == 0, (e_fp.shape, tile)
    pad = (-rows) % tr

    def tiles(e):
        e = torch.nn.functional.pad(e, (0, 0, 0, pad))
        return e.reshape((rows + pad) // tr, tr, kb // bc, bc).sum(dim=(1, 3))
    pick = uniform_wins(tiles(e_int), tiles(e_fp))
    return pick.repeat_interleave(tr, 0)[:rows].repeat_interleave(bc, 1)


MIXING_TILES = ((8, 64), (16, 64), (256, 64))


def mixing(uni, tiles=MIXING_TILES):
    """Within-tile mixing of per-block choices uni [rows, cols / 16] (True = uniform format): per tile shape, the tiles,
    the tiles whose blocks disagree, and the sum over those of the minority share (min(uniform, E2M1) / blocks).
    Rows padded to a whole tile are left out of the counts (the partial last tile counts its real blocks only)."""
    rows, kb = uni.shape
    out = {}
    for tr, tc in tiles:
        bc = tc // 16
        pad = (-rows) % tr
        u = torch.nn.functional.pad(uni.float(), (0, 0, 0, pad)).reshape((rows + pad) // tr, tr, kb // bc, bc)
        real = torch.nn.functional.pad(torch.ones_like(uni, dtype=torch.float32), (0, 0, 0, pad)).reshape(u.shape)
        n_uni, n = u.sum(dim=(1, 3)), real.sum(dim=(1, 3))
        mixed = (n_uni > 0) & (n_uni < n)
        minority = torch.minimum(n_uni, n - n_uni) / n
        out[f'{tr}x{tc}'] = dict(tiles=int(n.numel()), mixed=int(mixed.sum()), minority_sum=float(minority[mixed].sum()))
    return out


@torch.no_grad()
def quantize(w, rule, tile=None):
    """-> (dequantized weight in w's dtype, statistics). tile: None / (1, 16) per block, else (rows, cols).
    sq_error is the total squared error of the installed weight (in w's dtype) against w; for the per-block rule the
    statistics also hold the within-tile mixing of the choices at 8x64, 16x64 and 256x64."""
    d_fp, d_int, e_fp, e_int, zero = candidates(w, rule)
    uni = select(e_fp, e_int, rule, tile)
    out = torch.where(uni.repeat_interleave(16, 1), d_int, d_fp).to(w.dtype)
    stats = dict(blocks=uni.numel(), uniform_blocks=int(uni.sum()), zero_scale_blocks=zero,
                 sq_error=float(((out.double() - w.double()) ** 2).sum()))
    if tile is None or tuple(tile) == (1, 16):
        stats['mixing'] = mixing(uni)
    return out, stats
