"""
    How coarse can the `hess` importance be before it stops doing anything?

    `hess` weights each element's squared error by E[x_j^2] of the input channel it multiplies, and
    that weight is applied PER ELEMENT -- 16 different values inside one 1x16 scale block. This asks
    what survives when the weight is replaced by its mean over a run of channels.

    Two claims are provable, and this file is where they are checked rather than assumed:

    1.  A per-TYPE-BLOCK constant weight is an EXACT no-op. The alpha search compares candidates
        within a scale block and every election rule compares quantities homogeneous of the same
        degree in the loss, so a positive constant cancels out of both. `impg64` at 8x64 must
        therefore reproduce the unweighted quantizer bit for bit. If it does not, the election is
        not scale-invariant and some rule has an absolute threshold hiding in it.

    2.  A per-SCALE-BLOCK constant weight cancels out of the ALPHA SEARCH ONLY. So `impg16` must
        agree bit for bit with the unweighted quantizer when the election is disabled (`_e2m1`),
        and differ from it once the election is on.

    Together these localize `hess`: whatever it is worth, claim 1 says none of it can come from a
    tile-level envelope, and claim 2 says the part that survives block-mean coarsening is entirely
    an election effect.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantize.quantizer import quant_mix_4_6


def _w(seed=0, m=128, k=256):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, k, generator=g, dtype=torch.float32) * 0.02


def _imp(k=256, seed=1):
    g = torch.Generator().manual_seed(seed)
    # heavy-tailed, like a real E[x_j^2]: a few channels dominate
    return torch.rand(k, generator=g).pow(6) * 1000 + 1e-3


def run(dtype_suffix, w, imp, tb="8x64"):
    from quantize.quantizer import parse_mix_4_6_dtype
    name = "mix_4_6" + dtype_suffix
    (metric, elect, margin, use_imp, clip, cg, ag, perm, rot, rot_n, rot_g, rot_o, pv,
     ia, ie, ig) = parse_mix_4_6_dtype(name)
    bm, bk = (int(x) for x in tb.split("x"))
    return quant_mix_4_6(w, 4, 16, type_block=(bm, bk), metric=metric, elect=elect, margin=margin,
                         clip=clip, clip_min_gain=cg, alpha_min_gain=ag, permute=perm,
                         importance=imp if use_imp else None,
                         imp_alpha=ia, imp_elect=ie, imp_gran=ig)


def main():
    w, imp = _w(), _imp()
    ok = True

    for elect in ("_h1.5", "_m1", "_argmin", "_dom", "_v0.7"):
        e = "" if elect == "_argmin" else elect
        plain = run("_clipheadx" + e, w, imp)
        g64   = run("_clipheadx_hess_impg64" + e, w, imp)
        g16   = run("_clipheadx_hess_impg16" + e, w, imp)
        hess  = run("_clipheadx_hess" + e, w, imp)

        same64 = torch.equal(plain, g64)
        d16    = (g16.float() - plain.float()).abs().max().item()
        dh     = (hess.float() - plain.float()).abs().max().item()
        print(f"{elect:>8}  impg64==plain: {same64}   |impg16-plain|max {d16:.3e}   "
              f"|hess-plain|max {dh:.3e}")
        if not same64:
            n = (plain != g64).sum().item()
            print(f"           FAIL: per-tile importance changed {n} elements -- "
                  f"an election rule is NOT scale-invariant")
            ok = False
        if dh == 0.0:
            print("           FAIL: full hess is a no-op too; importance is not reaching the loss")
            ok = False

    # claim 2: with the election off, a per-scale-block constant cancels in the alpha search
    plain_e = run("_clipheadx_e2m1", w, imp)
    g16_e   = run("_clipheadx_hess_impg16_e2m1", w, imp)
    hess_e  = run("_clipheadx_hess_e2m1", w, imp)
    same = torch.equal(plain_e, g16_e)
    print(f"\n   e2m1 (election off)  impg16==plain: {same}   "
          f"|hess-plain|max {(hess_e.float()-plain_e.float()).abs().max().item():.3e}")
    if not same:
        print("           FAIL: a per-scale-block constant did NOT cancel in the alpha search")
        ok = False

    # and with the election on, impg16 must actually differ from plain -- otherwise the whole
    # variant is vacuous and any perplexity difference would be noise
    if (run("_clipheadx_hess_impg16_h1.5", w, imp).float()
            - run("_clipheadx_h1.5", w, imp).float()).abs().max().item() == 0.0:
        print("           FAIL: impg16 with election on is identical to plain -- vacuous variant")
        ok = False

    print()
    ok = test_alpha_tiebreak() and ok

    print("\nALL OK" if ok else "\nFAILURES ABOVE")
    return 0 if ok else 1

def test_alpha_tiebreak():
    """
        Why impg64 is bit-exact under a single-alpha preset but NOT under a multi-alpha one.

        `sum_j (c * d_j^2)` and `c * sum_j d_j^2` are not the same float. So a per-tile constant
        rescales the loss only up to rounding, and the alpha search's `err < best_err` can flip on a
        near-tie. With one candidate alpha (`clipa1`) there is no comparison to flip, so equality is
        exact; with five (`clipheadx`) it is exact only up to tie-breaking.

        This matters because it sets the NOISE FLOOR for the whole impg comparison: any difference
        this mechanism can produce is not a real effect.
    """
    w, imp = _w(m=512, k=1024).to(torch.bfloat16).float(), _imp(k=1024)
    for clip, alphas in (("a1", 1), ("headx", "many")):
        plain = run(f"_clip{clip}_m1", w, imp)
        g64   = run(f"_clip{clip}_hess_impg64_m1", w, imp)
        n = int((plain != g64).sum())
        print(f"   clip={clip:<8} ({alphas} alpha) impg64 vs plain: {n} of {plain.numel()} "
              f"elements differ  ({100*n/plain.numel():.4f}%)")
        if clip == "a1" and n:
            print("           FAIL: single-alpha preset must be bit-exact")
            return False
    return True

if __name__ == "__main__":
    raise SystemExit(main())
