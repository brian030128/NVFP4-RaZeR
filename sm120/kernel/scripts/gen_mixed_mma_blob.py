#!/usr/bin/env python3
"""Emit one k_block of mixed-format MMAs as a SINGLE opaque inline-PTX blob per pattern.

WHAT THIS IS FOR
----------------
The cheap dispatch is the C++ one: a branch that encloses a whole k_tile costs ~2.4 cycles per
level, because it sits outside the loop body and so does not stop ptxas interleaving the next
k_block's LDSM loads with this k_block's MMAs. (A dispatch *inside* the body costs ~90 cycles --
see scripts/gen_mixed_mma_ptx.py.) Its limit was arm count: past 8 arms cicc stopped inlining the
specialized k_tile body, outlined it into an ABI call and spilled the accumulators to a 912-byte
frame, taking 1166 -> 33 TFLOP/s. docs/mixed_nvfp4_report.md recorded that cliff as resisting
every lever tried: -inline-threshold, __attribute__((always_inline)), and halving the code.

This moves it. The lever it uses is not code size but *statement count*: emitting a k_block's 16
MMAs as one opaque asm blob rather than 16 separate cute::gemm statements shrinks what cicc sees
for a k_tile body from 32 inline-asm statements to 2. Measured on an RTX 5090:

    arms   granule       C++ + cute::gemm        C++ + blob
      8    32x32x128     clean, 1167 TFLOP/s     clean
     16    16x32x128     STACK:912, 36.7         clean, STACK:0, 1116
     32    32x16x128     (not reached)           clean, STACK:0, 1084
     64    16x16x128     (not reached)           STACK:864, outlined

So the cliff moves from 8 arms to between 32 and 64 -- a 4x larger arm budget for the cheap
dispatch. Note the throughput still falls with arm count (code footprint), so 16 and 32 arms cost
7.5% and 10.2% against stock NVFP4 rather than the 3.3% that 8 arms costs.

K granularity stays at 128 here: the dispatch encloses the whole k_tile, which is exactly why it
is cheap. For K=64 use gen_mixed_mma_ptx.py and pay the in-body dispatch.

    python3 scripts/gen_mixed_mma_blob.py                 # 64 arms, 16x16x128
    A_ATOMS=1 B_ATOMS=4 python3 scripts/gen_mixed_mma_blob.py   # 16 arms, 16x32x128

Then build with -DMIXFP4_BLOB=1 and matching -DMIXFP4_A/B_ATOMS_PER_GRANULE.
"""
from __future__ import annotations
import os, pathlib

# Atoms a warp covers. Derived from the CTA tile and the warp arrangement: with
# Shape<_128,_128,_128> and a 4x2 warp grid a warp owns 32 rows x 64 cols, i.e. 2 x 8 atoms.
# Halving the CTA tile's N to Shape<_128,_64,_128> makes it 2 x 4, which halves the number of B
# granules -- and so squares down the arm count for a given column granularity.
MMA_M = int(os.environ.get("MMA_M", 2))
MMA_N = int(os.environ.get("MMA_N", 8))
# Default to 16 arms (16 rows x 32 cols x 128 K): the cheapest point that is finer than the
# shipped 32x32x128 default and still well inside the moved cliff.
A_ATOMS = int(os.environ.get("A_ATOMS", 1))
B_ATOMS = int(os.environ.get("B_ATOMS", 4))
A_GRAN = MMA_M // A_ATOMS
B_GRAN = MMA_N // B_ATOMS
NPAT = 1 << (A_GRAN + B_GRAN)

# Measured on an RTX 5090 / CUDA 13.1: 32 arms still inlines cleanly, 64 outlines the k_tile body
# and spills the accumulators to a 864-byte frame. Emitting past that produces a correct but
# ~30x slower kernel, which is worth refusing to do silently.
if NPAT > 32 and os.environ.get("ALLOW_OUTLINE") != "1":
    raise SystemExit(
        "error: %d arms (a_atoms=%d, b_atoms=%d). Past 32 arms cicc outlines the k_tile body and\n"
        "       spills the accumulators -- verified at 64 arms (STACK:864). Use fewer granules,\n"
        "       or scripts/gen_mixed_mma_ptx.py if you need K=64 granularity.\n"
        "       Set ALLOW_OUTLINE=1 to emit it anyway (correct, but ~30x slower)."
        % (NPAT, A_ATOMS, B_ATOMS))

SEL = ["0x3210", "0x3214", "0x3254", "0x3654"]
# [NVFP4-RaZeR local change, kernel-opt t0; see sm120/kernel/LOCAL_CHANGES.md] Two options, both defaulting to the
# original blob (byte-identical output):
# TAG0: '0' emits site-0 (e2m1 x e2m1) MMAs without the identity prmt tag; they read the scale word directly. Site 0 is
#   never patched, and the tags cost one PRMT per (m-atom, k_block) and k_tile in every arm -- the whole gap between
#   the no-dispatch kernel and stock NVFP4. The E0M3 sites keep their tags; patch with --untagged-site0.
# ORDER: the MMA issue order inside a k_block. 'm' is the m-major serpentine below. 'n' is the order of CUTLASS's
#   cute::gemm, as in stock: n outer, m serpentine inside, so consecutive MMAs share B and its scale word and ptxas can
#   mark them .reuse. Measured without effect (results/kernel_opt/t0/exploration); kept for that diagnostic. Every
#   accumulator still receives its MMAs in the same k order, so outputs are unchanged.
ORDER = os.environ.get("ORDER", "m")
TAG0 = os.environ.get("TAG0", "1") == "1"
assert ORDER in ("m", "n"), ORDER
# [NVFP4-RaZeR local change, kernel-opt K-tile ablation; see sm120/kernel/LOCAL_CHANGES.md] BRANCH: 'a' or 'b' appends
# mma_kblock_branch() to the header: the same k_block of MMAs, in the same order and with the same per-MMA PTX as the
# blob arms, but each MMA's format chosen at run time by an ordinary C++ if/else on the format flag of the weight
# operand's granule ('a': A's, sites 0/1; 'b': B's, sites 0/2) -- the per-MMA branch of docs/mixed_nvfp4_report.md
# section 1, which ptxas if-converts into predicated OMMA pairs. The mainloop uses it with -DMIXFP4_PER_MMA_BRANCH=1.
# Unset, the output is the blob header byte for byte.
BRANCH = os.environ.get("BRANCH", "")
assert BRANCH in ("", "a", "b"), BRANCH
# Inline-asm operand numbering, derived from the warp tile: accumulators first (4 per MMA), then
# the A fragments (4 regs per m-atom), B (2 per n-atom), and one scale word per atom.
N_OUT = 4 * MMA_M * MMA_N
A_BASE = N_OUT
B_BASE = A_BASE + 4 * MMA_M
SFA_BASE = B_BASE + 2 * MMA_N
SFB_BASE = SFA_BASE + MMA_M
MMA = ("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
       "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3")


def acc(m, n, v):
    return (m * MMA_N + n) * 4 + v


def serpentine(i):
    if ORDER == "n":
        n = i // MMA_M
        j = i % MMA_M
        return (MMA_M - 1 - j) if (n & 1) else j, n
    m = i // MMA_N
    j = i % MMA_N
    return m, (MMA_N - 1 - j) if (m & 1) else j


def emit_pattern(p):
    out = []
    for i in range(MMA_M * MMA_N):
        m, ns = serpentine(i)
        a_flag = (p >> (m // A_ATOMS)) & 1
        b_flag = (p >> (A_GRAN + ns // B_ATOMS)) & 1
        site = a_flag | (b_flag << 1)
        d = [acc(m, ns, v) for v in range(4)]
        a = [A_BASE + m * 4 + v for v in range(4)]
        b = [B_BASE + ns * 2 + v for v in range(2)]
        if site == 0 and not TAG0:
            sfa = "%%%d" % (SFA_BASE + m)
        else:
            out.append("    prmt.b32 %%sf%d, %%%d, %%%d, %s;"
                       % (i, SFA_BASE + m, SFA_BASE + m, SEL[site]))
            sfa = "%%sf%d" % i
        out.append(
            "    %s {%%%d,%%%d,%%%d,%%%d}, {%%%d,%%%d,%%%d,%%%d}, {%%%d,%%%d}, "
            "{%%%d,%%%d,%%%d,%%%d}, {%s}, {0, 0}, {%%%d}, {0, 0};"
            % (MMA, d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b[0], b[1],
               d[0], d[1], d[2], d[3], sfa, SFB_BASE + ns))
    return out


def emit_branch():
    """mma_kblock_branch(): one asm statement per MMA, in the blob's issue order, behind a C++ if/else on the weight
    operand's per-granule flag (bit 7 of the granule's first atom's scale word, as the mainloop's read_site)."""
    body = []
    for i in range(MMA_M * MMA_N):
        m, ns = serpentine(i)
        if BRANCH == "a":
            flag = "sfa(cute::Int<0>{}, cute::Int<%d>{})" % ((m // A_ATOMS) * A_ATOMS)
            site_w = 1
        else:
            flag = "sfb(cute::Int<0>{}, cute::Int<%d>{})" % ((ns // B_ATOMS) * B_ATOMS)
            site_w = 2
        outs = ", ".join('"+f"(acc(cute::Int<%d>{}, cute::Int<%d>{}, cute::Int<%d>{}))' % (v, m, ns) for v in range(4))
        ins = ", ".join(['"r"(a(cute::Int<%d>{}, cute::Int<%d>{}))' % (v, m) for v in range(4)]
                        + ['"r"(b(cute::Int<%d>{}, cute::Int<%d>{}))' % (v, ns) for v in range(2)]
                        + ['"r"(sfa(cute::Int<0>{}, cute::Int<%d>{}))' % m, '"r"(sfb(cute::Int<0>{}, cute::Int<%d>{}))' % ns])

        def one(site):
            # operands: %0-%3 accumulators, %4-%7 A, %8-%9 B, %10 SFA, %11 SFB
            asm = ["{"]
            if site == 0 and not TAG0:
                sfa = "%10"
            else:
                asm += ["  .reg .b32 %sf;", "  prmt.b32 %%sf, %%10, %%10, %s;" % SEL[site]]
                sfa = "%sf"
            asm.append("  %s {%%0,%%1,%%2,%%3}, {%%4,%%5,%%6,%%7}, {%%8,%%9}, {%%0,%%1,%%2,%%3}, {%s}, {0, 0}, {%%11}, "
                       "{0, 0};" % (MMA, sfa))
            asm.append("}")
            text = "\n".join('        "%s\\n"' % ln for ln in asm)
            return "      asm volatile(\n%s\n        : %s\n        : %s);" % (text, outs, ins)
        body.append("    if ((%s & 0x80u) != 0u) {\n%s\n    } else {\n%s\n    }" % (flag, one(site_w), one(0)))
    return (
        "// Per-MMA branch (BRANCH=%s): each MMA's format chosen at run time by its own C++ if/else.\n"
        "#define MIXFP4_BLOBGEN_BRANCH %d\n"
        "namespace mixfp4 {\n"
        "template <class TAcc, class TA, class TB, class TSFA, class TSFB>\n"
        "CUTLASS_DEVICE void\n"
        "mma_kblock_branch(TAcc& acc, TA const& a, TB const& b, TSFA const& sfa, TSFB const& sfb) {\n"
        "%s\n"
        "}\n"
        "} // namespace mixfp4\n" % (BRANCH, 1 if BRANCH == "a" else 2, "\n".join(body)))


def main():
    outs = ",\n".join(
        '        "+f"(acc(cute::Int<%d>{}, cute::Int<%d>{}, cute::Int<%d>{}))' % (v, m, n)
        for m in range(MMA_M) for n in range(MMA_N) for v in range(4))
    ins = []
    for m in range(MMA_M):
        for v in range(4):
            ins.append('        "r"(a(cute::Int<%d>{}, cute::Int<%d>{}))' % (v, m))
    for n in range(MMA_N):
        for v in range(2):
            ins.append('        "r"(b(cute::Int<%d>{}, cute::Int<%d>{}))' % (v, n))
    for m in range(MMA_M):
        ins.append('        "r"(sfa(cute::Int<0>{}, cute::Int<%d>{}))' % m)
    for n in range(MMA_N):
        ins.append('        "r"(sfb(cute::Int<0>{}, cute::Int<%d>{}))' % n)
    ins_s = ",\n".join(ins)

    arms = []
    for pat in range(NPAT):
        asm = ["{", "  .reg .b32 %%sf<%d>;" % (MMA_M * MMA_N)] + emit_pattern(pat) + ["}"]
        body = "\n".join('      "%s\\n"' % ln for ln in asm)
        kw = "if" if pat == 0 else "else if"
        arms.append("  %s constexpr (P == %d) {\n    asm volatile(\n%s\n      :\n%s\n      :\n%s);\n  }"
                    % (kw, pat, body, outs, ins_s))

    header = (
        "// GENERATED by gen_blob.py -- do not edit.\n"
        "// One k_block of mixed-format MMAs as a SINGLE opaque inline-PTX blob whose formats are\n"
        "// all fixed at compile time by P. No dispatch inside; the caller's per-k_tile C++\n"
        "// pattern specialization picks P.\n"
        "//\n"
        "// Granule: %d rows of A x %d columns of B x 128 K (one k_tile), %d arms.\n"
        "// Built for a warp tile of %d x %d atoms; the mainloop static_asserts this.\n"
        "#pragma once\n"
        "#define MIXFP4_BLOBGEN_MMA_M %d\n"
        "#define MIXFP4_BLOBGEN_MMA_N %d\n"
        "#include \"cute/tensor.hpp\"\n"
        "namespace mixfp4 {\n"
        "template <uint32_t P, class TAcc, class TA, class TB, class TSFA, class TSFB>\n"
        "CUTLASS_DEVICE void\n"
        "mma_kblock_blob(TAcc& acc, TA const& a, TB const& b, TSFA const& sfa, TSFB const& sfb) {\n"
        "%s\n"
        "}\n"
        "} // namespace mixfp4\n"
        % (A_ATOMS * 16, B_ATOMS * 8, NPAT, MMA_M, MMA_N, MMA_M, MMA_N, "\n".join(arms)))

    if BRANCH:
        header += emit_branch()
    dst = (pathlib.Path(__file__).resolve().parent.parent
           / "src/collective/mixed_mma_blob_generated.hpp")
    dst.write_text(header)
    print("wrote %s: %d arms x %d MMAs, granule %d x %d x 128"
          % (dst.name, NPAT, MMA_M * MMA_N, A_ATOMS * 16, B_ATOMS * 8))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
