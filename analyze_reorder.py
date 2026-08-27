"""
    Summarize a `run_reorder_sim.py --diagnostics_only` CSV against the i.i.d. noise expectation.

    The two-way decomposition G = mu + a_i + b_j + e_ij assigns a share of the variance to the row
    effects and to the column effects even when there are none: for an M x N grid of i.i.d. cells,

        E[row_share] = (M - 1) / (MN - 1)        E[col_share] = (N - 1) / (MN - 1)

    which for a 4096 x 256 tag grid is 0.0039 and 0.0002. So a raw "row effects are 2% of the
    variance" is not interpretable on its own -- it has to be read against that baseline. This
    prints both, per projection.

        python analyze_reorder.py results/reorder/diag_llama-3.1-8b_heade0.csv
"""
import csv
import re
import statistics as st
import sys


def main(path):
    rows = list(csv.DictReader(open(path)))
    by_proj = {}

    for r in rows:
        m = re.search(r"\((\d+), (\d+)\)", r["tensor"])
        if m is None:
            continue
        M, K = int(m.group(1)), int(m.group(2))
        N = K // 16
        cells = M * N
        rec = dict(
            row=float(r["row_share"]),          col=float(r["col_share"]),
            row_noise=(M - 1) / (cells - 1),    col_noise=(N - 1) / (cells - 1),
            resid=float(r["resid_share"]),      rank1=float(r["rank1_sign_fit"]),
            pos=float(r["pos_share"]),
        )
        by_proj.setdefault(r["tensor"].split()[0].split(".")[-1], []).append(rec)

    print(f"{len(rows)} tensors from {path}\n")
    hdr = (f"{'projection':>11} {'row':>7} {'noise':>7} {'x':>5} {'col':>7} {'noise':>7} {'x':>5} "
           f"{'row+col':>8} {'resid':>7} {'rank1':>6} {'pos':>6}")
    print(hdr)
    print("-" * len(hdr))

    every = []
    for proj, v in sorted(by_proj.items(), key=lambda kv: -len(kv[1])):
        every += v
        _line(proj, v)
    print("-" * len(hdr))
    _line("ALL", every)

    a = lambda k: st.mean(x[k] for x in every)
    print(f"\nRow and column effects together carry {100 * (a('row') + a('col')):.1f}% of the "
          f"variance in the E0M3 preference.")
    print(f"A reordering can only exploit that part; the remaining "
          f"{100 * a('resid'):.1f}% is idiosyncratic to the individual 16-element scale block and "
          f"is invariant to any row or column permutation.")


def _line(label, v):
    a = lambda k: st.mean(x[k] for x in v)
    print(f"{label:>11} {a('row'):7.4f} {a('row_noise'):7.4f} {a('row')/a('row_noise'):5.1f} "
          f"{a('col'):7.4f} {a('col_noise'):7.4f} {a('col')/a('col_noise'):5.1f} "
          f"{a('row')+a('col'):8.4f} {a('resid'):7.4f} {a('rank1'):6.4f} {a('pos'):6.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/reorder/diag_llama-3.1-8b_heade0.csv")
