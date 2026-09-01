"""
    What does reordering actually cost?

    Two costs, and they are very different in kind.

    OFFLINE -- the co-clustering search runs once per weight tensor at quantization time. Measured
    from the perplexity logs it is ~22 minutes for a full Llama-3.1-8B pass against ~30 seconds
    without it, so it is a one-off quantization expense, not a serving expense.

    RUNTIME -- this is the one that decides deployability. Permuting the reduction axis of W means
    the activation entering that GEMM must arrive in the same order. Whether that is free depends
    entirely on WHICH axis (ALGORITHM.md §7):

      free, absorbed offline:
        down_proj columns   -- the FF intermediate axis, absorbed into gate/up_proj's rows
        o_proj  columns     -- within a head, absorbed into v_proj's rows
        one GLOBAL residual permutation -- absorbed into RMSNorm gains, embedding and lm_head,
                                          but then shared by q/k/v/gate/up across EVERY layer

      NOT free:
        a per-layer, per-matrix column permutation of q/k/v/gate/up. Those all read the same
        residual stream, so giving each its own order needs a runtime gather on the activation.

    `coclcol` as measured permutes every matrix independently, so on 5 of 7 matrices it implies a
    gather per layer. This benchmarks that gather against the GEMM it precedes, which is the ratio
    that decides whether the measured -0.007 to -0.011 wikitext is affordable.
"""
import argparse
import time

import torch


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3          # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    args = ap.parse_args()

    dt = getattr(torch, args.dtype)
    dev = "cuda"
    torch.manual_seed(0)

    # (name, in_features, out_features) for a Llama-3.1-8B block
    shapes = [
        ("q_proj",    4096,  4096),
        ("k_proj",    4096,  1024),
        ("v_proj",    4096,  1024),
        ("o_proj",    4096,  4096),
        ("gate_proj", 4096, 14336),
        ("up_proj",   4096, 14336),
        ("down_proj", 14336, 4096),
    ]

    print(f"batch={args.batch} seq={args.seq} dtype={args.dtype} on {torch.cuda.get_device_name()}\n")
    print(f"{'matrix':>10} {'K':>7} {'N':>7} {'gather ms':>10} {'gemm ms':>9} {'overhead':>9} "
          f"{'free?':>6}")

    free = {"down_proj": "yes", "o_proj": "yes"}          # absorbable per layer, see module docstring
    tot_g = tot_m = 0.0
    for name, k, n in shapes:
        x = torch.randn(args.batch, args.seq, k, device=dev, dtype=dt)
        w = torch.randn(n, k, device=dev, dtype=dt)
        perm = torch.randperm(k, device=dev)

        g_ms = bench(lambda: x.index_select(-1, perm))
        m_ms = bench(lambda: torch.nn.functional.linear(x, w))
        tot_g += g_ms
        tot_m += m_ms
        print(f"{name:>10} {k:>7} {n:>7} {g_ms:10.3f} {m_ms:9.3f} {100 * g_ms / m_ms:8.1f}% "
              f"{free.get(name, 'NO'):>6}")

    print(f"\n{'ALL 7':>10} {'':>7} {'':>7} {tot_g:10.3f} {tot_m:9.3f} "
          f"{100 * tot_g / tot_m:8.1f}%")

    # The deployable subset: only the two matrices whose column permutation is absorbed offline
    # contribute no runtime cost at all; the other five would each need a gather.
    paid = sum(bench(lambda x=torch.randn(args.batch, args.seq, k, device=dev, dtype=dt),
                     p=torch.randperm(k, device=dev): x.index_select(-1, p))
               for name, k, _ in shapes if name not in free)
    print(f"\ngather cost for the 5 matrices that are NOT absorbable: {paid:.3f} ms per layer")
    print(f"as a share of the block's GEMM time: {100 * paid / tot_m:.1f}%")
    print("\nA per-layer per-matrix column permutation is not free. Only down_proj and o_proj")
    print("absorb theirs; q/k/v/gate/up share the residual stream and admit ONE global order.")


if __name__ == "__main__":
    main()
