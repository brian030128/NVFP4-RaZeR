import torch
from quantize.modulewise_selection import modulewise_maps
from quantize.relinearized_format import stale_map


def main():
    torch.manual_seed(20260928)
    ce = torch.randn(192, 400) * .01 - .02
    kl = torch.randn(192, 400) * .01 - .02
    # Exact ties span module boundaries; later tiles are ineligible.
    ce[:, 0:12] = -.03; kl[:, 0:12] = -.03
    ce[:, -10:] = 1.; kl[:, -10:] = 1.
    ranges = {'first': (0, 7), 'second': (7, 157), 'third': (157, 400)}
    tables = {n: (ce[:, lo:hi].contiguous(), kl[:, lo:hi].contiguous()) for n, (lo, hi) in ranges.items()}
    mixed = list(range(22)) + list(range(64, 85)) + list(range(128, 149))
    for count in (5, 256, 500):
        maps = modulewise_maps(tables, list(ranges), count)
        expected = {'c4_64': stale_map(ce[:64], kl[:64], count),
                    'mixed64': stale_map(ce[mixed], kl[mixed], count),
                    'pooled192': stale_map(ce, kl, count)}
        for p in maps:
            assert torch.equal(torch.cat(list(maps[p].values())), expected[p]), p
    print('Modulewise and full-table selection match, including cross-module ties and eligibility')


if __name__ == '__main__': main()
